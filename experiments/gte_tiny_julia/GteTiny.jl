module GteTiny
using JSON3, SHA, Unicode
export Model, Tokenizer, tokenize, encode, hidden_states

const REVISION = "4cc5e73d86a67c601897257b467187234aa3bca3"
const GRAPH_SHA = "ceab9964bcdb19ad013a2fe44bc1ddb3657ab0af1f48cddc349014d6085762ca"
const WIDTH = 384
const HEADS = 12
const HEAD_DIM = 32
const SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

struct Tokenizer
    vocab::Dict{String,Int}
end
function Tokenizer(path::AbstractString)
    t = JSON3.read(read(path, String))
    t.model.type == "WordPiece" && t.normalizer.type == "BertNormalizer" || error("unsupported tokenizer")
    t.normalizer.lowercase && t.normalizer.clean_text && t.normalizer.handle_chinese_chars || error("unsupported normalization")
    t.model.max_input_chars_per_word == 100 || error("unsupported word limit")
    Tokenizer(Dict(String(k)=>Int(v) for (k,v) in pairs(t.model.vocab)))
end

iscjk(c) = 0x4e00 <= UInt32(c) <= 0x9fff || 0x3400 <= UInt32(c) <= 0x4dbf ||
    0x20000 <= UInt32(c) <= 0x2a6df || 0x2a700 <= UInt32(c) <= 0x2b73f ||
    0x2b740 <= UInt32(c) <= 0x2b81f || 0x2b820 <= UInt32(c) <= 0x2ceaf ||
    0xf900 <= UInt32(c) <= 0xfaff || 0x2f800 <= UInt32(c) <= 0x2fa1f
category(c) = Base.Unicode.category_abbrev(c)
isbertspace(c) = c in (' ', '\t', '\n', '\r') || category(c) == "Zs"
isbertcontrol(c) = !(c in ('\t','\n','\r')) && startswith(category(c), "C")
isbertpunct(c) = (0x21 <= UInt32(c) <= 0x2f) || (0x3a <= UInt32(c) <= 0x40) ||
    (0x5b <= UInt32(c) <= 0x60) || (0x7b <= UInt32(c) <= 0x7e) || startswith(category(c), "P")

function basic_tokens(text)
    clean = IOBuffer()
    for c in text
        (c == '\0' || c == '\ufffd' || isbertcontrol(c)) && continue
        if isbertspace(c)
            print(clean, ' ')
        elseif iscjk(c)
            print(clean, ' ', c, ' ')
        else
            print(clean, c)
        end
    end
    # BertNormalizer lowercases each character, then strips nonspacing marks after NFD.
    lowered = join(lowercase(string(c)) for c in String(take!(clean)))
    normalized = Unicode.normalize(lowered, :NFD)
    out = String[]; current = IOBuffer()
    flushword() = position(current) > 0 && push!(out, String(take!(current)))
    for c in normalized
        category(c) == "Mn" && continue
        if isspace(c) || category(c) in ("Zl", "Zp")
            flushword()
        elseif isbertpunct(c)
            flushword(); push!(out, string(c))
        else
            print(current, c)
        end
    end
    flushword()
    out
end

function wordpieces!(ids, tok, word)
    chars = collect(word)
    length(chars) > 100 && (push!(ids, 100); return)
    pieces = Int[]; first = 1
    while first <= length(chars)
        found = false
        for last in length(chars):-1:first
            piece = (first == 1 ? "" : "##") * String(chars[first:last])
            if haskey(tok.vocab, piece)
                push!(pieces, tok.vocab[piece]); first = last+1; found = true; break
            end
        end
        found || (push!(ids, 100); return)
    end
    append!(ids, pieces)
end

function tokenize(tok::Tokenizer, text::AbstractString; max_tokens=256)
    2 <= max_tokens <= 256 || throw(ArgumentError("max_tokens must be in 2:256"))
    ids = Int[101]
    # Added special tokens are recognized before normalization, even inside words.
    start = firstindex(text)
    for m in eachmatch(r"\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\]", text)
        if start < m.offset
            for word in basic_tokens(SubString(text, start, prevind(text,m.offset)))
                wordpieces!(ids,tok,word)
            end
        end
        push!(ids,tok.vocab[String(m.match)])
        start = nextind(text,m.offset,length(m.match))
    end
    if start <= lastindex(text)
        for word in basic_tokens(SubString(text,start))
            wordpieces!(ids,tok,word)
        end
    end
    resize!(ids, min(length(ids),max_tokens-1))
    push!(ids,102)
    ids
end

struct QEmbedding
    weight::Matrix{UInt8}
    scale::Float32
    zero::Int32
end
struct QDense
    weight::Matrix{Int8} # input x output; contiguous reduction dimension
    scale::Float32
    bias::Vector{Float32}
    sums::Vector{Int32}
end
struct Norm
    weight::Vector{Float32}
    bias::Vector{Float32}
end
struct Layer
    query::QDense
    key::QDense
    value::QDense
    attention::QDense
    up::QDense
    down::QDense
    attnorm::Norm
    outnorm::Norm
end
struct Model
    tokenizer::Tokenizer
    word::QEmbedding
    position::QEmbedding
    token_type::QEmbedding
    norm::Norm
    layers::Vector{Layer}
end

function Model(dir::AbstractString)
    ENDIAN_BOM == 0x04030201 || error("little-endian host required")
    meta = JSON3.read(read(joinpath(dir,"weights.json"),String))
    meta.format == 1 && meta.revision == REVISION && meta.source_sha256 == GRAPH_SHA || error("wrong model bundle")
    bytes2hex(sha256(read(joinpath(dir,"tokenizer.json")))) == meta.tokenizer_sha256 || error("tokenizer checksum mismatch")
    function tensor(name)
        entry = meta.tensors[Symbol(name)]
        data = read(joinpath(dir,entry.file))
        length(data) == entry.bytes && bytes2hex(sha256(data)) == entry.sha256 || error("tensor checksum mismatch: $name")
        T = entry.dtype == "float32" ? Float32 : entry.dtype == "int8" ? Int8 : entry.dtype == "uint8" ? UInt8 : error("dtype")
        shape = Tuple(Int.(entry.shape))
        reshape(copy(reinterpret(T,data)),shape)
    end
    embedding(name) = QEmbedding(tensor(name*".weight"),only(tensor(name*".scale")),Int32(only(tensor(name*".zero"))))
    function dense(name)
        w = tensor(name*".weight")
        QDense(w,only(tensor(name*".scale")),vec(tensor(name*".bias")),vec(sum(Int32.(w);dims=1)))
    end
    norm(name) = Norm(vec(tensor(name*".weight")),vec(tensor(name*".bias")))
    layers = [Layer((dense("$i.$name") for name in ("query","key","value","attention","up","down"))...,
                    norm("$i.attnorm"),norm("$i.outnorm")) for i in 0:5]
    m = Model(Tokenizer(joinpath(dir,"tokenizer.json")),embedding("word"),embedding("position"),
              embedding("token_type"),norm("embedding.norm"),layers)
    size(m.word.weight) == (384,30522) && size(m.position.weight) == (384,512) || error("embedding dimensions")
    for l in m.layers
        all(size(d.weight)==(384,384) for d in (l.query,l.key,l.value,l.attention)) || error("attention dimensions")
        size(l.up.weight)==(384,1536) && size(l.down.weight)==(1536,384) || error("MLP dimensions")
    end
    m
end

function norm!(x::Matrix{Float32}, n::Norm)
    width, tokens = size(x)
    for j in 1:tokens
        mu = 0f0
        @inbounds @simd for k in 1:width
            mu += x[k,j]
        end
        mu /= Float32(width)
        variance = 0f0
        @inbounds @simd for k in 1:width
            variance += (x[k,j]-mu)^2
        end
        denom = sqrt(variance / Float32(width) + 1f-12)
        @inbounds @simd for k in 1:width
            x[k,j] = ((x[k,j]-mu) / denom) * n.weight[k] + n.bias[k]
        end
    end
    x
end

function dynamic_quantize(x::AbstractArray{Float32})
    lo, hi = extrema(x)
    lo = min(0f0,lo); hi = max(0f0,hi)
    scale = hi == lo ? 1f0 : (hi-lo)/255f0
    zero = Int32(clamp(round(-lo/scale),0f0,255f0))
    q = similar(x,UInt8)
    @inbounds for i in eachindex(x)
        q[i] = UInt8(clamp(round(Int32,x[i]/scale)+zero,Int32(0),Int32(255)))
    end
    q, scale, zero
end

function dense(d::QDense, quant)
    a, scale, zero = quant
    width, tokens = size(a); outputs = length(d.bias)
    size(d.weight,1) == width || throw(DimensionMismatch())
    y = Matrix{Float32}(undef,outputs,tokens)
    factor = scale*d.scale
    Threads.@threads :static for task in 1:tokens*outputs
        o = mod1(task,outputs); t = (task-1)÷outputs+1
        acc = Int32(0)
        @inbounds @simd for k in 1:width
            acc += Int32(d.weight[k,o])*Int32(a[k,t])
        end
        @inbounds y[o,t] = Float32(acc-zero*d.sums[o])*factor+d.bias[o]
    end
    y
end
dense(d::QDense, x::Matrix{Float32}) = dense(d,dynamic_quantize(x))

function attention(q, k, v, mask)
    seq, batch = size(mask)
    q8, qs, qz = dynamic_quantize(q)
    k8, ks, kz = dynamic_quantize(k)
    v8, vs, vz = dynamic_quantize(v)
    probs = Array{Float32}(undef,seq,seq,HEADS,batch)
    factor = qs*ks
    Threads.@threads :static for task in 1:batch*HEADS*seq
        query = mod1(task,seq); head = mod1((task-1)÷seq+1,HEADS); b = (task-1)÷(seq*HEADS)+1
        offset = (head-1)*HEAD_DIM; colq = (b-1)*seq+query
        best = -Inf32
        for key in 1:seq
            acc = Int32(0); colk = (b-1)*seq+key
            @inbounds @simd for d in 1:HEAD_DIM
                acc += (Int32(q8[offset+d,colq])-qz)*(Int32(k8[offset+d,colk])-kz)
            end
            score = Float32(acc)*factor/sqrt(32f0)
            mask[key,b] || (score += -floatmax(Float32))
            probs[key,query,head,b] = score; best = max(best,score)
        end
        total = 0f0
        for key in 1:seq
            value = exp(probs[key,query,head,b]-best)
            probs[key,query,head,b] = value; total += value
        end
        for key in 1:seq
            probs[key,query,head,b] /= total
        end
    end
    p8, ps, pz = dynamic_quantize(probs)
    context = Matrix{Float32}(undef,WIDTH,seq*batch)
    Threads.@threads :static for task in 1:batch*HEADS*seq
        query = mod1(task,seq); head = mod1((task-1)÷seq+1,HEADS); b = (task-1)÷(seq*HEADS)+1
        offset = (head-1)*HEAD_DIM
        for d in 1:HEAD_DIM
            acc = Int32(0)
            @inbounds @simd for key in 1:seq
                acc += (Int32(p8[key,query,head,b])-pz)*(Int32(v8[offset+d,(b-1)*seq+key])-vz)
            end
            context[offset+d,(b-1)*seq+query] = Float32(acc)*(ps*vs)
        end
    end
    context
end

# Abramowitz-Stegun 7.1.26; max erf approximation error ~1.5e-7.
# Julia arithmetic only; no Python, BLAS, libm erf, ONNX, or model framework calls.
function gelu(x::Float32)
    z = x / sqrt(2f0)
    t = 1f0 / (1f0+0.3275911f0*abs(z))
    poly = ((((1.061405429f0*t-1.453152027f0)*t+1.421413741f0)*t-0.284496736f0)*t+0.254829592f0)*t
    e = copysign(1f0-poly*exp(-z*z),z)
    (x*(1f0+e))*0.5f0
end

function hidden_states(m::Model, ids::Matrix{Int}, mask::Matrix{Bool}; trace=nothing)
    size(ids) == size(mask) || throw(DimensionMismatch())
    seq,batch = size(ids)
    1 <= seq <= 256 && 1 <= batch <= 32 || throw(ArgumentError("expected 1..256 tokens, 1..32 texts"))
    all(id->0<=id<30522,ids) || throw(ArgumentError("token ID outside vocabulary"))
    all(any(mask;dims=1)) || throw(ArgumentError("empty attention mask"))
    x = Matrix{Float32}(undef,WIDTH,seq*batch)
    for b in 1:batch, j in 1:seq, d in 1:WIDTH
        word = Float32(Int32(m.word.weight[d,ids[j,b]+1])-m.word.zero)*m.word.scale
        typ = Float32(Int32(m.token_type.weight[d,1])-m.token_type.zero)*m.token_type.scale
        pos = Float32(Int32(m.position.weight[d,j])-m.position.zero)*m.position.scale
        x[d,(b-1)*seq+j] = (word+typ)+pos
    end
    norm!(x,m.norm)
    trace === nothing || push!(trace,copy(x))
    for layer in m.layers
        quant = dynamic_quantize(x)
        q = dense(layer.query,quant); k = dense(layer.key,quant); v = dense(layer.value,quant)
        a = dense(layer.attention,attention(q,k,v,mask))
        a .+= x; norm!(a,layer.attnorm)
        z = dense(layer.up,a)
        z .= gelu.(z)
        x = dense(layer.down,z)
        x .+= a; norm!(x,layer.outnorm)
        trace === nothing || push!(trace,copy(x))
    end
    x
end

function encode_batch(m, texts; max_tokens=256)
    tokens = [tokenize(m.tokenizer,t;max_tokens) for t in texts]
    seq = maximum(length,tokens)
    ids = zeros(Int,seq,length(texts)); mask = fill(false,seq,length(texts))
    for (b,t) in enumerate(tokens)
        ids[1:length(t),b] .= t; mask[1:length(t),b] .= true
    end
    h = hidden_states(m,ids,mask)
    out = zeros(Float32,WIDTH,length(texts))
    for b in eachindex(texts)
        for j in eachindex(tokens[b]), d in 1:WIDTH
            out[d,b] += h[d,(b-1)*seq+j]
        end
        out[:,b] ./= Float32(length(tokens[b]))
        norm = sqrt(sum(abs2,view(out,:,b)))
        isfinite(norm) && norm > 0 || error("invalid embedding")
        out[:,b] ./= norm
    end
    out
end

function encode(m::Model,texts::AbstractVector{<:AbstractString};batch_size=8,max_tokens=256)
    1 <= batch_size <= 32 || throw(ArgumentError("batch_size must be in 1:32"))
    isempty(texts) && return zeros(Float32,WIDTH,0)
    out = Matrix{Float32}(undef,WIDTH,length(texts))
    for first in 1:batch_size:length(texts)
        last = min(first+batch_size-1,length(texts))
        out[:,first:last] = encode_batch(m,view(texts,first:last);max_tokens)
    end
    out
end
end # module
