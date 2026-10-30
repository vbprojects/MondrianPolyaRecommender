isdefined(@__MODULE__, :GteTiny) || include("GteTiny.jl")
module KernelGte
using ..GteTiny
using KernelAbstractions
export KernelModel, encode
include("CPUKernels.jl")

const V16 = NTuple{16,VecElement{Int32}}
@inline zero16() = ntuple(_->VecElement(Int32(0)),Val(16))
@inline splat16(x::UInt32) = ntuple(_->VecElement(reinterpret(Int32,x)),Val(16))
@inline load16(w,offset) = ntuple(i->VecElement(reinterpret(Int32,@inbounds(w[offset+i]))),Val(16))

# Unsigned activation bytes × signed weight bytes, four products per INT32 lane.
# No saturating 16-bit intermediate (unlike the AVX2 vpmaddubsw idiom).
@inline function dpbusd(c::V16,a::V16,b::V16)
    Base.llvmcall(("""
        declare <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32>, <16 x i32>, <16 x i32>)
        define <16 x i32> @entry(<16 x i32> %c, <16 x i32> %a, <16 x i32> %b) alwaysinline nounwind {
            %r = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %c, <16 x i32> %a, <16 x i32> %b)
            ret <16 x i32> %r
        }
        ""","entry"),V16,Tuple{V16,V16,V16},c,a,b)
end

function has_vnni()
    Sys.ARCH == :x86_64 && Sys.islinux() && isfile("/proc/cpuinfo") || return false
    flags = read("/proc/cpuinfo",String)
    occursin(r"\bavx512_vnni\b",flags) && occursin(r"\bavx512f\b",flags)
end

struct PackedDense
    source::GteTiny.QDense
    packed::Array{UInt32,3} # 16 outputs × K/4 × output blocks
end
function PackedDense(d::GteTiny.QDense)
    k,n = size(d.weight)
    k%4 == 0 && n%16 == 0 || error("GTE dimensions must be divisible by 4/16")
    packed = Array{UInt32}(undef,16,k÷4,n÷16)
    for o in 1:n, chunk in 1:k÷4
        word = UInt32(0)
        for b in 0:3
            word |= UInt32(reinterpret(UInt8,d.weight[4(chunk-1)+b+1,o])) << (8b)
        end
        packed[mod1(o,16),chunk,(o-1)÷16+1] = word
    end
    PackedDense(d,packed)
end

@kernel function portable_dense!(y,w,a,bias,sums,factor,zero)
    o,t = @index(Global,NTuple)
    acc = Int32(0)
    @inbounds @simd for k in 1:size(w,1)
        acc += Int32(w[k,o])*Int32(a[k,t])
    end
    @inbounds y[o,t] = Float32(acc-zero*sums[o])*factor+bias[o]
end

@inline function vnni_tile!(y,w,a,bias,sums,factor,zero,ob,tb)
    k4 = size(w,2); first = 4(tb-1)+1
    c1=zero16(); c2=zero16(); c3=zero16(); c4=zero16()
    if first+3 <= size(a,2)
        @inbounds for k in 1:k4
            weights = load16(w,16*((ob-1)*k4+k-1))
            c1 = dpbusd(c1,splat16(a[k,first]),weights)
            c2 = dpbusd(c2,splat16(a[k,first+1]),weights)
            c3 = dpbusd(c3,splat16(a[k,first+2]),weights)
            c4 = dpbusd(c4,splat16(a[k,first+3]),weights)
        end
        @inbounds for lane in 1:16
            o=16(ob-1)+lane; correction=zero*sums[o]
            y[o,first] = Float32(c1[lane].value-correction)*factor+bias[o]
            y[o,first+1] = Float32(c2[lane].value-correction)*factor+bias[o]
            y[o,first+2] = Float32(c3[lane].value-correction)*factor+bias[o]
            y[o,first+3] = Float32(c4[lane].value-correction)*factor+bias[o]
        end
    else
        for t in first:size(a,2)
            c=zero16()
            @inbounds for k in 1:k4
                c=dpbusd(c,splat16(a[k,t]),load16(w,16*((ob-1)*k4+k-1)))
            end
            @inbounds for lane in 1:16
                o=16(ob-1)+lane
                y[o,t]=Float32(c[lane].value-zero*sums[o])*factor+bias[o]
            end
        end
    end
end

@kernel function vnni_dense!(y,w,a,bias,sums,factor,zero)
    ob,tb = @index(Global,NTuple)
    vnni_tile!(y,w,a,bias,sums,factor,zero,ob,tb)
end

struct KernelModel
    reference::GteTiny.Model
    packed::IdDict{GteTiny.QDense,PackedDense}
    vnni::Bool
    polynomial_math::Bool
end
function KernelModel(m::GteTiny.Model;vnni=has_vnni(),polynomial_math=false)
    vnni && !has_vnni() && error("AVX-512 VNNI is unavailable")
    packed=IdDict{GteTiny.QDense,PackedDense}()
    for l in m.layers, d in (l.query,l.key,l.value,l.attention,l.up,l.down)
        packed[d]=PackedDense(d)
    end
    KernelModel(m,packed,vnni,polynomial_math)
end

function dense(m::KernelModel,d::GteTiny.QDense,quant::Tuple,ws::Workspace=Workspace(),slot=:dense_out)
    a,s,z=quant
    y=buffer(ws,Float32,slot,(length(d.bias),size(a,2)))
    if m.vnni
        # Materialize packed words: ReinterpretArray indexing otherwise adds pointer
        # conversion/alias checks to every innermost multiply iteration.
        words=buffer(ws,UInt32,:words,(size(a,1)÷4,size(a,2)))
        copyto!(vec(words),reinterpret(UInt32,vec(a)))
        vnni_dense!(CPU(),(4,4))(y,m.packed[d].packed,words,d.bias,d.sums,s*d.scale,z;
            ndrange=(size(y,1)÷16,cld(size(y,2),4)))
    else
        portable_dense!(CPU(),(16,4))(y,d.weight,a,d.bias,d.sums,s*d.scale,z;ndrange=size(y))
    end
    KernelAbstractions.synchronize(CPU())
    y
end
dense(m::KernelModel,d,x::Matrix{Float32},ws::Workspace=Workspace(),slot=:dense_out)=dense(m,d,quantize(x,ws),ws,slot)

function encode_batch(m::KernelModel,texts;max_tokens=256,trace=nothing)
    ref=m.reference
    ws=Workspace();math=Val(m.polynomial_math)
    tokens=[GteTiny.tokenize(ref.tokenizer,t;max_tokens) for t in texts]
    seq=maximum(length,tokens); batch=length(tokens)
    ids=zeros(Int,seq,batch); mask=fill(false,seq,batch)
    for (b,t) in enumerate(tokens)
        ids[1:length(t),b]=t; mask[1:length(t),b].=true
    end
    x=Matrix{Float32}(undef,384,seq*batch)
    for b in 1:batch,j in 1:seq,d in 1:384
        w=Float32(Int32(ref.word.weight[d,ids[j,b]+1])-ref.word.zero)*ref.word.scale
        t=Float32(Int32(ref.token_type.weight[d,1])-ref.token_type.zero)*ref.token_type.scale
        p=Float32(Int32(ref.position.weight[d,j])-ref.position.zero)*ref.position.scale
        x[d,(b-1)*seq+j]=(w+t)+p
    end
    GteTiny.norm!(x,ref.norm)
    trace===nothing || push!(trace,copy(x))
    for l in ref.layers
        quant=quantize(x,ws)
        q=dense(m,l.query,quant,ws,:query); k=dense(m,l.key,quant,ws,:key); v=dense(m,l.value,quant,ws,:value)
        a=dense(m,l.attention,attention(q,k,v,mask,ws,math,m.vnni),ws,:attention)
        a .+= x; GteTiny.norm!(a,l.attnorm)
        z=dense(m,l.up,a,ws,:up); gelu!(z,math)
        x=dense(m,l.down,z,ws,:down); x .+= a; GteTiny.norm!(x,l.outnorm)
        trace===nothing || push!(trace,copy(x))
    end
    out=zeros(Float32,384,batch)
    for b in 1:batch
        for j in eachindex(tokens[b]),d in 1:384
            out[d,b]+=x[d,(b-1)*seq+j]
        end
        out[:,b]./=Float32(length(tokens[b]))
        out[:,b]./=sqrt(sum(abs2,view(out,:,b)))
    end
    out
end

function encode(m::KernelModel,texts::AbstractVector{<:AbstractString};batch_size=8,max_tokens=256)
    1<=batch_size<=32 || throw(ArgumentError("batch_size must be in 1:32"))
    2<=max_tokens<=256 || throw(ArgumentError("max_tokens must be in 2:256"))
    out=Matrix{Float32}(undef,384,length(texts))
    for first in 1:batch_size:length(texts)
        last=min(first+batch_size-1,length(texts))
        out[:,first:last]=encode_batch(m,view(texts,first:last);max_tokens)
    end
    out
end
end
