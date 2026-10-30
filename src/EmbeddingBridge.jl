module EmbeddingBridge
using JSON3, Base64
export Worker, embed, verify_nostr
const QWEN_MODEL="futur/Qwen3-Embedding-0.6B-model2vec-onnx@a3f60a85a2570e7976ea7f2d6505ad3e5152a96f"
const GTE_MODEL="TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3"
# Quantized vectors must never be silently appended to the existing FP32 index.
const GTE_INT8_MODEL=GTE_MODEL*";onnx-int8-ceab9964bcdb"
mutable struct Worker
    process::Base.Process
    model::String
    dimension::Int
    lock::ReentrantLock
end
function Worker(;python=get(ENV,"MONDRIAN_EMBEDDING_PYTHON",
        normpath(joinpath(@__DIR__,"../.venv-embeddings/bin/python"))),device="cpu",model=QWEN_MODEL,
        backend="auto",rust_worker=get(ENV,"MONDRIAN_EMBEDDING_RUST",
            normpath(joinpath(@__DIR__,"../native/gte-worker/target/release/mondrian-gte-worker"))),
        bundle=get(ENV,"MONDRIAN_GTE_BUNDLE",normpath(joinpath(@__DIR__,"../output/gte_tiny_rust"))),
        threads=2,batch_size=8)
    device in ("cpu","cuda","auto") || throw(ArgumentError("invalid embedding device"))
    model in (QWEN_MODEL,GTE_MODEL,GTE_INT8_MODEL) || throw(ArgumentError("unsupported pinned embedding model"))
    backend in ("auto","python","rust") || throw(ArgumentError("invalid embedding backend"))
    selected=backend=="auto" ? (model==GTE_INT8_MODEL ? "rust" : "python") : backend
    command=if selected=="rust"
        model==GTE_INT8_MODEL || throw(ArgumentError("Rust backend requires GTE_INT8_MODEL"))
        device!="cuda" || throw(ArgumentError("Rust GTE worker currently supports CPU only"))
        1<=threads<=256 && 1<=batch_size<=32 || throw(ArgumentError("invalid thread/batch count"))
        isfile(rust_worker) || error("Rust worker missing; build native/gte-worker with cargo build --release --locked")
        Cmd([rust_worker,"--bundle",bundle,"--device","cpu","--model",model,
            "--threads",string(threads),"--batch-size",string(batch_size)])
    else
        model!=GTE_INT8_MODEL || throw(ArgumentError("GTE_INT8_MODEL requires the Rust backend"))
        script=normpath(joinpath(@__DIR__,"../scripts/live_embedding_worker.py"))
        Cmd([python,"-u",script,"--device",device,"--model",model])
    end
    process=open(command,"r+")
    try
        ready=JSON3.read(readline(process));get(ready,:ready,false) || error("worker failed to initialize")
        ready.model==model && ready.dimension==(model==QWEN_MODEL ? 256 : 384) || error("worker model contract mismatch")
        Worker(process,String(ready.model),Int(ready.dimension),ReentrantLock())
    catch
        close(process);rethrow()
    end
end
function request(w::Worker,data)
    lock(w.lock) do
        write(w.process,JSON3.write(data)*"\n");flush(w.process)
        result=JSON3.read(readline(w.process))
        haskey(result,:error) && error("embedding/verification worker: $(result.error)")
        result
    end
end
function embed(w::Worker,texts)
    isempty(texts) && return zeros(Float32,w.dimension,0)
    r=request(w,(op="embed",texts=texts))
    r.model==w.model && r.dimension==w.dimension && r.count==length(texts) || error("embedding model contract changed")
    data=base64decode(r.vectors);length(data)==4w.dimension*length(texts) || error("invalid vector byte count")
    words=ltoh.(reinterpret(UInt32,data));X=reshape(copy(reinterpret(Float32,words)),w.dimension,length(texts))
    all(isfinite,X) || error("nonfinite embeddings");X
end
verify_nostr(w::Worker,events)=Bool.(request(w,(op="verify_nostr",events=events)).valid)
function Base.close(w::Worker)
    try request(w,(op="close",)) finally close(w.process) end
end
end
