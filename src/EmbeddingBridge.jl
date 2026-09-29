module EmbeddingBridge
using JSON3, Base64
export Worker, embed, verify_nostr
const QWEN_MODEL="futur/Qwen3-Embedding-0.6B-model2vec-onnx@a3f60a85a2570e7976ea7f2d6505ad3e5152a96f"
const GTE_MODEL="TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3"
mutable struct Worker
    process::Base.Process
    model::String
    dimension::Int
    lock::ReentrantLock
end
function Worker(;python=get(ENV,"MONDRIAN_EMBEDDING_PYTHON",
        normpath(joinpath(@__DIR__,"../.venv-embeddings/bin/python"))),device="cpu",model=QWEN_MODEL)
    device in ("cpu","cuda","auto") || throw(ArgumentError("invalid embedding device"))
    script=normpath(joinpath(@__DIR__,"../scripts/live_embedding_worker.py"))
    model in (QWEN_MODEL,GTE_MODEL) || throw(ArgumentError("unsupported pinned embedding model"))
    process=open(Cmd([python,"-u",script,"--device",device,"--model",model]),"r+")
    try
        ready=JSON3.read(readline(process));get(ready,:ready,false) || error("worker failed to initialize")
        ready.model==model && ready.dimension==(model==GTE_MODEL ? 384 : 256) || error("worker model contract mismatch")
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
