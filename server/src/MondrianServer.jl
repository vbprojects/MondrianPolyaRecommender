# Preserve the core modules' persisted Julia Serialization paths (Main.*).
isdefined(@__MODULE__, :PostRecords) || include("../../src/PostRecords.jl")
isdefined(@__MODULE__, :LeafLatentTrees) || include("../../src/LeafLatentTree.jl")
isdefined(@__MODULE__, :DiskCodecs) || include("../../src/DiskCodecs.jl")
isdefined(@__MODULE__, :KeywordExtraction) || include("../../src/KeywordExtraction.jl")
isdefined(@__MODULE__, :DiskTreeStore) || include("../../src/DiskTreeStore.jl")
isdefined(@__MODULE__, :SourceAdapters) || include("../../src/SourceAdapters.jl")
isdefined(@__MODULE__, :LiveSources) || include("../../src/LiveSources.jl")
isdefined(@__MODULE__, :DiskIngestionRunner) || include("../../src/DiskIngestionRunner.jl")
isdefined(@__MODULE__, :EmbeddingBridge) || include("../../src/EmbeddingBridge.jl")

module MondrianServer
using JSON3, SHA, UUIDs, Random, LinearAlgebra, Base64, TOML, HTTP, CodecZlib
import ..PostRecords, ..LeafLatentTrees, ..DiskCodecs, ..KeywordExtraction, ..DiskTreeStore, ..SourceAdapters, ..LiveSources, ..DiskIngestionRunner, ..EmbeddingBridge
const DS=DiskTreeStore
const DR=DiskIngestionRunner
const KE=KeywordExtraction
const LT=LeafLatentTrees
using .PostRecords
export start_server, stop_server, submit, tree_pack, nearby, terms, ingest_batch!, load_config
struct APIError <: Exception
    status::Int
    code::String
end
Base.showerror(io::IO,e::APIError)=print(io,e.code)
fail(code,status=400)=throw(APIError(status,code))
Base.@kwdef struct Settings
    queue_capacity::Int=64
    timeout::Float64=5.
    body_limit::Int=65536
    server_budget::Int=32*1024^2
    pack_refresh::Float64=5.
    maintenance_interval::Float64=1.
    kappa::Float64=2.
    default_limit::Int=25
    leaf_budget::Int=4
    max_leaves::Int=16
    candidates::Int=256
    max_candidates::Int=1024
    max_draws::Int=8192
    term_limit::Int=25
    term_page::Int=100
    cors::Vector{String}=String[]
end
function validate(c::Settings)
    1<=c.queue_capacity<=256 && isfinite(c.timeout) && c.timeout>0 &&
    1024<=c.body_limit<=1024^2 && c.server_budget>=8*1024^2 &&
    isfinite(c.pack_refresh) && c.pack_refresh>0 && isfinite(c.maintenance_interval) && c.maintenance_interval>0 &&
    isfinite(c.kappa) && c.kappa>=0 && 1<=c.default_limit<=25 &&
    1<=c.leaf_budget<=c.max_leaves<=16 &&
    c.default_limit<=c.candidates<=c.max_candidates<=1024 &&
    c.max_candidates<=c.max_draws<=32768 && 1<=c.term_limit<=c.term_page<=100 ||
        error("invalid server settings")
    c
end
mutable struct Context
    store::DS.Store
    settings::Settings
    identity::Any
    secret::Vector{UInt8}
    packs::Dict{Symbol,Any}
end
function Context(s,c=Settings();descriptor=nothing)
    validate(c)
    r=DS.rows(s,"SELECT value FROM meta WHERE key='server_identity'")
    if isempty(r)
        # Unknown encoders require an explicit descriptor rather than guessed pooling.
        model=s.config.model
        desc=if descriptor!==nothing
            descriptor
        elseif startswith(model,"futur/Qwen3-Embedding-0.6B-model2vec-onnx@") && s.config.dimension==256
            (model=model,dimension=256,preparation="live_embedding_worker/static-token-pooling-v1",normalization="l2",coordinate_dtype="float32")
        elseif startswith(model,"TaylorAI/gte-tiny@") && s.config.dimension==384
            (model=model,dimension=384,preparation="gte-tiny/masked-mean/max-tokens-256/v1",normalization="l2",coordinate_dtype="float32")
        else
            error("unknown embedding contract; provide an explicit descriptor")
        end
        space=bytes2hex(sha256(JSON3.write(desc)))
        identity=(store_id=string(uuid4()),embedding_space_id=space,embedding=desc)
        DS.commit!(s) do
            DS.exec(s,"INSERT INTO meta VALUES ('server_identity',?)",(collect(codeunits(JSON3.write(identity))),))
        end
    else
        identity=JSON3.read(String(only(r).value))
        identity.embedding.model==s.config.model && identity.embedding.dimension==s.config.dimension || error("embedding contract mismatch")
        descriptor!==nothing && JSON3.write(identity.embedding)!=JSON3.write(descriptor) && error("stored embedding descriptor mismatch")
    end
    Context(s,c,identity,rand(RandomDevice(),UInt8,32),Dict())
end
revision(s)=bytes2hex(sha256(JSON3.write((root_bounds=(-1.,1.),dimension=s.config.dimension,nodes=DS.tree_signature(s)))))
function envelope(ctx,t)
    s=ctx.store
    (api_version="1",store_id=ctx.identity.store_id,embedding_space_id=ctx.identity.embedding_space_id,
     generation=string(s.generation),effective_at=t,tree_revision=revision(s))
end
function valid_time(ctx,t)
    isfinite(t) && t>=ctx.store.clock || fail("invalid_query_time",422)
    ctx.store.poisoned && fail("storage_unavailable",503)
end
function next_expiry(s)
    r=DS.one(s,"SELECT MIN(created) AS t FROM posts WHERE status=1").t
    ismissing(r) || r===nothing ? Inf : Float64(r)+s.config.retention_seconds
end
function fresh(ctx,t)
    valid_time(ctx,t)
    next_expiry(ctx.store)<t && fail("maintenance_pending",503)
end
function vector_input(ctx,input)
    input isa AbstractDict || input isa JSON3.Object || fail("expected_object")
    get(input,"embedding_space_id",nothing)==ctx.identity.embedding_space_id || fail("embedding_space_mismatch",422)
    v=get(input,"vector",nothing)
    v isa AbstractVector && length(v)==ctx.store.config.dimension || fail("invalid_dimension",422)
    all(x->x isa Real && !(x isa Bool) && isfinite(x) && -1<=x<=1,v) || fail("invalid_vector",422)
    Float64.(v)
end
function integer_option(input,name,default,lo,hi)
    x=get(input,name,default)
    x isa Integer && !(x isa Bool) && lo<=x<=hi || fail("invalid_"*name)
    Int(x)
end
include("TreePack.jl")
include("Nearby.jl")
include("Terms.jl")
include("Maintenance.jl")
include("StoreOwner.jl")
include("HTTPServer.jl")
include("Ingestion.jl")
end
