# Offline fixture: creates only a new directory; never opens an existing store.
include("../src/MondrianServer.jl")
using .PostRecords, Random, LinearAlgebra, TOML
const DS=DiskTreeStore
length(ARGS)==2 || error("usage: smoke_fixture.jl DATA_DIRECTORY sqlite|rocksdb")
base=abspath(ARGS[1]);backend=Symbol(ARGS[2])
backend in (:sqlite,:rocksdb) || error("invalid backend")
root=joinpath(base,"store")
ispath(root) && error("fixture refuses an existing store directory")
mkpath(base)
model="futur/Qwen3-Embedding-0.6B-model2vec-onnx@a3f60a85a2570e7976ea7f2d6505ad3e5152a96f"
s=DS.open_store(root;backend,budget=32*1024^2,
    create_config=DS.config(;dimension=256,model,max_leaves=1))
try
    t=time();rng=MersenneTwister(82)
    X=randn(rng,Float32,256,32);X./=sqrt.(sum(abs2,X;dims=1))
    records=[PostRecord(PostKey(:threads,"container-$i");text="Julia trees and public data",tags=["Julia"],
        created_at=t,observed_at=t) for i in 1:32]
    e=KeywordExtraction.JuliaYakeExtractor()
    DS.enable_keywords!(s,e)
    DS.append_batch!(s,records,X;time=t,keywords=KeywordExtraction.prepare_keywords(e,KeywordExtraction.keyword_input.(records)))
    @assert DS.audit!(s)==32
finally
    DS.close_store(s)
end
cfg=Dict("mode"=>"serve","store"=>root,
    "ingestion_config"=>joinpath(@__DIR__,"sources.toml"),
    "http"=>Dict("host"=>"0.0.0.0","port"=>parse(Int,get(ENV,"MONDRIAN_SMOKE_PORT","8080"))),
    "storage"=>Dict("backend"=>String(backend),"managed_budget_mib"=>32))
open(joinpath(base,"smoke.toml"),"w") do io;TOML.print(io,cfg);end
println("Created 32-post ",backend," fixture")
