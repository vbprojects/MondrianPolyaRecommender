# Integrated storage/tree ingestion, synthetic geometry, precomputed embeddings/keywords.
using JSON3, Random, LinearAlgebra, Statistics
include("../src/MondrianServer.jl")
const DS=DiskTreeStore;const KE=KeywordExtraction
using .PostRecords
BLAS.set_num_threads(1)
function benchmark()
    cfg=DS.config(dimension=384,model="synthetic-benchmark",max_leaves=32,split_after=64,reservoir_cap=32,half_life=3600.)
    e=KE.HashtagsOnlyExtractor();rng=MersenneTwister(82);N=2048
    X=randn(rng,Float32,384,N);X./=sqrt.(sum(abs2,X;dims=1))
    records=[PostRecord(PostKey(:threads,"bench-$i");text="Synthetic topic $(i%16)",tags=["topic$(i%16)","shared"],created_at=100.,observed_at=100.) for i in 1:N]
    kws=vcat([KE.prepare_keywords(e,KE.keyword_input.(records[first:min(first+1023,N)])) for first in 1:1024:N]...);results=[];signatures=[]
    for backend in (:sqlite,:rocksdb)
        mktempdir() do warm
            s=DS.open_store(warm;backend,budget=32*1024^2,create_config=cfg);DS.enable_keywords!(s,e)
            for first in 1:64:256
                ids=first:first+63;DS.append_batch!(s,records[ids],X[:,ids];time=100.,keywords=kws[ids])
            end
            DS.close_store(s)
        end
        mktempdir() do root
            s=DS.open_store(root;backend,budget=32*1024^2,create_config=cfg);DS.enable_keywords!(s,e)
            timings=Float64[]
            for first in 1:256:N
                ids=first:first+255
                push!(timings,@elapsed DS.append_batch!(s,records[ids],X[:,ids];time=100.,keywords=kws[ids],cursor=("benchmark",string(last(ids)))))
            end
            @assert DS.audit!(s)==N
            push!(signatures,DS.tree_signature(s))
            push!(results,(backend=String(backend),posts=N,dimension=384,batch_size=256,
                seconds=sum(timings),posts_per_second=N/sum(timings),batch_seconds=timings,
                keyword_method="hashtags-only, prepared outside timing",embedding_inference="excluded; seeded normalized random Float32",
                tree_density="common rank-4 PPCA",statistics=DS.storage_stats(s)))
            DS.close_store(s)
        end
    end
    @assert signatures[1]==signatures[2]
    root=normpath(joinpath(@__DIR__,"../.."));mkpath(joinpath(root,"output/rocks_backend"))
    write(joinpath(root,"output/rocks_backend/ingestion.json"),JSON3.write((results,topology_equal=true,seed=82,julia=string(VERSION),blas_threads=BLAS.get_num_threads(),julia_threads=Threads.nthreads())))
    println(JSON3.write(results))
end
benchmark()
