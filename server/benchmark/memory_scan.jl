# Run from repository root; private temporary SQLite store, no live-store writes.
mkpath("output/memory_scan")
include(joinpath(pwd(),"examples/disk_live_modules.jl"))
using .PostRecords, Random, LinearAlgebra, JSON3
const DS=DiskTreeStore;const KE=KeywordExtraction
BLAS.set_num_threads(1)
function batch!(s,e,rng,step)
    t=10.0step
    records=[PostRecord(PostKey(:threads,"memory-$(step)-$(i)");text="Julia trees probability public data topic $(i%8)",tags=["topic$(i%8)"],created_at=t,observed_at=t) for i in 1:64]
    X=randn(rng,Float32,256,64);X./=sqrt.(sum(abs2,X;dims=1))
    DS.append_batch!(s,records,X;time=t,keywords=KE.prepare_keywords(e,KE.keyword_input.(records)))
    DS.expire!(s;time=t);DS.maintain!(s;time=t)
    if step%16==0;DS.collect_garbage!(s);end
end
function capture(s,step)
    GC.gc(true)
    fields=Dict(first(split(line,':'))=>split(last(split(line,':'))) for line in eachline("/proc/self/status") if startswith(line,"VmRSS:") || startswith(line,"VmHWM:"))
    (;step,live_bytes=Base.gc_live_bytes(),rss=1024parse(Int,fields["VmRSS"][1]),
      peak_rss=1024parse(Int,fields["VmHWM"][1]),nodes=length(s.nodes),statements=length(s.statements),
      state_cache=s.state_cache.used,block_cache=s.block_cache.used,block_entries=length(s.block_cache.entries),
      block_evictions=s.block_cache.evictions,metrics=length(s.metrics),fds=length(readdir("/proc/self/fd")),
      active=DS.one(s,"SELECT COALESCE(SUM(count),0) AS n FROM leaves").n,
      sqlite_cache=DS.sqlite_usage(s),request_blocks=length(s.request_blocks),splits=get(s.metrics,"splits",0.),merges=get(s.metrics,"merges",0.))
end
mktempdir() do root
    s=DS.open_store(root;budget=32*1024^2,create_config=DS.config(dimension=256,retention_seconds=80.,half_life=3.,max_leaves=8,split_after=32,reservoir_cap=16,merge_below=4.))
    # Deliberately force frame eviction early, so growing cache occupancy cannot
    # be confused with a leak in this short diagnostic.
    s.block_cache.capacity=512*1024
    e=KE.JuliaYakeExtractor();DS.enable_keywords!(s,e);rng=MersenneTwister(82);samples=[]
    try
        for step in 1:96
            batch!(s,e,rng,step)
            if step%16==0
                push!(samples,capture(s,step));println(JSON3.write(last(samples)));flush(stdout)
            end
        end
        DS.clear_caches!(s);push!(samples,capture(s,97))
        write("output/memory_scan/replay.json",JSON3.write(samples))
    finally DS.close_store(s) end
end
