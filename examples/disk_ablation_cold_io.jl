# Per-file cache eviction is advisory; /proc read_bytes establishes actual I/O.
include("disk_ablation_worker.jl")
function advise_drop(root)
    failures=0
    for name in readdir(root)
        (endswith(name,".bin") || startswith(name,"index.sqlite")) || continue
        open(joinpath(root,name)) do io
            failures+=ccall(:posix_fadvise,Cint,(Cint,Int64,Int64,Cint),Base.fd(io),0,0,4)!=0
        end
    end
    failures
end
function main()
    warmup();results=Any[]
    for seed in (17,29,43),mode in DC.FORMATS,layout in (:append,:leaf),budget in (32,256,1024,4096)
        name="$(seed)-$(mode)-$(layout)";root=joinpath(STORE,name);s=DS.open_store(root;budget=budget*1024^2)
        X,_=fixture_vectors(seed);queries=Any[];failures=0
        measure_query(s,X[:,1];leaf_budget=1,limit=20,rng=MersenneTwister(1),workload="warmup",condition="warmup")
        for leaves in (1,4,16),j in 1:8
            DS.clear_caches!(s);failures+=advise_drop(root)
            push!(queries,measure_query(s,X[:,j];leaf_budget=leaves,limit=20,rng=MersenneTwister(seed+j),
                workload="uniform",condition="file_cache_eviction_hint"))
        end
        push!(results,(seed=seed,mode=mode,layout=layout,budget_mib=budget,advice_failures=failures,queries=queries))
        DS.close_store(s);savejson(joinpath(OUT,"cold_io.json"),results)
        println("COLD IO ",name," ",budget);flush(stdout)
    end
end
main()
