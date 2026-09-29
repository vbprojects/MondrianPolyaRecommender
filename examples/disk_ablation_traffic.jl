# Persistent-cache traffic: caches survive between queries, unlike cold controls.
include("disk_ablation_worker.jl")
function main()
    warmup();results=Any[]
    for seed in (17,29,43),mode in DC.FORMATS,layout in (:append,:leaf),budget in (32,256,1024,4096)
        name="$(seed)-$(mode)-$(layout)";s=DS.open_store(joinpath(STORE,name);budget=budget*1024^2)
        X,ids=fixture_vectors(seed);queries=Any[];lookups=Any[]
        keys=[begin p=DS.one(s,"SELECT kind,native FROM posts WHERE id=?",(id,));key_from_bytes(p.kind,p.native);end for id in ids]
        measure_query(s,X[:,1];leaf_budget=1,limit=20,rng=MersenneTwister(1),workload="warmup",condition="warmup")
        DS.clear_caches!(s)
        for leaves in (1,4,16),workload in ("uniform","topic_burst","cold_revisit"),j in 1:40
            col=workload=="topic_burst" ? 1 : workload=="cold_revisit" && j%5==0 ? 1 : (j*13)%48+1
            push!(queries,measure_query(s,X[:,col];leaf_budget=leaves,limit=20,rng=MersenneTwister(seed+j),
                workload,condition="continuous_"*workload))
            a=time_ns();p=DS.lookup(s,keys[col]);dt=(time_ns()-a)/1e6
            p.id==ids[col]||error("identity lookup disagrees")
            push!(lookups,(ms=dt,id=p.id))
        end
        push!(results,(seed=seed,mode=mode,layout=layout,budget_mib=budget,queries=queries,identity_lookups=lookups,stats=DS.storage_stats(s)))
        DS.close_store(s);savejson(joinpath(OUT,"traffic.json"),results)
        println("TRAFFIC ",name," ",budget);flush(stdout)
    end
end
main()
