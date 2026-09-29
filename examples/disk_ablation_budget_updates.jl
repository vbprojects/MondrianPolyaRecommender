# Durable 1,024-post update probes at each cache budget, on disposable snapshots.
include("disk_ablation_common.jl")
function snapshot(root,dest)
    # The source has no writer and its WAL has been checkpointed on close.
    cp(joinpath(root,"index.sqlite"),joinpath(dest,"index.sqlite"))
    for name in readdir(root)
        endswith(name,".bin") || continue
        ccall(:link,Cint,(Cstring,Cstring),joinpath(root,name),joinpath(dest,name))==0 || error("snapshot link failed")
    end
end
function main()
    warmup();results=Any[]
    for seed in (17,29,43),mode in DC.FORMATS,layout in (:append,:leaf),budget in (32,256,1024,4096)
        root=joinpath(STORE,"$(seed)-$(mode)-$(layout)")
        result=mktempdir(STORE) do dest
            snapshot(root,dest);s=DS.open_store(dest;budget=budget*1024^2)
            t=s.clock
            rs,X=open(joinpath(INPUT,"posts.jsonl")) do p
                open(joinpath(INPUT,"embeddings.f32")) do v
                    readbatch(p,v,1,1024;shift=t,prefix="budget-probe-")
                end
            end
            b=osinfo();start=time_ns();DS.append_batch!(s,rs,X;time=t+10.24)
            seconds=(time_ns()-start)/1e9;mem=osinfo()
            stats=DS.storage_stats(s);DS.close_store(s)
            (seed=seed,mode=mode,layout=layout,budget_mib=budget,posts=1024,seconds=seconds,posts_per_second=1024/seconds,
                os_write_bytes=delta(b,mem,"write_bytes"),memory=mem,stats=stats)
        end
        push!(results,result);savejson(joinpath(OUT,"budget_updates.json"),results)
        println("BUDGET UPDATE ",seed," ",mode," ",layout," ",budget);flush(stdout)
    end
end
main()
