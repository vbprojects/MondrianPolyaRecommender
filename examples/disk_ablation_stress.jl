include("disk_ablation_common.jl")
using Test
function synth(d,n,rng,phase)
    X=randn(rng,Float32,d,n).*.035
    for i in 1:n
        a=2pi*rand(rng)+phase;cluster=rand(rng,(-1.,1.))
        X[1,i]+=.55cluster+.12cos(a);X[2,i]+=.15sin(a)+.16sin(phase)
    end
    clamp.(X,-.99f0,.99f0)
end
records(first,n,t)=[PostRecord(PostKey(:threads,"synthetic-$i");text="synthetic $i",created_at=t,observed_at=t) for i in first:first+n-1]
function stresscase(mode,layout,seed,d)
    mktempdir() do root
        s=DS.open_store(root;budget=32*1024^2,create_config=DS.config(;dimension=d,mode,layout,seed,
            max_leaves=16,split_after=48,reservoir_cap=32,half_life=30.,merge_below=2.))
        rng=MersenneTwister(seed);next=1;timeline=Any[];unchanged=true
        for cycle in 0:2
            time=cycle*1000.;X=synth(d,1024,rng,Float64(cycle));ps=records(next,1024,time);next+=1024
            a=time_ns();DS.append_batch!(s,ps,X;time);ingest=(time_ns()-a)/1e9
            digest=fingerprints(s);active=DS.audit!(s);splitstats=DS.storage_stats(s);tree=DS.tree_signature(s)
            a=time_ns();DS.maintain!(s;time=time+900.);merge=(time_ns()-a)/1e9
            unchanged &=digest==fingerprints(s);DS.audit!(s)==active||error("membership changed")
            push!(timeline,(cycle=cycle,tree=tree,ingest_seconds=ingest,merge_seconds=merge,after_split=splitstats,after_merge=DS.storage_stats(s)))
        end
        unchanged||error("repeated migration changed encoded values")
        # Replay, metadata update, semantic edit, deletion and non-resurrection.
        id=PostKey(:threads,"synthetic-1");old=DS.lookup(s,id);clock=s.clock+1
        edited=PostRecord(id;text="edited synthetic",created_at=0.,observed_at=clock,revision_at=clock)
        DS.append_batch!(s,[edited],synth(d,1,rng,0.);time=clock)
        DS.delete_post!(s,id;revision_at=clock+1,time=clock+1)
        ignored=DS.append_batch!(s,[edited],synth(d,1,rng,0.);time=clock+1).ignored
        ignored==1 && DS.lookup(s,id).status==2 || error("deleted record resurrected")
        daily=Any[]
        for day in 1:9
            t=day*86400.;X=synth(d,128,rng,Float64(day));ps=records(next,128,t);next+=128
            DS.append_batch!(s,ps,X;time=t)
            tree=DS.tree_signature(s);qt=t+900.
            expired=DS.expire!(s;time=qt);DS.maintain!(s;time=qt)
            DS.storage_stats(s).leaves==1 || error("daily decay fixture did not collapse")
            prior=DS.storage_stats(s);a=time_ns();DS.compact!(s);elapsed=(time_ns()-a)/1e9
            DS.audit!(s)
            q=DS.sample_ids(s,X[:,1];limit=20,leaf_budget=1,time=qt,rng=MersenneTwister(seed+500+day))
            for row in DS.fetch_records(s,q.ids;time=qt);row.created_at>=qt-7*86400||error("expired sample");end
            push!(daily,(day=day,tree=tree,expired=expired,compaction_seconds=elapsed,before=prior,after=DS.storage_stats(s)))
        end
        result=(mode=mode,layout=layout,seed=seed,dimension=d,exact_relocation=unchanged,
            replay_ignored=ignored,timeline=timeline,daily=daily,metrics=copy(s.metrics))
        DS.close_store(s);result
    end
end
function main()
    results=Any[]
    for seed in (17,29,43),d in (2,32),mode in DC.FORMATS,layout in (:append,:leaf)
        push!(results,stresscase(mode,layout,seed,d));println("STRESS ",seed," ",d," ",mode," ",layout);flush(stdout)
        savejson(joinpath(OUT,"stress.json"),results)
    end
end
main()
