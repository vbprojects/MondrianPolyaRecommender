include("disk_ablation_common.jl")
function measure_query(s,x;leaf_budget,limit,rng,workload,condition)
    empty!(s.request_blocks);empty!(s.request_decoders)
    before=osinfo();read0=metric(s,"vector_read_bytes")+metric(s,"text_read_bytes")+metric(s,"state_read_bytes")
    hit0=s.block_cache.hits+s.state_cache.hits;miss0=s.block_cache.misses+s.state_cache.misses
    sql0=DS.sqlite_usage(s)
    start=time_ns()
    result=DS.sample_ids(s,x;leaf_budget,limit,rng)
    # Density factors are fetched only for the selected candidate leaves.
    for leaf in result.leaves;DS.get_state(s,leaf);end
    V=DS.fetch_vectors(s,result.ids);DS.fetch_records(s,result.ids)
    elapsed=(time_ns()-start)/1e6;after=osinfo()
    (workload=workload,condition=condition,leaf_budget=leaf_budget,requested=limit,returned=length(result.ids),
     ms=elapsed,blocks=length(s.request_blocks),decoders=length(s.request_decoders),
     application_read_bytes=metric(s,"vector_read_bytes")+metric(s,"text_read_bytes")+metric(s,"state_read_bytes")-read0,
     os_read_bytes=delta(before,after,"read_bytes"),os_read_chars=delta(before,after,"rchar"),
     cache_hits=s.block_cache.hits+s.state_cache.hits-hit0,cache_misses=s.block_cache.misses+s.state_cache.misses-miss0,
     sqlite_hits=DS.sqlite_usage(s)["cache_hits"]-sql0["cache_hits"],sqlite_misses=DS.sqlite_usage(s)["cache_misses"]-sql0["cache_misses"],
     sqlite_bytes=DS.sqlite_usage(s)["cache_bytes"],
     underfilled=result.underfilled,rss=after["rss"])
end
function run_queries(root,seed,budget)
    GC.gc()
    s=DS.open_store(root;budget);restart_empty=isempty(s.state_cache.entries)&&isempty(s.block_cache.entries)
    X,_=fixture_vectors(seed);rng=MersenneTwister(seed+77);measures=Any[]
    # Compile query paths outside measurements, then explicitly clear all managed caches.
    measure_query(s,X[:,1];leaf_budget=1,limit=20,rng,workload="warmup",condition="warmup")
    DS.clear_caches!(s)
    for workload in ("uniform","topic_burst","cold_revisit"), leaves in (1,4,16), j in 1:8
        col=workload=="topic_burst" ? 1 : (workload=="cold_revisit" ? 40+j : (leaves+j)%40+1)
        x=view(X,:,col)
        DS.clear_caches!(s)
        push!(measures,measure_query(s,x;leaf_budget=leaves,limit=20,rng=MersenneTwister(seed+j),workload,condition="application_cold"))
        push!(measures,measure_query(s,x;leaf_budget=leaves,limit=20,rng=MersenneTwister(seed+j),workload,condition="immediate_repeat"))
    end
    for leaves in (1,4,16),j in 1:6
        push!(measures,measure_query(s,view(X,:,j);leaf_budget=leaves,limit=256,rng,workload="candidate_pool",condition="mixed"))
    end
    # Walk each stored frame once to exercise eviction without a corpus-sized
    # output buffer or an artificial global-arrival scan of leaf-oriented data.
    readstart=osinfo();DS.clear_caches!(s);afterid=0;scan_start=time_ns()
    while true
        ps=DS.rows(s,"SELECT id FROM blocks WHERE live>0 AND id>? ORDER BY id LIMIT 256",(afterid,))
        isempty(ps)&&break
        for p in ps;DC.decode(DS.frame_for(s,p.id);rows=[1]);end
        afterid=last(ps).id
    end
    scan=(protocol="each live frame once; decode one row",seconds=(time_ns()-scan_start)/1e9,os_read_bytes=delta(readstart,osinfo(),"read_bytes"))
    stats=DS.storage_stats(s);mem=osinfo();DS.close_store(s)
    (budget=budget,restart_empty=restart_empty,stats=stats,memory=mem,scan=scan,queries=measures)
end
function build(mode,layout,seed;count=100000)
    name="$(seed)-$(mode)-$(layout)";root=joinpath(STORE,name)
    isdir(root) && error("store exists: $root; use a new directory or explicitly remove this experiment store")
    warmup();cfg=DS.config(;mode,layout,seed)
    s=DS.open_store(root;create_config=cfg,budget=256*1024^2)
    batches=Any[];start=osinfo();t0=time_ns()
    open(joinpath(INPUT,"posts.jsonl")) do textio
        open(joinpath(INPUT,"embeddings.f32")) do vecio
            for first in 1:1024:count
                n=min(1024,count-first+1);rs,X=readbatch(textio,vecio,first,n)
                b0=osinfo();a=time_ns();splits=metric(s,"splits");bytes=metric(s,"payload_write_bytes")
                DS.append_batch!(s,rs,X;time=(first+n-1)/100,cursor=("fixture",string(first+n-1)))
                push!(batches,(posts=n,seconds=(time_ns()-a)/1e9,splits=metric(s,"splits")-splits,
                    written=metric(s,"payload_write_bytes")-bytes,os_write_bytes=delta(b0,osinfo(),"write_bytes"),rss=osinfo()["rss"]))
                (first==1 || (first-1)%10240==0) && (println(name," ",first+n-1,"/",count);flush(stdout))
            end
        end
    end
    ingestion=(seconds=(time_ns()-t0)/1e9,durable_seconds=sum(b.seconds for b in batches),memory=osinfo(),
        os_read_bytes=delta(start,osinfo(),"read_bytes"),os_write_bytes=delta(start,osinfo(),"write_bytes"),batches=batches)
    # Reference auditing is a separate process; worker retains only bounded buffers.
    before=DS.storage_stats(s);m0=copy(s.metrics);f0=fingerprints(s);a=time_ns()
    DS.compact!(s);compact_seconds=(time_ns()-a)/1e9
    f1=fingerprints(s);f0==f1 || error("relocation changed encoded values")
    DS.rows(s,"PRAGMA wal_checkpoint(TRUNCATE)")
    after=DS.storage_stats(s);breakdown=db_breakdown(s)
    occupancy=DS.rows(s,"SELECT n,rank,COUNT(*) AS blocks,SUM(decoder_bytes) AS decoder_bytes FROM blocks WHERE live>0 GROUP BY n,rank ORDER BY n,rank")
    tree=DS.tree_signature(s);metrics=copy(s.metrics)
    audit=DS.audit!(s);audit==count || error("missing posts")
    DS.close_store(s)
    partial=(name=name,mode=mode,layout=layout,seed=seed,count=count,config=cfg,ingestion=ingestion,
        before_compaction=before,storage=after,db_breakdown=breakdown,occupancy=occupancy,metrics=metrics,
        compaction=(seconds=compact_seconds,bytes=metric_from(metrics,"payload_write_bytes")-metric_from(m0,"payload_write_bytes")),
        fingerprint=f1,tree=tree,correctness=(relocation_exact=true,audited_posts=audit),queries=[])
    savejson(joinpath(OUT,"raw",name*".json"),partial)
    runs=[run_queries(root,seed,budget*1024^2) for budget in (32,256,1024,4096)]
    savejson(joinpath(OUT,"raw",name*".json"),merge(partial,(queries=runs,)))
    println("DONE ",name);flush(stdout)
end
metric_from(m,k)=get(m,k,0.)
if abspath(PROGRAM_FILE)==@__FILE__
    if length(ARGS)>3 && ARGS[4]=="queries"
        name="$(ARGS[3])-$(ARGS[1])-$(ARGS[2])";path=joinpath(OUT,"raw",name*".json")
        result=JSON3.read(read(path,String),Dict{String,Any});warmup()
        result["queries"]=[run_queries(joinpath(STORE,name),parse(Int,ARGS[3]),b*1024^2) for b in (32,256,1024,4096)]
        savejson(path,result)
    else
        build(Symbol(ARGS[1]),Symbol(ARGS[2]),parse(Int,ARGS[3]);count=length(ARGS)>3 ? parse(Int,ARGS[4]) : 100000)
    end
end
