struct NeedEmbeddings <: Exception
    texts::Vector{String}
end
"""Replan against the latest generation on every commit attempt. Only embedding
inference leaves the owner; queries never run inference or create user state."""
function ingest_batch!(service,source,batch,embed;extractor=KE.JuliaYakeExtractor(),now=nothing)
    vectors=Dict{String,Vector{Float32}}()
    while true
        try
            return submit(service.owner;timeout=Inf) do ctx
                t=max(ctx.store.clock,now===nothing ? service.clock() : now)
                cached=texts->begin
                    missing=unique([String(x) for x in texts if !haskey(vectors,x)])
                    !isempty(missing) && throw(NeedEmbeddings(missing))
                    isempty(texts) ? zeros(Float32,ctx.store.config.dimension,0) : reduce(hcat,[vectors[x] for x in texts])
                end
                DR.commit_batch!(ctx.store,source,batch,cached;extractor,now=t)
            end
        catch e
            e isa NeedEmbeddings || rethrow()
            X=Float32.(embed(e.texts))
            d=submit(ctx->ctx.store.config.dimension,service.owner;timeout=Inf)
            size(X)==(d,length(e.texts)) && all(isfinite,X) && all(x->abs(x)<=1,X) || error("invalid embedding result")
            for (j,text) in enumerate(e.texts);vectors[text]=X[:,j];end
        end
    end
end
function start_ingestion!(service,cfg;device="cpu",max_events=256,seconds=5.)
    extractor=KE.configured_extractor(get(cfg,"keywords",Dict()))
    sources=get(cfg,"sources",[Dict("name"=>"bluesky","kind"=>"bluesky")])
    names=[String(s["name"]) for s in sources]
    length(unique(names))==length(names) && all(x->!startswith(x,"__"),names) || error("invalid source names")
    1<=max_events<=1024 && seconds>0 || error("invalid source budget")
    service.ingest_task=Threads.@spawn begin
        worker=nothing
        try
            model=get(get(cfg,"index",Dict()),"embedding_model",EmbeddingBridge.QWEN_MODEL)
            worker=EmbeddingBridge.Worker(;device,model)
            submit(service.owner;timeout=Inf) do ctx
                ctx.store.config.model==worker.model && ctx.store.config.dimension==worker.dimension || error("embedding model mismatch")
            end
            println(JSON3.write((embedding_model=worker.model,dimension=worker.dimension)));flush(stdout)
            while !service.stopped[]
                for source in sources
                    service.stopped[] && break
                    name=String(source["name"])
                    cursor=submit(ctx->DR.source_cursor(ctx.store,name),service.owner;timeout=Inf)
                    batch=nothing
                    for attempt in 1:3
                        service.stopped[] && break
                        try
                            batch=LiveSources.collect_source(source,cursor;limit=max_events,seconds,
                                verify=x->only(EmbeddingBridge.verify_nostr(worker,[x])),now=service.clock());break
                        catch e
                            e isa InterruptException && rethrow()
                            println(stderr,JSON3.write((source=name,attempt,error=string(typeof(e)))))
                            sleep(.25attempt)
                        end
                    end
                    batch===nothing && continue
                    result=ingest_batch!(service,name,batch,texts->EmbeddingBridge.embed(worker,texts);extractor)
                    println(JSON3.write((source=name,received=batch.received,result=result.counts,index=result.index)))
                end
                # Short sleeps make shutdown responsive without busy polling.
                until=time()+Float64(get(cfg,"poll_interval",2))
                while !service.stopped[] && time()<until;sleep(.1);end
            end
        catch e
            service.error[]=typeof(e)
            println(stderr,JSON3.write((error="ingestion_stopped",exception=string(typeof(e)))))
        finally
            worker!==nothing && close(worker)
        end
    end
end
function load_config(path)
    cfg=TOML.parsefile(path);base=dirname(abspath(path))
    for key in ("store","ingestion_config")
        haskey(cfg,key) && !isabspath(cfg[key]) && (cfg[key]=normpath(joinpath(base,cfg[key])))
    end
    cfg
end
function configured_settings(cfg)
    h=get(cfg,"http",Dict());s=get(cfg,"storage",Dict());t=get(cfg,"tree",Dict())
    n=get(cfg,"nearby",Dict());k=get(cfg,"terms",Dict())
    Settings(queue_capacity=get(h,"queue_capacity",64),timeout=Float64(get(h,"request_timeout_seconds",5)),
        body_limit=get(h,"max_body_bytes",65536),server_budget=get(s,"server_budget_mib",32)*1024^2,
        pack_refresh=Float64(get(t,"pack_refresh_seconds",5)),
        maintenance_interval=Float64(get(s,"maintenance_interval_seconds",1)),kappa=Float64(get(t,"polya_kappa",2)),
        default_limit=get(n,"default_limit",25),leaf_budget=get(n,"default_leaf_budget",4),
        max_leaves=get(n,"max_leaf_budget",16),candidates=get(n,"default_candidate_budget",256),
        max_candidates=get(n,"max_candidate_budget",1024),max_draws=get(n,"max_posting_draws",8192),
        term_limit=get(k,"default_limit",25),term_page=get(k,"max_page_size",100),
        cors=String.(get(h,"cors_origins",String[])))
end
function run_config(path;seconds=Inf)
    cfg=load_config(path);mode=get(cfg,"mode","combined");mode in ("combined","serve") || error("invalid mode")
    ingestion=TOML.parsefile(cfg["ingestion_config"])
    root=cfg["store"];storage=get(cfg,"storage",Dict());http=get(cfg,"http",Dict())
    existing=DS.detect_backend(root)!==nothing
    create=nothing
    if !existing
        mode=="serve" && error("serve mode requires an existing disk store")
        index=get(ingestion,"index",Dict())
        opts=Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in index if k in ("dimension","half_life","retention_seconds","max_leaves","split_after","reservoir_cap","max_depth","merge_below","seed"))
        opts[:dimension]=get(index,"dimension",256)
        opts[:model]=get(index,"embedding_model","futur/Qwen3-Embedding-0.6B-model2vec-onnx@a3f60a85a2570e7976ea7f2d6505ad3e5152a96f")
        disk=get(ingestion,"disk",Dict())
        opts[:mode]=Symbol(get(disk,"format","f16"));opts[:layout]=Symbol(get(disk,"layout","append"))
        create=DS.config(;opts...)
    end
    isfile(joinpath(root,"CURRENT")) && error("use a disk store; Arrow migration is not automatic")
    s=DS.open_store(root;budget=get(storage,"managed_budget_mib",256)*1024^2,create_config=create,backend=Symbol(get(storage,"backend","auto")),require_existing=mode=="serve")
    service=nothing
    try
        kw=get(ingestion,"keywords",Dict())
        if mode=="combined"
            index=get(ingestion,"index",Dict())
            s.config.model==get(index,"embedding_model",EmbeddingBridge.QWEN_MODEL) &&
                s.config.dimension==get(index,"dimension",256) || error("configured encoder differs from stored encoder; use a new store")
            get(kw,"enabled",true) || error("combined server requires keyword summaries")
            extractor=KE.configured_extractor(kw)
            DS.enable_keywords!(s,extractor;half_life=get(kw,"half_life",s.config.half_life))
            DS.backfill_keywords!(s,extractor)
        end
        service=start_server(s;settings=configured_settings(cfg),host=get(http,"host","127.0.0.1"),port=get(http,"port",8080))
        mode=="combined" && start_ingestion!(service,ingestion;device=get(cfg,"embedding_device","cpu"),
            max_events=get(cfg,"max_events",256),seconds=Float64(get(cfg,"source_seconds",5)))
        println(JSON3.write((server="ready",mode,backend=string(DS.metadata_backend(s)),port=HTTP.port(service.http),store=root)))
        deadline=time()+seconds
        while time()<deadline && service.error[]===nothing;sleep(.1);end
        service.error[]!==nothing && error("background ingestion or maintenance failed")
    finally
        if service===nothing;DS.close_store(s)
        else;stop_server(service);end
    end
end
