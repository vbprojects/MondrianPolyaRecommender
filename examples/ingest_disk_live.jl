#!/usr/bin/env julia
using TOML, JSON3, LinearAlgebra, SHA
include("disk_live_modules.jl")
include("../src/EmbeddingBridge.jl")
include("../src/SourceAdapters.jl")
include("../src/LiveSources.jl")
include("../src/DiskIngestionRunner.jl")
using .PostRecords, .LiveSources
const DS=DiskTreeStore
const DR=DiskIngestionRunner
const KE=KeywordExtraction
const LIVE_MODEL="futur/Qwen3-Embedding-0.6B-model2vec-onnx@a3f60a85a2570e7976ea7f2d6505ad3e5152a96f"
function main(args)
    options=Dict{String,String}();i=1
    while i<=length(args)
        key=args[i]
        key in ("--help","-h") && (println("julia --project=environments/ingestion examples/ingest_live.jl --storage disk [--config path.toml] [--store data/live_disk] [--replay file.jsonl] [--max-events 256] [--seconds 10] [--rounds 1] [--loop] [--backend cpu] [--metadata-backend auto|sqlite|rocksdb] [--embedding-device cpu|cuda|auto] [--cache-mib 256] [--keyword-rebuild]");return)
        if key in ("--loop","--keyword-rebuild");options[key]="true"
        else
            key in ("--config","--store","--replay","--max-events","--seconds","--rounds","--backend","--metadata-backend","--embedding-device","--cache-mib") || error("unknown option $key")
            i<length(args) || error("missing value for $key");i+=1;options[key]=args[i]
        end
        i+=1
    end
    cfg=haskey(options,"--config") ? TOML.parsefile(options["--config"]) : Dict{String,Any}()
    root=get(options,"--store",get(cfg,"store","data/live_disk"))
    isfile(joinpath(root,"CURRENT")) && error("Arrow checkpoint found; use --storage arrow or a new disk store")
    limit=parse(Int,get(options,"--max-events","256"));seconds=parse(Float64,get(options,"--seconds","10"));rounds=parse(Int,get(options,"--rounds","1"))
    1<=limit<=1024 && isfinite(seconds) && seconds>0 && rounds>=1 || error("invalid run budget")
    get(options,"--backend","cpu")=="cpu" || error("disk storage/tree path currently requires --backend cpu; embedding inference may use CUDA")
    BLAS.set_num_threads(1)
    index=get(cfg,"index",Dict{String,Any}());storage=get(cfg,"disk",Dict{String,Any}())
    get(index,"rank",4)==4 || error("disk tree uses rank-4 PPCA")
    keys_allowed=Set(["dimension","half_life","retention_seconds","max_leaves","split_after","reservoir_cap","max_depth","merge_below","seed"])
    opts=Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in index if k in keys_allowed)
    opts[:dimension]=get(index,"dimension",256)
    opts[:model]=get(index,"embedding_model",LIVE_MODEL)
    opts[:mode]=Symbol(get(storage,"format","f16"));opts[:layout]=Symbol(get(storage,"layout","append"))
    budget=parse(Int,get(options,"--cache-mib",string(get(storage,"cache_mib",256))))*1024^2
    sources=get(cfg,"sources",[Dict("name"=>"bluesky","kind"=>"bluesky")])
    names=String[s["name"] for s in sources]
    length(names)==length(unique(names)) && all(n->!startswith(n,"__"),names) || error("invalid source names")
    s=DS.open_store(root;budget,create_config=DS.config(;opts...),backend=Symbol(get(options,"--metadata-backend",get(storage,"backend","auto"))));worker=nothing
    try
        s.config.model==opts[:model] && s.config.dimension==opts[:dimension] || error("configured encoder differs from stored encoder; use a new store")
        worker=EmbeddingBridge.Worker(;device=get(options,"--embedding-device","cpu"),model=opts[:model])
        s.config.model==worker.model && s.config.dimension==worker.dimension || error("stored embedding model/dimension does not match worker")
        println(JSON3.write((embedding_model=worker.model,dimension=worker.dimension,store=root)));flush(stdout)
        kwcfg=get(cfg,"keywords",Dict{String,Any}());extractor=KE.configured_extractor(kwcfg)
        if get(kwcfg,"enabled",true)
            DS.enable_keywords!(s,extractor;half_life=get(kwcfg,"half_life",s.config.half_life),rebuild=haskey(options,"--keyword-rebuild"))
            # Each page is committed and resumable; no full-corpus load or re-embedding.
            while true
                n=DS.backfill_keywords!(s,extractor;batch_size=128,max_batches=1)
                n==0 && break
                println(JSON3.write((backfilled=n,generation=s.generation)));flush(stdout)
            end
        elseif DS.keyword_enabled(s)
            error("store has keywords enabled; keep enabled or use a separate store")
        end
        embed=texts->EmbeddingBridge.embed(worker,texts)
        verify=x->only(EmbeddingBridge.verify_nostr(worker,[x]))
        function publish(source,batch,now)
            result=DR.commit_batch!(s,source,batch,embed;extractor,now)
            DR.reclaim_daily!(s)
            println(JSON3.write((source,received=batch.received,complete=batch.complete,result=result.counts,index=DR.status(s))))
            flush(stdout)
        end
        if haskey(options,"--replay")
            path=options["--replay"];source="replay:"*bytes2hex(open(sha256,path))
            cursor=DR.source_cursor(s,source);firstline=isempty(cursor) ? 0 : parse(Int,cursor)
            pending=PostEvent[];lastline=firstline;count=0;now=s.clock
            for (line,text) in enumerate(eachline(path))
                line<=firstline && continue
                events,observed=DR.normalize_fixture(JSON3.read(text,Dict{String,Any});verify)
                # Do not split one source record's expanded events across cursors.
                length(events)<=1024 || error("source record expands beyond event budget")
                if !isempty(pending) && length(pending)+length(events)>1024
                    publish(source,SourceBatch(pending,string(lastline),count,true),now)
                    pending=PostEvent[];count=0
                end
                append!(pending,events);now=max(now,observed);lastline=line;count+=1
                if count>=limit
                    publish(source,SourceBatch(pending,string(lastline),count,true),now)
                    pending=PostEvent[];count=0
                end
            end
            count>0 && publish(source,SourceBatch(pending,string(lastline),count,true),now)
        else
            round=0
            while haskey(options,"--loop") || round<rounds
                round+=1
                for source in sources
                    name=String(source["name"]);now=max(time(),s.clock);batch=nothing
                    for attempt in 1:3
                        try
                            batch=collect_source(source,DR.source_cursor(s,name);limit,seconds,verify,now);break
                        catch e
                            e isa InterruptException && rethrow()
                            println(stderr,JSON3.write((source=name,attempt,error=e isa SourceError ? e.message : string(typeof(e)))))
                            attempt<3 && sleep(min(2.,.25*2^(attempt-1)))
                        end
                    end
                    if batch===nothing
                        DR.maintenance!(s;now=max(time(),s.clock));DR.reclaim_daily!(s);continue
                    end
                    publish(name,batch,max(time(),s.clock))
                end
                (haskey(options,"--loop") || round<rounds) && sleep(Float64(get(cfg,"poll_interval",2)))
            end
        end
        println(JSON3.write((store=s.root,index=DR.status(s),storage=DS.storage_stats(s),keywords=DS.keyword_stats(s))))
    finally
        try worker!==nothing && close(worker) finally DS.close_store(s) end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main(ARGS)
