#!/usr/bin/env julia
# Run with --project=environments/ingestion. All source operations are reads.
using TOML, JSON3, LinearAlgebra, CUDA, SHA
include("live_modules.jl")
include("../src/EmbeddingBridge.jl")
include("../src/SourceAdapters.jl")
include("../src/LiveSources.jl")
include("../src/IngestionRunner.jl")
using .PostRecords, .LeafTreeKernels, .LiveIndex, .ArrowPostStore, .LiveSources, .IngestionRunner
function main(args)
    options=Dict{String,String}();i=1
    while i<=length(args)
        key=args[i]
        key in ("--help","-h") && (println("julia --project=environments/ingestion examples/ingest_live.jl [--config path.toml] [--store data/live] [--replay file.jsonl] [--max-events 256] [--seconds 10] [--rounds 1] [--loop] [--backend cpu|cuda] [--embedding-device cpu|cuda|auto]");return)
        if key=="--loop";options[key]="true"
        else
            key in ("--config","--store","--replay","--max-events","--seconds","--rounds","--backend","--embedding-device") || error("unknown option $key")
            i<length(args) || error("missing value for $key");i+=1;options[key]=args[i]
        end
        i+=1
    end
    cfg=haskey(options,"--config") ? TOML.parsefile(options["--config"]) : Dict{String,Any}()
    root=get(options,"--store",get(cfg,"store","data/live"));limit=parse(Int,get(options,"--max-events","256"))
    seconds=parse(Float64,get(options,"--seconds","10"));rounds=parse(Int,get(options,"--rounds","1"))
    1<=limit<=1024 && seconds>0 && rounds>=1 || error("invalid run budget")
    backendname=get(options,"--backend","cpu");backendname in ("cpu","cuda") || error("unknown backend")
    BLAS.set_num_threads(1)
    backendname=="cuda" && !CUDA.functional() && error("CUDA backend unavailable")
    CUDA.allowscalar(false)
    backend=KernelTreeBackend(backendname=="cpu" ? Array : CUDA.CuArray)
    sources=get(cfg,"sources",[Dict("name"=>"bluesky","kind"=>"bluesky")])
    names=String[s["name"] for s in sources]
    length(names)==length(unique(names)) && all(n->!startswith(n,"__"),names) || error("source names must be unique and cannot start with __")
    with_store_lock(root) do
        config=IndexConfig(;Dict(Symbol(k)=>v for (k,v) in get(cfg,"index",Dict()))...)
        existing=isfile(joinpath(root,"CURRENT"));index=existing ? restore(root;backend) : Index(config;backend)
        existing && haskey(cfg,"index") && JSON3.write(index.config)!=JSON3.write(config) && error("index configuration differs from checkpoint; use the original configuration")
        worker=EmbeddingBridge.Worker(;device=get(options,"--embedding-device","cpu"))
        try
            index.config.embedding_model==worker.model && index.config.dimension==worker.dimension || error("checkpoint embedding model does not match worker")
            embed=texts->EmbeddingBridge.embed(worker,texts);verify=x->only(EmbeddingBridge.verify_nostr(worker,[x]))
            !existing && checkpoint!(index,root)
            if haskey(options,"--replay")
                path=options["--replay"];source="replay:"*bytes2hex(open(sha256,path))
                firstline=parse(Int,get(index.cursors,source,"0"));pending=PostEvent[];lastline=firstline;count=0;now=index.clock
                for (line,text) in enumerate(eachline(path))
                    line<=firstline && continue
                    row=JSON3.read(text,Dict{String,Any});events,observed=normalize_fixture(row;verify)
                    append!(pending,events);now=max(now,observed);lastline=line;count+=1
                    if count>=limit
                        batch=SourceBatch(pending,string(lastline),count,true)
                        result=commit_batch!(index,root,source,batch,embed;now)
                        println(JSON3.write((source="replay",received=count,result=result.counts,index=result.audit)))
                        pending=PostEvent[];count=0
                    end
                end
                if count>0
                    result=commit_batch!(index,root,source,SourceBatch(pending,string(lastline),count,true),embed;now)
                    println(JSON3.write((source="replay",received=count,result=result.counts,index=result.audit)))
                end
            else
                round=0
                while haskey(options,"--loop") || round<rounds
                    round+=1
                    for source in sources
                        name=String(source["name"]);now=max(time(),index.clock);batch=nothing
                        for attempt in 1:3
                            try
                                batch=collect_source(source,get(index.cursors,name,"");limit,seconds,verify,now);break
                            catch e
                                e isa InterruptException && rethrow()
                                # No payload, authorization header or token-bearing
                                # URL is included in diagnostics.
                                println(stderr,JSON3.write((source=name,attempt=attempt,error=e isa SourceError ? e.message : string(typeof(e)))))
                                attempt<3 && sleep(min(2.,.25*2^(attempt-1)))
                            end
                        end
                        if batch===nothing
                            daily_maintenance!(index;now);checkpoint!(index,root);prune_checkpoints!(root)
                            continue
                        end
                        result=commit_batch!(index,root,name,batch,embed;now=max(time(),index.clock))
                        println(JSON3.write((source=name,received=batch.received,complete=batch.complete,result=result.counts,index=result.audit)))
                    end
                    (haskey(options,"--loop") || round<rounds) && sleep(Float64(get(cfg,"poll_interval",2)))
                end
            end
            println(JSON3.write((checkpoint=checkpoint_path(root),index=audit(index))))
        finally close(worker) end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main(ARGS)
