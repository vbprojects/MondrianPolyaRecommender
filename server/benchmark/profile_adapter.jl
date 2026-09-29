# Profile the storage adapter. Run each case in a separate process:
# julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl rocksdb ingestion
# BACKEND = sqlite|rocksdb; WORKLOAD = ingestion|terms
using Profile, JSON3, Random, LinearAlgebra, Statistics, SHA
include("../src/MondrianServer.jl")
const DS=DiskTreeStore; const KE=KeywordExtraction; const MS=MondrianServer
using .PostRecords
const ROOT=normpath(joinpath(@__DIR__,"../.."))
const OUT=joinpath(ROOT,get(ENV,"MONDRIAN_PROFILE_OUTPUT","output/adapter_profile"))
const BASELINE_ONLY=get(ENV,"MONDRIAN_PROFILE_BASELINE_ONLY","0")=="1"
const SAMPLE_RATE=0.0001
BLAS.set_num_threads(1)

@noinline function profile_workload_ingestion(s,records,X,keywords)
    for first in 1:256:length(records)
        ids=first:first+255
        DS.append_batch!(s,records[ids],X[:,ids];time=100.,keywords=keywords[ids],cursor=("benchmark",string(last(ids))))
    end
    nothing
end
@noinline function profile_workload_terms(ctx,inputs)
    [MS.terms(ctx,input;time=ctx.store.clock) for input in inputs]
end

function classification(f)
    file=String(f.file);name=String(f.func)
    # Generic Julia methods can reside in another package's precompile image.
    # Its .so filename does not identify the subsystem executing that method.
    endswith(file,".so") && (startswith(name,"jfptr_") || startswith(name,"julia_")) && return nothing
    if endswith(file,"RocksRowCodec.jl")
        occursin("normalize_row",name) && return "other_adapter"
        occursin("rowkey",name) && return "key_encoding"
        return "serialization"
    elseif endswith(file,"RocksMetadata.jl")
        parts=split(name,'#';keepempty=false);isempty(parts) || (name=first(parts))
        name in ("pack","unpack") && return "serialization"
        name in ("part","key","rkey","rprefix","bstr") && return "key_encoding"
        name in ("execute","simple_select","predicates","splitfields","literal","project","scalar") && return "statement_adapter"
        name in ("flush_pending!","finish_journal!","direct!","begin!","commit!","recover!","rollback!") && return "journal_orchestration"
        return "other_adapter"
    elseif occursin("/packages/RocksDB/",file) || occursin("librocksdb",file) || occursin("rocksdb",lowercase(name))
        return "rocksdb_binding_native"
    elseif occursin("/packages/SQLite/",file) || occursin("libsqlite",file) || occursin("sqlite3",name)
        return "sqlite_binding_native"
    elseif occursin("/Serialization/",file) || endswith(file,"Serialization.jl")
        return "serialization"
    elseif endswith(file,"Terms.jl") || endswith(file,"DiskKeywords.jl")
        return "keywords_and_ranking"
    elseif endswith(file,"LeafLatentTree.jl") || occursin("/LinearAlgebra/",file)
        return "density_and_linear_algebra"
    elseif endswith(file,"DiskTreeStore.jl") || endswith(file,"DiskCodecs.jl")
        return "other_tree_payload"
    end
    nothing
end
function reclassify_saved!(prefix)
    path=joinpath(OUT,prefix*".json");result=JSON3.read(read(path,String),Dict{String,Any})
    for (kind,suffix) in (("cpu","cpu"),("alloc","alloc"))
        exportdata=JSON3.read(read(joinpath(OUT,prefix*"-"*suffix*".speedscope.json"),String))
        frames=[(func=f.name,file=f.file,line=f.line) for f in exportdata.shared.frames]
        p=only(exportdata.profiles)
        stacks=[[frames[Int(i)+1] for i in reverse(sample)] for sample in p.samples]
        weights=Int.(p.weights)
        row=only(filter(r->r["kind"]==kind,result["result"]["results"]))
        if kind=="cpu";row["metrics"]["profile"]=summarize(stacks,weights)
        else
            row["metrics"]["profile"]["bytes"]=summarize(stacks,weights)
            row["metrics"]["profile"]["counts"]=summarize(stacks,ones(Int,length(weights)))
        end
    end
    result["analysis_source_sha256"]=bytes2hex(open(sha256,@__FILE__))
    write(path,JSON3.write(result))
end
isworkload(frames)=any(f->startswith(String(f.func),"profile_workload_"),frames)
function summarize(stacks,weights)
    exclusive=Dict{String,Int}();inclusive=Dict{String,Int}();sites=Dict{String,Int}()
    for (frames,w) in zip(stacks,weights)
        cats=String[c for f in frames for c in (classification(f),) if c!==nothing]
        firstcat=isempty(cats) ? "unattributed" : first(cats)
        exclusive[firstcat]=get(exclusive,firstcat,0)+w
        for cat in unique(cats);inclusive[cat]=get(inclusive,cat,0)+w;end
        for f in frames
            classification(f)===nothing && continue
            site="$(f.func) ($(f.file):$(f.line))";sites[site]=get(sites,site,0)+w
            break
        end
    end
    total=sum(weights;init=0)
    rows(d)=[(category=k,weight=v,percent=100v/max(1,total)) for (k,v) in sort!(collect(d);by=x->-last(x))]
    (;total_weight=total,exclusive=rows(exclusive),inclusive=rows(inclusive),top_sites=first(rows(sites),min(40,length(sites))))
end
function stack_export(stacks,weights,name,unit)
    frames=Any[];ids=Dict{Tuple{String,String,Int,Bool},Int}();samples=Vector{Int}[]
    for stack in stacks
        sample=Int[]
        for f in reverse(stack)
            k=(String(f.func),String(f.file),Int(f.line),f.from_c)
            id=get!(ids,k) do
                push!(frames,(name=k[1],file=k[2],line=max(0,k[3])))
                length(frames)-1
            end
            push!(sample,id)
        end
        push!(samples,sample)
    end
    Dict("\$schema"=>"https://www.speedscope.app/file-format-schema.json","shared"=>(;frames),
        "profiles"=>[(type="sampled",name,unit,startValue=0,endValue=sum(weights;init=0),samples,weights)],"activeProfileIndex"=>0)
end
function save_cpu(prefix)
    data,dict=Profile.retrieve(;include_meta=false)
    stacks=Vector{Base.StackTraces.StackFrame}[];weights=Int[];counts=Dict{Tuple,Int}();ips=UInt[]
    for ip in data
        if ip==0
            if !isempty(ips)
                frames=reduce(vcat,(get(dict,i,Base.StackTraces.StackFrame[]) for i in ips);init=Base.StackTraces.StackFrame[])
                if isworkload(frames)
                    k=Tuple(ips)
                    if haskey(counts,k);weights[counts[k]]+=1
                    else;push!(stacks,frames);push!(weights,1);counts[k]=length(stacks);end
                end
                empty!(ips)
            end
        else;push!(ips,ip);end
    end
    isempty(weights) && error("no workload samples captured")
    write(joinpath(OUT,prefix*"-cpu.speedscope.json"),JSON3.write(stack_export(stacks,weights,prefix*" sampled workload stacks","none")))
    # Standard profiler reports retain idle/other-thread samples for inspection.
    open(joinpath(OUT,prefix*"-cpu-flat.txt"),"w") do io
        Profile.print(IOContext(io,:displaysize=>(10000,220));format=:flat,C=true,sortedby=:count,mincount=10)
    end
    summarize(stacks,weights)
end
function save_allocations(prefix)
    data=Profile.Allocs.fetch()
    selected=filter(a->isworkload(a.stacktrace),data.allocs)
    stacks=[a.stacktrace for a in selected];bytes=[a.size for a in selected]
    isempty(bytes) && error("no allocation samples captured")
    write(joinpath(OUT,prefix*"-alloc.speedscope.json"),JSON3.write(stack_export(stacks,bytes,prefix*" sampled allocation bytes","bytes")))
    (;sample_rate=SAMPLE_RATE,sampled_allocations=length(bytes),sampled_bytes=sum(bytes),
        bytes=summarize(stacks,bytes),counts=summarize(stacks,ones(Int,length(bytes))))
end
function timed_workload(f)
    GC.gc()
    v=@timed f()
    metrics=(seconds=v.time,allocated_bytes=v.bytes,gc_seconds=v.gctime,
        compile_seconds=get(v,:compile_time,0.),recompile_seconds=get(v,:recompile_time,0.))
    metrics,v.value
end
function collect_pass(f,kind,prefix)
    if kind==:baseline
        timed_workload(f)
    elseif kind==:cpu
        GC.gc();Profile.clear();Profile.init(n=8_000_000,delay=.001)
        t=@elapsed Profile.@profile f()
        result=(seconds_instrumented=t,profile=save_cpu(prefix))
        Profile.clear();result,nothing
    else
        GC.gc();Profile.Allocs.clear()
        t=@elapsed Profile.Allocs.@profile sample_rate=SAMPLE_RATE f()
        result=(seconds_instrumented=t,profile=save_allocations(prefix))
        Profile.Allocs.clear();result,nothing
    end
end
function ingestion(backend,prefix)
    N=2048;rng=MersenneTwister(82)
    X=randn(rng,Float32,384,N);X./=sqrt.(sum(abs2,X;dims=1))
    records=[PostRecord(PostKey(:threads,"bench-$i");text="Synthetic topic $(i%16)",tags=["topic$(i%16)","shared"],created_at=100.,observed_at=100.) for i in 1:N]
    e=KE.HashtagsOnlyExtractor()
    keywords=vcat([KE.prepare_keywords(e,KE.keyword_input.(records[a:a+1023])) for a in (1,1025)]...)
    cfg=DS.config(dimension=384,model="synthetic-benchmark",max_leaves=32,split_after=64,reservoir_cap=32,half_life=3600.)
    results=Any[];signatures=String[]
    for kind in (BASELINE_ONLY ? (:warmup,:baseline,:baseline,:baseline) : (:warmup,:baseline,:baseline,:baseline,:cpu,:alloc))
        println(prefix," ",kind);flush(stdout)
        mktempdir() do root
            s=DS.open_store(root;backend,budget=32*1024^2,create_config=cfg)
            try
                DS.enable_keywords!(s,e)
                f=()->profile_workload_ingestion(s,records,X,keywords)
                if kind==:warmup;f()
                else
                    metrics,_=collect_pass(f,kind,prefix)
                    push!(results,(;kind,metrics))
                end
                @assert DS.audit!(s)==N
                push!(signatures,bytes2hex(sha256(JSON3.write(DS.tree_signature(s)))))
            finally DS.close_store(s) end
        end
    end
    @assert length(unique(signatures))==1
    (;posts=N,batch_size=256,dimension=384,seed=82,results,topology_digest=first(signatures),audit_passed=true)
end
function terms(backend,prefix)
    root=joinpath(ROOT,backend==:rocksdb ? get(ENV,"MONDRIAN_PROFILE_ROCKS_STORE","data/rocks_backend_100k") : "data/server_http_pilot")
    s=DS.open_store(root;backend,budget=32*1024^2,require_existing=true)
    try
        generation=s.generation
        ctx=MS.Context(s,MS.Settings(timeout=120.))
        ids=[r.id for r in DS.rows(s,"SELECT id FROM posts WHERE status=1 ORDER BY id LIMIT 6")]
        X=DS.fetch_vectors(s,ids)
        inputs=[Dict("embedding_space_id"=>ctx.identity.embedding_space_id,"vector"=>X[:,i]) for i in axes(X,2)]
        f=()->profile_workload_terms(ctx,inputs)
        println(prefix," warmup");flush(stdout);reference=f()
        canonical(x)=JSON3.write([(leaf=r.leaf_id,terms=r.terms) for r in x])
        digest=bytes2hex(sha256(canonical(reference)));results=Any[]
        for kind in (BASELINE_ONLY ? (:baseline,:baseline,:baseline) : (:baseline,:baseline,:baseline,:cpu,:alloc))
            println(prefix," ",kind);flush(stdout)
            metrics,value=collect_pass(f,kind,prefix)
            kind==:baseline && @assert canonical(value)==canonical(reference)
            push!(results,(;kind,metrics))
        end
        @assert generation==s.generation
        @assert canonical(f())==canonical(reference)
        (;posts=DS.one(s,"SELECT COUNT(*) AS n FROM posts WHERE status=1").n,
            requests=length(inputs),query_ids=ids,leaf_ids=[r.leaf_id for r in reference],results,
            term_digest=digest,generation_unchanged=true,storage=DS.storage_stats(s))
    finally DS.close_store(s) end
end
function main()
    length(ARGS)==2 || error("usage: profile_adapter.jl sqlite|rocksdb ingestion|terms")
    backend=Symbol(ARGS[1]);workload=Symbol(ARGS[2])
    backend in (:sqlite,:rocksdb) && workload in (:ingestion,:terms) || error("invalid case")
    mkpath(OUT);prefix="$(backend)-$(workload)"
    result=workload==:ingestion ? ingestion(backend,prefix) : terms(backend,prefix)
    sources=Dict(p=>bytes2hex(open(sha256,joinpath(ROOT,p))) for p in ("src/RocksMetadata.jl","src/RocksRowCodec.jl","src/DiskTreeStore.jl","src/DiskKeywords.jl","server/src/Terms.jl","server/benchmark/profile_adapter.jl","server/Manifest.toml"))
    write(joinpath(OUT,prefix*".json"),JSON3.write((;backend,workload,julia=string(VERSION),
        julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),sources,result)))
    println("saved ",prefix);flush(stdout)
end
abspath(PROGRAM_FILE)==(@__FILE__) && main()
