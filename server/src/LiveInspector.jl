# SQLite-only observer: never constructs a writable DiskTreeStore.Store.
if !isdefined(@__MODULE__, :DiskTreeStore)
    for (name,file) in ((:PostRecords,"PostRecords.jl"),(:LeafLatentTrees,"LeafLatentTree.jl"),
            (:DiskCodecs,"DiskCodecs.jl"),(:KeywordExtraction,"KeywordExtraction.jl"),(:DiskTreeStore,"DiskTreeStore.jl"))
        isdefined(@__MODULE__,name) || include(joinpath(@__DIR__,"../../src",file))
    end
end
module LiveInspector
using SQLite, DBInterface, Tables, JSON3, Serialization, Random, LinearAlgebra, HTTP
import ..DiskTreeStore, ..LeafLatentTrees
const DS=DiskTreeStore
const ASSETS=normpath(joinpath(@__DIR__,"../inspector"))
struct ReadError <: Exception
    code::String
    status::Int
end
fail(code,status=503)=throw(ReadError(code,status))
mutable struct Inspector
    root::String
    run_dir::String
    gate::ReentrantLock
    cache::Any
    cached_at::Float64
end
function Inspector(root;run_dir="")
    root=realpath(root)
    isfile(joinpath(root,"index.sqlite")) || error("Live inspector requires SQLite; RocksDB must be read through its store owner")
    Inspector(root,abspath(run_dir),ReentrantLock(),nothing,0.)
end
q(db,sql,args=())=Tables.rowtable(DBInterface.execute(db,sql,args))
one(db,sql,args=())=only(q(db,sql,args))
function deadline_callback(p::Ptr{Cvoid})::Cint
    time_ns()>unsafe_load(Ptr{UInt64}(p)) ? Cint(1) : Cint(0)
end
const DEADLINE_CALLBACK=@cfunction(deadline_callback,Cint,(Ptr{Cvoid},))

"""Short WAL read transaction. Read-only URI and query_only are both enforced."""
function read_snapshot(f,ins;seconds=4.)
    path=joinpath(ins.root,"index.sqlite")
    # Encode reserved URI bytes; preserve slashes.
    uri="file:"*replace(path,"%"=>"%25","?"=>"%3F","#"=>"%23")*"?mode=ro"
    db=SQLite.DB(uri)
    deadline=Ref(time_ns()+UInt64(round(seconds*1e9)))
    try
        SQLite.C.sqlite3_db_readonly(db.handle,"main")==1 || error("read-only open failed")
        q(db,"PRAGMA query_only=ON");q(db,"PRAGMA busy_timeout=250")
        q(db,"PRAGMA cache_size=-8192");q(db,"PRAGMA mmap_size=0");q(db,"PRAGMA temp_store=FILE")
        GC.@preserve deadline begin
            SQLite.C.sqlite3_progress_handler(db.handle,1000,DEADLINE_CALLBACK,Base.unsafe_convert(Ptr{UInt64},deadline))
            q(db,"BEGIN")
            try
                runtime=deserialize(IOBuffer(one(db,"SELECT value FROM meta WHERE key='runtime'").value))
                runtime.julia_version==string(VERSION) || fail("julia_version_mismatch")
                cfg=JSON3.read(String(one(db,"SELECT value FROM meta WHERE key='config'").value))
                f(db,runtime,cfg)
            finally
                SQLite.C.sqlite3_progress_handler(db.handle,0,C_NULL,C_NULL)
                q(db,"ROLLBACK") # Ends the read snapshot, never publishes.
            end
        end
    finally
        SQLite.C.sqlite3_progress_handler(db.handle,0,C_NULL,C_NULL)
        SQLite.close(db)
    end
end

function process_stats(ins)
    result=Any[];path=joinpath(ins.run_dir,"runner.pid")
    isfile(path) || return result
    pid=tryparse(Int,strip(read(path,String)));pid===nothing && return result
    pending=[pid];seen=Set{Int}()
    while !isempty(pending) && length(seen)<16
        p=popfirst!(pending);p in seen && continue;push!(seen,p)
        try
            cmd=replace(read("/proc/$p/cmdline",String),'\0'=>' ')
            kind=occursin("scripts/live_embedding_worker.py",cmd) ? "embedding" :
                 occursin(r"\bjulia\b.*examples/ingest_live.jl",cmd) && !occursin("timeout ",cmd) ? "julia" : ""
            if !isempty(kind)
                fields=Dict(split(l,':';limit=2)[1]=>strip(split(l,':';limit=2)[2]) for l in eachline("/proc/$p/status") if occursin(':',l))
                kb(k)=parse(Int,first(split(fields[k])))*1024
                push!(result,(kind,pid=p,rss_bytes=kb("VmRSS"),peak_bytes=kb("VmHWM"),threads=parse(Int,fields["Threads"])))
            end
            append!(pending,parse.(Int,split(read("/proc/$p/task/$p/children",String))))
        catch
            # Process may exit between procfs reads; the database is still inspectable.
        end
    end
    result
end

function snapshot(ins)
    data=read_snapshot(ins) do db,rt,cfg
        rs=q(db,"SELECT * FROM nodes ORDER BY id LIMIT 8192")
        length(rs)<8192 || fail("inspector_node_limit")
        nodes=Dict(Int(r.id)=>r for r in rs)
        counts=Dict(Int(r.id)=>Int(r.count) for r in q(db,"SELECT id,count FROM leaves"))
        mass=Dict{Int,Float64}();subcounts=Dict{Int,Int}()
        function aggregate(id)
            n=nodes[id]
            if n.axis==0
                mass[id]=n.mass*exp2(-(rt.clock-n.at)/cfg.half_life)
                subcounts[id]=get(counts,id,0)
            else
                aggregate(n.lft);aggregate(n.rgt)
                mass[id]=mass[n.lft]+mass[n.rgt];subcounts[id]=subcounts[n.lft]+subcounts[n.rgt]
            end
        end
        aggregate(1);out=Any[];lo=fill(-1.,cfg.dimension);hi=fill(1.,cfg.dimension)
        function walk(id,p)
            n=nodes[id];lv=sum(log,hi.-lo)
            push!(out,(id=string(id),parent=string(n.parent),axis=Int(n.axis),cut=n.cut,
                left=string(n.lft),right=string(n.rgt),depth=Int(n.depth),leaf=n.axis==0,
                count=subcounts[id],mass=mass[id],probability=p,log_volume=lv,
                log_density=p>0 ? log(p)-lv : nothing))
            n.axis==0 && return
            prior=(n.cut-lo[n.axis])/(hi[n.axis]-lo[n.axis])
            frac=(mass[n.lft]+2prior)/(mass[id]+2)
            old=hi[n.axis];hi[n.axis]=n.cut;walk(n.lft,p*frac);hi[n.axis]=old
            old=lo[n.axis];lo[n.axis]=n.cut;walk(n.rgt,p*(1-frac));lo[n.axis]=old
        end
        walk(1,1.)
        kw=q(db,"SELECT value FROM meta WHERE key='keywords'")
        covered=isempty(kw) ? 0 : (v=q(db,"SELECT n FROM kw_counts WHERE leaf=0 AND term=-1");isempty(v) ? 0 : only(v).n)
        sizes=q(db,"SELECT kind,SUM(bytes) AS bytes FROM files GROUP BY kind")
        (generation=string(rt.generation),at=rt.clock,model=cfg.model,dimension=cfg.dimension,
            half_life=cfg.half_life,retention_seconds=cfg.retention_seconds,format=cfg.mode,
            active=subcounts[1],leaves=length(counts),mass=mass[1],nodes=out,keyword_covered=covered,
            keyword_extractor=isempty(kw) ? nothing : String(JSON3.read(String(only(kw).value)).extractor),
            payload_bytes=sum(r.bytes for r in sizes;init=0),kappa=2.)
    end
    metadata=sum(filesize(joinpath(ins.root,n)) for n in ("index.sqlite","index.sqlite-wal","index.sqlite-shm") if isfile(joinpath(ins.root,n));init=0)
    merge(data,(store=basename(ins.root),store_id=bytes2hex(DS.sha256(ins.root)),
        observed_at=time(),metadata_bytes=metadata,processes=process_stats(ins)))
end

function addbest!(best,row,limit=10)
    push!(best,row);sort!(best;by=x->(-x.score,x.term));length(best)>limit && pop!(best)
end
function keyword_summary(db,leaf,at)
    raw=q(db,"SELECT value FROM meta WHERE key='keywords'")
    isempty(raw) && return (enabled=false,common=[],distinctive=[],covered=0,eligible=0,extractor=nothing)
    cfg=JSON3.read(String(only(raw).value))
    function countat(leaf,term)
        r=q(db,"SELECT n,mass,at FROM kw_counts WHERE leaf=? AND term=?",(leaf,term))
        isempty(r) ? (n=0,mass=0.) : (n=only(r).n,mass=only(r).mass*exp2(-(at-only(r).at)/cfg.half_life))
    end
    covered=countat(leaf,-1);eligible=countat(leaf,0);global_count=countat(0,-1)
    common=Any[];distinctive=Any[]
    sql="SELECT t.term,c.n,c.mass,c.at,g.mass AS gm,g.at AS ga FROM kw_counts c JOIN kw_terms t ON t.id=c.term JOIN kw_counts g ON g.leaf=0 AND g.term=c.term WHERE c.leaf=? AND c.term>0"
    # Stream rows into two top-ten buffers instead of retaining the vocabulary.
    for r in DBInterface.execute(db,sql,(leaf,))
        support=r.mass*exp2(-(at-r.at)/cfg.half_life)
        gm=r.gm*exp2(-(at-r.ga)/cfg.half_life)
        lift=max(0.,log(((support+1)/(covered.mass+2))/((gm+1)/(global_count.mass+2))))
        row=(term=String(r.term),count=Int(r.n),support,score=support)
        addbest!(common,row)
        r.n>=3 && lift>0 && addbest!(distinctive,merge(row,(score=support*lift,)))
    end
    (enabled=true,common,distinctive,covered=covered.n,eligible=eligible.n,
        decayed_covered=covered.mass,extractor=cfg.extractor)
end

function leaf_detail(ins,id;generation=nothing)
    read_snapshot(ins) do db,rt,cfg
        generation!==nothing && generation!=string(rt.generation) && fail("snapshot_changed",409)
        rows=q(db,"SELECT count FROM leaves WHERE id=?",(id,))
        isempty(rows) && fail("leaf_changed",409)
        count=Int(only(rows).count);posts=Any[];unavailable=0
        # Evenly spaced posting slots: bounded, illustrative examples, not a
        # semantic ranking or random scan of all posts in the leaf.
        slots=count==0 ? Int[] : unique(round.(Int,range(1,count;length=min(6,count))))
        for slot in slots
            ps=q(db,"SELECT id,kind,native,textfile,textoff,textlen FROM posts WHERE status=1 AND leaf=? AND slot=?",(id,slot))
            isempty(ps) && continue;p=only(ps)
            if basename(p.textfile)!=p.textfile || !(0<=p.textlen<=1024^2)
                unavailable+=1;continue
            end
            try
                bytes=open(joinpath(ins.root,p.textfile)) do io;seek(io,p.textoff);read(io,p.textlen);end
                length(bytes)==p.textlen || error("short payload")
                record=JSON3.read(String(bytes));key=DS.PostRecords.key_from_bytes(p.kind,p.native)
                push!(posts,(id=string(p.id),platform=String(record.platform),native_id=key.native,
                    text=first(String(record.text),8000),truncated=length(record.text)>8000,
                    author=String(record.author),created_at=record.created_at,permalink=String(record.permalink)))
            catch e
                # A concurrent compactor may unlink a payload after our metadata
                # snapshot. Skip it; never open a writer or mutate the source.
                e isa InterruptException && rethrow()
                unavailable+=1
            end
        end
        ppca=nothing
        row=one(db,"SELECT payload FROM states WHERE id=?",(id,))
        if length(row.payload)<=2*1024^2
            st=deserialize(IOBuffer(row.payload));p=LeafLatentTrees.parameters(st.sketch)
            signal=sum(abs2,p.W);noise=sum(p.D)
            ppca=(rank=size(p.W,2),reservoir=size(st.reservoir,2),signal_fraction=signal/max(eps(),signal+noise),
                mean_norm=norm(p.mean),noise_variance=sum(p.D)/length(p.D))
        end
        (id=string(id),generation=string(rt.generation),at=rt.clock,count,
            keywords=keyword_summary(db,id,rt.clock),ppca,posts,unavailable,
            sample_method="Evenly spaced posting slots; illustrative, not keyword-filtered or ranked")
    end
end

function response(code,data)
    HTTP.Response(code,["Content-Type"=>"application/json","Cache-Control"=>"no-store"],JSON3.write(data))
end
function handler(ins,req)
    req.method=="GET" || return response(405,(error="read_only_inspector",))
    uri=HTTP.URI(req.target)
    assets=Dict("/"=>"index.html","/inspector.js"=>"inspector.js","/inspector.css"=>"inspector.css")
    if haskey(assets,uri.path)
        file=assets[uri.path];mime=endswith(file,".js") ? "text/javascript" : endswith(file,".css") ? "text/css" : "text/html"
        return HTTP.Response(200,["Content-Type"=>mime,"Cache-Control"=>"no-cache"],read(joinpath(ASSETS,file)))
    end
    uri.path=="/healthz" && return response(200,(status="ok",))
    uri.path in ("/api/snapshot","/api/leaf") || return response(404,(error="not_found",))
    trylock(ins.gate) || return response(429,(error="inspector_busy",))
    try
        if uri.path=="/api/snapshot"
            if ins.cache===nothing || time()-ins.cached_at>=2
                ins.cache=snapshot(ins);ins.cached_at=time()
            end
            return response(200,ins.cache)
        end
        params=HTTP.queryparams(uri);id=tryparse(Int,get(params,"id",""))
        id!==nothing && id>0 || return response(400,(error="invalid_leaf",))
        response(200,leaf_detail(ins,id;generation=get(params,"generation",nothing)))
    catch e
        e isa InterruptException && rethrow()
        if e isa ReadError
            e.status==409 && (ins.cache=nothing) # Retry must not reuse an older tree generation.
            return response(e.status,(error=e.code,))
        end
        println(stderr,JSON3.write((inspector_error=string(typeof(e)),)))
        response(503,(error="snapshot_unavailable",))
    finally
        unlock(ins.gate)
    end
end
end
