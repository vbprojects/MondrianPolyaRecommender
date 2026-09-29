# Isolated metadata-engine ablation. This is NOT a replacement tree backend.
using SQLite, DBInterface, Tables, RocksDB, JSON3, Random, Statistics, SHA, Serialization
const ROOT=normpath(joinpath(@__DIR__,"../.."))
const OUT=joinpath(ROOT,"output/server")
pad(n)=lpad(string(n),12,'0')
pkey(id)="P/"*pad(id)
skey(leaf,slot)="S/"*pad(leaf)*"/"*pad(slot)
tkey(leaf,term)="T/"*pad(leaf)*"/"*pad(term)
mutable struct Engine
    kind::String
    db::Any
    put::Any
    del::Any
    get::Any
    range::Any
    write_options::Any
    root::String
end
function engine(kind,root)
    mkpath(root)
    if kind=="sqlite"
        db=SQLite.DB(joinpath(root,"index.sqlite"))
        for sql in ("PRAGMA journal_mode=WAL","PRAGMA synchronous=FULL","PRAGMA cache_size=-32768","PRAGMA mmap_size=0",
                    "CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY,v BLOB NOT NULL) WITHOUT ROWID")
            Tables.rowtable(DBInterface.execute(db,sql))
        end
        Engine(kind,db,SQLite.Stmt(db,"INSERT OR REPLACE INTO kv VALUES (?,?)"),
            SQLite.Stmt(db,"DELETE FROM kv WHERE k=?"),SQLite.Stmt(db,"SELECT v FROM kv WHERE k=?"),
            SQLite.Stmt(db,"SELECT k,v FROM kv WHERE k>=? AND k<? ORDER BY k LIMIT ?"),nothing,root)
    else
        db=RocksDB.opendb(root;options=RocksDB.Options(compression=:none,
            block_cache_size=16*1024^2,write_buffer_size=8*1024^2,max_background_jobs=2))
        Engine(kind,db,nothing,nothing,nothing,nothing,RocksDB.WriteOptions(sync=true),root)
    end
end
function commit(e,updates)
    if e.kind=="sqlite"
        DBInterface.execute(e.db,"BEGIN IMMEDIATE")
        try
            for (k,v) in updates
                v===nothing ? DBInterface.execute(e.del,(k,)) : DBInterface.execute(e.put,(k,v))
            end
            DBInterface.execute(e.db,"COMMIT")
        catch
            DBInterface.execute(e.db,"ROLLBACK");rethrow()
        end
    else
        wb=RocksDB.WriteBatch()
        try
            for (k,v) in updates
                v===nothing ? delete!(wb,k) : put!(wb,k,v)
            end
            RocksDB.write!(e.db,wb;options=e.write_options)
        finally finalize(wb) end
    end
end
function getvalue(e,k)
    if e.kind=="sqlite"
        r=Tables.rowtable(DBInterface.execute(e.get,(k,)))
        isempty(r) ? nothing : Vector{UInt8}(only(r).v)
    else
        get(e.db,k)
    end
end
function rangevalues(e,prefix;limit=25)
    if e.kind=="sqlite"
        [(String(r.k),Vector{UInt8}(r.v)) for r in Tables.rowtable(DBInterface.execute(e.range,(prefix,prefix*"~",limit)))]
    else
        it=RocksDB.DBIterator(e.db);out=Tuple{String,Vector{UInt8}}[]
        try
            RocksDB.seek!(it,prefix)
            while RocksDB.valid(it) && length(out)<limit
                key=String(RocksDB.key(it));startswith(key,prefix) || break
                push!(out,(key,copy(RocksDB.value(it))));RocksDB.advance!(it)
            end
        finally finalize(it) end
        out
    end
end
function settle(e)
    if e.kind=="sqlite"
        Tables.rowtable(DBInterface.execute(e.db,"PRAGMA wal_checkpoint(TRUNCATE)"))
    else
        RocksDB.compact!(e.db)
    end
end
function closeengine(e)
    if e.kind=="sqlite"
        for st in (e.put,e.del,e.get,e.range);DBInterface.close!(st);end
        SQLite.close(e.db)
    else
        close(e.db);finalize(e.write_options)
    end
end
function io_bytes()
    Dict(split(x,':')[1]=>parse(Int,strip(split(x,':')[2])) for x in eachline("/proc/self/io"))
end
function rss()
    for line in eachline("/proc/self/status")
        startswith(line,"VmHWM:") && return parse(Int,split(line)[2])*1024
    end
    0
end
bytes(x)=Vector{UInt8}(codeunits(JSON3.write(x)))
valueint(e,k)=let v=getvalue(e,k);v===nothing ? 0 : parse(Int,String(v));end
function fixtures()
    # Read-only sources; join stable native identity, not inferred row correspondence.
    db=SQLite.DB(joinpath(ROOT,"data/keyword_experiment/yake/index.sqlite"))
    DBInterface.execute(db,"PRAGMA query_only=ON")
    DBInterface.execute(db,"ATTACH DATABASE ? AS layout",(joinpath(ROOT,"data/disk_ablation/17-f16-append/index.sqlite"),))
    records=Tables.rowtable(DBInterface.execute(db,"SELECT p.id,p.kind,p.native,p.created,p.sequence,p.block,p.row,p.leaf,k.terms FROM layout.posts p JOIN main.posts q ON p.id=q.id AND p.kind=q.kind AND p.native=q.native AND p.digest=q.digest JOIN main.kw_posts k ON k.id=q.id WHERE p.status=1 ORDER BY p.id"))
    out=[(id=r.id,leaf=r.leaf,kind=r.kind,native=String(Vector{UInt8}(r.native)),block=r.block,row=r.row,
        terms=deserialize(IOBuffer(Vector{UInt8}(r.terms)))) for r in records]
    SQLite.close(db);length(out)==100000 || error("fixture mismatch");out
end
function trial(kind,seed)
    records=fixtures();rng=MersenneTwister(seed)
    path=joinpath(ROOT,"data/server_engine_ablation","$kind-$seed")
    isdir(path) && error("benchmark directory already exists: $path")
    e=engine(kind,path);count=Dict{Int,Int}();termcounts=Dict{String,Int}()
    function updates(batch)
        ops=Pair{String,Union{Nothing,Vector{UInt8}}}[];dirty=Set{String}()
        for r in batch
            slot=get(count,r.leaf,0)+1;count[r.leaf]=slot
            push!(ops,pkey(r.id)=>bytes((leaf=r.leaf,slot,block=r.block,row=r.row)))
            push!(ops,"I/$(r.kind)/$(r.native)"=>bytes(r.id))
            push!(ops,skey(r.leaf,slot)=>bytes(r.id))
            for term in r.terms,leaf in (0,r.leaf)
                k=tkey(leaf,term);termcounts[k]=get(termcounts,k,0)+1;push!(dirty,k)
            end
        end
        for k in sort!(collect(dirty));push!(ops,k=>bytes(termcounts[k]));end
        for leaf in unique(r.leaf for r in batch);push!(ops,"C/"*pad(leaf)=>bytes(count[leaf]));end
        push!(ops,"cursor"=>bytes(last(batch).id));ops
    end
    # Compile each engine path outside measurement.
    mktempdir() do tmp
        w=engine(kind,joinpath(tmp,"warm"));commit(w,["a"=>bytes(1)]);getvalue(w,"a");rangevalues(w,"a");settle(w);closeengine(w)
    end
    logical=0;write_seconds=0.;read_seconds=Float64[];before=io_bytes();beg=time_ns()
    for start in 1:1024:length(records)
        batch=records[start:min(start+1023,end)];ops=updates(batch)
        logical+=sum(ncodeunits(k)+(v===nothing ? 0 : length(v)) for (k,v) in ops)
        write_seconds+=@elapsed commit(e,ops)
        # Fixed 256-candidate metadata lookup: leaf slot -> post location.
        leaves=sort(collect(keys(count)));chosen=rand(rng,leaves,min(4,length(leaves)))
        t=@elapsed begin
            for leaf in chosen, _ in 1:64
                id=valueint(e,skey(leaf,rand(rng,1:count[leaf])))
                id>0 && getvalue(e,pkey(id))!==nothing || error("missing posting")
            end
            # A bounded term prefix read, not the server's full support ranking.
            rangevalues(e,"T/"*pad(first(chosen))*"/";limit=25)
        end
        push!(read_seconds,t)
    end
    elapsed=(time_ns()-beg)/1e9
    settle_seconds=@elapsed settle(e)
    ingestion_io=io_bytes();stage=Dict{String,Any}()
    # Model split then merge: move one substantial leaf's postings twice;
    # vectors remain external and unchanged. Rebuild only affected leaf aggregates.
    leaf=first(sort(collect(keys(count));by=l->-count[l]))
    members=filter(r->r.leaf==leaf,records);half=members[1:2:end];newleaf=maximum(keys(count))+1
    function move_rows(rows,target)
        staged=Dict{String,Union{Nothing,Vector{UInt8}}}()
        readkey(k)=haskey(staged,k) ? staged[k] : getvalue(e,k)
        readint(k)=parse(Int,String(readkey(k)))
        ops=Pair{String,Union{Nothing,Vector{UInt8}}}[];dirty=Set{String}()
        for r in rows
            old=JSON3.read(String(readkey(pkey(r.id))))
            lastid=readint(skey(old.leaf,count[old.leaf]))
            if old.slot<count[old.leaf]
                last=JSON3.read(String(readkey(pkey(lastid))))
                push!(ops,skey(old.leaf,old.slot)=>bytes(lastid))
                push!(ops,pkey(lastid)=>bytes((leaf=old.leaf,slot=old.slot,block=last.block,row=last.row)))
            end
            # Overlay makes the next dense-slot swap see staged relocations.
            push!(ops,skey(old.leaf,count[old.leaf])=>nothing);count[old.leaf]-=1
            count[target]=get(count,target,0)+1
            push!(ops,skey(target,count[target])=>bytes(r.id))
            push!(ops,pkey(r.id)=>bytes((leaf=target,slot=count[target],block=r.block,row=r.row)))
            for term in r.terms
                for (l,delta) in ((old.leaf,-1),(target,1))
                    k=tkey(l,term);termcounts[k]=get(termcounts,k,0)+delta;push!(ops,k=>bytes(termcounts[k]))
                end
            end
            push!(ops,"C/"*pad(old.leaf)=>bytes(count[old.leaf]))
            push!(ops,"C/"*pad(target)=>bytes(count[target]))
            for (k,v) in ops;staged[k]=v;end
            empty!(ops)
        end
        commit(e,sort!(collect(staged);by=first))
    end
    # Publish each bounded 128-post relocation atomically.
    stage["maintenance_policy"]="one atomic batch per relocation/deletion operation"
    stage["migration_posts"]=min(128,length(half));rows=half[1:stage["migration_posts"]]
    stage["split_seconds"]=@elapsed move_rows(rows,newleaf)
    stage["merge_seconds"]=@elapsed move_rows(rows,leaf)
    # Expiry/deletion batch: remove a contiguous identity range and its metadata.
    deleted=records[end-1023:end]
    staged=Dict{String,Union{Nothing,Vector{UInt8}}}()
    readkey(k)=haskey(staged,k) ? staged[k] : getvalue(e,k)
    readint(k)=parse(Int,String(readkey(k)))
    t=@elapsed begin
    for r in reverse(deleted)
        old=JSON3.read(String(readkey(pkey(r.id))))
        lastid=readint(skey(old.leaf,count[old.leaf]))
        ops=Pair{String,Union{Nothing,Vector{UInt8}}}[]
        if lastid!=r.id
            last=JSON3.read(String(readkey(pkey(lastid))))
            push!(ops,skey(old.leaf,old.slot)=>bytes(lastid))
            push!(ops,pkey(lastid)=>bytes((leaf=old.leaf,slot=old.slot,block=last.block,row=last.row)))
        end
        push!(ops,skey(old.leaf,count[old.leaf])=>nothing);count[old.leaf]-=1
        push!(ops,pkey(r.id)=>nothing);push!(ops,"I/$(r.kind)/$(r.native)"=>nothing)
        for term in r.terms,l in (0,old.leaf)
            k=tkey(l,term);termcounts[k]-=1;push!(ops,k=>bytes(termcounts[k]))
        end
        push!(ops,"C/"*pad(old.leaf)=>bytes(count[old.leaf]))
        for (k,v) in ops;staged[k]=v;end
    end
    commit(e,sort!(collect(staged);by=first))
    end
    stage["delete_1024_seconds"]=t
    compact_seconds=@elapsed settle(e);after=io_bytes()
    digest=bytes2hex(sha256(JSON3.write([(l,count[l]) for l in sort(collect(keys(count)))])))
    @assert valueint(e,"cursor")==100000
    @assert all(getvalue(e,pkey(r.id))===nothing for r in deleted)
    @assert all(valueint(e,k)==v for (k,v) in termcounts)
    for l in keys(count),slot in 1:count[l]
        id=valueint(e,skey(l,slot));v=JSON3.read(String(getvalue(e,pkey(id))))
        @assert v.leaf==l && v.slot==slot
    end
    closeengine(e)
    e=engine(kind,path);@assert valueint(e,"cursor")==100000;closeengine(e)
    files=sum(filesize(joinpath(dir,f)) for (dir,_,fs) in walkdir(path) for f in fs)
    result=(;kind,seed,posts=length(records),durable_batch=1024,write_seconds,posts_per_write_second=length(records)/write_seconds,
        elapsed_with_reads_and_fixture_ops=elapsed,settle_seconds,compact_seconds,
        query_p50_ms=1000median(read_seconds),query_p95_ms=1000quantile(read_seconds,.95),query_p99_ms=1000quantile(read_seconds,.99),
        logical_ingestion_bytes=logical,ingestion_os_write_bytes=ingestion_io["write_bytes"]-before["write_bytes"],
        all_os_write_bytes=after["write_bytes"]-before["write_bytes"],final_files_bytes=files,
        process_peak_rss=rss(),stage,counts_digest=digest,correctness="passed")
    mkpath(OUT);write(joinpath(OUT,"metadata-$kind-$seed.json"),JSON3.write(result));println(JSON3.write(result))
end
trial(ARGS[1],parse(Int,ARGS[2]))
