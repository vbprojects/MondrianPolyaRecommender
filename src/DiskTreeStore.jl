"""Experimental single-writer disk tree. Include LeafLatentTree, PostRecords,
and DiskCodecs first. SQLite or RocksDB metadata is authoritative; immutable payload files are fsynced
before their references commit. Live ingestion uses this store through DiskIngestionRunner.
"""
module DiskTreeStore
using SQLite, DBInterface, Tables, JSON3, Random, LinearAlgebra, Serialization, SHA, Statistics
using ..LeafLatentTrees, ..PostRecords, ..DiskCodecs
const LT=LeafLatentTrees
const DC=DiskCodecs
export open_store, close_store, append_batch!, lookup, fetch_vectors, fetch_records,
       sample_ids, candidate_leaves, leaf_refs, delete_post!, expire!, maintain!, compact!,
       rebuild_density!, compact_texts!, storage_stats, audit!, clear_caches!, get_state, frame_for,
       config, load_nodes!, commit!, collect_garbage!, tree_signature, polya_weights,
       metadata_backend, detect_backend, import_sqlite_to_rocks
const MiB=1024^2
Base.find_package("RocksDB")!==nothing && include("RocksMetadata.jl")
metadata_backend(s)=s.db isa SQLite.DB ? :sqlite : :rocksdb
function detect_backend(root)
    sql=isfile(joinpath(root,"index.sqlite"));rocks=isfile(joinpath(root,"metadata.rocks","CURRENT"))
    sql && rocks && error("ambiguous store: both SQLite and RocksDB metadata exist")
    sql ? :sqlite : rocks ? :rocksdb : nothing
end
serialize_bytes(x)=take!(let io=IOBuffer();serialize(io,x);io;end)
deserialize_bytes(x)=deserialize(IOBuffer(Vector{UInt8}(x)))
function config(;dimension=384,mode=:f16,layout=:append,half_life=21600.,retention_seconds=7*86400.,
        max_leaves=128,split_after=256,reservoir_cap=64,max_depth=24,merge_below=8.,seed=17,
        model="TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3")
    0<dimension<=4096 && Symbol(mode) in DC.FORMATS && Symbol(layout) in (:append,:leaf) || error("invalid storage format")
    half_life>0 && retention_seconds>0 && max_leaves>0 && split_after>=16 && reservoir_cap>=16 &&
        0<=merge_below<split_after && max_depth>0 || error("invalid tree configuration")
    (dimension=Int(dimension),mode=Symbol(mode),layout=Symbol(layout),half_life=Float64(half_life),
     retention_seconds=Float64(retention_seconds),max_leaves=Int(max_leaves),split_after=Int(split_after),
     reservoir_cap=Int(reservoir_cap),max_depth=Int(max_depth),merge_below=Float64(merge_below),seed=Int(seed),model=String(model))
end
mutable struct Cache
    capacity::Int
    used::Int
    peak::Int
    tick::Int
    entries::Dict{Int64,Tuple{Any,Int,Int,Bool}}
    hits::Int
    misses::Int
    evictions::Int
end
Cache(cap)=Cache(cap,0,0,0,Dict(),0,0,0)
mutable struct Node
    id::Int64
    parent::Int64
    axis::Int
    cut::Float64
    left::Int64
    right::Int64
    depth::Int
    mass::Float64
    at::Float64
end
mutable struct LeafState
    sketch::LT.Sketch
    reservoir::Matrix{Float32}
    sequences::Vector{Int64}
    at::Float64
    arrivals::Float64
    seen::Int64
end
LeafState(d,t)=LeafState(LT.Sketch(:ppca,min(4,d),d),zeros(Float32,d,0),Int64[],t,0.,0)
mutable struct Store
    root::String
    db::Any
    config::NamedTuple
    nodes::Dict{Int64,Node}
    state_cache::Cache
    block_cache::Cache
    budget::Int
    sqlite_budget::Int
    workspace::Int
    rng::MersenneTwister
    basisrng::MersenneTwister
    next_id::Int64
    next_node::Int64
    next_block::Int64
    sequence::Int64
    generation::Int64
    clock::Float64
    lockfile::IOStream
    transaction::Bool
    poisoned::Bool
    packs::Dict{Symbol,Tuple{String,IOStream}}
    statements::Dict{String,SQLite.Stmt}
    metrics::Dict{String,Float64}
    request_blocks::Set{Int64}
    request_decoders::Set{String}
    keyword_config::Any
    keyword_cache::Any
end
function statement(s,sql)
    get!(s.statements,sql) do;SQLite.Stmt(s.db,sql);end
end
exec(s,sql,args=())=s.db isa SQLite.DB ? DBInterface.execute(statement(s,sql),args) : RocksMetadata.execute(s.db,sql,args)
rows(s,sql,args=())=s.db isa SQLite.DB ? Tables.rowtable(exec(s,sql,args)) : exec(s,sql,args)
one(s,sql,args=())=only(rows(s,sql,args))
metric!(s,key,value=1.)=(s.metrics[key]=get(s.metrics,key,0.)+value)
function sqlite_usage(s)
    s.db isa SQLite.DB || return Dict{String,Int}()
    result=Dict{String,Int}()
    for (name,code) in (("cache_bytes",1),("cache_hits",7),("cache_misses",8),("cache_writes",9),("cache_spills",12))
        current=Ref{Cint}(0);high=Ref{Cint}(0)
        SQLite.C.sqlite3_db_status(s.db.handle,code,current,high,0)==0 || error("SQLite memory instrumentation failed")
        result[name]=Int(current[])
    end
    result
end
function save_state(s,id,state)
    s.transaction || error("dirty leaf outside transaction")
    data=serialize_bytes(state)
    exec(s,"INSERT OR REPLACE INTO states(id,payload) VALUES (?,?)",(id,data))
    metric!(s,"state_write_bytes",length(data))
end
function evict!(s,c)
    isempty(c.entries) && return
    id=first(keys(c.entries));oldest=typemax(Int)
    for (k,(_,_,tick,_)) in c.entries
        if tick<oldest;id=k;oldest=tick;end
    end
    value,n,_,dirty=pop!(c.entries,id);dirty && save_state(s,id,value)
    c.used-=n;c.evictions+=1
end
function cache_put!(s,c,id,value;dirty=false)
    id=Int64(id)
    if haskey(c.entries,id)
        _,n,_,wasdirty=pop!(c.entries,id);c.used-=n;dirty|=wasdirty
    end
    n=Base.summarysize(value);n<=c.capacity || error("one cached item exceeds budget")
    while c.used+n>c.capacity;evict!(s,c);end
    c.tick+=1;c.entries[id]=(value,n,c.tick,dirty);c.used+=n;c.peak=max(c.peak,c.used)
    value
end
function cache_get(s,c,id,loader)
    id=Int64(id);c.tick+=1
    if haskey(c.entries,id)
        v,n,_,dirty=c.entries[id];c.entries[id]=(v,n,c.tick,dirty);c.hits+=1;return v
    end
    c.misses+=1;cache_put!(s,c,id,loader())
end
function remove_cached!(c,id)
    if haskey(c.entries,id);_,n,_,_=pop!(c.entries,id);c.used-=n;end
end
function get_state(s,id)
    cache_get(s,s.state_cache,id,()->begin
        data=one(s,"SELECT payload FROM states WHERE id=?",(id,)).payload
        metric!(s,"state_read_bytes",length(data));deserialize_bytes(data)
    end)
end
function dirty_state!(s,id,st)
    cache_put!(s,s.state_cache,id,st;dirty=true)
    n=s.nodes[id];n.mass=st.sketch.mass;n.at=st.at
end
function flush_states!(s)
    for (id,(v,n,t,dirty)) in s.state_cache.entries
        dirty || continue;save_state(s,id,v);s.state_cache.entries[id]=(v,n,t,false)
    end
end
function clear_caches!(s)
    s.keyword_cache=nothing
    s.transaction && flush_states!(s)
    for c in (s.state_cache,s.block_cache)
        any(x->x[4],values(c.entries)) && error("uncommitted cache")
        empty!(c.entries);c.used=0
    end
    exec(s,"PRAGMA shrink_memory");nothing
end
function schema!(db)
    for sql in [
        "CREATE TABLE meta(key TEXT PRIMARY KEY,value BLOB NOT NULL)",
        "CREATE TABLE nodes(id INTEGER PRIMARY KEY,parent INTEGER,axis INTEGER,cut REAL,lft INTEGER,rgt INTEGER,depth INTEGER,mass REAL,at REAL)",
        "CREATE TABLE states(id INTEGER PRIMARY KEY,payload BLOB NOT NULL)",
        "CREATE TABLE leaves(id INTEGER PRIMARY KEY,count INTEGER NOT NULL)",
        "CREATE TABLE identities(kind INTEGER,native BLOB,id INTEGER,PRIMARY KEY(kind,native)) WITHOUT ROWID",
        "CREATE TABLE posts(id INTEGER PRIMARY KEY,kind INTEGER,native BLOB,revision REAL,observed REAL,created REAL,density_at REAL,sequence INTEGER,status INTEGER,leaf INTEGER,slot INTEGER,block INTEGER,row INTEGER,textfile TEXT,textoff INTEGER,textlen INTEGER,digest TEXT)",
        "CREATE UNIQUE INDEX leaf_slots ON posts(leaf,slot) WHERE status=1",
        "CREATE INDEX leaf_sequence ON posts(leaf,sequence,id) WHERE status=1",
       "CREATE INDEX active_sequence ON posts(sequence,id) WHERE status=1",
        "CREATE INDEX post_blocks ON posts(block,row) WHERE status=1",
        "CREATE INDEX expiry ON posts(status,created,id)",
        "CREATE INDEX text_files ON posts(textfile,status)",
        "CREATE TABLE blocks(id INTEGER PRIMARY KEY,file TEXT,offset INTEGER,bytes INTEGER,sha TEXT,n INTEGER,live INTEGER,vector_bytes INTEGER,decoder_bytes INTEGER,rank INTEGER)",
        "CREATE INDEX block_files ON blocks(file,live)",
        "CREATE TABLE files(name TEXT PRIMARY KEY,kind TEXT,bytes INTEGER)",
        "CREATE TABLE cursors(source TEXT PRIMARY KEY,cursor TEXT)"
    ];DBInterface.execute(db,sql);end
end
function syncdir(path)
    fd=ccall(:open,Cint,(Cstring,Cint),path,0);fd>=0 || error("directory open failed")
    try ccall(:fsync,Cint,(Cint,),fd)==0 || error("directory fsync failed") finally ccall(:close,Cint,(Cint,),fd) end
end
function load_nodes!(s)
    empty!(s.nodes)
    for r in rows(s,"SELECT * FROM nodes ORDER BY id")
        s.nodes[r.id]=Node(r.id,r.parent,r.axis,r.cut,r.lft,r.rgt,r.depth,r.mass,r.at)
    end
end
function open_store(root;budget=256MiB,create_config=nothing,create_fault=nothing,backend=:auto,recovery_fault=nothing,require_existing=false)
    budget>=20MiB || error("budget must be at least 20 MiB")
    mkpath(root);root=abspath(root);lockfile=open(joinpath(root,".writer.lock"),"a+")
    if ccall(:flock,Cint,(Cint,Cint),Base.fd(lockfile),2|4)!=0;close(lockfile);error("store already open by a writer");end
    db=nothing
    try
    backend=Symbol(backend);backend in (:auto,:sqlite,:rocksdb) || error("unknown metadata backend")
    existing=detect_backend(root)
    existing!==nothing && backend!=:auto && backend!=existing && error("metadata backend mismatch; use a new directory for another backend")
    selected=backend==:auto ? something(existing,:sqlite) : backend
    workspace=16MiB;remaining=budget-workspace;dbbudget=min(64MiB,remaining÷4)
    if selected==:sqlite
        db=SQLite.DB(joinpath(root,"index.sqlite"))
        fresh=isempty(Tables.rowtable(DBInterface.execute(db,"SELECT name FROM sqlite_master WHERE name='meta'")))
        for sql in ["PRAGMA journal_mode=WAL","PRAGMA synchronous=FULL","PRAGMA mmap_size=0",
            "PRAGMA temp_store=FILE","PRAGMA cache_size=-$(9*(dbbudget÷1024)÷10)","PRAGMA wal_autocheckpoint=1000","PRAGMA busy_timeout=5000"]
            Tables.rowtable(DBInterface.execute(db,sql))
        end
    else
        isdefined(@__MODULE__,:RocksMetadata) || error("RocksDB.jl is required; use the server environment")
        db=RocksMetadata.open_db(joinpath(root,"metadata.rocks"),dbbudget;recovery_fault)
        fresh=RocksMetadata.getrow(db,"meta","config")===nothing
    end
    require_existing && fresh && error("existing committed store required")
    cfg=if fresh
        create_config===nothing ? config() : create_config
    else
        row=db isa SQLite.DB ? only(Tables.rowtable(DBInterface.execute(db,"SELECT value FROM meta WHERE key='config'"))) : RocksMetadata.getrow(db,"meta","config")
        stored=config(;Dict(Symbol(k)=>v for (k,v) in JSON3.read(String(row.value),Dict{String,Any}))...)
        create_config!==nothing && stored!=create_config && error("stored configuration/model mismatch")
        stored
    end
    s=Store(root,db,cfg,Dict(),Cache((remaining-dbbudget)÷3),Cache(2*((remaining-dbbudget)÷3)),budget,dbbudget,workspace,
        MersenneTwister(cfg.seed),MersenneTwister(cfg.seed+1),1,2,1,0,0,0.,lockfile,false,false,Dict(),Dict(),Dict(),Set(),Set(),nothing,nothing)
    if fresh
        commit!(s;fault=create_fault) do
            # Bootstrap is one publication too: interrupted DDL must not leave
            # a meta table without its configuration/root generation.
            db isa SQLite.DB && schema!(db)
            exec(s,"INSERT INTO meta VALUES ('config',?)",(collect(codeunits(JSON3.write(cfg))),))
            s.nodes[1]=Node(1,0,0,0.,0,0,0,0.,0.)
            exec(s,"INSERT INTO leaves VALUES (1,0)")
            dirty_state!(s,1,LeafState(cfg.dimension,0.))
        end
    else
        meta=deserialize_bytes(one(s,"SELECT value FROM meta WHERE key='runtime'").value)
        meta.julia_version==string(VERSION) || error("trusted state continuation requires Julia $(meta.julia_version)")
        for name in (:next_id,:next_node,:next_block,:sequence,:generation,:clock,:rng,:basisrng);setfield!(s,name,getproperty(meta,name));end
        load_nodes!(s)
    end
    load_keywords!(s)
    s
    catch
        db!==nothing && (db isa SQLite.DB ? SQLite.close(db) : RocksMetadata.close_db(db))
        close(lockfile)
        rethrow()
    end
end
function close_store(s)
    for (_,io) in values(s.packs);isopen(io) && close(io);end
    for stmt in values(s.statements);DBInterface.close!(stmt);end
    (s.db isa SQLite.DB ? SQLite.close(s.db) : RocksMetadata.close_db(s.db));close(s.lockfile);nothing
end
function commit!(f,s;fault=nothing)
    s.poisoned && error("discard this instance and reopen after a failed commit")
    s.transaction && error("nested transaction")
    s.keyword_cache=nothing
    s.transaction=true;empty!(s.packs);exec(s,"BEGIN IMMEDIATE")
    try
        result=f();flush_states!(s)
        for n in values(s.nodes)
            exec(s,"INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?)",(n.id,n.parent,n.axis,n.cut,n.left,n.right,n.depth,n.mass,n.at))
        end
        fault==:exit_before_files && ccall(:_exit,Cvoid,(Cint,),93)
        fault==:before_files && error("injected before files")
        for (kind,(name,io)) in s.packs
            flush(io);ccall(:fsync,Cint,(Cint,),Base.fd(io))==0 || error("payload fsync failed")
            size=position(io);close(io);mv(joinpath(s.root,"."*name),joinpath(s.root,name));metric!(s,"payload_write_bytes",size)
            exec(s,"INSERT INTO files VALUES (?,?,?)",(name,String(kind),size))
        end
        !isempty(s.packs) && syncdir(s.root)
        fault==:exit_before_commit && ccall(:_exit,Cvoid,(Cint,),93)
        fault==:before_commit && error("injected before commit")
        s.generation+=1
        runtime=(next_id=s.next_id,next_node=s.next_node,next_block=s.next_block,sequence=s.sequence,
            generation=s.generation,clock=s.clock,rng=s.rng,basisrng=s.basisrng,julia_version=string(VERSION))
        exec(s,"INSERT OR REPLACE INTO meta VALUES ('runtime',?)",(serialize_bytes(runtime),))
        exec(s,"COMMIT");s.transaction=false;empty!(s.packs)
        fault==:exit_after_commit && ccall(:_exit,Cvoid,(Cint,),93)
        fault==:after_commit && error("injected after commit")
        result
    catch
        s.transaction && try exec(s,"ROLLBACK") catch end
        s.transaction=false;s.poisoned=true
        for (_,io) in values(s.packs);isopen(io) && close(io);end
        rethrow()
    end
end
# Compose storage operations under one source-batch publication. The outer
# commit owns rollback, fsync and generation; nested fault injection is forbidden.
function mutation!(f,s;fault=nothing)
    s.poisoned && error("discard this instance and reopen after a failed commit")
    if s.transaction
        fault===nothing || error("fault injection belongs to the outer transaction")
        return f()
    end
    commit!(f,s;fault)
end
function packio(s,kind)
    s.transaction || error("payload write outside transaction")
    get!(s.packs,kind) do
        name="$(kind)-$(s.generation+1)-$(randstring(RandomDevice(),16)).bin"
        (name,open(joinpath(s.root,"."*name),"w+"))
    end
end
function write_frame!(s,f)
    name,io=packio(s,:vectors);bytes=DC.frame_bytes(f);offset=position(io);write(io,bytes)
    id=s.next_block;s.next_block+=1;st=DC.payload_stats(f)
    exec(s,"INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?)",(id,name,offset,length(bytes),bytes2hex(sha256(bytes)),f.n,0,st.vector_bytes,st.decoder_bytes,st.rank))
    cache_put!(s,s.block_cache,id,f)
    id
end
function frame_for(s,id)
    length(s.request_blocks)<4096 && push!(s.request_blocks,id)
    f=cache_get(s,s.block_cache,id,()->begin
        b=one(s,"SELECT file,offset,bytes,sha FROM blocks WHERE id=?",(id,))
        # A newly written frame can be evicted before the transaction publishes.
        path=joinpath(s.root,b.file)
        if !isfile(path)
            path=joinpath(s.root,"."*b.file)
            for (name,io) in values(s.packs);name==b.file && flush(io);end
        end
        bytes=open(path) do io;seek(io,b.offset);read(io,b.bytes);end
        length(bytes)==b.bytes && bytes2hex(sha256(bytes))==b.sha || error("corrupt codec frame")
        metric!(s,"vector_read_bytes",length(bytes));metric!(s,"vector_read_calls")
        DC.read_frame(bytes)
    end)
    !isempty(f.center) && length(s.request_decoders)<4096 && push!(s.request_decoders,DC.decoder_key(f))
    f
end
function write_text!(s,record)
    name,io=packio(s,:texts)
    payload=JSON3.write((text=record.text,platform=record.platform,author=record.author,created_at=record.created_at,
        observed_at=record.observed_at,revision_at=record.revision_at,permalink=record.permalink,reply_to=record.reply_to,
        tags=record.tags,metadata=record.metadata))
    offset=position(io);write(io,payload);name,offset,ncodeunits(payload)
end
function lookup(s,key::PostKey)
    r=rows(s,"SELECT p.* FROM identities i JOIN posts p ON p.id=i.id WHERE i.kind=? AND i.native=?",(Int(key.kind),native_bytes(key)))
    isempty(r) ? nothing : only(r)
end
"""Resolve a sampled persistent ID without loading the identity catalog.
Use PostRecords.key_from_bytes(row.kind, row.native) for its platform key.
"""
function lookup(s,id::Integer)
    r=rows(s,"SELECT * FROM posts WHERE id=?",(id,))
    isempty(r) ? nothing : only(r)
end
function fetch_vectors(s,ids;time=s.clock)
    length(ids)<=1024 || error("bounded fetch requires at most 1024 IDs")
    X=Matrix{Float32}(undef,s.config.dimension,length(ids));cutoff=time-s.config.retention_seconds
    for (j,id) in enumerate(ids)
        p=one(s,"SELECT status,created,block,row FROM posts WHERE id=?",(id,))
        p.status==1 && p.created>=cutoff || throw(KeyError(id))
        X[:,j]=vec(DC.decode(frame_for(s,p.block);rows=[p.row]))
    end
    X
end
function fetch_records(s,ids;time=s.clock)
    length(ids)<=1024 || error("bounded fetch requires at most 1024 IDs")
    out=Any[]
    for id in ids
        p=one(s,"SELECT status,created,textfile,textoff,textlen FROM posts WHERE id=?",(id,))
        p.status==1 && p.created>=time-s.config.retention_seconds || throw(KeyError(id))
        data=open(joinpath(s.root,p.textfile)) do io;seek(io,p.textoff);read(io,p.textlen);end
        length(data)==p.textlen || error("short text payload")
        metric!(s,"text_read_bytes",length(data));metric!(s,"text_read_calls")
        push!(out,JSON3.read(String(data)))
    end
    out
end
function route(s,x)
    i=Int64(1)
    while s.nodes[i].axis!=0
        n=s.nodes[i];i=x[n.axis]<n.cut ? n.left : n.right
    end
    i
end
function leaf_refs(s,leaf;after=0,limit=256)
    0<limit<=1024 || error("unbounded leaf scan")
    rows(s,"SELECT id,sequence,block,row,density_at FROM posts WHERE status=1 AND leaf=? AND sequence>? ORDER BY sequence,id LIMIT ?",(leaf,after,limit))
end
function add_slot!(s,id,leaf)
    count=one(s,"SELECT count FROM leaves WHERE id=?",(leaf,)).count
    exec(s,"UPDATE posts SET leaf=?,slot=? WHERE id=?",(leaf,count+1,id))
    exec(s,"UPDATE leaves SET count=count+1 WHERE id=?",(leaf,))
    kw_contribute!(s,id,leaf,1)
end
function remove_slot!(s,p)
    kw_contribute!(s,p.id,p.leaf,-1)
    last=one(s,"SELECT count FROM leaves WHERE id=?",(p.leaf,)).count
    exec(s,"UPDATE posts SET leaf=NULL,slot=NULL WHERE id=?",(p.id,))
    if p.slot!=last
        exec(s,"UPDATE posts SET slot=? WHERE status=1 AND leaf=? AND slot=?",(p.slot,p.leaf,last))
    end
    exec(s,"UPDATE leaves SET count=count-1 WHERE id=?",(p.leaf,))
end
function move_post!(s,id,leaf)
    p=one(s,"SELECT id,leaf,slot FROM posts WHERE id=?",(id,));p.leaf==leaf && return
    remove_slot!(s,p);add_slot!(s,id,leaf)
end

function replace_frame_refs!(s,f,ids)
    block=write_frame!(s,f)
    for (row,id) in enumerate(ids)
        old=one(s,"SELECT block FROM posts WHERE id=?",(id,)).block
        !ismissing(old) && exec(s,"UPDATE blocks SET live=live-1 WHERE id=?",(old,))
        exec(s,"UPDATE posts SET block=?,row=? WHERE id=?",(block,row,id))
    end
    exec(s,"UPDATE blocks SET live=? WHERE id=?",(length(ids),block))
end
"""Stream one leaf in sequence order, or preserve arrival frames for leaf=0.
Only encoded bytes/decoder parameters move; no encoder is called here.
"""
function repack!(s,leaf)
    if leaf==0
        # Append compaction preserves frozen arrival-batch groups. Sorting every
        # individual post by arrival would interleave different decoder groups
        # and needlessly duplicate their parameters into near-singleton frames.
        ceiling=s.next_block-1;after=0
        while true
            blocks=rows(s,"SELECT id FROM blocks WHERE live>0 AND id>? AND id<=? ORDER BY id LIMIT 128",(after,ceiling))
            isempty(blocks) && break
            for b in blocks
                refs=rows(s,"SELECT id,row FROM posts WHERE status=1 AND block=? ORDER BY row",(b.id,))
                length(refs)<=256 || error("oversized codec group")
                isempty(refs) && continue
                f=DC.subset(frame_for(s,b.id),[p.row for p in refs])
                replace_frame_refs!(s,f,[p.id for p in refs])
            end
            after=last(blocks).id
        end
        return nothing
    end
    after=0;pending=nothing;ids=Int64[];key="";before=get(s.metrics,"payload_write_bytes",0.)
    while true
        refs=leaf_refs(s,leaf;after)
        isempty(refs) && break
        for ref in refs
            f=DC.subset(frame_for(s,ref.block),[ref.row]);nextkey=DC.decoder_key(f)
            if pending!==nothing && (key!=nextkey || length(ids)==256)
                replace_frame_refs!(s,pending,ids);pending=nothing;empty!(ids)
            end
            pending=pending===nothing ? f : DC.concatenate(pending,f)
            key=nextkey;push!(ids,ref.id)
        end
        after=last(refs).sequence
    end
    pending!==nothing && replace_frame_refs!(s,pending,ids)
    nothing
end
function split_leaf!(s,id,axis,cut)
    started=time_ns()
    n=s.nodes[id];n.axis==0 || error("split requires leaf")
    st=get_state(s,id);mask=vec(st.reservoir[axis,:].<cut)
    any(mask) && !all(mask) || error("empty split reservoir")
    RL=st.reservoir[:,mask];RR=st.reservoir[:,.!mask];rank=st.sketch.rank
    Lseed=LT.reservoir_seed(RL,rank,s.basisrng);Rseed=LT.reservoir_seed(RR,rank,s.basisrng)
    l,r,_=LT.conditional_children(st.sketch,axis,cut,Lseed,Rseed)
    li=s.next_node;ri=li+1;s.next_node+=2
    n.axis=axis;n.cut=cut;n.left=li;n.right=ri
    for (j,sketch,R,seq) in ((li,l,RL,st.sequences[mask]),(ri,r,RR,st.sequences[.!mask]))
        s.nodes[j]=Node(j,id,0,0.,0,0,n.depth+1,sketch.mass,st.at)
        exec(s,"INSERT INTO leaves VALUES (?,0)",(j,))
        dirty_state!(s,j,LeafState(sketch,R,seq,st.at,0.,length(seq)))
    end
    remove_cached!(s.state_cache,id);exec(s,"DELETE FROM states WHERE id=?",(id,))
    after=0
    while true
        refs=leaf_refs(s,id;after);isempty(refs) && break
        for ref in refs
            x=vec(DC.decode(frame_for(s,ref.block);rows=[ref.row]))
            move_post!(s,ref.id,x[axis]<cut ? li : ri);metric!(s,"split_posts_routed")
        end
        after=last(refs).sequence
    end
    exec(s,"DELETE FROM leaves WHERE id=?",(id,))
    if s.config.layout==:leaf;repack!(s,li);repack!(s,ri);end
    metric!(s,"splits");metric!(s,"split_seconds",(time_ns()-started)/1e9);li,ri
end
function train_point!(s,x,sequence,time)
    id=route(s,x);n=s.nodes[id];st=get_state(s,id);elapsed=time-st.at
    LT.advance_state!(st.sketch,x,elapsed,s.config.half_life)
    st.arrivals=st.arrivals*exp2(-elapsed/s.config.half_life)+1;st.at=time;st.seen+=1
    if size(st.reservoir,2)==s.config.reservoir_cap
        st.reservoir[:,1:end-1].=st.reservoir[:,2:end];st.reservoir[:,end]=x
        popfirst!(st.sequences);push!(st.sequences,sequence)
    else
        st.reservoir=hcat(st.reservoir,Float32.(x));push!(st.sequences,sequence)
    end
    dirty_state!(s,id,st)
    if count(n->n.axis==0,values(s.nodes))<s.config.max_leaves && st.arrivals>=s.config.split_after &&
            n.depth<s.config.max_depth && size(st.reservoir,2)>=16
        proposal=LT.cut_proposal(st.reservoir,s.rng)
        proposal!==nothing && split_leaf!(s,id,proposal[1],proposal[2])
    end
end
function advance!(s,st,time)
    elapsed=time-st.at;LT.decay!(st.sketch,elapsed,s.config.half_life)
    st.arrivals*=exp2(-elapsed/s.config.half_life);st.at=time;st
end
function merge_nodes!(s,parent,time)
    n=s.nodes[parent];left,right=n.left,n.right
    l=advance!(s,get_state(s,left),time);r=advance!(s,get_state(s,right),time)
    sketch=LT.merge_sketches(l.sketch,r.sketch);seq=vcat(l.sequences,r.sequences);R=hcat(l.reservoir,r.reservoir)
    order=sortperm(seq);keep=order[max(1,end-s.config.reservoir_cap+1):end]
    n.axis=0;n.cut=0.;n.left=0;n.right=0
    exec(s,"INSERT INTO leaves VALUES (?,0)",(parent,))
    dirty_state!(s,parent,LeafState(sketch,R[:,keep],seq[keep],time,0.,length(keep)))
    for child in (left,right)
        after=0
        while true
            refs=leaf_refs(s,child;after);isempty(refs) && break
            for ref in refs;move_post!(s,ref.id,parent);metric!(s,"merge_posts_moved");end
            after=last(refs).sequence
        end
        remove_cached!(s.state_cache,child);delete!(s.nodes,child)
        for table in ("nodes","states","leaves");exec(s,"DELETE FROM $table WHERE id=?",(child,));end
    end
    s.config.layout==:leaf && repack!(s,parent)
    metric!(s,"merges")
end
function merge_cold!(s,time)
    for id in sort(collect(keys(s.nodes));rev=true)
        haskey(s.nodes,id) || continue;n=s.nodes[id];n.axis==0 && continue
        l,r=s.nodes[n.left],s.nodes[n.right];l.axis==0 && r.axis==0 || continue
        mass=l.mass*exp2(-(time-l.at)/s.config.half_life)+r.mass*exp2(-(time-r.at)/s.config.half_life)
        mass<=s.config.merge_below && merge_nodes!(s,id,time)
    end
end
function weighted_update!(sk,x,w)
    w<=0 && return
    total=sk.mass+w;delta=Float64.(x).-sk.mean;v=sqrt(sk.mass*w/total).*delta
    sk.mean .+= (w/total).*delta;sk.diagonal .+= v.^2;sk.mass=total
    sk.rank==0 && return
    a=sk.U'*v;res=v-sk.U*a;n=norm(res)
    if n>1e-12;Q=hcat(sk.U,res/n);z=vcat(a,n);C=diagm(vcat(sk.eigenvalues,0.))+z*z'
    elseif isempty(a);return
    else;Q=sk.U;C=diagm(sk.eigenvalues)+a*a';end
    e=eigen(Symmetric(C));ids=length(e.values):-1:max(1,length(e.values)-sk.rank+1)
    sk.U=Q*e.vectors[:,ids];sk.eigenvalues=max.(0.,e.values[ids])
end
function rebuild_leaf!(s,leaf)
    st=LeafState(s.config.dimension,s.clock);after=0
    while true
        refs=leaf_refs(s,leaf;after);isempty(refs) && break
        for ref in refs
            x=vec(DC.decode(frame_for(s,ref.block);rows=[ref.row]))
            weighted_update!(st.sketch,x,exp2(-(s.clock-ref.density_at)/s.config.half_life))
            st.reservoir=hcat(st.reservoir,x);push!(st.sequences,ref.sequence)
            if length(st.sequences)>s.config.reservoir_cap
                st.reservoir=st.reservoir[:,2:end];popfirst!(st.sequences)
            end
            metric!(s,"rebuild_posts")
        end
        after=last(refs).sequence
    end
    st.seen=length(st.sequences);dirty_state!(s,leaf,st)
end
function deactivate!(s,p,status)
    remove_slot!(s,p);kw_forget!(s,p.id);exec(s,"UPDATE blocks SET live=live-1 WHERE id=?",(p.block,))
    exec(s,"UPDATE posts SET status=?,block=NULL,row=NULL,textfile=NULL,textoff=NULL,textlen=NULL WHERE id=?",(status,p.id))
end
function append_batch!(s,records,X;time=s.clock,cursor=nothing,fault=nothing,keywords=nothing)
    validate_keywords(s,records,keywords)
    n=length(records);0<n<=1024 && size(X)==(s.config.dimension,n) && all(isfinite,X) && maximum(abs,X)<=1 || error("invalid bounded input batch")
    isfinite(time) && time>=s.clock || error("clock moved backwards")
    names=PostKey[]
    for p in records;append!(names,unique(vcat([p.key],p.aliases)));end
    length(unique(names))==length(names) || error("collapse duplicate or aliased identities before a batch")
    started=time_ns()
    result=mutation!(s;fault) do
        s.clock=Float64(time);work=Tuple{Int,Int64,Int64}[];affected=Set{Int64}();seen_existing=Set{Int64}();ignored=0;metadata=0
        for (col,record) in enumerate(records)
            old=lookup(s,record.key);digest=content_digest(record)
            if old!==nothing
                old.id in seen_existing && error("collapse multiple events for an aliased identity before a batch")
                push!(seen_existing,old.id)
            end
            if record.created_at<time-s.config.retention_seconds || (old!==nothing &&
                (record.revision_at<old.revision || (record.revision_at==old.revision &&
                (old.status!=1 || record.observed_at<old.observed))))
                ignored+=1;continue
            end
            old!==nothing && record.revision_at==old.revision && old.status==1 && digest!=old.digest && error("conflicting same-revision content")
            if old!==nothing && old.status==1 && old.digest==digest
                for key in record.aliases
                    prior=lookup(s,key);prior!==nothing && prior.id!=old.id && error("identity alias conflict")
                    exec(s,"INSERT OR REPLACE INTO identities VALUES (?,?,?)",(Int(key.kind),native_bytes(key),old.id))
                end
                if record.observed_at>old.observed || record.revision_at>old.revision
                    keywords!==nothing && kw_attach!(s,old.id,keywords[col])
                    name,offset,len=write_text!(s,record)
                    exec(s,"UPDATE posts SET revision=?,observed=?,textfile=?,textoff=?,textlen=? WHERE id=?",(record.revision_at,record.observed_at,name,offset,len,old.id));metadata+=1
                else;ignored+=1;end
                continue
            end
            if old!==nothing && old.status==1;push!(affected,old.leaf);deactivate!(s,old,0);end
            id=old===nothing ? s.next_id : old.id
            old===nothing && (s.next_id+=1)
            for key in vcat([record.key],record.aliases)
                prior=lookup(s,key);prior!==nothing && prior.id!=id && error("identity alias conflict")
                exec(s,"INSERT OR REPLACE INTO identities VALUES (?,?,?)",(Int(key.kind),native_bytes(key),id))
            end
            s.sequence+=1;push!(work,(col,id,s.sequence))
        end
        for leaf in affected;rebuild_leaf!(s,leaf);end
        for (col,_,seq) in work;train_point!(s,view(X,:,col),seq,time);end
        if !isempty(work)
            original_leaves=[route(s,view(X,:,col)) for (col,_,_) in work]
            groups=Vector{Int}[]
            if s.config.mode in (:centroid_leaf,:pca16)
                for leaf in sort(unique(original_leaves))
                    ids=findall(==(leaf),original_leaves)
                    for a in 1:256:length(ids);push!(groups,ids[a:min(a+255,end)]);end
                end
            else
                ids=s.config.mode==:centroid_shared ? sortperm(original_leaves) : collect(eachindex(work))
                for a in 1:256:length(ids);push!(groups,ids[a:min(a+255,end)]);end
            end
            for (col,id,seq) in work
                p=records[col];file,offset,len=write_text!(s,p)
                exec(s,"INSERT OR REPLACE INTO posts VALUES (?,?,?,?,?,?,?,?,1,NULL,NULL,NULL,NULL,?,?,?,?)",
                    (id,Int(p.key.kind),native_bytes(p.key),p.revision_at,p.observed_at,p.created_at,time,seq,file,offset,len,content_digest(p)))
            end
            for members in groups
                f=DC.encode(s.config.mode,X[:,[work[j][1] for j in members]])
                assigned=[route(s,x) for x in eachcol(DC.decode(f))]
                parts=s.config.layout==:leaf ? [findall(==(leaf),assigned) for leaf in sort(unique(assigned))] : [collect(eachindex(members))]
                metric!(s,"encoded_groups");metric!(s,"encoded_posts",f.n);metric!(s,"encoded_rank_sum",f.rank*f.n)
                metric!(s,"fit_group_size_$(f.n)");metric!(s,"fit_rank_$(f.rank)")
                for part in parts
                    localframe=DC.subset(f,part);ids=[work[members[j]][2] for j in part]
                    replace_frame_refs!(s,localframe,ids)
                    for (j,id) in zip(part,ids);add_slot!(s,id,assigned[j]);end
                end
            end
        end
        if keywords!==nothing
            for (col,id,_) in work;kw_attach!(s,id,keywords[col]);end
        end
        cursor!==nothing && exec(s,"INSERT OR REPLACE INTO cursors VALUES (?,?)",cursor)
        (encoded=length(work),ignored=ignored,metadata=metadata)
    end
    metric!(s,"ingest_seconds",(time_ns()-started)/1e9);result
end
function candidate_leaves(s,query;leaf_budget=4)
    length(query)==s.config.dimension && all(isfinite,query) && 0<leaf_budget<=s.config.max_leaves || error("invalid regional query")
    lo=fill(-1.,s.config.dimension);hi=fill(1.,s.config.dimension);scores=Tuple{Float64,Int64}[]
    function visit(id)
        n=s.nodes[id]
        if n.axis==0
            distance=sum(abs2,max.(lo.-query,0.).+max.(query.-hi,0.));push!(scores,(distance,id));return
        end
        old=hi[n.axis];hi[n.axis]=n.cut;visit(n.left);hi[n.axis]=old
        old=lo[n.axis];lo[n.axis]=n.cut;visit(n.right);lo[n.axis]=old
    end
    visit(1);sort!(scores);last.(scores[1:min(leaf_budget,length(scores))])
end
function sample_ids(s,query;limit=20,leaf_budget=4,time=s.clock,rng=Random.default_rng(),max_attempts=max(256,32limit))
    0<limit<=1024 || error("invalid sample size")
    chosen=candidate_leaves(s,query;leaf_budget)
    counts=[one(s,"SELECT count FROM leaves WHERE id=?",(id,)).count for id in chosen]
    cumulative=cumsum(counts);total=sum(counts);ids=Int64[];seen=Set{Int64}();attempts=0
    while length(ids)<limit && attempts<max_attempts && length(seen)<total && total>0
        at=rand(rng,1:total);j=searchsortedfirst(cumulative,at);slot=at-(j==1 ? 0 : cumulative[j-1]);attempts+=1
        p=one(s,"SELECT id,created FROM posts WHERE status=1 AND leaf=? AND slot=?",(chosen[j],slot))
        p.id in seen && continue;push!(seen,p.id)
        p.created>=time-s.config.retention_seconds && push!(ids,p.id)
    end
    (ids=ids,attempts=attempts,underfilled=length(ids)<limit,leaves=chosen)
end
function delete_post!(s,key;revision_at,time=s.clock,fault=nothing)
    isfinite(time) && isfinite(revision_at) && time>=s.clock || error("invalid clock/revision")
    mutation!(s;fault) do
        s.clock=Float64(time);p=lookup(s,key)
        if p===nothing
            id=s.next_id;s.next_id+=1
            exec(s,"INSERT INTO identities VALUES (?,?,?)",(Int(key.kind),native_bytes(key),id))
            exec(s,"INSERT INTO posts VALUES (?,?,?,?,?,0,0,0,2,NULL,NULL,NULL,NULL,NULL,NULL,NULL,?)",
                (id,Int(key.kind),native_bytes(key),revision_at,time,""))
            return true
        end
        revision_at<p.revision && return false
        if p.status==1;deactivate!(s,p,2);rebuild_leaf!(s,p.leaf);end
        exec(s,"UPDATE posts SET status=2,revision=?,observed=? WHERE id=?",(revision_at,time,p.id));true
    end
end
function expire!(s;time=s.clock)
    time>=s.clock || error("clock moved backwards")
    mutation!(s) do
        s.clock=Float64(time);affected=Set{Int64}();removed=0
        while true
            ps=rows(s,"SELECT id,leaf,slot,block FROM posts WHERE status=1 AND created<? ORDER BY created,id LIMIT 256",(time-s.config.retention_seconds,))
            isempty(ps) && break
            for p in ps
                # Dense-slot swaps can change slots of later rows in this page.
                current=one(s,"SELECT id,leaf,slot,block FROM posts WHERE id=?",(p.id,))
                push!(affected,current.leaf);deactivate!(s,current,3);removed+=1
            end
        end
        for leaf in affected;rebuild_leaf!(s,leaf);end
        removed
    end
end
function maintain!(s;time=s.clock)
    time>=s.clock || error("clock moved backwards")
    started=time_ns();mutation!(s) do;s.clock=Float64(time);merge_cold!(s,time);end
    metric!(s,"merge_seconds",(time_ns()-started)/1e9);s
end
function rebuild_density!(s,leaves=sort([n.id for n in values(s.nodes) if n.axis==0]))
    commit!(s) do;for leaf in leaves;rebuild_leaf!(s,leaf);end;end;s
end
function compact!(s)
    started=time_ns();commit!(s) do
        if s.config.layout==:append;repack!(s,0)
        else;for id in sort([n.id for n in values(s.nodes) if n.axis==0]);repack!(s,id);end;end
    end
    metric!(s,"compaction_seconds",(time_ns()-started)/1e9);collect_garbage!(s);s
end
"""Copy retained text bytes out of partially obsolete packs in bounded pages.
Vector compaction alone cannot reclaim texts mixed with deleted/edited posts."""
function compact_texts!(s)
    # Freeze the file list via keyset order; newly written packs are not candidates
    # because their live byte count equals their physical size.
    after=""
    while true
        files=rows(s,"SELECT f.name FROM files f WHERE f.kind='texts' AND f.name>? AND f.bytes>(SELECT COALESCE(SUM(p.textlen),0) FROM posts p WHERE p.textfile=f.name AND p.status=1) ORDER BY f.name LIMIT 32",(after,))
        isempty(files) && break
        for file in files
            while true
                ps=rows(s,"SELECT id,textoff,textlen FROM posts WHERE status=1 AND textfile=? ORDER BY id LIMIT 128",(file.name,))
                isempty(ps) && break
                commit!(s) do
                    open(joinpath(s.root,file.name)) do input
                        for p in ps
                            seek(input,p.textoff);payload=read(input,p.textlen)
                            length(payload)==p.textlen || error("short text payload")
                            name,output=packio(s,:texts);offset=position(output);write(output,payload)
                            exec(s,"UPDATE posts SET textfile=?,textoff=? WHERE id=?",(name,offset,p.id))
                            metric!(s,"text_compaction_bytes",length(payload))
                        end
                    end
                end
            end
        end
        after=String(last(files).name)
    end
    collect_garbage!(s);s
end
function collect_garbage!(s)
    keyword_enabled(s) && keyword_gc!(s)
    removed=0
    while true
        files=rows(s,"SELECT f.name,f.bytes FROM files f WHERE NOT EXISTS (SELECT 1 FROM blocks b WHERE b.file=f.name AND b.live>0) AND NOT EXISTS (SELECT 1 FROM posts p WHERE p.textfile=f.name AND p.status=1) LIMIT 128")
        isempty(files) && break
        commit!(s) do
            for f in files
                exec(s,"DELETE FROM blocks WHERE file=? AND live=0",(f.name,));exec(s,"DELETE FROM files WHERE name=?",(f.name,))
            end
        end
        for f in files;rm(joinpath(s.root,f.name);force=true);removed+=f.bytes;end
    end
    # Recover files created before a failed publication, without scanning post rows.
    for name in readdir(s.root)
        (startswith(name,"vectors-") || startswith(name,"texts-") || startswith(name,".vectors-") || startswith(name,".texts-")) || continue
        isempty(rows(s,"SELECT name FROM files WHERE name=?",(name,))) || continue
        path=joinpath(s.root,name);removed+=filesize(path);rm(path)
    end
    removed>0 && syncdir(s.root);metric!(s,"reclaimed_bytes",removed);removed
end
tree_signature(s)=[(n.id,n.parent,n.axis,n.cut,n.left,n.right,n.depth) for n in sort(collect(values(s.nodes));by=n->n.id)]
"""Read-only geometric-prior Pólya masses/probabilities keyed by persistent ID.
Uses only compact routing metadata; no PPCA/reservoir state needs to be loaded.
"""
function polya_weights(s;time=s.clock,kappa=2.)
    isfinite(time) && time>=s.clock && isfinite(kappa) && kappa>=0 || error("invalid probability query")
    ids=sort(collect(keys(s.nodes)));mass=Dict{Int64,Float64}();prob=Dict{Int64,Float64}()
    for id in reverse(ids)
        n=s.nodes[id]
        mass[id]=n.axis==0 ? n.mass*exp2(-(time-n.at)/s.config.half_life) : mass[n.left]+mass[n.right]
        prob[id]=0.
    end
    lo=fill(-1.,s.config.dimension);hi=fill(1.,s.config.dimension)
    function visit(id,p)
        n=s.nodes[id]
        if n.axis==0;prob[id]=p;return;end
        prior=(n.cut-lo[n.axis])/(hi[n.axis]-lo[n.axis])
        q=mass[id]+kappa>0 ? (mass[n.left]+kappa*prior)/(mass[id]+kappa) : prior
        old=hi[n.axis];hi[n.axis]=n.cut;visit(n.left,p*q);hi[n.axis]=old
        old=lo[n.axis];lo[n.axis]=n.cut;visit(n.right,p*(1-q));lo[n.axis]=old
    end
    visit(1,1.);(probabilities=prob,masses=mass)
end
function storage_stats(s)
    aggregate=one(s,"SELECT COUNT(*) AS blocks,COALESCE(SUM(vector_bytes),0) AS vectors,COALESCE(SUM(decoder_bytes),0) AS decoders,COALESCE(SUM(bytes),0) AS frames FROM blocks WHERE live>0")
    files=Dict(String(r.kind)=>Int(r.bytes) for r in rows(s,"SELECT kind,SUM(bytes) AS bytes FROM files GROUP BY kind"))
    sqlitebytes=sum(filesize(joinpath(s.root,n)) for n in ("index.sqlite","index.sqlite-wal","index.sqlite-shm") if isfile(joinpath(s.root,n));init=0)
    rocksbytes=s.db isa SQLite.DB ? 0 : RocksMetadata.metadata_bytes(s.db)
    states=one(s,"SELECT COALESCE(SUM(length(payload)),0) AS bytes FROM states").bytes
    (active=one(s,"SELECT COUNT(*) AS n FROM posts WHERE status=1").n,
     identities=one(s,"SELECT COUNT(*) AS n FROM identities").n,blocks=aggregate.blocks,
     vector_bytes=aggregate.vectors,decoder_bytes=aggregate.decoders,frame_bytes=aggregate.frames,
     vector_files=get(files,"vectors",0),text_files=get(files,"texts",0),sqlite_bytes=sqlitebytes,rocksdb_bytes=rocksbytes,metadata_bytes=sqlitebytes+rocksbytes,metadata_backend=string(metadata_backend(s)),
     leaf_state_bytes=states,total_bytes=sum(values(files);init=0)+sqlitebytes+rocksbytes,
     topology_bytes=Base.summarysize(s.nodes),managed_budget=s.budget,
     cache_bytes=s.state_cache.used+s.block_cache.used,state_cache_peak=s.state_cache.peak,block_cache_peak=s.block_cache.peak,
     metadata_cache_budget=s.sqlite_budget,sqlite_cache_budget=s.db isa SQLite.DB ? s.sqlite_budget : 0,sqlite_actual=sqlite_usage(s),
     rocksdb_actual=s.db isa SQLite.DB ? nothing : RocksMetadata.usage(s.db),workspace_reservation=s.workspace,
     cache_hits=s.state_cache.hits+s.block_cache.hits,cache_misses=s.state_cache.misses+s.block_cache.misses,
     state_evictions=s.state_cache.evictions,block_evictions=s.block_cache.evictions,
     leaves=count(n->n.axis==0,values(s.nodes)),generation=s.generation)
end
function audit!(s;check_vectors=true)
    bad=one(s,"SELECT COUNT(*) AS n FROM posts p WHERE status=1 AND (block IS NULL OR leaf IS NULL OR slot IS NULL OR NOT EXISTS (SELECT 1 FROM blocks b WHERE b.id=p.block))").n
    bad==0 || error("dangling active post")
    for r in rows(s,"SELECT l.id,l.count,COUNT(p.id) AS actual,COALESCE(MAX(p.slot),0) AS maxslot FROM leaves l LEFT JOIN posts p ON p.leaf=l.id AND p.status=1 GROUP BY l.id")
        r.count==r.actual==r.maxslot || error("posting slots have holes")
    end
    after=0;checked=0
    while true
        ps=rows(s,"SELECT id,kind,native,leaf,block,row FROM posts WHERE status=1 AND id>? ORDER BY id LIMIT 256",(after,))
        isempty(ps) && break
        for p in ps
            identity=one(s,"SELECT id FROM identities WHERE kind=? AND native=?",(p.kind,p.native))
            identity.id==p.id || error("identity mismatch")
            check_vectors && route(s,vec(DC.decode(frame_for(s,p.block);rows=[p.row])))!=p.leaf && error("stale routed membership")
            checked+=1
        end
        after=last(ps).id
    end
    checked
end
include("DiskKeywords.jl")
include("DiskBackendMigration.jl")

end
