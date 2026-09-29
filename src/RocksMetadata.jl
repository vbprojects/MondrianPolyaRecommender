"""Native RocksDB metadata adapter for the disk tree's internal relational operations.
No SQLite mirror. Ordered secondary keys implement leaf slots, sequence, expiry,
identity, block/text references and keyword lookups. This is not a public SQL engine.

A bounded write buffer and on-disk undo journal allow large single-owner mutations
without retaining their complete WriteBatch in RAM. Recovery finishes before open
returns. Payload durability remains the outer DiskTreeStore transaction's job.
"""
module RocksMetadata
using RocksDB, Serialization
const VERSION_TAG="mondrian-rocks-metadata-v2"
const LEGACY_VERSION_TAG="mondrian-rocks-metadata-v1"
const SCHEMA=Dict(
 "meta"=>(:key,:value), "nodes"=>(:id,:parent,:axis,:cut,:lft,:rgt,:depth,:mass,:at),
 "states"=>(:id,:payload), "leaves"=>(:id,:count), "identities"=>(:kind,:native,:id),
 "posts"=>(:id,:kind,:native,:revision,:observed,:created,:density_at,:sequence,:status,:leaf,:slot,:block,:row,:textfile,:textoff,:textlen,:digest),
 "blocks"=>(:id,:file,:offset,:bytes,:sha,:n,:live,:vector_bytes,:decoder_bytes,:rank),
 "files"=>(:name,:kind,:bytes), "cursors"=>(:source,:cursor),
 "kw_terms"=>(:id,:term), "kw_posts"=>(:id,:digest,:eligible,:assumed,:truncated,:terms),
 "kw_counts"=>(:leaf,:term,:n,:mass,:at))
const PRIMARY=Dict(t=>(t=="meta" ? (:key,) : t=="files" ? (:name,) : t=="cursors" ? (:source,) :
    t=="identities" ? (:kind,:native) : t=="kw_counts" ? (:leaf,:term) : (:id,)) for t in keys(SCHEMA))
const REAL_FIELDS=Set([:revision,:observed,:created,:density_at,:cut,:mass,:at])
const INTEGER_FIELDS=Set([:id,:parent,:axis,:lft,:rgt,:depth,:count,:sequence,:status,:leaf,:slot,:block,:row,:textoff,:textlen,:offset,:bytes,:n,:live,:vector_bytes,:decoder_bytes,:rank,:eligible,:assumed,:truncated])
const INDEXES=Dict(
 "posts"=>[("active_id",(:id,),true),("leaf_slot",(:leaf,:slot),true),
    ("leaf_sequence",(:leaf,:sequence,:id),true),("active_sequence",(:sequence,:id),true),
    ("post_blocks",(:block,:row,:id),true),("expiry",(:status,:created,:id),false),
    ("text_files",(:textfile,:status,:id),false)],
 "blocks"=>[("block_files",(:file,:live,:id),false),("live_blocks",(:id,),true)],
 "kw_terms"=>[("term_name",(:term,),false)])
include("RocksRowCodec.jl")
bstr(s)=Vector{UInt8}(codeunits(s))
part(x::Integer)="i"*string(reinterpret(UInt64,Int64(x)) ⊻ (UInt64(1)<<63);base=16,pad=16)
function part(x::AbstractFloat)
    bits=reinterpret(UInt64,Float64(x==0 ? 0. : x))
    "f"*string((bits>>63)==1 ? ~bits : bits ⊻ (UInt64(1)<<63);base=16,pad=16)
end
part(x::AbstractString)="s"*bytes2hex(codeunits(x))
part(x::AbstractVector{UInt8})="b"*bytes2hex(x)
keysize(x::Union{Integer,AbstractFloat})=17
keysize(x::AbstractString)=1+2ncodeunits(x)
keysize(x::AbstractVector{UInt8})=1+2length(x)
const HEX=codeunits("0123456789abcdef")
function keypart!(b,p,x::Union{Integer,AbstractFloat})
    if x isa Integer
        b[p]=0x69;bits=reinterpret(UInt64,Int64(x)) ⊻ (UInt64(1)<<63)
    else
        b[p]=0x66;bits=reinterpret(UInt64,Float64(x==0 ? 0. : x))
        bits=(bits>>63)==1 ? ~bits : bits ⊻ (UInt64(1)<<63)
    end
    @inbounds for j in 1:16;b[p+j]=HEX[Int((bits>>((16-j)*4))&0xf)+1];end
    p+17
end
function keypart!(b,p,x::Union{AbstractString,AbstractVector{UInt8}})
    b[p]=x isa AbstractString ? 0x73 : 0x62;p+=1
    @inbounds for v in (x isa AbstractString ? codeunits(x) : x)
        b[p]=HEX[Int(v>>4)+1];b[p+1]=HEX[Int(v&0xf)+1];p+=2
    end
    p
end
function key(prefix,args...)
    n=ncodeunits(prefix);size=n+sum(keysize,args;init=0)+max(0,length(args)-1)
    b=Vector{UInt8}(undef,size);copyto!(b,1,codeunits(prefix),1,n);p=n+1
    for (i,x) in enumerate(args)
        i>1 && (b[p]=0x2f;p+=1);p=keypart!(b,p,x)
    end
    String(b)
end
rprefix(table)="R/"*table*"/"
rkey(table,args...)=key(rprefix(table),args...)
mutable struct DB
    native::RocksDB.DB
    path::String
    pending::Dict{String,Union{Nothing,Vector{UInt8}}}
    pending_bytes::Int
    buffer_limit::Int
    peak_buffer_bytes::Int
    active::Bool
    sync::RocksDB.WriteOptions
    async::RocksDB.WriteOptions
    block_budget::Int
    memtable_budget::Int
    flushes::Int
    recovered::Bool
    commit_fault::Union{Nothing,Symbol}
end
function direct!(db,entries;sync=false)
    wb=RocksDB.WriteBatch()
    try
        for (k,v) in entries
            v===nothing ? delete!(wb,k) : put!(wb,k,v)
        end
        RocksDB.write!(db.native,wb;options=sync ? db.sync : db.async)
    finally finalize(wb) end
end
rawget(db,k)=haskey(db.pending,k) ? db.pending[k] : get(db.native,k)
function stage!(db,k,v)
    db.active || error("metadata mutation outside transaction")
    if haskey(db.pending,k)
        old=db.pending[k];db.pending_bytes-=ncodeunits(k)+(old===nothing ? 0 : length(old))+64
    end
    db.pending[k]=v
    db.pending_bytes+=ncodeunits(k)+(v===nothing ? 0 : length(v))+64
    db.peak_buffer_bytes=max(db.peak_buffer_bytes,db.pending_bytes)
    db.pending_bytes>=db.buffer_limit && flush_pending!(db)
end
function flush_pending!(db)
    isempty(db.pending) && return
    wb=RocksDB.WriteBatch()
    try
        for (k,v) in db.pending
            journal="J/"*k
            if get(db.native,journal)===nothing
                old=get(db.native,k)
                put!(wb,journal,old===nothing ? UInt8[0] : vcat(UInt8[1],old))
            end
            v===nothing ? delete!(wb,k) : put!(wb,k,v)
        end
        RocksDB.write!(db.native,wb;options=db.async);db.flushes+=1
        empty!(db.pending);db.pending_bytes=0
    finally finalize(wb) end
end
function rawscan(f,db,prefix;start=prefix)
    flush_pending!(db)
    length(db.native.children)>1024 && filter!(w->(obj=w.value;obj!==nothing && obj.ptr!=C_NULL),db.native.children)
    it=RocksDB.DBIterator(db.native)
    try
        RocksDB.seek!(it,start)
        while RocksDB.valid(it)
            k=String(RocksDB.key(it));startswith(k,prefix) || break
            f(k,RocksDB.value(it))===false && break
            RocksDB.advance!(it)
        end
        RocksDB._check_iter_error(it)
    finally finalize(it) end
end
function finish_journal!(db,rollback;fault=nothing)
    after="J/"
    while true
        page=Pair{String,Vector{UInt8}}[];size=0
        rawscan(db,"J/";start=after) do k,v
            push!(page,k=>v);size+=ncodeunits(k)+length(v)
            length(page)<128 && size<db.buffer_limit
        end
        isempty(page) && break
        after=first(last(page))*"\0"
        wb=RocksDB.WriteBatch()
        try
            for (k,v) in page
                if rollback
                    target=k[3:end]
                    first(v)==0 ? delete!(wb,target) : put!(wb,target,v[2:end])
                end
                delete!(wb,k)
            end
            RocksDB.write!(db.native,wb;options=db.async)
        finally finalize(wb) end
        fault==:exit_during_recovery && ccall(:_exit,Cvoid,(Cint,),94)
    end
    direct!(db,["!transaction"=>nothing];sync=true)
end
function recover!(db;fault=nothing)
    state=get(db.native,"!transaction");state===nothing && return
    label=String(state);label in ("open","committed") || error("invalid transaction marker")
    finish_journal!(db,label=="open";fault);db.recovered=true
end
function open_db(path,budget;buffer_limit=512*1024,recovery_fault=nothing)
    block=max(256*1024,budget÷2);mem=max(256*1024,budget÷4)
    opts=RocksDB.Options(compression=:none,block_cache_size=nothing,bloom_bits_per_key=nothing,
        write_buffer_size=mem,max_background_jobs=2,max_open_files=128)
    # Charge SST index/filter blocks to the same LRU as data, instead of pinning
    # every table's filter outside the configured database-cache allowance.
    table_options=RocksDB.Lib.rocksdb_block_based_options_create()
    try
        opts.cache=RocksDB.Cache(block)
        opts.filter=RocksDB.FilterPolicy(;bits_per_key=10.)
        RocksDB.Lib.rocksdb_block_based_options_set_block_cache(table_options,opts.cache.ptr)
        RocksDB.Lib.rocksdb_block_based_options_set_filter_policy(table_options,opts.filter.ptr)
        opts.filter.owned=false
        RocksDB.Lib.rocksdb_block_based_options_set_cache_index_and_filter_blocks(table_options,Cuchar(1))
        # Whole-file filters can exceed a forced-small cache and be reread on
        # every lookup. Partition both metadata structures into ~4 KiB pieces.
        RocksDB.Lib.rocksdb_block_based_options_set_index_type(table_options,
            Cint(RocksDB.Lib.rocksdb_block_based_table_index_type_two_level_index_search))
        RocksDB.Lib.rocksdb_block_based_options_set_partition_filters(table_options,Cuchar(1))
        RocksDB.Lib.rocksdb_block_based_options_set_metadata_block_size(table_options,UInt64(4096))
        RocksDB.Lib.rocksdb_options_set_block_based_table_factory(opts.ptr,table_options)
    finally RocksDB.Lib.rocksdb_block_based_options_destroy(table_options) end
    # Limit two memtables; the block cache plus both memtables fit the DB allowance.
    RocksDB.Lib.rocksdb_options_set_max_write_buffer_number(opts.ptr,Cint(2))
    native=RocksDB.opendb(path;options=opts)
    db=DB(native,path,Dict(),0,buffer_limit,0,false,RocksDB.WriteOptions(sync=true),RocksDB.WriteOptions(sync=false),block,mem,0,false,nothing)
    try
        tag=get(native,"!format")
        tag!==nothing && !(String(tag) in (VERSION_TAG,LEGACY_VERSION_TAG)) && error("unsupported RocksDB metadata format")
        recover!(db;fault=recovery_fault)
        tag===nothing && direct!(db,["!format"=>bstr(VERSION_TAG)];sync=true)
        db
    catch;close_db(db);rethrow();end
end
function close_db(db)
    empty!(db.pending);db.pending_bytes=0
    close(db.native);finalize(db.sync);finalize(db.async)
end
"""Explicit offline SST rewrite, e.g. to apply partitioned indexes to a pilot
created with older table options. Does not alter logical records/generations.
This can generate substantial IO and is not run on ordinary startup.
"""
function rewrite_tables!(db)
    db.active && error("cannot compact during a metadata transaction")
    opts=RocksDB.Lib.rocksdb_compactoptions_create()
    try
        # RocksDB BottommostLevelCompaction::kForce = 2. The Julia wrapper's
        # Bool argument cannot express this enum value.
        RocksDB.Lib.rocksdb_compactoptions_set_bottommost_level_compaction(opts,Cuchar(2))
        RocksDB.Lib.rocksdb_compact_range_opt(db.native.ptr,opts,C_NULL,0,C_NULL,0)
    finally RocksDB.Lib.rocksdb_compactoptions_destroy(opts) end
end
function begin!(db)
    db.active && error("nested metadata transaction")
    get(db.native,"!transaction")===nothing || error("unfinished transaction")
    direct!(db,["!transaction"=>bstr("open")];sync=true);db.active=true
    # Journal the format gate with the first new-format mutation. Read-only
    # opens do not upgrade v1; rollback restores its original gate and records.
    String(copy(rawget(db,"!format")))==LEGACY_VERSION_TAG && stage!(db,"!format",bstr(VERSION_TAG))
end
function commit!(db)
    db.active || error("no metadata transaction")
    flush_pending!(db)
    direct!(db,["!transaction"=>bstr("committed")];sync=true)
    db.commit_fault==:exit_after_marker && ccall(:_exit,Cvoid,(Cint,),95)
    db.active=false;finish_journal!(db,false)
end
function rollback!(db)
    empty!(db.pending);db.pending_bytes=0;db.active=false;recover!(db)
end
function getrow(db,table,args...)
    v=rawget(db,rkey(table,args...));v===nothing ? nothing : unpack(v)
end
indexkeys(table,r)=String[]
index_signature(r)=nothing
for (table,indexes) in INDEXES
    schema=SCHEMA[table];body=Expr[];signature=Symbol[]
    for (name,fields,active) in indexes
        append!(signature,fields)
        condition=foldl((a,b)->:($a && $b),[:(!ismissing(r.$c)) for c in fields])
        if active
            c=table=="posts" ? :status : :live;push!(signature,c)
            condition=table=="posts" ? :($condition && r.status==1) : :($condition && r.live>0)
        end
        push!(body,:($condition && push!(out,key($("X/"*name*"/"),$([:(r.$c) for c in fields]...)))))
    end
    @eval begin
        function indexkeys(table,r::NamedTuple{$schema})
            out=String[];$(body...);out
        end
        index_signature(r::NamedTuple{$schema})=($([:(r.$c) for c in unique(signature)]...),)
    end
end
function counter!(db,name,delta)
    delta==0 && return nothing
    k="C/"*name;old=rawget(db,k);n=(old===nothing ? 0 : unpack(old))+delta
    n>=0 || error("negative metadata counter");stage!(db,k,pack(n))
end
countrows(db,table)=let x=rawget(db,"C/"*table);x===nothing ? 0 : unpack(x);end
function putrow!(db,table,input;replace=true,ignore=false)
    r=normalize_row(table,input);pk=rowkey(r)
    bytes=rawget(db,pk);old=bytes===nothing ? nothing : unpack(bytes)
    old!==nothing && ignore && return old
    old!==nothing && !replace && error("duplicate metadata primary key")
    old!==nothing && isequal(old,r) && return old
    old===nothing && counter!(db,table,1)
    if old===nothing || !isequal(index_signature(old),index_signature(r))
        oldkeys=old===nothing ? String[] : indexkeys(table,old);newkeys=indexkeys(table,r)
        for k in oldkeys;k in newkeys || stage!(db,k,nothing);end
        for k in newkeys
            k in oldkeys && continue
            if startswith(k,"X/leaf_slot/") || startswith(k,"X/term_name/")
                prior=rawget(db,k);prior!==nothing && String(copy(prior))!=pk && error("duplicate metadata secondary key")
            end
            stage!(db,k,bstr(pk))
        end
    end
    if table=="posts"
        counter!(db,"active_posts",Int(r.status==1)-Int(old!==nothing && old.status==1))
    end
    table=="kw_terms" && r.id>=next_term_id(db) && stage!(db,"C/max_term_id",pack(r.id))
    stage!(db,pk,pack(r));r
end
function deleterow!(db,table,r)
    for k in indexkeys(table,r);stage!(db,k,nothing);end
    stage!(db,rowkey(r),nothing)
    counter!(db,table,-1)
    table=="posts" && r.status==1 && counter!(db,"active_posts",-1)
    table=="kw_terms" && r.id==next_term_id(db)-1 && reset_term_max!(db)
    nothing
end
# Closed, internal statement vocabulary: known tables and scalar predicates only.
# Unsupported joins/subqueries are rejected, never mirrored into an SQL database.
function splitfields(text)
    out=String[];start=firstindex(text);depth=0;quoted=false
    for i in eachindex(text)
        c=text[i]
        c=='\'' && (quoted=!quoted)
        !quoted && c=='(' && (depth+=1)
        !quoted && c==')' && (depth-=1)
        if c==',' && depth==0 && !quoted
            push!(out,strip(text[start:prevind(text,i)]));start=nextind(text,i)
        end
    end
    push!(out,strip(text[start:end]));out
end
function literal(s,args,at)
    s=strip(s)
    if s=="?";at[]+=1;return args[at[]]
    elseif s=="NULL";return missing
    elseif startswith(s,"'") && endswith(s,"'");return s[2:end-1]
    else
        x=tryparse(Int64,s);x!==nothing && return x
        y=tryparse(Float64,s);y!==nothing && return y
        error("unsupported metadata literal: $s")
    end
end
function predicates(text,args,at)
    isempty(text) && return Tuple{Symbol,String,Any}[]
    result=Tuple{Symbol,String,Any}[]
    for p in split(text," AND ")
        m=match(r"^(?:\w+\.)?(\w+)\s*(IS NULL|IS NOT NULL|!=|>=|<=|=|>|<)\s*(.*)$",strip(p))
        m===nothing && error("unsupported metadata predicate: $p")
        push!(result,(Symbol(m[1]),String(m[2]),startswith(m[2],"IS") ? missing : literal(m[3],args,at)))
    end
    result
end
function matches(r,conds)
    for (c,op,v) in conds
        x=getproperty(r,c)
        ok=op=="IS NULL" ? ismissing(x) : op=="IS NOT NULL" ? !ismissing(x) :
            (ismissing(x)||ismissing(v)) ? false :
            op=="=" ? x==v : op=="!=" ? x!=v : op==">" ? x>v : op=="<" ? x<v : op==">=" ? x>=v : x<=v
        ok || return false
    end
    true
end
function visitrows(f,db,table,conds=Tuple{Symbol,String,Any}[];order="",limit=typemax(Int))
    eq=Dict(c=>v for (c,op,v) in conds if op=="=")
    primary=PRIMARY[table]
    if all(c->haskey(eq,c),primary)
        r=getrow(db,table,(eq[c] for c in primary)...)
        r!==nothing && matches(r,conds) && limit>0 && f(r)
        return
    end
    plans=Tuple{String,Tuple,Bool}[(rprefix(table),primary,false)]
    for (name,fields,active) in get(INDEXES,table,[])
        if active
            table=="posts" && get(eq,:status,0)!=1 && continue
            table=="blocks" && !any(x->x==(:live,">",0),conds) && continue
        end
        # Indexes omit NULLs: use only when predicates/order imply the prefix exists.
        first(fields) in (:leaf,:block,:textfile,:file,:term) && !haskey(eq,first(fields)) && continue
        push!(plans,("X/"*name*"/",fields,true))
    end
    sortfields=isempty(order) ? Symbol[] : Symbol.(strip.(replace.(split(order,','),r"^\w+\."=>"")))
    best=nothing;bestscore=-1
    for (prefix,fields,index) in plans
        n=0
        for c in fields;haskey(eq,c) ? (n+=1) : break;end
        remaining=[c for c in fields if !haskey(eq,c)]
        wanted=[c for c in sortfields if !haskey(eq,c)]
        !isempty(wanted) && (length(remaining)<length(wanted) || remaining[1:length(wanted)]!=wanted) && continue
        score=10n+Int(index)
        if score>bestscore;best=(prefix,fields,index,n);bestscore=score;end
    end
    best===nothing && error("unsupported metadata ordering: $table $order")
    prefix,fields,index,n=best
    if n==length(fields)
        v=rawget(db,key(prefix,(eq[c] for c in fields)...))
        r=v===nothing ? nothing : index ? let b=rawget(db,String(copy(v)));b===nothing ? nothing : unpack(b);end : unpack(v)
        r!==nothing && matches(r,conds) && limit>0 && f(r)
        return
    end
    prefix=key(prefix,(eq[c] for c in fields[1:n])...)*(n>0 ? "/" : "")
    start=prefix
    if n<length(fields)
        for (c,op,v) in conds
            c==fields[n+1] && op in (">",">=") && (start=prefix*part(v))
        end
    end
    count=0
    rawscan(db,prefix;start) do k,v
        r=index ? let b=rawget(db,String(copy(v)));b===nothing ? nothing : unpack(b);end : unpack(v)
        r===nothing && return true
        if n<length(fields)
            for (c,op,val) in conds
                c==fields[n+1] && op in ("<","<=") || continue
                x=getproperty(r,c)
                (op=="<" ? x>=val : x>val) && return false
            end
        end
        matches(r,conds) || return true
        count+=1
        f(r)===false && return false
        count<limit
    end
end
function project(r,columns)
    columns=="*" && return r
    cols=splitfields(columns);names=Symbol[];vals=Any[]
    for col in cols
        m=match(r"^(?:\w+\.)?(\w+)(?: AS (\w+))?$",col)
        m===nothing && error("unsupported metadata projection: $col")
        push!(names,Symbol(m[2]===nothing ? m[1] : m[2]));push!(vals,getproperty(r,Symbol(m[1])))
    end
    NamedTuple{Tuple(names)}(Tuple(vals))
end
function scalar(r,expr)
    m=match(r"^length\((\w+)\)$",expr);m!==nothing && return length(getproperty(r,Symbol(m[1])))
    m=match(r"^(\w+)=(-?\d+)$",expr);m!==nothing && return Int(getproperty(r,Symbol(m[1]))==parse(Int,m[2]))
    getproperty(r,Symbol(replace(expr,r"^\w+\."=>"")))
end
function simple_select(db,sql,args)
    m=match(r"^SELECT (.*?) FROM (\w+)(?: (?:p|t|c|k))?(?: WHERE (.*?))?(?: ORDER BY (.*?))?(?: LIMIT (\?|\d+))?$",sql)
    m===nothing && error("unsupported metadata read: $sql")
    columns,table=m[1],m[2];haskey(SCHEMA,table) || error("unknown metadata table")
    at=Ref(0);conds=predicates(something(m[3],""),args,at)
    order=String(something(m[4],""));limit=m[5]===nothing ? typemax(Int) : Int(literal(m[5],args,at))
    at[]==length(args) || error("metadata argument mismatch")
    fields=splitfields(columns)
    if any(x->occursin(r"^(COUNT|COALESCE|MIN|MAX|SUM)\(",x),fields)
        specs=[];names=Symbol[];vals=Any[]
        for col in fields
            q=match(r"^(.*?) AS (\w+)$",col);q===nothing && error("aggregate alias required")
            expr=q[1];push!(names,Symbol(q[2]))
            ag=match(r"^(?:COALESCE\()?((?:COUNT|SUM|MIN|MAX))\((.*?)\)(?:,0\))?$",expr)
            ag===nothing && error("unsupported aggregate $expr")
            push!(specs,(String(ag[1]),String(ag[2])));push!(vals,ag[1] in ("MIN","MAX") ? missing : 0)
        end
        if length(specs)==1 && specs[1]==("COUNT","*") && (isempty(conds) || (table=="posts" && conds==[(:status,"=",1)]))
            vals[1]=countrows(db,isempty(conds) ? table : "active_posts")
        elseif length(specs)==1 && specs[1]==("MIN","created") && table=="posts" && conds==[(:status,"=",1)]
            visitrows(db,table,conds;order="created,id",limit=1) do r;vals[1]=r.created;end
        else
            visitrows(db,table,conds;order) do r
                for (i,(op,arg)) in enumerate(specs)
                    x=arg=="*" ? 1 : scalar(r,arg);ismissing(x) && continue
                    vals[i]=op=="COUNT" ? vals[i]+1 : op=="SUM" ? vals[i]+x : ismissing(vals[i]) ? x : op=="MIN" ? min(vals[i],x) : max(vals[i],x)
                end
            end
        end
        return [NamedTuple{Tuple(names)}(Tuple(vals))]
    end
    out=NamedTuple[]
    limit>0 && visitrows(db,table,conds;order,limit) do r;push!(out,project(r,String(columns)));end
    out
end
function file_usage(db,name)
    vb=0;nb=0;tb=0;np=0
    visitrows(db,"blocks",[(:file,"=",name),(:live,">",0)]) do r;vb+=r.bytes;nb+=1;end
    visitrows(db,"posts",[(:textfile,"=",name),(:status,"=",1)]) do r;tb+=r.textlen;np+=1;end
    (;vb,nb,tb,np)
end
function keyword_page(db,leaf,after;distinctive=true)
    out=NamedTuple[];prefix=key(rprefix("kw_counts"),leaf)*"/"
    rawscan(db,prefix;start=prefix*part(after)) do _,v
        c=unpack(v);c.term>after || return true
        t=getrow(db,"kw_terms",c.term)
        if t!==nothing
            if distinctive
                g=getrow(db,"kw_counts",0,c.term)
                g===nothing || push!(out,(term=c.term,n=c.n,mass=c.mass,at=c.at,label=t.term,gm=g.mass,ga=g.at))
            else
                push!(out,(term=c.term,n=c.n,mass=c.mass,at=c.at,label=t.term))
            end
        end
        length(out)<256
    end
    out
end
function special_select(db,sql,args)
    if startswith(sql,"SELECT p.* FROM identities")
        i=getrow(db,"identities",args...);p=i===nothing ? nothing : getrow(db,"posts",i.id)
        return p===nothing ? NamedTuple[] : [p]
    elseif startswith(sql,"SELECT k.*,p.density_at")
        k=getrow(db,"kw_posts",args[1]);p=getrow(db,"posts",args[1])
        return k===nothing || p===nothing ? NamedTuple[] : [merge(k,(density_at=p.density_at,))]
    elseif startswith(sql,"SELECT c.term,c.n,c.mass,c.at,t.term AS label")
        return keyword_page(db,args[1],args[2];distinctive=occursin("JOIN kw_counts g",sql))
    elseif startswith(sql,"SELECT p.* FROM posts p WHERE p.status=1 AND p.id>?")
        out=NamedTuple[]
        visitrows(db,"posts",[(:status,"=",1),(:id,">",args[1])];order="id") do p
            getrow(db,"kw_posts",p.id)===nothing && push!(out,p)
            length(out)<args[2]
        end
        return out
    elseif startswith(sql,"SELECT id FROM kw_terms t WHERE")
        out=NamedTuple[]
        visitrows(db,"kw_terms",[(:id,">",args[1])];order="id") do t
            getrow(db,"kw_counts",0,t.id)===nothing && push!(out,(id=t.id,))
            length(out)<256
        end
        return out
    elseif startswith(sql,"SELECT f.name") || startswith(sql,"SELECT COUNT(*) AS n FROM files f WHERE")
        countonly=startswith(sql,"SELECT COUNT")
        out=NamedTuple[];n=0
        lm=match(r" LIMIT (\d+)$",sql);limit=lm===nothing ? typemax(Int) : parse(Int,lm[1])
        conds=occursin("f.kind='texts'",sql) ? [(:kind,"=","texts")] : Tuple{Symbol,String,Any}[]
        occursin("f.name>?",sql) && push!(conds,(:name,">",args[1]))
        visitrows(db,"files",conds;order="name") do f
            u=file_usage(db,f.name)
            valid=occursin("NOT EXISTS",sql) ? u.nb==0 && u.np==0 :
                occursin("f.bytes!=",sql) ? f.bytes!=u.tb : f.bytes>u.tb
            occursin("AND EXISTS",sql) && (valid &= u.np>0)
            if valid;n+=1;push!(out,occursin("f.name,f.bytes",sql) ? (name=f.name,bytes=f.bytes) : (name=f.name,));end
            countonly && empty!(out)
            n<limit
        end
        return countonly ? [(n=n,)] : out
    elseif startswith(sql,"SELECT b.id FROM blocks b JOIN files")
        out=NamedTuple[]
        visitrows(db,"blocks",[(:live,">",0),(:id,">",args[1]),(:id,"<=",args[2])];order="id") do b
            f=getrow(db,"files",b.file)
            (b.live<b.n || f.bytes>file_usage(db,b.file).vb) && push!(out,(id=b.id,))
            isempty(out)
        end
        return out
    elseif sql=="SELECT kind,SUM(bytes) AS bytes FROM files GROUP BY kind"
        sums=Dict{String,Int}();visitrows(db,"files") do f;sums[f.kind]=get(sums,f.kind,0)+f.bytes;end
        return [(kind=k,bytes=v) for (k,v) in sort!(collect(sums);by=first)]
    elseif startswith(sql,"SELECT COUNT(*) AS n FROM posts p WHERE status=1 AND (block IS NULL")
        n=0
        visitrows(db,"posts",[(:status,"=",1)]) do p
            (ismissing(p.block)||ismissing(p.leaf)||ismissing(p.slot)||getrow(db,"blocks",p.block)===nothing) && (n+=1)
            nothing
        end
        return [(n=n,)]
    elseif startswith(sql,"SELECT l.id,l.count,COUNT(p.id)")
        out=NamedTuple[]
        visitrows(db,"leaves") do l
            n=0;slot=0
            visitrows(db,"posts",[(:status,"=",1),(:leaf,"=",l.id)]) do p;n+=1;slot=max(slot,p.slot);end
            push!(out,(id=l.id,count=l.count,actual=n,maxslot=slot))
        end
        return out
    elseif occursin("FROM posts p JOIN kw_posts k",sql)
        columns=split(sql," FROM ")[1][8:end];out=NamedTuple[]
        visitrows(db,"posts",[(:status,"=",1)]) do p
            k=getrow(db,"kw_posts",p.id);k===nothing || push!(out,project(merge(p,k),String(columns)))
        end
        return out
    elseif startswith(sql,"SELECT COUNT(*) AS n FROM kw_posts k LEFT JOIN")
        n=0
        visitrows(db,"kw_posts") do k
            p=getrow(db,"posts",k.id);(p===nothing || p.status!=1) && (n+=1)
            nothing
        end
        return [(n=n,)]
    elseif startswith(sql,"SELECT n FROM kw_counts WHERE leaf=0 AND term=(SELECT id FROM kw_terms WHERE term=")
        name=match(r"term='(.*)'\)",sql)[1]
        found=simple_select(db,"SELECT id FROM kw_terms WHERE term=?",(String(name),))
        isempty(found) && return NamedTuple[]
        r=getrow(db,"kw_counts",0,only(found).id);return r===nothing ? NamedTuple[] : [(n=r.n,)]
    end
    nothing
end
next_term_id(db)=let x=rawget(db,"C/max_term_id");(x===nothing ? 0 : unpack(x))+1;end
function reset_term_max!(db)
    flush_pending!(db);it=RocksDB.DBIterator(db.native);prefix=rprefix("kw_terms")
    try
        RocksDB.seek!(it,prefix*"~")
        if RocksDB.valid(it);RocksDB.Lib.rocksdb_iter_prev(it.ptr)
        else;RocksDB.seek_to_last!(it);end
        n=RocksDB.valid(it) && startswith(String(RocksDB.key(it)),prefix) ? unpack(RocksDB.value(it)).id : 0
        RocksDB._check_iter_error(it)
        stage!(db,"C/max_term_id",pack(n))
    finally finalize(it) end
end
# Finite, application-owned prepared operations. No growing query cache and no
# regex/projection planning on these frequent inserts and scalar lookups.
const HOT_OPERATIONS=Dict{String,Function}()
for table in ROW_TABLES, policy in (""," OR REPLACE"," OR IGNORE")
    fields=SCHEMA[table];placeholders=join(fill("?",length(fields)),",")
    op=let table=table,fields=fields,policy=policy
        function(db,args)
            length(args)==length(fields) || error("metadata argument mismatch")
            putrow!(db,table,NamedTuple{fields}(Tuple(args));replace=policy==" OR REPLACE",ignore=policy==" OR IGNORE")
            NamedTuple[]
        end
    end
    for cols in ("","("*join(string.(fields),",")*")")
        HOT_OPERATIONS["INSERT$policy INTO $table$cols VALUES ($placeholders)"]=op
    end
end
for (sql,table,pk,fields) in (
    ("SELECT n,mass,at FROM kw_counts WHERE leaf=? AND term=?","kw_counts",2,(:n,:mass,:at)),
    ("SELECT count FROM leaves WHERE id=?","leaves",1,(:count,)),
    ("SELECT digest FROM kw_posts WHERE id=?","kw_posts",1,(:digest,)))
    HOT_OPERATIONS[sql]=let table=table,pk=pk,fields=fields
        function(db,args)
            length(args)==pk || error("metadata argument mismatch")
            r=getrow(db,table,args...)
            r===nothing ? NamedTuple[] : [NamedTuple{fields}(map(c->getproperty(r,c),fields))]
        end
    end
end
function term_row(db,term)
    p=rawget(db,key("X/term_name/",term))
    p===nothing && return nothing
    v=rawget(db,String(copy(p)));v===nothing ? nothing : unpack(v)
end
HOT_OPERATIONS["SELECT id FROM kw_terms WHERE term=?"]=(db,args)->begin
    length(args)==1 || error("metadata argument mismatch")
    r=term_row(db,args[1]);r===nothing ? NamedTuple[] : [(id=r.id,)]
end
HOT_OPERATIONS["INSERT OR IGNORE INTO kw_terms(term) VALUES (?)"]=(db,args)->begin
    length(args)==1 || error("metadata argument mismatch")
    term_row(db,args[1])===nothing && putrow!(db,"kw_terms",(id=next_term_id(db),term=args[1]);replace=false)
    NamedTuple[]
end
function execute(db,sql,args=())
    sql=String(sql)
    hot=get(HOT_OPERATIONS,sql,nothing);hot!==nothing && return hot(db,args)
    if sql=="BEGIN IMMEDIATE";begin!(db);return NamedTuple[]
    elseif sql=="COMMIT";commit!(db);return NamedTuple[]
    elseif sql=="ROLLBACK";rollback!(db);return NamedTuple[]
    elseif sql=="PRAGMA shrink_memory";return NamedTuple[]
    elseif startswith(sql,"CREATE TABLE") || startswith(sql,"CREATE INDEX") || startswith(sql,"CREATE UNIQUE INDEX")
        db.active || error("DDL outside transaction");return NamedTuple[]
    elseif startswith(sql,"SELECT ")
        special=special_select(db,sql,args);special!==nothing && return special
        return simple_select(db,sql,args)
    elseif startswith(sql,"INSERT ")
        m=match(r"^INSERT(?: OR (REPLACE|IGNORE))? INTO (\w+)(?:\((.*?)\))? VALUES \((.*)\)$",sql)
        m===nothing && error("unsupported metadata insert: $sql")
        policy,table=m[1],String(m[2]);at=Ref(0)
        columns=m[3]===nothing ? SCHEMA[table] : Tuple(Symbol.(strip.(split(m[3],','))))
        vals=literal.(splitfields(m[4]),Ref(args),Ref(at));at[]==length(args) || error("metadata argument mismatch")
        r=NamedTuple{columns}(Tuple(vals))
        if table=="kw_terms" && columns==(:term,)
            prior=simple_select(db,"SELECT id FROM kw_terms WHERE term=?",(r.term,))
            !isempty(prior) && policy=="IGNORE" && return NamedTuple[]
            r=(id=next_term_id(db),term=r.term)
        end
        putrow!(db,table,r;replace=policy=="REPLACE",ignore=policy=="IGNORE")
        return NamedTuple[]
    elseif startswith(sql,"UPDATE ")
        m=match(r"^UPDATE (\w+) SET (.*?) WHERE (.*)$",sql);m===nothing && error("unsupported metadata update")
        table=String(m[1]);at=Ref(0);assignments=[]
        for assignment in splitfields(m[2])
            field,rhs=strip.(split(assignment,'=';limit=2));op=match(r"^(\w+)([+-])(\d+)$",rhs)
            push!(assignments,(Symbol(field),op===nothing ? literal(rhs,args,at) : (Symbol(op[1]),op[2]=="+" ? parse(Int,op[3]) : -parse(Int,op[3]))))
        end
        conds=predicates(m[3],args,at);at[]==length(args) || error("metadata argument mismatch")
        visitrows(db,table,conds) do r
            updated=r
            for (field,v) in assignments
                value=v isa Tuple ? getproperty(r,v[1])+v[2] : v
                updated=merge(updated,NamedTuple{(field,)}((value,)))
            end
            putrow!(db,table,updated)
        end
        return NamedTuple[]
    elseif startswith(sql,"DELETE FROM ")
        m=match(r"^DELETE FROM (\w+)(?: WHERE (.*))?$",sql);m===nothing && error("unsupported metadata delete")
        at=Ref(0);table=String(m[1]);conds=predicates(something(m[2],""),args,at)
        at[]==length(args) || error("metadata argument mismatch")
        visitrows(db,table,conds) do r;deleterow!(db,table,r);end
        return NamedTuple[]
    end
    error("unsupported metadata operation: $sql")
end
function usage(db)
    props=Dict{String,Any}()
    for k in ("block-cache-usage","cur-size-all-mem-tables","estimate-table-readers-mem","estimate-pending-compaction-bytes")
        raw=RocksDB.property(db.native,"rocksdb."*k);props[k]=raw===nothing ? nothing : tryparse(Int,raw)
    end
    merge(props,Dict("block_cache_budget"=>db.block_budget,"memtable_budget_each"=>db.memtable_budget,
        "format_version"=>String(copy(rawget(db,"!format"))),
        "write_buffer_bytes"=>db.pending_bytes,"write_buffer_peak_bytes"=>db.peak_buffer_bytes,
        "write_buffer_limit"=>db.buffer_limit,"journal_flushes"=>db.flushes,"recovered"=>db.recovered))
end
metadata_bytes(db)=sum(filesize(joinpath(dir,f)) for (dir,_,files) in walkdir(db.path) for f in files;init=0)
end
