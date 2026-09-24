"""Arrow IPC checkpoints for the live index. Explicit tables, atomic generation pointer.
Only the two Julia RNG states use Julia serialization; post and model data are portable.
Read checkpoints from a trusted local store, not from arbitrary remote uploads.
"""
module ArrowPostStore
using Arrow, Tables, JSON3, SHA, Serialization, Random
using ..PostRecords, ..LeafLatentTrees, ..LeafTreeKernels, ..LiveIndex
export checkpoint!, restore, checkpoint_path, prune_checkpoints!
const SCHEMA_VERSION=1
const TABLES=("metadata","posts","aliases","codes","decoders","nodes","states","cursors")
function sync_directory(path)
    Sys.isunix() || return
    fd=ccall(:open,Cint,(Cstring,Cint),abspath(path),0)
    fd>=0 || error("cannot open checkpoint directory")
    try ccall(:fsync,Cint,(Cint,),fd)==0 || error("directory fsync failed")
    finally ccall(:close,Cint,(Cint,),fd) end
end
bytes(x)=take!(let io=IOBuffer();serialize(io,x);io;end)
function durable_write(path,writer)
    open(path,"w") do io
        writer(io);flush(io)
        Sys.isunix() && ccall(:fsync,Cint,(Cint,),Base.fd(io))!=0 && error("fsync failed")
    end
end
function write_table(path,table)
    durable_write(path,io->Arrow.write(io,table;file=true,compress=:zstd,ntasks=1,
        metadata=["mondrian.schema_version"=>string(SCHEMA_VERSION)]))
end
function checkpoint_path(root)
    name=strip(read(joinpath(root,"CURRENT"),String))
    occursin(r"^generation-[0-9]+-[0-9a-f]+$",name) || error("invalid checkpoint pointer")
    joinpath(root,name)
end
"""Publish a complete checkpoint. A crash before CURRENT replacement leaves the old
generation authoritative. `fault_after` is for failure-injection tests only.
One writer per directory; the caller must serialize checkpoint ownership.
"""
function checkpoint!(s::Index,root;fault_after=nothing)
    lock(s.lock) do
        audit(s);mkpath(root)
        generation="generation-$(s.generation)-$(randstring(RandomDevice(),collect("0123456789abcdef"),16))"
        staging=joinpath(root,"."*generation);mkdir(staging)
        f=s.fitted;c=s.config
        meta=(schema_version=[SCHEMA_VERSION],julia_version=[string(VERSION)],
            config_json=[JSON3.write(c)],clock=[s.clock],generation=[s.generation],
            next_id=[s.next_id],next_block=[s.next_block],has_tree=[f!==nothing],
            processed=[f===nothing ? 0 : f.processed],split_count=[f===nothing ? 0 : f.split_count],
            merge_count=[f===nothing ? 0 : f.merge_count],last_merge_time=[f===nothing ? s.clock : f.last_merge_time],
            rng=[f===nothing ? UInt8[] : bytes(f.rng)],basisrng=[f===nothing ? UInt8[] : bytes(f.basisrng)])
        postids=sort!(collect(keys(s.posts)));ps=[s.posts[id] for id in postids]
        hasrecord=[p.record!==nothing for p in ps]
        posts=(post_id=postids,kind=UInt8[p.key.kind for p in ps],native_id=[native_bytes(p.key) for p in ps],
            revision_at=Float64[p.revision_at for p in ps],observed_at=Float64[p.observed_at for p in ps],status=UInt8[p.status for p in ps],
            density_at=Float64[p.density_at for p in ps],sequence=Int64[p.sequence for p in ps],has_record=hasrecord,
            platform=String[p.record===nothing ? "" : p.record.platform for p in ps],
            author=String[p.record===nothing ? "" : p.record.author for p in ps],
            text=String[p.record===nothing ? "" : p.record.text for p in ps],
            created_at=Float64[p.record===nothing ? 0. : p.record.created_at for p in ps],
            permalink=String[p.record===nothing ? "" : p.record.permalink for p in ps],
            reply_to=String[p.record===nothing ? "" : p.record.reply_to for p in ps],
            tags=Vector{String}[p.record===nothing ? String[] : p.record.tags for p in ps],
            metadata_json=String[p.record===nothing ? "{}" : p.record.metadata for p in ps])
        keysorted=sort!([k for (k,id) in s.identities if k!=s.posts[id].key];by=k->(k.kind,k.native))
        aliases=(kind=UInt8[k.kind for k in keysorted],native_id=[native_bytes(k) for k in keysorted],
            post_id=UInt64[s.identities[k] for k in keysorted])
        codetable=(post_id=UInt64[],decoder_id=UInt64[],region=Int32[],scale=Float32[],residual=Vector{UInt8}[])
        for b in s.blocks,j in eachindex(b.post_ids)
            push!(codetable.post_id,b.post_ids[j]);push!(codetable.decoder_id,b.id);push!(codetable.region,Int32(b.regions[j]))
            push!(codetable.scale,b.codes.scales[j]);push!(codetable.residual,copy(b.codes.residuals[:,j]))
        end
        decoders=(decoder_id=UInt64[b.id for b in s.blocks],center=Vector{Float32}[b.codes.center for b in s.blocks])
        nodes=(node_id=Int32[],lo=Vector{Float64}[],hi=Vector{Float64}[],axis=Int32[],cut=Float64[],left=Int32[],right=Int32[],
            depth=Int32[],arrivals=Int64[],seen=Int64[],last_time=Float64[],effective_arrivals=Float64[],
            reservoir=Vector{Float32}[],reservoir_rows=Vector{Int64}[])
        states=(node_id=Int32[],mass=Float64[],mean=Vector{Float64}[],scatter=Vector{Float64}[],
            basis=Vector{Float64}[],eigenvalues=Vector{Float64}[])
        if f!==nothing
            for (i,n) in enumerate(f.nodes)
                for (name,value) in ((:node_id,i),(:lo,n.lo),(:hi,n.hi),(:axis,n.axis),(:cut,n.cut),(:left,n.left),(:right,n.right),
                    (:depth,n.depth),(:arrivals,n.arrivals),(:seen,n.seen),(:last_time,f.last_times[i]),(:effective_arrivals,f.effective_arrivals[i]),
                    (:reservoir,vec(n.reservoir)),(:reservoir_rows,f.reservoir_rows[i]))
                    push!(getproperty(nodes,name),value)
                end
                if n.axis==0
                    p=only(n.states)
                    for (name,value) in ((:node_id,i),(:mass,p.mass),(:mean,p.mean),(:scatter,p.diagonal),(:basis,vec(p.U)),(:eigenvalues,p.eigenvalues))
                        push!(getproperty(states,name),value)
                    end
                end
            end
        end
        ck=sort!(collect(keys(s.cursors)));cursors=(source=ck,cursor=[s.cursors[k] for k in ck])
        payloads=(meta,posts,aliases,codetable,decoders,nodes,states,cursors)
        for (name,table) in zip(TABLES,payloads)
            write_table(joinpath(staging,name*".arrow"),table)
            fault_after==name && error("injected checkpoint failure after $name")
        end
        names=[name*".arrow" for name in TABLES]
        write_table(joinpath(staging,"checksums.arrow"),(file=names,sha256=[bytes2hex(open(sha256,joinpath(staging,n))) for n in names]))
        sync_directory(staging)
        final=joinpath(root,generation);Base.Filesystem.rename(staging,final)
        sync_directory(root)
        pointer=joinpath(root,".CURRENT-"*generation)
        durable_write(pointer,io->write(io,generation*"\n"))
        fault_after=="publish" && error("injected checkpoint failure before CURRENT replacement")
        Base.Filesystem.rename(pointer,joinpath(root,"CURRENT"))
        # Persist the directory entry on POSIX. Checkpoints are local-filesystem artifacts.
        sync_directory(root)
        final
    end
end

function restore(root;backend=KernelTreeBackend())
    path=checkpoint_path(root)
    sums=Arrow.Table(joinpath(path,"checksums.arrow"))
    Set(String.(sums.file))==Set(name*".arrow" for name in TABLES) || error("incomplete checkpoint manifest")
    for row in Tables.rows(sums)
        bytes2hex(open(sha256,joinpath(path,row.file)))==row.sha256 || error("checkpoint checksum mismatch: $(row.file)")
    end
    m=only(Tables.rows(Arrow.Table(joinpath(path,"metadata.arrow"))))
    m.schema_version==SCHEMA_VERSION || error("unsupported checkpoint schema")
    m.julia_version==string(VERSION) || error("RNG continuation requires Julia $(m.julia_version); use a migration before restoring with $VERSION")
    config=IndexConfig(;Dict(Symbol(k)=>v for (k,v) in pairs(JSON3.read(m.config_json)))...)
    s=Index(config;backend,time=m.clock);s.generation=m.generation;s.next_id=m.next_id;s.next_block=m.next_block
    posts=Arrow.Table(joinpath(path,"posts.arrow"))
    for p in Tables.rows(posts)
        key=key_from_bytes(p.kind,p.native_id)
        record=p.has_record ? PostRecord(key;platform=p.platform,author=p.author,text=p.text,created_at=p.created_at,
            observed_at=p.observed_at,revision_at=p.revision_at,permalink=p.permalink,reply_to=p.reply_to,
            tags=collect(p.tags),metadata=p.metadata_json) : nothing
        s.posts[p.post_id]=PostState(p.post_id,key,p.revision_at,p.observed_at,p.status,record,p.density_at,p.sequence)
        s.identities[key]=p.post_id
    end
    for row in Tables.rows(Arrow.Table(joinpath(path,"aliases.arrow")))
        s.identities[key_from_bytes(row.kind,row.native_id)]=row.post_id
    end
    centers=Dict(row.decoder_id=>Float32.(row.center) for row in Tables.rows(Arrow.Table(joinpath(path,"decoders.arrow"))))
    code=Arrow.Table(joinpath(path,"codes.arrow"))
    for block in unique(code.decoder_id)
        ids=findall(==(block),code.decoder_id);payload=hcat([Vector{UInt8}(code.residual[j]) for j in ids]...)
        c=CentroidCodes(centers[block],Float32.(code.scale[ids]),payload)
        push!(s.blocks,CodeBlock(block,UInt64.(code.post_id[ids]),Int.(code.region[ids]),c))
    end
    if m.has_tree
        ns=LeafNode[];lasttimes=Float64[];arrivals=Float64[];reservoirrows=Vector{Int}[]
        for row in Tables.rows(Arrow.Table(joinpath(path,"nodes.arrow")))
            row.node_id==length(ns)+1 || error("invalid node ordering")
            R=reshape(Float32.(row.reservoir),config.dimension,:)
            push!(ns,LeafNode(Float64.(row.lo),Float64.(row.hi),row.axis,row.cut,row.left,row.right,row.depth,row.arrivals,row.seen,R,Sketch[]))
            push!(lasttimes,row.last_time);push!(arrivals,row.effective_arrivals);push!(reservoirrows,Int.(row.reservoir_rows))
        end
        for row in Tables.rows(Arrow.Table(joinpath(path,"states.arrow")))
            values=Float64.(row.eigenvalues);basis=reshape(Float64.(row.basis),config.dimension,length(values))
            push!(ns[row.node_id].states,Sketch(:ppca,config.rank,row.mass,Float64.(row.mean),Float64.(row.scatter),basis,values))
        end
        s.fitted=(nodes=ns,errors=[NamedTuple[]],update_seconds=[0.],split_seconds=[0.],configs=[(:ppca,config.rank)],
            half_life=config.half_life,time=m.clock,last_times=lasttimes,effective_arrivals=arrivals,reservoir_rows=reservoirrows,
            reservoir_cap=config.reservoir_cap,split_after=config.split_after,merge_below=config.merge_below,
            merge_interval=config.merge_interval,last_merge_time=m.last_merge_time,merge_count=m.merge_count,split_count=m.split_count,
            processed=m.processed,rng=deserialize(IOBuffer(Vector{UInt8}(m.rng))),basisrng=deserialize(IOBuffer(Vector{UInt8}(m.basisrng))))
    end
    for row in Tables.rows(Arrow.Table(joinpath(path,"cursors.arrow")));s.cursors[row.source]=row.cursor;end
    LiveIndex.relocate!(s);audit(s);s
end
"""Reclaim old completed generations after a successful checkpoint. Never remove
CURRENT or unrelated files. Use while holding exclusive ownership of this store.
"""
function prune_checkpoints!(root;keep=2)
    keep>=1 || throw(ArgumentError("keep must be positive"))
    current=checkpoint_path(root)
    paths=[joinpath(root,n) for n in readdir(root) if occursin(r"^generation-[0-9]+-[0-9a-f]+$",n) && isdir(joinpath(root,n))]
    sort!(paths;by=p->stat(p).mtime,rev=true)
    retained=Set(vcat([current],[p for p in paths if p!=current][1:min(keep-1,length(paths)-1)]))
    removed=0
    for p in paths
        p in retained && continue
        rm(p;recursive=true);removed+=1
    end
    removed
end
end
