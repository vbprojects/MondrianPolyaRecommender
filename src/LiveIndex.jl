"""Resumable density tree + independent immutable post codes and stable identities.
Include PostRecords, LeafLatentTrees and LeafTreeKernels before this module.
"""
module LiveIndex
using Random, LinearAlgebra
using ..PostRecords, ..LeafLatentTrees, ..LeafTreeKernels
const LT=LeafLatentTrees
const LK=LeafTreeKernels
export Index, IndexConfig, ingest!, maintain!, expire!, lookup, embedding, candidates,
       audit, rebuild_density!, CodeBlock, PostState, gc_tombstones!

Base.@kwdef struct IndexConfig
    dimension::Int=256
    rank::Int=4
    half_life::Float64=6*3600.
    max_leaves::Int=128
    split_after::Int=256
    reservoir_cap::Int=64
    max_depth::Int=16
    merge_below::Float64=8.
    merge_interval::Float64=3600.
    codec_batch::Int=256
    retention_seconds::Float64=86400.
    seed::Int=918
    embedding_model::String="futur/Qwen3-Embedding-0.6B-model2vec-onnx@a3f60a85a2570e7976ea7f2d6505ad3e5152a96f"
end
function validate(c::IndexConfig)
    c.dimension>0 && 0<=c.rank<=c.dimension && c.max_leaves>=1 && c.split_after>=16 &&
        c.reservoir_cap>=16 && c.max_depth>=1 && c.codec_batch>=1 &&
        isfinite(c.half_life) && c.half_life>0 && isfinite(c.merge_interval) && c.merge_interval>0 &&
        0<=c.merge_below<c.split_after && isfinite(c.retention_seconds) && c.retention_seconds>0 &&
        !isempty(c.embedding_model) || throw(ArgumentError("invalid live index configuration"))
    c
end
struct CodeBlock
    id::UInt64
    post_ids::Vector{UInt64}
    regions::Vector{Int}
    codes::CentroidCodes
end
mutable struct PostState
    id::UInt64
    key::PostKey
    revision_at::Float64
    observed_at::Float64
    status::UInt8 # 1 active, 2 deleted, 3 expired
    record::Union{Nothing,PostRecord}
    density_at::Float64
    sequence::Int
end
mutable struct Index
    config::IndexConfig
    backend::KernelTreeBackend
    fitted::Any
    posts::Dict{UInt64,PostState}
    identities::Dict{PostKey,UInt64}
    blocks::Vector{CodeBlock}
    locations::Dict{UInt64,Tuple{Int,Int}}
    next_id::UInt64
    next_block::UInt64
    clock::Float64
    generation::UInt64
    cursors::Dict{String,String}
    lock::ReentrantLock
end
function Index(config=IndexConfig();backend=KernelTreeBackend(),time=0.)
    validate(config);isfinite(time) || throw(ArgumentError("invalid clock"))
    Index(config,backend,nothing,Dict{UInt64,PostState}(),Dict{PostKey,UInt64}(),CodeBlock[],
        Dict{UInt64,Tuple{Int,Int}}(),1,1,Float64(time),0,Dict{String,String}(),ReentrantLock())
end
options(c)=(max_leaves=c.max_leaves,split_after=c.split_after,reservoir_cap=c.reservoir_cap,seed=c.seed,
    configs=[(:ppca,c.rank)],max_depth=c.max_depth,half_life=c.half_life,
    merge_below=c.merge_below,merge_interval=c.merge_interval)
lookup(s::Index,key::PostKey)=lock(s.lock) do
    get(s.posts,get(s.identities,key,UInt64(0)),nothing)
end
function existing_event(s,e)
    keys=e.record===nothing ? [e.key] : vcat([e.key],e.record.aliases)
    ids=unique([s.identities[k] for k in keys if haskey(s.identities,k)])
    length(ids)<=1 || throw(ArgumentError("event aliases resolve to different posts"))
    isempty(ids) ? nothing : s.posts[only(ids)]
end
function embedding(s::Index,id::Integer)
    lock(s.lock) do
    at=get(s.locations,UInt64(id),nothing);at===nothing && throw(KeyError(id))
    b,j=at;vec(decode_centroid(s.backend,s.blocks[b].codes;indices=[j]))
    end
end
function relocate!(s)
    empty!(s.locations)
    for (i,b) in enumerate(s.blocks),j in eachindex(b.post_ids)
        haskey(s.locations,b.post_ids[j]) && error("duplicate encoded post ID")
        s.locations[b.post_ids[j]]=(i,j)
    end
end
function reroute!(s)
    s.fitted===nothing && return
    for b in s.blocks;b.regions.=route_centroid(s.backend,s.fitted.nodes,b.codes);end
end
function remove_codes!(s,removed)
    isempty(removed) && return
    kept=CodeBlock[]
    for b in s.blocks
        ids=findall(id->!(id in removed),b.post_ids)
        isempty(ids) && continue
        if length(ids)==length(b.post_ids);push!(kept,b)
        else
            # Slice bytes, never decode/re-encode. Shared center remains unchanged.
            c=CentroidCodes(b.codes.center,b.codes.scales[ids],b.codes.residuals[:,ids])
            push!(kept,CodeBlock(b.id,b.post_ids[ids],b.regions[ids],c))
        end
    end
    s.blocks=kept;relocate!(s)
end

"""Rebase specified leaf states on the retained compressed observations at one clock.
Edits/deletes cannot be exactly undone from a rank-truncated sketch. This explicit
rebuild avoids accumulating ghost posts. It changes the density by codec/sketch error.
"""
function rebuild_density!(s::Index,regions=Set(b.regions[j] for b in s.blocks for j in eachindex(b.regions)))
    s.fitted===nothing && return s
    nodes=s.fitted.nodes;regions=Set(Int.(collect(regions)))
    ids=sort!([i for i in regions if 1<=i<=length(nodes) && nodes[i].axis==0])
    isempty(ids) && return s
    slots=Dict(id=>j for (j,id) in enumerate(ids));c=s.config
    states=[Sketch(:ppca,c.rank,c.dimension) for _ in ids]
    # Stable observation order across blocks, including edited posts with reused IDs.
    refs=Tuple{Int,Int,Int}[]
    for (bi,b) in enumerate(s.blocks),j in eachindex(b.post_ids)
        haskey(slots,b.regions[j]) && push!(refs,(s.posts[b.post_ids[j]].sequence,bi,j))
    end
    sort!(refs;by=first)
    reservoirs=[Vector{Tuple{Int,Vector{Float32}}}() for _ in ids]
    for start in 1:c.codec_batch:length(refs)
        chosen=refs[start:min(start+c.codec_batch-1,length(refs))]
        X=zeros(Float32,c.dimension,length(chosen));groups=Int[];weights=Float64[]
        for (col,(seq,bi,j)) in enumerate(chosen)
            b=s.blocks[bi];state=s.posts[b.post_ids[j]];slot=slots[b.regions[j]]
            # Small transient decode; no retained full embedding archive.
            for a in 1:c.dimension;X[a,col]=LK.centroid_value(b.codes.center,b.codes.residuals,b.codes.scales,a,j);end
            push!(groups,slot);push!(weights,exp2(-(s.clock-state.density_at)/c.half_life))
            push!(reservoirs[slot],(seq,copy(X[:,col])))
            length(reservoirs[slot])>c.reservoir_cap && popfirst!(reservoirs[slot])
        end
        kernel_weighted_update!(s.backend,states,X,groups,weights)
    end
    for (slot,i) in enumerate(ids)
        n=nodes[i];n.states=[states[slot]];R=reservoirs[slot]
        n.reservoir=isempty(R) ? zeros(Float32,c.dimension,0) : hcat(last.(R)...)
        s.fitted.reservoir_rows[i]=first.(R);n.seen=length(R);n.arrivals=0
        s.fitted.effective_arrivals[i]=0.;s.fitted.last_times[i]=s.clock
    end
    s.fitted=merge(s.fitted,(time=s.clock,));s
end

function touch_aliases!(s,id,record)
    for key in vcat([record.key],record.aliases)
        prior=get(s.identities,key,id);prior==id || throw(ArgumentError("alias belongs to a different post"))
        s.identities[key]=id
    end
end
"""Ingest normalized events with an embedding callback `texts -> d×n Float32 matrix`.
All embedding/validation work finishes before mutating the index. Duplicate deliveries
and unchanged text do not update density. One commit clock per microbatch; source
creation/revision times are retained separately. Call checkpoint! to commit to Arrow.
"""
function ingest!(s::Index,events::AbstractVector{PostEvent},embed;time=nothing)
    lock(s.lock) do
        isempty(events) && return (inserted=0,updated=0,deleted=0,ignored=0)
        now=time===nothing ? max(s.clock,maximum(e.observed_at for e in events)) : Float64(time)
        isfinite(now) && now>=s.clock || throw(ArgumentError("ingestion clock moved backwards"))
        # Collapse deliveries for the same known identity to the newest source revision.
        selected=Dict{PostKey,PostEvent}();order=PostKey[]
        batch_primary=Dict{PostKey,PostKey}()
        for e in events
            e.operation==:upsert || continue
            old=existing_event(s,e);key=old===nothing ? e.key : old.key
            for alias in vcat([e.key],e.record.aliases)
                get(batch_primary,alias,key)==key || throw(ArgumentError("conflicting batch aliases"))
                batch_primary[alias]=key
            end
        end
        batch_authors=Dict(e.key=>e.record.author for e in events if e.operation==:upsert && e.key.kind==3)
        for e in events
            old=existing_event(s,e);key=old===nothing ? get(batch_primary,e.key,e.key) : old.key
            # Authenticate deletion ownership before collapsing deliveries: an
            # invalid deletion must not suppress a valid note in the same batch.
            if e.operation==:delete && key.kind==3
                author=old!==nothing && old.record!==nothing ? old.record.author : get(batch_authors,key,nothing)
                (author===nothing || author!=e.author) && continue
            end
            if !haskey(selected,key);push!(order,key);selected[key]=e
            elseif e.revision_at>selected[key].revision_at ||
                (e.revision_at==selected[key].revision_at && (e.operation==:delete ||
                    (selected[key].operation==:upsert && e.observed_at>selected[key].observed_at)))
                selected[key]=e
            end
        end
        actions=Tuple{PostKey,PostEvent,Union{Nothing,PostState},Bool}[];texts=String[];ignored=length(events)-length(order)
        claimed=Dict{PostKey,UInt64}();next=s.next_id
        for key in order
            e=selected[key];old=lookup(s,key)
            if old!==nothing && (e.revision_at<old.revision_at ||
                (e.revision_at==old.revision_at && (old.status!=1 ||
                    (e.operation==:upsert && e.observed_at<old.observed_at))))
                ignored+=1;continue
            end
            if e.operation==:upsert && e.record.created_at<now-s.config.retention_seconds
                ignored+=1;continue
            end
            if old!==nothing && e.operation==:upsert && e.revision_at==old.revision_at &&
                    old.record!==nothing && content_digest(old.record)!=content_digest(e.record)
                throw(ArgumentError("conflicting content at an identical source revision"))
            end
            changed=e.operation==:upsert && (old===nothing || old.record===nothing || content_digest(old.record)!=content_digest(e.record))
            if e.operation==:upsert
                id=old===nothing ? next : old.id
                for alias in vcat([e.record.key],e.record.aliases)
                    get(claimed,alias,get(s.identities,alias,id))==id || throw(ArgumentError("conflicting identity alias"));claimed[alias]=id
                end
            end
            old===nothing && (next+=1)
            changed && push!(texts,e.record.text)
            push!(actions,(key,e,old,changed))
        end
        X=isempty(texts) ? zeros(Float32,s.config.dimension,0) : Float32.(embed(texts))
        size(X)==(s.config.dimension,length(texts)) && all(isfinite,X) && all(x->abs(x)<=1,X) || throw(ArgumentError("embedding callback returned invalid coordinates or dimensions"))
        s.clock=now;removed=Set{UInt64}();affected=Set{Int}()
        topology=s.fitted===nothing ? nothing : [(n.axis,n.cut,n.left,n.right) for n in s.fitted.nodes]
        for (_,e,old,changed) in actions
            if old!==nothing && old.status==1 && (e.operation==:delete || changed)
                bi,j=s.locations[old.id];push!(affected,s.blocks[bi].regions[j]);push!(removed,old.id)
            end
        end
        remove_codes!(s,removed)
        # Rebuild before fitting new observations; do not train on old and new versions.
        rebuild_density!(s,affected)
        before=s.fitted===nothing ? 0 : s.fitted.processed
        if !isempty(texts)
            s.fitted=fit_kernel_tree(s.backend,X;options(s.config)...,timestamps=fill(now,length(texts)),_initial=s.fitted)
        elseif s.fitted!==nothing
            s.fitted=kernel_merge_tree!(s.backend,s.fitted;time=now,merge_below=s.config.merge_below)
        end
        # Compaction and new cuts can invalidate all old node indices. This routes
        # directly from frozen bytes; no historical vector is recompressed.
        if s.fitted!==nothing && topology!=[(n.axis,n.cut,n.left,n.right) for n in s.fitted.nodes];reroute!(s);end
        newids=UInt64[];inserted=0;updated=0;deleted=0;column=0
        for (key,e,old,changed) in actions
            id=old===nothing ? s.next_id : old.id
            old===nothing && (s.next_id+=1)
            if e.operation==:delete
                s.posts[id]=PostState(id,key,e.revision_at,e.observed_at,2,nothing,now,0)
                s.identities[key]=id;deleted+=1
                for (alias,primary) in batch_primary
                    primary==key && (s.identities[alias]=id)
                end
            else
                changed && (column+=1;push!(newids,id))
                state=PostState(id,key,e.revision_at,e.observed_at,1,e.record,
                    changed ? now : old.density_at,changed ? before+column : old.sequence)
                s.posts[id]=state;touch_aliases!(s,id,e.record);s.identities[key]=id
                old===nothing ? (inserted+=1) : (updated+=1)
            end
        end
        # Pack bounded frozen batches in leaf order. A decoder may span leaves:
        # forcing one tiny decoder per leaf per microbatch can cost more than the
        # original vectors. Membership remains a per-post property.
        if !isempty(newids)
            assigned=kernel_route(s.backend,s.fitted.nodes,X)
            members=sortperm(assigned)
            for start in 1:s.config.codec_batch:length(members)
                cols=members[start:min(start+s.config.codec_batch-1,length(members))];ids=newids[cols]
                codes=encode_centroid(s.backend,X[:,cols]);regions=route_centroid(s.backend,s.fitted.nodes,codes)
                push!(s.blocks,CodeBlock(s.next_block,ids,regions,codes));s.next_block+=1
            end
        end
        relocate!(s);s.generation+=1
        (inserted=inserted,updated=updated,deleted=deleted,ignored=ignored)
    end
end

function maintain!(s::Index;time)
    lock(s.lock) do
        isfinite(time) && time>=s.clock || throw(ArgumentError("maintenance clock moved backwards"))
        s.clock=Float64(time)
        if s.fitted!==nothing
            old=s.fitted.merge_count
            s.fitted=kernel_merge_tree!(s.backend,s.fitted;time,merge_below=s.config.merge_below)
            s.fitted.merge_count!=old && reroute!(s)
        end
        s.generation+=1;s
    end
end
"""Forget inactive identity markers outside the configured replay horizon. IDs are
never reused. Call only with a cutoff older than the source replay/retention window.
"""
function gc_tombstones!(s::Index;before)
    lock(s.lock) do
        isfinite(before) && before<=s.clock-s.config.retention_seconds || throw(ArgumentError("tombstone GC must be outside retention window"))
        remove=Set(id for (id,p) in s.posts if p.status!=1 && p.observed_at<before)
        for id in remove;delete!(s.posts,id);end
        filter!(pair->last(pair) ∉ remove,s.identities)
        !isempty(remove) && (s.generation+=1)
        length(remove)
    end
end
"""Expire source posts before a supplied UTC cutoff; remove their text and codes.
Identity tombstones remain for replay protection; their GC policy is independent.
"""
function expire!(s::Index;before,time=s.clock)
    lock(s.lock) do
        isfinite(before) && isfinite(time) && time>=s.clock || throw(ArgumentError("invalid expiry clock"))
        s.clock=Float64(time);removed=Set{UInt64}();affected=Set{Int}()
        for (id,state) in s.posts
            if state.status==1 && state.record.created_at<before
                bi,j=s.locations[id];push!(affected,s.blocks[bi].regions[j]);push!(removed,id)
                state.status=3;state.record=nothing
            end
        end
        remove_codes!(s,removed);rebuild_density!(s,affected);maintain!(s;time)
        length(removed)
    end
end
"""A bounded regional sample, not exact kNN. Probe at most leaf_budget boxes and
return at most limit retained posts. Geometry work is linear in the bounded tree;
the posting-list scan is currently linear in retained IDs, not embedding dimension.
"""
function candidates(s::Index,query;limit=20,leaf_budget=4,rng=Random.default_rng(),platform=nothing)
    lock(s.lock) do
        length(query)==s.config.dimension && all(isfinite,query) && 0<limit<=1000 && 0<leaf_budget<=s.config.max_leaves || throw(ArgumentError("invalid regional query"))
        s.fitted===nothing && return UInt64[]
        ns=s.fitted.nodes;leaves=findall(n->n.axis==0,ns)
        distances=[sum(abs2,max.(ns[i].lo.-query,0).+max.(query.-ns[i].hi,0)) for i in leaves]
        chosen=Set(leaves[partialsortperm(distances,1:min(leaf_budget,length(leaves)))])
        # Reservoir sampling bounds query memory independently of corpus size.
        result=UInt64[];seen=0
        for b in s.blocks,j in eachindex(b.post_ids)
            b.regions[j] in chosen || continue
            id=b.post_ids[j];state=s.posts[id]
            platform!==nothing && state.record.platform!=platform && continue
            seen+=1
            if length(result)<limit;push!(result,id)
            else;slot=rand(rng,1:seen);slot<=limit && (result[slot]=id);end
        end
        result
    end
end
function audit(s::Index)
    active=Set(id for (id,p) in s.posts if p.status==1)
    active==Set(keys(s.locations)) || error("metadata/code identity mismatch")
    for (key,id) in s.identities;haskey(s.posts,id) || error("dangling identity");end
    for b in s.blocks
        LK.validate_codes(b.codes)
        length(b.post_ids)==length(b.regions)==length(b.codes.scales) || error("invalid block columns")
        b.regions==route_centroid(s.backend,s.fitted.nodes,b.codes) || error("stale membership")
        for id in b.post_ids
            state=s.posts[id];state.record!==nothing && state.status==1 || error("deleted post remains queryable")
            s.identities[state.key]==id || error("identity mapping mismatch")
        end
    end
    isempty(s.posts) || s.next_id>maximum(keys(s.posts)) || error("post IDs would be reused")
    isempty(s.blocks) || s.next_block>maximum(b.id for b in s.blocks) || error("decoder IDs would be reused")
    (active=length(active),identities=length(s.identities),blocks=length(s.blocks),
     leaves=s.fitted===nothing ? 0 : count(n->n.axis==0,s.fitted.nodes),
     splits=s.fitted===nothing ? 0 : s.fitted.split_count,merges=s.fitted===nothing ? 0 : s.fitted.merge_count)
end
end
