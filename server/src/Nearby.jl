function candidate_ids(ctx,q,budget,leafbudget,t,rng)
    s=ctx.store;containing=DS.route(s,q)
    chosen=DS.candidate_leaves(s,q;leaf_budget=min(leafbudget,s.config.max_leaves))
    chosen=unique(vcat([containing],chosen))[1:min(leafbudget,length(unique(vcat([containing],chosen))))]
    counts=[DS.one(s,"SELECT count FROM leaves WHERE id=?",(id,)).count for id in chosen]
    # Sparse Fisher-Yates maps draw distinct slots without loading a leaf catalog.
    remaining=copy(counts);swaps=[Dict{Int,Int}() for _ in chosen]
    ids=Int64[];draws=0
    function draw(j)
        n=remaining[j];n==0 && return
        slot=rand(rng,1:n);actual=get(swaps[j],slot,slot)
        swaps[j][slot]=get(swaps[j],n,n);delete!(swaps[j],n);remaining[j]-=1;draws+=1
        p=DS.one(s,"SELECT id,created FROM posts WHERE status=1 AND leaf=? AND slot=?",(chosen[j],actual))
        p.created>=t-s.config.retention_seconds && push!(ids,p.id)
    end
    for j in eachindex(chosen)
        length(ids)>=budget && break
        draw(j)
    end
    while length(ids)<budget && sum(remaining)>0 && draws<ctx.settings.max_draws
        at=rand(rng,1:sum(remaining));draw(searchsortedfirst(cumsum(remaining),at))
    end
    (;ids,leaves=chosen,draws,remaining=sum(remaining))
end
function nearby(ctx,input;time=ctx.store.clock,rng=MersenneTwister(rand(RandomDevice(),UInt64)))
    valid_time(ctx,time);s=ctx.store;c=ctx.settings;q=vector_input(ctx,input)
    norm(q)>0 || fail("zero_query_vector",422)
    limit=integer_option(input,"limit",c.default_limit,1,25)
    budget=integer_option(input,"candidate_budget",c.candidates,limit,c.max_candidates)
    leaves=integer_option(input,"leaf_budget",c.leaf_budget,1,c.max_leaves)
    empty!(s.request_blocks);empty!(s.request_decoders)
    read_before=get(s.metrics,"vector_read_bytes",0.)
    misses_before=s.block_cache.misses
    pool=candidate_ids(ctx,q,budget,leaves,time,rng)
    X=DS.fetch_vectors(s,pool.ids;time)
    scores=Tuple{Float64,Int64,Int}[];zero=0
    for (j,id) in enumerate(pool.ids)
        x=view(X,:,j);nx=norm(x)
        if nx==0;zero+=1;continue;end
        push!(scores,(clamp(1-dot(q,x)/(norm(q)*nx),0.,2.),id,j))
    end
    sort!(scores);selected=scores[1:min(limit,length(scores))]
    records=DS.fetch_records(s,[r[2] for r in selected];time)
    items=Any[]
    for ((distance,id,j),record) in zip(selected,records)
        p=DS.lookup(s,id);key=key_from_bytes(p.kind,p.native)
        push!(items,(post_id=string(id),identity=(kind=platform_name(key),native_id=key.native),
            platform=String(record.platform),leaf_id=string(p.leaf),distance,
            delta=Float32.(Float64.(X[:,j]).-q)))
    end
    env=envelope(ctx,time)
    reason=length(items)>=limit ? nothing : zero>0 ? "zero_vectors_or_insufficient_candidates" :
        pool.draws>=c.max_draws ? "posting_draw_budget" : "insufficient_eligible_candidates"
    merge(env,(query_origin=q,delta_dtype="float32",delta_definition="decoded_stored_vector - query_origin",
        metric="cosine",leaf_selection_metric="squared_euclidean_box_distance",
        storage_dtype=string(s.config.mode),approximate=true,
        client_tree_changed=haskey(input,"client_tree_revision") && input["client_tree_revision"]!=env.tree_revision,
        routed_leaf=string(DS.route(s,q)),probed_leaves=string.(pool.leaves),
        candidate_budget=budget,candidates_scored=length(scores),posting_draws=pool.draws,
        vector_blocks=length(s.request_blocks),decoders=length(s.request_decoders),
        vector_file_bytes_read=Int(get(s.metrics,"vector_read_bytes",0.)-read_before),
        block_cache_misses=s.block_cache.misses-misses_before,
        zero_vectors_skipped=zero,returned=length(items),underfill_reason=reason,items=items))
end
