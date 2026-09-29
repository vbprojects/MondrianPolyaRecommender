function cursor_encode(ctx,payload)
    text=JSON3.write(payload)
    base64encode(text)*"."*bytes2hex(hmac_sha256(ctx.secret,text))
end
function cursor_decode(ctx,text)
    text isa String && ncodeunits(text)<=2048 || fail("invalid_cursor")
    parts=split(text,'.');length(parts)==2 || fail("invalid_cursor")
    data=try String(base64decode(parts[1])) catch;fail("invalid_cursor");end
    bytes2hex(hmac_sha256(ctx.secret,data))==parts[2] || fail("invalid_cursor")
    try JSON3.read(data,Dict{String,Any}) catch;fail("invalid_cursor");end
end
"""Pure bounded-memory keyset/ranking read of existing combined term aggregates."""
function terms(ctx,input;time=ctx.store.clock)
    fresh(ctx,time);s=ctx.store;c=ctx.settings;q=vector_input(ctx,input)
    DS.keyword_enabled(s) || fail("keywords_disabled",503)
    leaf=DS.route(s,q);limit=integer_option(input,"limit",c.term_limit,1,c.term_page)
    ranking=get(input,"ranking","common");ranking in ("common","distinctive") || fail("invalid_ranking")
    effective=time;after_score=Inf;after_label=""
    if haskey(input,"page_cursor")
        token=cursor_decode(ctx,input["page_cursor"])
        token["store"]==ctx.identity.store_id && token["generation"]==string(s.generation) &&
        token["leaf"]==string(leaf) && token["ranking"]==ranking || fail("snapshot_changed",409)
        effective=Float64(token["at"]);effective<=time || fail("invalid_cursor")
        after_score=Float64(token["score"]);after_label=String(token["term"])
    end
    covered=DS.kw_count(s,leaf,-1,effective);eligible=DS.kw_count(s,leaf,0,effective)
    global_count=ranking=="distinctive" ? DS.kw_count(s,0,-1,effective) : nothing
    total=DS.one(s,"SELECT count FROM leaves WHERE id=?",(leaf,)).count
    best=NamedTuple[];after=0;started=time_ns()
    while true
        (time_ns()-started)/1e9<c.timeout || fail("term_scan_timeout",503)
        page=DS.keyword_page(s,leaf,after;distinctive=ranking=="distinctive")
        isempty(page) && break
        for r in page
            support=r.mass*exp2(-(effective-r.at)/s.keyword_config.half_life)
            score=support
            if ranking=="distinctive"
                gm=r.gm*exp2(-(effective-r.ga)/s.keyword_config.half_life)
                lift=log(((support+1)/(covered.mass+2))/((gm+1)/(global_count.mass+2)))
                score=support*max(0.,lift)
            end
            ranking=="distinctive" && (r.n<3 || score<=0) && continue
            label=String(r.label)
            (score<after_score || (score==after_score && label>after_label)) || continue
            push!(best,(term=label,count=r.n,support,score,
                rate=covered.n==0 ? nothing : r.n/covered.n,
                decayed_rate=covered.mass==0 ? nothing : support/covered.mass))
            sort!(best;by=x->(-x.score,x.term));length(best)>limit+1 && pop!(best)
        end
        after=last(page).term
    end
    more=length(best)>limit;more && pop!(best)
    next=more ? cursor_encode(ctx,(store=ctx.identity.store_id,generation=string(s.generation),leaf=string(leaf),
        ranking,at=effective,score=last(best).score,term=last(best).term)) : nothing
    env=envelope(ctx,effective)
    merge(env,(leaf_id=string(leaf),terms=best,total,covered=covered.n,eligible=eligible.n,
        coverage=total==0 ? 1. : covered.n/total,decayed_covered=covered.mass,
        extractor=s.keyword_config.extractor,fingerprint=s.keyword_config.fingerprint,
        policy=s.keyword_config.policy,half_life=s.keyword_config.half_life,
        ranking,truncated=more,next_cursor=next,
        client_tree_changed=haskey(input,"client_tree_revision") && input["client_tree_revision"]!=env.tree_revision))
end
