"""Read-only export. Caller owns Store serialization; no query runs expiry."""
function tree_pack(ctx;view=:partitions,time=ctx.store.clock)
    view in (:partitions,:ppca) || fail("invalid_view")
    fresh(ctx,time);s=ctx.store;c=ctx.settings
    estimated=count(n->n.axis==0,values(s.nodes))*s.config.dimension*(view==:ppca ? 128 : 1)
    estimated<=c.server_budget÷4 || fail("pack_too_large",503)
    weights=DS.polya_weights(s;time,kappa=c.kappa)
    nodes=Any[];leaves=Any[]
    lo=fill(-1.,s.config.dimension);hi=fill(1.,s.config.dimension)
    function walk(id)
        n=s.nodes[id]
        common=(id=string(id),parent=n.parent==0 ? nothing : string(n.parent))
        if n.axis==0
            push!(nodes,merge(common,(kind="leaf",)))
            volume=sum(log,hi.-lo)
            isfinite(volume) || fail("invalid_partition",503)
            p=weights.probabilities[id];mass=weights.masses[id]
            leaf=merge((id=string(id),post_count=DS.one(s,"SELECT count FROM leaves WHERE id=?",(id,)).count,
                evidence_mass=mass,probability=p,log_volume=volume,
                log_density=p==0 ? nothing : log(p)-volume,zero_density=p==0,observed=mass>0),NamedTuple())
            if view==:ppca
                st=DS.get_state(s,id);params=LT.parameters(st.sketch)
                ppca=(mean=copy(params.mean),rank=size(params.W,2),
                    W=(shape=[size(params.W)...],order="row-major",values=vec(permutedims(params.W))),
                    D=copy(params.D),variance_floor=1e-7,
                    support_normalization="untruncated_gaussian",reliable=mass>=size(params.W,2)+2)
                leaf=merge(leaf,(ppca=ppca,))
            end
            push!(leaves,leaf)
        else
            push!(nodes,merge(common,(kind="split",axis=n.axis-1,cut=n.cut,left=string(n.left),right=string(n.right))))
            old=hi[n.axis];hi[n.axis]=n.cut;walk(n.left);hi[n.axis]=old
            old=lo[n.axis];lo[n.axis]=n.cut;walk(n.right);lo[n.axis]=old
        end
    end
    walk(1)
    merge(envelope(ctx,time),(view=String(view),embedding=ctx.identity.embedding,
        root=(id="1",lower=fill(-1.,s.config.dimension),upper=fill(1.,s.config.dimension)),
        density=(estimator="polya-piecewise-constant-v1",measure="ambient-coordinate-volume",
            half_life=s.config.half_life,kappa=c.kappa),retention_seconds=s.config.retention_seconds,
        storage_dtype=string(s.config.mode),created_at=time,
        fresh_until=min(time+c.pack_refresh,next_expiry(s)),
        maintenance_lag_seconds=time-s.clock,nodes=nodes,leaves=leaves))
end
function cached_pack(ctx,view,t)
    fresh(ctx,t)
    p=get(ctx.packs,view,nothing)
    if p!==nothing && p.generation==ctx.store.generation && t<p.until
        return p
    end
    payload=tree_pack(ctx;view,time=t)
    body=JSON3.write(payload)
    # Bound detached pack buffers. Two views + replacement buffers fit this share.
    ncodeunits(body)<=ctx.settings.server_budget÷32 || fail("pack_too_large",503)
    compressed=transcode(GzipCompressor,Vector{UInt8}(codeunits(body)))
    p=(body=body,gzip=compressed,etag="\""*bytes2hex(sha256(body))*"\"",
        generation=ctx.store.generation,until=payload.fresh_until)
    ctx.packs[view]=p;p
end
