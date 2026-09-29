# Bounded example export; include after disk_ablation_common.jl.
function snapshot(s,label)
    leaves=sort([n.id for n in values(s.nodes) if n.axis==0]);out=Any[]
    for leaf in leaves
        common=DS.leaf_keywords(s,leaf;ranking=:common)
        distinct=DS.leaf_keywords(s,leaf;ranking=:distinctive)
        refs=DS.leaf_refs(s,leaf;limit=32)
        examples=Any[]
        for ref in refs[1:min(end,8)]
            post=only(DS.fetch_records(s,[ref.id]))
            k=only(DS.rows(s,"SELECT terms FROM kw_posts WHERE id=?",(ref.id,)))
            ts=[DS.one(s,"SELECT term FROM kw_terms WHERE id=?",(id,)).term for id in DS.deserialize_bytes(k.terms)]
            push!(examples,(id=ref.id,text=post.text,terms=ts))
        end
        push!(out,(common=common,distinctive=distinct,examples=examples))
    end
    (label=label,at=s.clock,generation=s.generation,leaves=out)
end
