# Independent in-memory reference evaluation; never part of measured storage workers.
include("disk_ablation_common.jl")
reports=Dict{String,Any}()
for method in ("hashtags","yake")
    s=DS.open_store(joinpath(ROOT,"data/keyword_experiment",method);budget=32*1024^2)
    expected=Dict{Tuple{Int64,Int64},Tuple{Int64,Float64}}();after=0;posts=0
    while true
        page=DS.rows(s,"SELECT k.*,p.leaf,p.density_at,p.status FROM kw_posts k JOIN posts p ON p.id=k.id WHERE k.id>? ORDER BY k.id LIMIT 256",(after,))
        isempty(page) && break
        for p in page
            @assert p.status==1
            terms=vcat(Int64[-1],p.eligible==1 ? Int64[0] : Int64[],DS.deserialize_bytes(p.terms))
            @assert length(unique(terms))==length(terms)
            w=exp2(-(s.clock-p.density_at)/s.keyword_config.half_life)
            for leaf in (Int64(0),p.leaf),term in terms
                n,m=get(expected,(leaf,term),(Int64(0),0.))
                expected[(leaf,term)]=(n+1,m+w)
            end
            posts+=1
        end
        after=last(page).id
    end
    maxerror=0.;checked=0;leaf=-1;term=typemin(Int64)
    while true
        page=DS.rows(s,"SELECT * FROM kw_counts WHERE (leaf,term)>(?,?) ORDER BY leaf,term LIMIT 256",(leaf,term))
        isempty(page) && break
        for r in page
            n,m=pop!(expected,(r.leaf,r.term))
            @assert r.n==n
            actual=r.mass*exp2(-(s.clock-r.at)/s.keyword_config.half_life)
            @assert isapprox(actual,m;atol=1e-9,rtol=1e-9)
            maxerror=max(maxerror,abs(actual-m));checked+=1
        end
        leaf=last(page).leaf;term=last(page).term
    end
    @assert isempty(expected)
    @assert posts==100000
    @assert DS.audit!(s;check_vectors=false)==100000
    reports[method]=(posts=posts,counters_checked=checked,max_absolute_mass_error=maxerror,keyword_stats=DS.keyword_stats(s),leaves=length([n for n in values(s.nodes) if n.axis==0]))
    DS.close_store(s)
    println("$method: $posts posts, $checked counters agree; max mass error $maxerror");flush(stdout)
end
savejson(joinpath(ROOT,"output/keywords/reference_validation.json"),reports)
