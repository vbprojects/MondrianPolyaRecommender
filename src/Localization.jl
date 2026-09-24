export StaticLandscape, static_landscape, response_probability, posterior_update!,
       query_information, localization_estimate, choose_query, localization_trial

struct StaticLandscape
    tree::Tree
    points::Vector{NTuple{2,Float64}}
    posts::Vector{Int}
    density::Vector{Float64}
    distances::Matrix{Float64}
    likelihood::Matrix{Float64} # query location × hidden location
    likelihood_entropy::Matrix{Float64}
    prior::Vector{Float64}
    candidates::Vector{Int}
    representatives::Vector{Int}
    radius::Float64
    response_scale::Float64
    grid_size::Int
    predecessors::Matrix{Int}
    node_states::Vector{Vector{Int}}
    nearest_post::Vector{Int}
end

"""Bounded inverse-distance likelihood: 95% at zero distance, tends to 5%."""
response_probability(d,scale;noise=.05)=noise+(1-2noise)/(1+d/scale)

function static_landscape(;seed=42,grid_size=21,nposts=4000,alpha=.5,radius_fraction=.10)
    grid_size>=3 && nposts>0 && alpha>=0 && radius_fraction>0 || throw(ArgumentError("invalid landscape settings"))
    rng=MersenneTwister(seed); g=grid_size
    points=[((i-.5)/g,(j-.5)/g) for j in 1:g for i in 1:g]
    counts=zeros(Int,g*g)
    t=Tree(seed=7,half_life=Inf,max_nodes=255,lifetime=20.)
    # Three dense topic islands with background and a thin connecting population.
    for _ in 1:nposts
        u=rand(rng)
        x=if u<.08
            (rand(rng),rand(rng))
        elseif u<.20
            a=rand(rng); (.24+.5a,.28+.43a+.025randn(rng))
        else
            c=u<.49 ? (.24,.28) : u<.77 ? (.74,.71) : (.27,.77)
            (c[1]+.07randn(rng),c[2]+.07randn(rng))
        end
        if !all(v->0<=v<=1,x)
            # Rejection, not clamping: redraw from the uniform background.
            x=(rand(rng),rand(rng))
        end
        observe!(t,x;event_time=0.)
        i=min(g,floor(Int,x[1]*g)+1); j=min(g,floor(Int,x[2]*g)+1)
        counts[i+(j-1)*g]+=1
    end
    dens=[density(t,x) for x in points]
    # Bounded Fermat multiplier. Edge integration is endpoint trapezoidal
    # quadrature on an eight-neighbor grid, not an exact continuous geodesic.
    costs=clamp.(dens.^(-alpha),.2,5.)
    n=length(points); adjacency=[Tuple{Int,Float64}[] for _ in 1:n]
    for j in 1:g,i in 1:g
        a=i+(j-1)*g
        for dj in -1:1,di in -1:1
            di==0 && dj==0 && continue
            1<=i+di<=g && 1<=j+dj<=g || continue
            b=i+di+(j+dj-1)*g
            push!(adjacency[a],(b,hypot(di,dj)/g*(costs[a]+costs[b])/2))
        end
    end
    distances=fill(Inf,n,n); predecessors=zeros(Int,n,n)
    for source in 1:n
        d=fill(Inf,n); used=falses(n); d[source]=0.
        for _ in 1:n
            best=Inf; u=0
            for i in 1:n
                if !used[i] && d[i]<best; best=d[i]; u=i; end
            end
            u==0 && break
            used[u]=true
            for (v,w) in adjacency[u]
                if best+w<d[v]
                    d[v]=best+w; predecessors[source,v]=u
                end
            end
        end
        distances[source,:]=d
    end
    characteristic=sort(vec(copy(distances)))[cld(n*n,2)]
    scale=.12characteristic; radius=radius_fraction*characteristic
    likelihood=response_probability.(distances,scale)
    # Both simulator and inference know the stated dense-region restriction.
    threshold=sort(copy(dens))[ceil(Int,.65n)]
    prior=[dens[i]>=threshold ? dens[i] : 0. for i in 1:n]; prior./=sum(prior)
    candidates=findall(>(0),counts)
    # One post location per occupied tree leaf, selected by maximum observed
    # count without seeing the hidden user or feedback.
    representatives=Dict{Int,Int}()
    for i in candidates
        leaf=last(path(t,points[i]))
        old=get(representatives,leaf,0)
        if old==0 || counts[i]>counts[old]; representatives[leaf]=i; end
    end
    node_states=[Int[] for _ in t.nodes]
    for i in 1:n, node in path(t,points[i]); push!(node_states[node],i); end
    nearest_post=[candidates[argmin([distances[i,c] for c in candidates])] for i in 1:n]
    StaticLandscape(t,points,counts,dens,distances,likelihood,binary_entropy.(likelihood),prior,candidates,sort(collect(values(representatives))),radius,scale,g,predecessors,node_states,nearest_post)
end

function posterior_update!(belief,likelihood,positive)
    for i in eachindex(belief)
        belief[i]*=positive ? likelihood[i] : 1-likelihood[i]
    end
    z=sum(belief); z>0 || throw(ArgumentError("observation has zero probability"))
    belief./=z
end
binary_entropy(p)=p<=0 || p>=1 ? 0. : -p*log(p)-(1-p)*log1p(-p)
function query_information(belief,likelihood)
    p=sum(belief.*likelihood)
    binary_entropy(p)-sum(belief[i]*binary_entropy(likelihood[i]) for i in eachindex(belief))
end
function draw_index(rng,weights)
    u=rand(rng)*sum(weights); s=0.
    for i in eachindex(weights)
        s+=weights[i]; u<s && return i
    end
    lastindex(weights)
end

"""Choose the radius-ball with the most posterior mass. No truth access."""
function localization_estimate(belief,land::StaticLandscape)
    best=0.; center=1
    for j in eachindex(belief)
        p=0.
        for i in eachindex(belief)
            land.distances[j,i]<=land.radius && (p+=belief[i])
        end
        if p>best; best=p; center=j; end
    end
    center,best
end

"""Recover an actual shortest graph path using the shared Dijkstra cache."""
function fermat_path(land,a,b)
    result=[b]
    while last(result)!=a
        previous=land.predecessors[a,last(result)]
        previous>0 || error("missing shortest path")
        push!(result,previous)
        length(result)<=length(land.points) || error("cyclic predecessor path")
    end
    reverse!(result)
end

# Nearest graph vertex to a fraction of accumulated Fermat cost, not length.
function path_fraction(land,route,fraction)
    0<=fraction<=1 || throw(ArgumentError("invalid path fraction"))
    a=first(route); total=land.distances[a,last(route)]
    route[argmin([abs(land.distances[a,i]-fraction*total) for i in route])]
end

function path_proposals(rng,belief,land;midpoint_only=false,pairs=4)
    masses=[sum(belief[i] for i in ids;init=0.) for ids in land.node_states]
    splits=Int[]; weights=Float64[]
    for (i,n) in enumerate(land.tree.nodes)
        if n.axis!=0 && masses[n.left]>0 && masses[n.right]>0
            push!(splits,i); push!(weights,masses[n.left]*masses[n.right])
        end
    end
    candidates=Int[]; routes=Vector{Int}[]; edges=0
    fractions=midpoint_only ? (.5,) : (0.,.2,.35,.5,.65,.8,1.)
    if !isempty(splits) && sum(weights)>0
        for _ in 1:pairs
            n=land.tree.nodes[splits[draw_index(rng,weights)]]
            left=land.node_states[n.left]; right=land.node_states[n.right]
            a=left[draw_index(rng,[belief[i] for i in left])]
            b=right[draw_index(rng,[belief[i] for i in right])]
            route=fermat_path(land,a,b); push!(routes,route); edges+=length(route)-1
            for f in fractions
                push!(candidates,land.nearest_post[path_fraction(land,route,f)])
            end
        end
    end
    # If all posterior mass belongs to one tree leaf, sample two hypotheses
    # within that leaf instead. A point posterior falls back to its nearest post.
    if isempty(candidates)
        a=draw_index(rng,belief); b=draw_index(rng,belief)
        route=fermat_path(land,a,b); push!(routes,route); edges+=length(route)-1
        for f in fractions; push!(candidates,land.nearest_post[path_fraction(land,route,f)]); end
    end
    (candidates=unique(candidates),routes=routes,edges=edges)
end

function query_choice(rng,belief,land::StaticLandscape,policy::Symbol)
    if policy==:random
        q=draw_index(rng,land.posts)
        return (query=q,evaluated=0,routes=Vector{Int}[],proposals=[q],edges=0)
    end
    policy in (:greedy,:information,:tree_information,:tree_path_information,:tree_midpoint) || throw(ArgumentError("unknown query policy"))
    proposal=policy in (:tree_path_information,:tree_midpoint) ? path_proposals(rng,belief,land;midpoint_only=policy==:tree_midpoint) : nothing
    options=isnothing(proposal) ? (policy==:tree_information ? land.representatives : land.candidates) : proposal.candidates
    best=-Inf; selected=first(options)
    for q in options
        l=view(land.likelihood,q,:)
        p=0.; h=0.
        for i in eachindex(belief)
            p+=belief[i]*l[i]
            policy!=:greedy && (h+=belief[i]*land.likelihood_entropy[q,i])
        end
        score=policy==:greedy ? p : binary_entropy(p)-h
        if score>best; best=score; selected=q; end
    end
    (query=selected,evaluated=length(options),routes=isnothing(proposal) ? Vector{Int}[] : proposal.routes,
        proposals=isnothing(proposal) ? Int[] : options,edges=isnothing(proposal) ? 0 : proposal.edges)
end
choose_query(rng,belief,land::StaticLandscape,policy::Symbol)=query_choice(rng,belief,land,policy).query

"""Static hidden state, binary observations, calibrated confidence stopping.
Runs to budget for evaluation/animation even after first confidence stop.
The reported stopping time uses posterior confidence only; truth is used solely
for evaluation, never query selection or stopping.
"""
function localization_trial(land,policy;seed=1,budget=80,confidence=.9,target=nothing,history=false)
    budget>0 && 0<confidence<1 || throw(ArgumentError("invalid trial budget/confidence"))
    rng=MersenneTwister(seed); proposal_rng=MersenneTwister(seed+1_000_000)
    evaluated=0; path_edges=0; selection_seconds=0.
    truth=isnothing(target) ? draw_index(rng,land.prior) : target
    belief=copy(land.prior); snapshots=NamedTuple[]
    stop=0; correct=false; likes=0; stoplikes=0
    center,certainty=localization_estimate(belief,land)
    # Require at least one interaction for the reported experiment.
    for step in 0:budget
        query=0; positive=false; routes=Vector{Int}[]; proposals=Int[]; query_evaluations=0
        if step>0
            started=time_ns()
            choice=query_choice(policy in (:tree_path_information,:tree_midpoint) ? proposal_rng : rng,belief,land,policy)
            duration=(time_ns()-started)/1e9
            query=choice.query; routes=choice.routes; proposals=choice.proposals; query_evaluations=choice.evaluated
            if stop==0
                evaluated+=choice.evaluated; path_edges+=choice.edges; selection_seconds+=duration
            end
            positive=rand(rng)<land.likelihood[query,truth]
            likes+=positive
            posterior_update!(belief,view(land.likelihood,query,:),positive)
            center,certainty=localization_estimate(belief,land)
            if stop==0 && certainty>=confidence
                stop=step; correct=land.distances[center,truth]<=land.radius; stoplikes=likes
            end
        end
        if history
            push!(snapshots,(step=step,belief=copy(belief),estimate=center,confidence=certainty,
                error=land.distances[center,truth],query=query,positive=Int(positive),routes=routes,proposals=proposals,evaluated=query_evaluations))
        end
    end
    (target=truth,stop=stop,correct=correct,evaluated=evaluated,path_edges=path_edges,selection_seconds=selection_seconds,used=stop==0 ? budget : stop,
        likes=stop==0 ? likes : stoplikes,final_error=land.distances[center,truth],history=snapshots)
end
