"""Controlled density/proposal ablation on a fixed tree and fixed post codec.
The simulator's response geometry is supplied separately from inference.
"""
module DensityAblation
using LinearAlgebra, Random, Statistics
using ..LeafLatentTrees
const LT=LeafLatentTrees
export GaussianLeaf, gaussian_leaf, gaussian_logpdf, density_logs, draw_box,
       knn_edges, shortest_from, graph_geometry, attach_distances, density_cost,
       reference_cost, likelihood, information, trial, branch_states, select_candidates,
       distance_matrix, polyaprobs, acceptance

distance_matrix(X,Y)=sqrt.(max.(0.,sum(abs2,Float64.(X);dims=1)'.+sum(abs2,Float64.(Y);dims=1).-2Float64.(X)'*Float64.(Y)))
struct GaussianLeaf
    mean::Vector{Float64}
    W::Matrix{Float64}
    D::Vector{Float64}
    precisionW::Matrix{Float64}
    middle::Matrix{Float64}
    logdet::Float64
    acceptance::Float64
end
function acceptance(p,node,rng;draws=4096)
    if size(p.W,2)==0
        sd=sqrt.(p.D)
        return exp(sum(log(max(LT.cdf((node.hi[j]-p.mean[j])/sd[j])-LT.cdf((node.lo[j]-p.mean[j])/sd[j]),1e-300)) for j in eachindex(sd)))
    end
    X=p.mean.+p.W*randn(rng,size(p.W,2),draws)+sqrt.(p.D).*randn(rng,length(p.mean),draws)
    count(j->all(node.lo .<= view(X,:,j)) && all(view(X,:,j) .< node.hi),1:draws)/draws
end
function gaussian_leaf(s,node,rng;draws=4096)
    p=LT.parameters(s);B=p.W./p.D;K=Matrix{Float64}(I,size(p.W,2),size(p.W,2))+p.W'*B
    C=cholesky(Symmetric(K));mass=acceptance(p,node,rng;draws)
    mass>0 || error("zero Monte Carlo truncation mass; increase integration draws")
    GaussianLeaf(copy(p.mean),p.W,p.D,B,inv(C),sum(log,p.D)+2sum(log,diag(C.L);init=0.),mass)
end
function gaussian_logpdf(p::GaussianLeaf,X)
    R=Float64.(X).-p.mean;v=p.precisionW'*R
    quadratic=vec(sum(R.^2 ./p.D;dims=1)-sum(v.*(p.middle*v);dims=1))
    -.5 .* (length(p.mean)*log(2pi)+p.logdet.+max.(quadratic,0.))
end
function polyaprobs(nodes,counts;kappa=2.)
    mass=copy(counts);p=zeros(length(nodes));p[1]=1.
    for i in length(nodes):-1:1
        n=nodes[i];n.axis!=0 && (mass[i]=mass[n.left]+mass[n.right])
    end
    for (i,n) in enumerate(nodes)
        n.axis==0 && continue
        prior=(n.cut-n.lo[n.axis])/(n.hi[n.axis]-n.lo[n.axis])
        q=(mass[n.left]+kappa*prior)/(mass[i]+kappa)
        p[n.left]=p[i]*q;p[n.right]=p[i]*(1-q);p[i]=0.
    end
    p
end
function density_logs(nodes,probs,models,X)
    ids=[LT.leaf_index(nodes,x) for x in eachcol(X)];logs=zeros(size(X,2))
    for leaf in unique(ids)
        positions=findall(==(leaf),ids);node=nodes[leaf]
        if models===nothing
            logs[positions].=log(probs[leaf])-sum(log,node.hi-node.lo)
        else
            p=models[leaf]
            logs[positions]=log(probs[leaf]).+gaussian_logpdf(p,X[:,positions]).-log(p.acceptance)
        end
    end
    logs
end
function density_cost(logs,d;center=median(logs))
    # p^(-1/d), normalized by a model's own landmark median; bounded leverage.
    clamp.(exp.(clamp.(-(logs.-center)/d,-log(4.),log(4.))),.25,4.),center
end
function reference_cost(X,reference;center=nothing,k=16)
    D=distance_matrix(X,reference)
    radii=[partialsort!(collect(view(D,i,:)),k) for i in axes(D,1)]
    scale=center===nothing ? median(radii) : center
    clamp.(radii/max(scale,1e-12),.25,4.),scale
end
function draw_box(rng,node,p,n;maxdraws=200000)
    if p===nothing
        return node.lo.+(node.hi-node.lo).*rand(rng,length(node.lo),n),n
    end
    values=Vector{Vector{Float64}}();attempts=0
    while length(values)<n && attempts<maxdraws
        x=p.mean+p.W*randn(rng,size(p.W,2))+sqrt.(p.D).*randn(rng,length(p.mean));attempts+=1
        all(node.lo .<= x) && all(x .< node.hi) && push!(values,x)
    end
    length(values)==n || error("truncated sampler exhausted: $(length(values))/$n")
    reduce(hcat,values),attempts
end
function knn_edges(D;k=12)
    n=size(D,1);edges=Set{Tuple{Int,Int}}()
    for i in 1:n
        values=copy(D[i,:]);values[i]=Inf
        for j in partialsortperm(values,1:min(k,n-1));push!(edges,minmax(i,j));end
    end
    # Prim's MST guarantees connectivity without a model-specific topology.
    used=falses(n);nearest=fill(Inf,n);parent=ones(Int,n);nearest[1]=0.
    for _ in 1:n
        u=argmin([used[i] ? Inf : nearest[i] for i in 1:n]);used[u]=true
        u!=1 && push!(edges,minmax(u,parent[u]))
        for v in 1:n
            if !used[v] && D[u,v]<nearest[v];nearest[v]=D[u,v];parent[v]=u;end
        end
    end
    sort!(collect(edges))
end
function shortest_from(n,edges,weights,sources)
    adjacent=[Tuple{Int,Float64}[] for _ in 1:n]
    for ((a,b),w) in zip(edges,weights);push!(adjacent[a],(b,w));push!(adjacent[b],(a,w));end
    out=fill(Inf,n,length(sources))
    for (column,start) in enumerate(sources)
        d=fill(Inf,n);used=falses(n);d[start]=0.
        for _ in 1:n
            best=Inf;u=0
            for i in 1:n
                if !used[i] && d[i]<best;u=i;best=d[i];end
            end
            u==0 && break;used[u]=true
            for (v,w) in adjacent[u];d[v]=min(d[v],best+w);end
        end
        out[:,column]=d
    end
    out
end
function graph_geometry(X,cost,edges,nh)
    D=distance_matrix(X,X);weights=[max(D[a,b],1e-10)*(cost[a]+cost[b])/2 for (a,b) in edges]
    paths=shortest_from(size(X,2),edges,weights,1:nh)
    scale=median([paths[i,j] for i in 1:nh for j in 1:i-1])
    @assert scale>0 && all(isfinite,paths)
    (paths=paths,cost=cost,scale=scale)
end
function attach_distances(Q,vertices,qcost,geometry,anchors)
    D=distance_matrix(Q,vertices);n=size(Q,2);h=size(geometry.paths,2);out=zeros(n,h)
    for j in 1:n,t in 1:h
        out[j,t]=minimum(D[j,a]*(qcost[j]+geometry.cost[a])/2+geometry.paths[a,t] for a in anchors[j])/geometry.scale
    end
    out
end
likelihood(D)=.05 .+ .9 ./ (1 .+ (D/.25).^2)
entropy(p)=p<=0 || p>=1 ? 0. : -p*log2(p)-(1-p)*log2(1-p)
function information(p,L)
    [entropy(dot(p,view(L,i,:)))-dot(p,entropy.(view(L,i,:))) for i in axes(L,1)]
end
function draw(rng,weights)
    total=sum(weights);total>0 || throw(ArgumentError("empty sampling distribution"))
    v=rand(rng)*total
    for i in eachindex(weights);v-=weights[i];v<=0 && return i;end
    lastindex(weights)
end
function branch_states(nodes,H)
    states=[Int[] for _ in nodes];leaves=zeros(Int,size(H,2))
    for j in axes(H,2)
        i=1
        while true
            push!(states[i],j);n=nodes[i]
            if n.axis==0;leaves[j]=i;break;end
            i=H[n.axis,j]<n.cut ? n.left : n.right
        end
    end
    states,leaves
end
function select_candidates(rng,p,nodes,states,hleaves,banks,seen,fallback;npairs=4,per_side=4)
    mass=[sum(p[j] for j in ids;init=0.) for ids in states]
    splits=[i for (i,n) in enumerate(nodes) if n.axis!=0 && mass[n.left]*mass[n.right]>0]
    weights=[mass[nodes[i].left]*mass[nodes[i].right] for i in splits];hypotheses=Int[]
    for _ in 1:npairs
        if isempty(splits)
            append!(hypotheses,(draw(rng,p),draw(rng,p)))
        else
            n=nodes[splits[draw(rng,weights)]]
            for child in (n.left,n.right)
                ids=states[child];push!(hypotheses,ids[draw(rng,p[ids])])
            end
        end
    end
    options=Int[]
    for h in hypotheses
        bank=banks[hleaves[h]]
        for _ in 1:per_side
            q=rand(rng,bank);!seen[q] && push!(options,q)
        end
    end
    unique!(options);used_fallback=isempty(options)
    if used_fallback
        remaining=findall(!,seen)
        isempty(remaining) && error("entire shared candidate corpus exhausted")
        eligible=[q for q in fallback if !seen[q]]
        isempty(eligible) && (eligible=remaining)
        options=unique(rand(rng,eligible,2npairs*per_side))
    end
    options,used_fallback
end
"""Target and feedback source are used only after the selected query is fixed.
Stopping uses a 90% posterior ball under inference geometry; correctness uses
the independently supplied simulator geometry. No truth-based early stopping.
"""
function trial(nodes,states,hleaves,banks,L,Ltruth,modelHH,trueHH,target,uniforms;
        seed=1,budget=80,radius=.10,confidence=.9,history=false,fallback=collect(axes(L,1)))
    h=size(L,2);p=fill(1/h,h);seen=falses(size(L,1));ent=entropy.(L)
    balls=Float64.(modelHH .<= radius);trueballs=trueHH .<= radius
    records=NamedTuple[];stop=0;correct=false;likes=0;stoplikes=0;usedscores=0;fallbacks=0
    selection_seconds=0.;brier=0.;logevidence=0.;center=1;certainty=0.;lastquery=0
    stoperror=0.;stopconfidence=0.
    for step in 1:budget
        rng=MersenneTwister(seed+100003step)
        started=time_ns()
        choices,used_fallback=select_candidates(rng,p,nodes,states,hleaves,banks,seen,fallback)
        best=-Inf;query=first(choices)
        for q in choices
            v=entropy(dot(p,view(L,q,:)))-dot(p,view(ent,q,:))
            if v>best;best=v;query=q;end
        end
        duration=(time_ns()-started)/1e9
        if stop==0;selection_seconds+=duration;usedscores+=length(choices);fallbacks+=used_fallback;end
        predictive=dot(p,view(L,query,:));positive=uniforms[step]<Ltruth[query,target];likes+=positive;seen[query]=true
        if stop==0
            brier+=(predictive-positive)^2;logevidence-=log(max(positive ? predictive : 1-predictive,1e-12))
        end
        for j in 1:h;p[j]*=positive ? L[query,j] : 1-L[query,j];end
        p./=sum(p);masses=balls*p;center=argmax(masses);certainty=masses[center];lastquery=query
        if stop==0 && certainty>=confidence
            stop=step;correct=trueballs[center,target];stoplikes=likes
            stoperror=trueHH[center,target];stopconfidence=certainty
        end
        history && push!(records,(step=step,belief=copy(p),query=query,positive=positive,estimate=center,
            certainty=certainty,error=trueHH[center,target],target=target,candidates=length(choices)))
        stop>0 && !history && break
    end
    used=stop==0 ? budget : stop
    (target=target,stop=stop,correct=correct,false_stop=stop>0 && !correct,exhausted=stop==0,
        used=used,capped_correct=correct ? stop : budget+1,likes=stop==0 ? likes : stoplikes,
        error=stop==0 ? trueHH[center,target] : stoperror,confidence=stop==0 ? certainty : stopconfidence,
        brier=brier/used,logloss=logevidence/used,
        mean_candidates=usedscores/used,fallback_fraction=fallbacks/used,selection_ms=1000selection_seconds/used,
        last_query=lastquery,history=records)
end
end
