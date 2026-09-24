using MondrianPolyaToy, LinearAlgebra, Random, Serialization, Printf
include("load_bluesky.jl")
const MP= MondrianPolyaToy
mutable struct LocalPPCA
    mass::Float64
    mean::Vector{Float64}
    basis::Matrix{Float64}
    scatter::Vector{Float64}
    trace::Float64
end
LocalPPCA(x)=LocalPPCA(0.,Float64.(x),zeros(length(x),0),Float64[],0.)
# Incremental rank-truncated scatter eigensystem. Discarded trace contributes
# isotropic noise. This is approximate online PPCA, not exact online EM.
function update!(c,x,w,rank)
    w<=0 && return
    total=c.mass+w; delta=Float64.(x).-c.mean
    v=sqrt(c.mass*w/total).*delta
    c.mean .+= (w/total).*delta; c.trace+=sum(abs2,v); c.mass=total
    q=c.basis'*v; residual=v-c.basis*q; normres=norm(residual)
    if normres>1e-12
        U=hcat(c.basis,residual/normres); z=vcat(q,normres)
        small=diagm(vcat(c.scatter,0.))+z*z'
    elseif isempty(q)
        return
    else
        U=c.basis; small=diagm(c.scatter)+q*q'
    end
    e=eigen(Symmetric(small)); ids=length(e.values):-1:max(1,length(e.values)-rank+1)
    c.scatter=max.(e.values[ids],0.); c.basis=U*e.vectors[:,ids]
end
function geometry(c)
    d=length(c.mean); r=length(c.scatter)
    noise=max(1e-6,(c.trace-sum(c.scatter))/max(c.mass*(d-r),1e-12))
    variances=max.(c.scatter/max(c.mass,1e-12),noise)
    noise,variances
end
function logscore(c,x)
    noise,vars=geometry(c); delta=x-c.mean; projected=c.basis'*delta
    distance=max(0.,sum(abs2,delta)-sum(abs2,projected))/noise+sum(projected.^2 ./ vars)
    log(max(c.mass,1e-12))-.5*((length(x)-length(vars))*log(noise)+sum(log,vars;init=0.)+distance)
end
function fit_local(t,X;rank=4,cap=2,split_count=128,radius=.35)
    mixtures=[LocalPPCA[] for _ in t.nodes]; active=falses(length(t.nodes))
    for row in axes(X,2)
        x=view(X,:,row); i=embedding_leaf(t,x,active); cs=mixtures[i]
        if isempty(cs) || (length(cs)<cap && minimum(sum(abs2,x-c.mean) for c in cs)>radius^2 && sum(c.mass for c in cs)>=16)
            c=LocalPPCA(x);update!(c,x,1.,rank);push!(cs,c)
        else
            scores=[logscore(c,x) for c in cs]; weights=exp.(scores.-maximum(scores)); weights ./= sum(weights)
            for (c,w) in zip(cs,weights);update!(c,x,w,rank);end
        end
        if t.nodes[i].axis!=0 && sum(c.mass for c in cs)>=split_count
            active[i]=true # historical mixture remains at owner, new children empty
        end
        row%20000==0 && println("MPPCA ingested ",row)
    end
    (mixtures=mixtures,active=active)
end
function leafids(t)
    findall(n->n.axis==0,t.nodes)
end
function publish(t,model;samples=256,seed=88)
    counts=zeros(length(t.nodes)); rng=MersenneTwister(seed)
    for (owner,cs) in enumerate(model.mixtures),c in cs
        n=t.nodes[owner]; noise,vars=geometry(c)
        # Monte Carlo allocation, conditioned only approximately: Gaussian draws
        # are clipped to owning bounds. This is a censored Gaussian approximation.
        for _ in 1:samples
            x=c.mean+sqrt(noise)*randn(rng,t.dimensions)
            if !isempty(vars); x+=c.basis*(sqrt.(max.(vars.-noise,0.)).*randn(rng,length(vars)));end
            x=clamp.(x,n.lo,prevfloat.(n.hi))
            i=1
            while true
                counts[i]+=c.mass/samples
                node=t.nodes[i]; node.axis==0 && break
                i=x[node.axis]<node.cut ? node.left : node.right
            end
        end
    end
    p=zeros(length(counts));p[1]=1.
    for (i,n) in enumerate(t.nodes)
        n.axis==0 && continue
        q=(n.cut-n.lo[n.axis])/(n.hi[n.axis]-n.lo[n.axis])
        q=(t.kappa*q+counts[n.left])/(t.kappa+counts[n.left]+counts[n.right])
        p[n.left]=p[i]*q;p[n.right]=p[i]*(1-q)
    end
    counts,p
end
function main()
    BLAS.set_num_threads(1)
    root=normpath(joinpath(@__DIR__,"..")); out=joinpath(root,"output/bluesky_mppca");mkpath(out)
    X=load_bluesky_embeddings(joinpath(root,"data/bluesky_100000"))
    saved=deserialize(joinpath(root,"output/bluesky_tree/models.jls"));t=saved.partition
    fit_local(t,X[:,1:256]) # compilation warmup
    timed=@timed fit_local(t,X); model=timed.value
    allocation=@timed publish(t,model);counts,p=allocation.value
    cs=[c for bucket in model.mixtures for c in bucket]
    total=sum(c.mass for c in cs)
    @assert isapprox(total,size(X,2);rtol=1e-10)
    @assert isapprox(counts[1],total;rtol=1e-10)
    @assert isapprox(sum(p[leafids(t)]),1.;atol=1e-10)
    reference=embedding_probabilities(t)
    l1=sum(abs.(p[leafids(t)]-reference[leafids(t)]))
    # Validate conserved first moment and total second moment, despite truncated basis.
    exactmean=vec(sum(Float64.(X);dims=2))
    @assert isapprox(sum(c.mass*c.mean for c in cs),exactmean;atol=1e-6)
    exactsecond=sum(abs2,Float64.(X))
    @assert isapprox(sum(c.trace+c.mass*sum(abs2,c.mean) for c in cs),exactsecond;rtol=1e-8)
    @assert all(c->all(isfinite,c.mean)&&all(isfinite,c.scatter)&&length(c.scatter)<=4,cs)
    @assert all(length(bucket)<=2 for bucket in model.mixtures)
    # Independent allocation seed assesses sensitivity to Monte Carlo, no retraining.
    _,p2=publish(t,model;seed=89)
    mc=sum(abs.(p[leafids(t)]-p2[leafids(t)]))
    serialize(joinpath(out,"models.jls"),(partition=t,model=model,counts=counts,probabilities=p,
        rank=4,cap=2,radius=.35,split_count=128,samples=256,input_manifest=saved.input_manifest))
    lines=["Bluesky local online MPPCA; 100000 original 256D embeddings; static, no decay.",
        "Shared scaffold from output/bluesky_tree/models.jls; rank 4, max 2 components per owner, split count 128, novelty radius .35, minimum seeding mass 16.",
        "Historical mixtures retained at ancestors. Approximate online soft responsibilities and incremental truncated scatter updates; lost scatter trace supplies isotropic noise.",
        "No per-vector codes stored. This is summary compression, not a versioned vector codec.",
        "Published tree counts: 256 fixed-seed Gaussian samples/component, clipped to owning bounds (censored Gaussian approximation, not exact conditional Gaussian integration).",
        "method,model_bytes,mean_L1_vs_reference,fit_seconds",
        "reference,$(Base.summarysize(t)),0,$(.572275287)"]
    for name in ("residual","microclusters")
        m=saved.models[name];prob=embedding_probabilities(t,m);err=0.
        for i in leafids(t)
            n=t.nodes[i];j=embedding_leaf(t,(n.lo+n.hi)/2,m.active)
            err+=abs(reference[i]-prob[j]*exp(MP.logvolume(n)-MP.logvolume(t.nodes[j])))
        end
        push!(lines,"$name,$(Base.summarysize(m)),$err,previous_run")
    end
    push!(lines,"local_mppca,$(Base.summarysize((t,model))),$l1,$(timed.time)",
        "Components: $(length(cs)); owning nodes: $(count(!isempty,model.mixtures)); mass: $total",
        "Allocation seconds: $(allocation.time); independent allocation seed L1 difference: $mc",
        "Normalization error: $(abs(sum(p[leafids(t)])-1))",
        "Memory excludes original vectors/text, runtime working memory and export. Budgets are not matched; same-data reference comparison is not held-out recommendation quality.",
        "A 256D Cartesian histogram is omitted: 16 bins per coordinate would require 16^256 cells.",
        "Global count, first moments, total second moment, rank/component caps and finite parameters validated.")
    open(joinpath(out,"leaves.csv"),"w") do io
        println(io,"node,probability,reference_probability,approximate_count,log_density")
        for i in leafids(t)
            println(io,join((i,p[i],reference[i],counts[i],log(p[i])-MP.logvolume(t.nodes[i])),','))
        end
    end
    write(joinpath(out,"summary.txt"),join(lines,'\n')*"\n")
    open(joinpath(out,"components.csv"),"w") do io
        println(io,"owner,component,mass,rank,noise,retained_scatter_fraction")
        for (i,bucket) in enumerate(model.mixtures),(j,c) in enumerate(bucket)
            println(io,join((i,j,c.mass,length(c.scatter),geometry(c)[1],sum(c.scatter)/max(c.trace,1e-12)),','))
        end
    end
    println(join(lines,'\n'))
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
