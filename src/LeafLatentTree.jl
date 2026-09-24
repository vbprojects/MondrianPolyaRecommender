module LeafLatentTrees
using LinearAlgebra, Random
export Sketch, LeafNode, fit_leaf_tree, parameters, conditional_children, leaf_index, sample_leaf, reconstruction, decay!, leaf_probabilities, merge_sketches, merge_tree!

mutable struct Sketch
    kind::Symbol
    rank::Int
    mass::Float64
    mean::Vector{Float64}
    diagonal::Vector{Float64} # central scatter diagonal
    U::Matrix{Float64}
    eigenvalues::Vector{Float64} # rank-truncated central scatter
end
Sketch(kind,r,d)=Sketch(kind,r,0.,zeros(d),zeros(d),zeros(d,0),Float64[])
"""Scale unnormalized statistics; elapsed and half_life use the same time unit."""
function decay!(s::Sketch, elapsed::Real, half_life::Real)
    isfinite(elapsed) && elapsed>=0 || throw(ArgumentError("elapsed must be finite and nonnegative"))
    half_life>0 && !isnan(half_life) || throw(ArgumentError("half_life must be positive"))
    factor=exp2(-elapsed/half_life)
    s.mass*=factor;s.diagonal .*= factor;s.eigenvalues .*= factor
    s
end
"""Read-only Pólya leaf probabilities at a common time (internal entries are zero).
The geometric branch prior kappa does not decay. No cached subtree masses are used.
"""
function leaf_probabilities(fitted; time=fitted.time, config=1, kappa=2.)
    isfinite(time) && time>=fitted.time || throw(ArgumentError("query time precedes fitted stream"))
    isfinite(kappa) && kappa>=0 || throw(ArgumentError("kappa must be finite and nonnegative"))
    nodes=fitted.nodes;mass=zeros(length(nodes));prob=zeros(length(nodes));prob[1]=1.
    for (i,n) in enumerate(nodes)
        n.axis==0 && (mass[i]=n.states[config].mass*exp2(-(time-fitted.last_times[i])/fitted.half_life))
    end
    for i in length(nodes):-1:1
        n=nodes[i];n.axis!=0 && (mass[i]=mass[n.left]+mass[n.right])
    end
    for (i,n) in enumerate(nodes)
        n.axis==0 && continue
        prior=(n.cut-n.lo[n.axis])/(n.hi[n.axis]-n.lo[n.axis])
        q=mass[i]+kappa>0 ? (mass[n.left]+kappa*prior)/(mass[i]+kappa) : prior
        prob[n.left]=prob[i]*q;prob[n.right]=prob[i]*(1-q);prob[i]=0.
    end
    (probabilities=prob,masses=mass)
end
function update!(s::Sketch,x)
    total=s.mass+1;delta=x-s.mean;v=sqrt(s.mass/total)*delta
    s.mean .+= delta/total;s.diagonal .+= v.^2;s.mass=total
    s.rank==0 && return
    a=s.U'*v;res=v-s.U*a;n=norm(res)
    if n>1e-12
        Q=hcat(s.U,res/n);z=vcat(a,n);C=diagm(vcat(s.eigenvalues,0.))+z*z'
    elseif isempty(a);return
    else;Q=s.U;C=diagm(s.eigenvalues)+a*a';end
    e=eigen(Symmetric(C));ids=length(e.values):-1:max(1,length(e.values)-s.rank+1)
    s.U=Q*e.vectors[:,ids];s.eigenvalues=max.(0.,e.values[ids])
end
"""Return covariance as W*W' + Diagonal(D), never a dense covariance."""
function parameters(s::Sketch;floor=1e-7)
    v=max.(s.diagonal/max(s.mass,1e-12),floor);d=length(v)
    s.kind==:diagonal && return (mean=s.mean,W=zeros(d,0),D=v)
    lam=s.eigenvalues/max(s.mass,1e-12);r=length(lam)
    noise=max(floor,(sum(v)-sum(lam))/max(1,d-r))
    W=s.U .* reshape(sqrt.(max.(lam.-noise,0.)),1,:)
    if s.kind==:ppca
        D=fill(noise,d)
    else
        # Keep the exact stored covariance diagonal, shrink factors globally
        # if necessary to leave a positive diagonal remainder.
        factor_diag=vec(sum(abs2,W;dims=2))
        scale=minimum(min.(1.,sqrt.(max.(v.-floor,0.)./max.(factor_diag,1e-30))))
        W .*= scale
        D=max.(floor,v-vec(sum(abs2,W;dims=2)))
    end
    (mean=s.mean,W=W,D=D)
end
cdf(x)=.5ccall((:erfc,Base.Math.libm),Float64,(Float64,),-x/sqrt(2.))
function compress_conditional(s,mass,mean,diag,op,seedbasis)
    d=length(mean);r=min(s.rank,d)
    r==0 && return Sketch(s.kind,0,mass,mean,mass.*diag,zeros(d,0),Float64[])
    # Child reservoir supplies an initial orientation, not extra moment mass.
    Q=Matrix(qr(seedbasis).Q)[:,1:r]
    for _ in 1:8;Q=Matrix(qr(op(Q)).Q)[:,1:r];end
    e=eigen(Symmetric(Q'*op(Q)));order=sortperm(e.values;rev=true)
    U=Q*e.vectors[:,order];lam=max.(e.values[order],0.)
    Sketch(s.kind,r,mass,mean,mass.*diag,U,mass.*lam)
end
"""Moment targets for one Gaussian halfspace, followed by low-rank compression.
Repeated leaf-box constraints are NOT integrated here: the parent is an
untruncated Gaussian approximation. Rejection sampling enforces support later.
"""
function conditional_children(s,axis,cut,leftseed,rightseed)
    p=parameters(s);W=p.W;D=p.D;mean=p.mean
    col=W*W[axis,:];col[axis]+=D[axis];variance=col[axis];sd=sqrt(variance)
    alpha=(cut-mean[axis])/sd;q=clamp(cdf(alpha),1e-12,1-1e-12)
    phi=exp(-alpha^2/2)/sqrt(2pi);ll=phi/q;lr=phi/(1-q)
    leftmean=mean-col*(ll/sd);rightmean=mean+col*(lr/sd)
    dl=clamp(ll*(ll+alpha),0.,1.);dr=clamp(lr*(lr-alpha),0.,1.)
    parentdiag=vec(sum(abs2,W;dims=2))+D
    makeop(delta)=Q-> W*(W'*Q)+D.*Q-(delta/variance)*col*(col'*Q)
    leftdiag=max.(parentdiag-dl/variance*col.^2,0.)
    rightdiag=max.(parentdiag-dr/variance*col.^2,0.)
    l=compress_conditional(s,s.mass*q,leftmean,leftdiag,makeop(dl),leftseed)
    r=compress_conditional(s,s.mass*(1-q),rightmean,rightdiag,makeop(dr),rightseed)
    momentmean=norm(q*l.mean+(1-q)*r.mean-mean)
    # Include between-child mean contribution in the covariance accounting.
    exactdiag=q*(leftdiag+(leftmean-mean).^2)+(1-q)*(rightdiag+(rightmean-mean).^2)
    lp=parameters(l);rp=parameters(r)
    approxdiag=q*(vec(sum(abs2,lp.W;dims=2))+lp.D+(leftmean-mean).^2)+(1-q)*(vec(sum(abs2,rp.W;dims=2))+rp.D+(rightmean-mean).^2)
    l,r,(mass_error=abs(l.mass+r.mass-s.mass),mean_error=momentmean,
        conditional_diagonal_error=norm(exactdiag-parentdiag),compressed_diagonal_error=norm(approxdiag-parentdiag))
end
"""Merge stored central-scatter sketches without forming a dense covariance.
Mass, first moment and scatter diagonal are conserved; the sum of the two
retained low-rank scatters plus between-mean scatter is truncated back to rank r.
Discarded correlations cannot be recovered from either child.
"""
function merge_sketches(a::Sketch,b::Sketch)
    a.kind==b.kind && a.rank==b.rank && length(a.mean)==length(b.mean) ||
        throw(ArgumentError("incompatible sibling sketches"))
    total=a.mass+b.mass;d=length(a.mean)
    total==0 && return Sketch(a.kind,a.rank,d)
    delta=b.mean-a.mean;ratio=b.mass/total;between=a.mass*ratio
    mean=a.mean+ratio*delta
    diagonal=a.diagonal+b.diagonal+between*delta.^2
    F=hcat(a.U.*reshape(sqrt.(max.(a.eigenvalues,0.)),1,:),
           b.U.*reshape(sqrt.(max.(b.eigenvalues,0.)),1,:),sqrt(between)*delta)
    compress_merged(a,total,mean,diagonal,F)
end
function compress_merged(a,total,mean,diagonal,F)
    r=min(a.rank,length(mean),size(F,2))
    r==0 && return Sketch(a.kind,a.rank,total,mean,diagonal,zeros(length(mean),0),Float64[])
    e=svd(F;full=false)
    Sketch(a.kind,a.rank,total,mean,diagonal,e.U[:,1:r],e.S[1:r].^2)
end
mutable struct LeafNode
    lo::Vector{Float64}
    hi::Vector{Float64}
    axis::Int
    cut::Float64
    left::Int
    right::Int
    depth::Int
    arrivals::Int
    seen::Int
    reservoir::Matrix{Float32}
    states::Vector{Sketch}
end
function leaf_index(nodes,x)
    i=1
    while nodes[i].axis!=0
        n=nodes[i];i=x[n.axis]<n.cut ? n.left : n.right
    end
    i
end
function reservoir_seed(R,r,rng)
    d=size(R,1)
    if size(R,2)>1
        centered=Float64.(R).-sum(R;dims=2)/size(R,2)
        # Retained examples propose a child-specific basis without full covariance.
        Q=centered*randn(rng,size(R,2),r)
        return Q+1e-10randn(rng,d,r)
    end
    randn(rng,d,r)
end
# Operation boundaries shared by the reference and portable-kernel implementations.
function advance_state!(s,x,elapsed,half_life)
    decay!(s,elapsed,half_life);update!(s,x)
end
function cut_proposal(R,rng)
    d=size(R,1);widths=vec(maximum(R;dims=2)-minimum(R;dims=2));rate=sum(widths)
    rate>0 || return nothing
    u=rand(rng)*rate;axis=d
    for j in 1:d;u-=widths[j];if u<0;axis=j;break;end;end
    values=sort(vec(R[axis,:]));a=values[max(1,length(values)÷4)];b=values[min(length(values),3length(values)÷4)]
    a<b || return nothing
    cut=Float64(a+rand(rng)*(b-a));mask=R[axis,:].<cut
    any(mask) && !all(mask) ? (axis,cut,mask) : nothing
end
# Bring all active leaves to one clock, merge bottom-up, then remove dead slots.
# Nodes remain parent-before-child after compaction, as required by query kernels.
function merge_siblings!(nodes,last_times,effective_arrivals,reservoir_rows;
        time,half_life,merge_below,reservoir_cap,operations=nothing)
    isfinite(time) && all(t->t<=time,last_times) || throw(ArgumentError("merge time precedes tree state"))
    isfinite(merge_below) && merge_below>=0 || throw(ArgumentError("invalid merge threshold"))
    for (i,n) in enumerate(nodes)
        n.axis==0 || continue
        elapsed=time-last_times[i]
        for s in n.states
            operations===nothing ? decay!(s,elapsed,half_life) : operations.decay(s,elapsed,half_life)
        end
        effective_arrivals[i]*=exp2(-elapsed/half_life);last_times[i]=time
    end
    removed=falses(length(nodes));merges=0
    for i in length(nodes):-1:1
        n=nodes[i];n.axis==0 && continue
        l=nodes[n.left];r=nodes[n.right]
        l.axis==0 && r.axis==0 || continue
        # Threshold is combined observed weight, not density or geometric prior.
        l.states[1].mass+r.states[1].mass<=merge_below || continue
        states=[operations===nothing || !hasproperty(operations,:merge) ? merge_sketches(a,b) : operations.merge(a,b)
                for (a,b) in zip(l.states,r.states)]
        ids=vcat(reservoir_rows[n.left],reservoir_rows[n.right])
        R=hcat(l.reservoir,r.reservoir)
        # Preserve chronological FIFO order across siblings, including ties in time.
        order=sortperm(ids);keep=order[max(1,length(order)-reservoir_cap+1):end]
        n.reservoir=R[:,keep];reservoir_rows[i]=ids[keep];n.states=states
        removed[n.left]=true;removed[n.right]=true
        n.axis=0;n.cut=0.;n.left=0;n.right=0;n.arrivals=0;n.seen=length(keep)
        effective_arrivals[i]=0.;last_times[i]=time;merges+=1
    end
    if merges>0
        keep=findall(!,removed);mapping=zeros(Int,length(nodes));mapping[keep]=eachindex(keep)
        for i in keep
            n=nodes[i];n.axis==0 && continue
            n.left=mapping[n.left];n.right=mapping[n.right]
        end
        for values in (nodes,last_times,effective_arrivals,reservoir_rows)
            retained=values[keep];empty!(values);append!(values,retained)
        end
    end
    merges
end
"""Materialize decay and reclaim weak sibling leaves, including during idle time.
Mutates the model arrays; assign the returned fit to keep its clock/counters current.
Node indices may change. Does not delete the source post archive.
"""
function merge_tree!(f;time=f.time,merge_below=f.merge_below,operations=nothing)
    isfinite(f.half_life) || throw(ArgumentError("merging requires a finite half-life and chronological reservoirs"))
    isfinite(time) && time>=f.time || throw(ArgumentError("merge time precedes fitted stream"))
    0<=merge_below<f.split_after || throw(ArgumentError("merge threshold must be below split threshold"))
    count=merge_siblings!(f.nodes,f.last_times,f.effective_arrivals,f.reservoir_rows;
        time,half_life=f.half_life,merge_below,reservoir_cap=f.reservoir_cap,operations)
    merge(f,(time=Float64(time),last_merge_time=Float64(time),merge_count=f.merge_count+count))
end
function fit_leaf_tree(X;max_leaves=64,split_after=512,reservoir_cap=64,seed=918,
        configs=[(:diagonal,0),(:ppca,4),(:ppca,8),(:lowrank_diagonal,4),(:lowrank_diagonal,8)],
        max_depth=16, observer=nothing, observe_every=10000, on_split=nothing,
        timestamps=nothing, half_life=Inf, on_checkpoint=nothing, operations=nothing,
        merge_below=0., merge_interval=half_life, on_merge=nothing,
        _initial=nothing)
    d,N=size(X);all(isfinite,X)&&all(x->abs(x)<=1,X) || error("expected normalized finite embeddings")
    N>0 || throw(ArgumentError("empty stream"))
    half_life>0 && !isnan(half_life) || throw(ArgumentError("half_life must be positive"))
    isfinite(half_life) && timestamps===nothing && throw(ArgumentError("decay requires timestamps"))
    ts=timestamps===nothing ? zeros(N) : Float64.(timestamps)
    length(ts)==N && all(isfinite,ts) && issorted(ts) || throw(ArgumentError("timestamps must be finite, chronological, and match columns"))
    isfinite(merge_below) && 0<=merge_below<split_after || throw(ArgumentError("merge_below must be nonnegative and below split_after"))
    merge_below>0 && !(isfinite(half_life) && isfinite(merge_interval) && merge_interval>0) &&
        throw(ArgumentError("merging requires a finite half-life and positive finite merge_interval"))
    rng=_initial===nothing ? MersenneTwister(seed) : _initial.rng
    basisrng=_initial===nothing ? MersenneTwister(seed+1) : _initial.basisrng
    root=LeafNode(fill(-1.,d),fill(1.,d),0,0.,0,0,0,0,0,zeros(Float32,d,0),[Sketch(k,r,d) for (k,r) in configs])
    nodes=[root];errors=[NamedTuple[] for _ in configs];times=zeros(length(configs));split_times=zeros(length(configs));leaves=1
    last_times=[ts[1]];effective_arrivals=[0.];reservoir_rows=[Int[]]
    processed=0;last_merge_time=ts[1];merge_count=0;split_count=0
    if _initial!==nothing
        ts[1]>=_initial.time || throw(ArgumentError("continuation precedes fitted stream"))
        length(_initial.nodes[1].lo)==d && _initial.configs==configs &&
            _initial.half_life==half_life && _initial.reservoir_cap==reservoir_cap &&
            _initial.split_after==split_after || throw(ArgumentError("continuation must preserve dimension, configs, half-life, reservoir capacity and split threshold"))
        nodes=_initial.nodes;last_times=_initial.last_times;effective_arrivals=_initial.effective_arrivals
        reservoir_rows=_initial.reservoir_rows;processed=_initial.processed
        last_merge_time=_initial.last_merge_time;merge_count=_initial.merge_count;split_count=_initial.split_count
        leaves=count(n->n.axis==0,nodes)
    end
    for row in 1:N
        if merge_below>0 && ts[row]-last_merge_time>=merge_interval
            nmerged=merge_siblings!(nodes,last_times,effective_arrivals,reservoir_rows;
                time=ts[row],half_life,merge_below,reservoir_cap,operations)
            leaves-=nmerged;merge_count+=nmerged;last_merge_time=ts[row]
            nmerged>0 && on_merge!==nothing && on_merge(processed+row,nmerged,nodes)
        end
        x=view(X,:,row);i=operations===nothing ? leaf_index(nodes,x) : operations.route(nodes,x);n=nodes[i];n.arrivals+=1;n.seen+=1
        elapsed=ts[row]-last_times[i]
        effective_arrivals[i]=effective_arrivals[i]*exp2(-elapsed/half_life)+1
        last_times[i]=ts[row]
        if operations===nothing || operations.advance!==nothing
            for (k,s) in enumerate(n.states)
                times[k]+=@elapsed (operations===nothing ? advance_state!(s,x,elapsed,half_life) : operations.advance(s,x,elapsed,half_life))
            end
        end
        if isfinite(half_life)
            push!(reservoir_rows[i],processed+row)
            length(reservoir_rows[i])>reservoir_cap && popfirst!(reservoir_rows[i])
        end
        if operations!==nothing
            n.reservoir=operations.reservoir(n,x,half_life,rng,reservoir_cap)
        else
        if size(n.reservoir,2)<reservoir_cap;n.reservoir=hcat(n.reservoir,Float32.(x))
        else
            if isfinite(half_life)
                # Recent FIFO window for split geometry; not a weighted reservoir.
                n.reservoir[:,1:end-1].=n.reservoir[:,2:end]
                n.reservoir[:,end].=x
            else
                j=rand(rng,1:n.seen);j<=reservoir_cap && (n.reservoir[:,j].=x)
            end
        end
        end
        if leaves<max_leaves && effective_arrivals[i]>=split_after && n.depth<max_depth && size(n.reservoir,2)>=16
            proposal=operations===nothing ? cut_proposal(n.reservoir,rng) : operations.proposal(n.reservoir,rng)
            if proposal!==nothing
            axis,cut,mask=proposal
            RL=n.reservoir[:,mask];RR=n.reservoir[:,.!mask]
            seedL=reservoir_seed(RL,maximum(last,configs),basisrng);seedR=reservoir_seed(RR,maximum(last,configs),basisrng)
            ls=Sketch[];rs=Sketch[]
            for (k,s) in enumerate(n.states)
                result=@timed (operations===nothing ? conditional_children(s,axis,cut,seedL,seedR) : operations.children(s,axis,cut,seedL,seedR))
                split_times[k]+=result.time;l,r,error=result.value;push!(ls,l);push!(rs,r);push!(errors[k],error)
            end
            lh=copy(n.hi);lh[axis]=cut;rl=copy(n.lo);rl[axis]=cut
            n.axis=axis;n.cut=cut;n.left=length(nodes)+1;n.right=length(nodes)+2
            push!(nodes,LeafNode(copy(n.lo),lh,0,0.,0,0,n.depth+1,0,size(RL,2),RL,ls))
            push!(nodes,LeafNode(rl,copy(n.hi),0,0.,0,0,n.depth+1,0,size(RR,2),RR,rs))
            append!(last_times,(ts[row],ts[row]));append!(effective_arrivals,(0.,0.))
            if isfinite(half_life)
                append!(reservoir_rows,(reservoir_rows[i][mask],reservoir_rows[i][.!mask]))
            else
                append!(reservoir_rows,(Int[],Int[]))
            end
            reservoir_rows[i]=Int[]
            n.states=Sketch[];n.reservoir=zeros(Float32,d,0);leaves+=1;split_count+=1
            on_split!==nothing && on_split(processed+row,i,nodes)
            end
        end
        if merge_below>0 && row==N
            nmerged=merge_siblings!(nodes,last_times,effective_arrivals,reservoir_rows;
                time=ts[end],half_life,merge_below,reservoir_cap,operations)
            leaves-=nmerged;merge_count+=nmerged;last_merge_time=ts[end]
            nmerged>0 && on_merge!==nothing && on_merge(processed+row,nmerged,nodes)
        end
        if on_checkpoint!==nothing && (row%observe_every==0 || row==N)
            on_checkpoint(row,(nodes=nodes,time=ts[row],half_life=Float64(half_life),last_times=last_times,
                merge_count=merge_count,split_count=split_count))
        end
        observer!==nothing && (row%observe_every==0 || row==N) && observer(row,nodes)
        row%20000==0 && println("Leaf-only Gaussian tree: ",row," arrivals; ",leaves," leaves")
    end
    # Materialize all remaining lazy decay at the final stream time.
    for (i,n) in enumerate(nodes)
        n.axis==0 || continue
        if operations===nothing || operations.decay!==nothing
            for s in n.states;operations===nothing ? decay!(s,ts[end]-last_times[i],half_life) : operations.decay(s,ts[end]-last_times[i],half_life);end
        end
        effective_arrivals[i]*=exp2(-(ts[end]-last_times[i])/half_life)
        last_times[i]=ts[end]
    end
    (nodes=nodes,errors=errors,update_seconds=times,split_seconds=split_times,configs=configs,
        half_life=Float64(half_life),time=ts[end],last_times=last_times,effective_arrivals=effective_arrivals,
        reservoir_rows=reservoir_rows,reservoir_cap=reservoir_cap,split_after=split_after,
        merge_below=Float64(merge_below),merge_interval=Float64(merge_interval),last_merge_time=last_merge_time,
        merge_count=merge_count,split_count=split_count,processed=processed+N,rng=rng,basisrng=basisrng)
end
function reconstruction(s,x)
    s.rank==0 ? s.mean : s.mean+s.U*(s.U'*(x-s.mean))
end
function sample_leaf(rng,node,s,count;max_draws=4096)
    p=parameters(s);d=length(p.mean);points=Vector{Vector{Float64}}();attempts=0
    while length(points)<count && attempts<max_draws
        x=p.mean+p.W*randn(rng,size(p.W,2))+sqrt.(p.D).*randn(rng,d);attempts+=1
        if all(node.lo .<= x) && all(x .< node.hi);push!(points,x);end
    end
    points,attempts
end
end
