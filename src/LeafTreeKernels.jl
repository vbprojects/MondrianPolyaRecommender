"""Portable numerical/tree kernels with a synchronous, reference-order host driver.
Include LeafLatentTree.jl first. No CUDA dependency: pass Array or a backend array constructor.
"""
module LeafTreeKernels
using KernelAbstractions, LinearAlgebra, Random
using ..LeafLatentTrees
const LT=LeafLatentTrees
export KernelTreeBackend, fit_kernel_tree, kernel_probabilities, kernel_route, kernel_sample_leaf,
       update_sketches!, kernel_decay!, kernel_children, kernel_operations, lapack_eigen_batch, kernel_merge_sketches, kernel_merge_tree!
struct KernelTreeBackend{F,S}
    array::F
    eigensolver::S
end
"""Solver returns ascending eigenvalues and column eigenvectors for an n×n×batch array.
Omit solver for the portable in-kernel Jacobi implementation.
"""
KernelTreeBackend(array=Array; eigensolver=nothing)=KernelTreeBackend(array,eigensolver)
function lapack_eigen_batch(C::Array)
    n,_,b=size(C);values=zeros(eltype(C),n,b);vectors=similar(C)
    for j in 1:b
        e=eigen(Symmetric(C[:,:,j]));values[:,j]=e.values;vectors[:,:,j]=e.vectors
    end
    values,vectors
end
function launch(kernel,args...;ndrange)
    backend=get_backend(first(args));kernel(backend,64)(args...;ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
end
@kernel function route_kernel!(out,X,axes,cuts,left,right)
    j=@index(Global,Linear)
    i=1
    while axes[i]!=0
        i=X[axes[i],j]<cuts[i] ? left[i] : right[i]
    end
    out[j]=i
end
function kernel_route(k,nodes,X::AbstractMatrix)
    size(X,1)==length(nodes[1].lo) && all(isfinite,X) || throw(ArgumentError("invalid query dimensions/values"))
    n=size(X,2);n==0 && return Int[]
    out=k.array(zeros(Int,n));cpu=get_backend(out) isa KernelAbstractions.CPU
    queries=cpu ? X : k.array(X)
    launch(route_kernel!,out,queries,k.array([n.axis for n in nodes]),k.array([n.cut for n in nodes]),k.array([n.left for n in nodes]),k.array([n.right for n in nodes]);ndrange=n)
    cpu ? out : Array(out)
end
# One work item owns one independent sketch; loops preserve within-leaf update order.
# Workspaces are preallocated arrays, so the device never allocates or calls LAPACK.
@inline basis_value(U::AbstractArray{T,3},i,j,l,d) where T = @inbounds U[i,j,l]
@inline basis_value(U::AbstractMatrix,i,j,l,d) = @inbounds U[i+(j-1)*d,l]
@inline basis_set!(U::AbstractArray{T,3},v,i,j,l,d) where T = (@inbounds U[i,j,l]=v)
@inline basis_set!(U::AbstractMatrix,v,i,j,l,d) = (@inbounds U[i+(j-1)*d,l]=v)
@inline function update_one!(mass,mu,diag,U,lam,nrank,X,row,elapsed,h,V,Q,C,E,z,work,solve_in_kernel,l,decay_factor=-1.,weight=1.)
    @inbounds begin
    d=size(mu,1);R=size(lam,1);r=Int(nrank[l]);factor=decay_factor<0 ? exp2(-elapsed/h) : decay_factor
    old=mass[l]*factor;total=old+weight;coef=sqrt(weight*(old/total))
    for a in 1:d
        delta=X[a,row]-mu[a,l];V[a,l]=coef*delta
        mu[a,l]+=weight*(delta/total);diag[a,l]=diag[a,l]*factor+V[a,l]^2
    end
    mass[l]=total
    for a in 1:r;lam[a,l]*=factor;end
    if R>0
        for a in 1:r
            dotv=0.
            for b in 1:d;dotv+=basis_value(U,b,a,l,d)*V[b,l];Q[b,a,l]=basis_value(U,b,a,l,d);end
            z[a,l]=dotv
        end
        norm2=0.
        for b in 1:d
            v=V[b,l]
            for a in 1:r;v-=basis_value(U,b,a,l,d)*z[a,l];end
            work[b,l]=v;norm2+=v*v
        end
        residual=sqrt(norm2);m=r
        if residual>1e-12
            m=r+1;z[m,l]=residual
            for b in 1:d;Q[b,m,l]=work[b,l]/residual;end
        end
        nrank[l]=m
        # Pad inactive dimensions below the nonnegative covariance spectrum.
        for a in 1:R+1,b in 1:R+1
            C[a,b,l]=a==b ? -1. : 0.
        end
        if m>0
            for a in 1:m,b in 1:m
                C[a,b,l]=z[a,l]*z[b,l]+(a==b && a<=r ? lam[a,l] : 0.)
                E[a,b,l]=a==b ? 1. : 0.
            end
            if solve_in_kernel
            # Cyclic Jacobi diagonalization of a symmetric (rank+1)-square matrix.
            for sweep in 1:40
                off=0.;scale=0.
                for a in 1:m;scale=max(scale,abs(C[a,a,l]));end
                for a in 1:m-1,b in a+1:m
                    ab=C[a,b,l];off=max(off,abs(ab))
                    if abs(ab)>max(1e-30,1e-15*scale)
                        tau=(C[b,b,l]-C[a,a,l])/(2ab)
                        t=ifelse(tau>=0,1.,-1.)/(abs(tau)+sqrt(1+tau*tau))
                        c=inv(sqrt(1+t*t));ss=t*c
                        aa=C[a,a,l];bb=C[b,b,l]
                        C[a,a,l]=aa-t*ab;C[b,b,l]=bb+t*ab;C[a,b,l]=0.;C[b,a,l]=0.
                        for v in 1:m
                            if v!=a && v!=b
                                va=C[v,a,l];vb=C[v,b,l]
                                C[v,a,l]=c*va-ss*vb;C[a,v,l]=C[v,a,l]
                                C[v,b,l]=ss*va+c*vb;C[b,v,l]=C[v,b,l]
                            end
                            ea=E[v,a,l];eb=E[v,b,l]
                            E[v,a,l]=c*ea-ss*eb;E[v,b,l]=ss*ea+c*eb
                        end
                    end
                end
                off<=max(1e-30,1e-15*scale) && break
            end
            keep=min(R,m)
            for a in 1:keep
                best=1
                for b in 2:m;C[b,b,l]>C[best,best,l] && (best=b);end
                lam[a,l]=max(0.,C[best,best,l]);C[best,best,l]=-Inf
                for b in 1:d
                    val=0.
                    for v in 1:m;val+=Q[b,v,l]*E[v,best,l];end
                    basis_set!(U,val,b,a,l,d)
                end
            end
            nrank[l]=keep
            end
        end
    end
    end
end
@kernel function update_kernel!(mass,mu,diag,U,lam,nrank,X,elapsed,h,V,Q,C,E,z,work,solve_in_kernel)
    l=@index(Global,Linear)
    update_one!(mass,mu,diag,U,lam,nrank,X,l,elapsed[l],h,V,Q,C,E,z,work,solve_in_kernel,l)
end
@kernel function apply_eigen_kernel!(U,lam,nrank,Q,values,vectors)
    l=@index(Global,Linear);m=Int(nrank[l]);keep=min(size(lam,1),m)
    for a in 1:keep
        source=size(values,1)-a+1;lam[a,l]=max(0.,values[source,l])
        for b in 1:size(U,1)
            val=0.
            for j in 1:m;val+=Q[b,j,l]*vectors[j,source,l];end
            U[b,a,l]=val
        end
    end
    nrank[l]=keep
end
function update_sketches!(k,states,X,elapsed,h)
    n=length(states);n>0 || return states
    d=length(states[1].mean);R=states[1].rank
    size(X)==(d,n) && length(elapsed)==n || throw(ArgumentError("one column and elapsed time required per sketch"))
    h>0 && !isnan(h) && all(t->isfinite(t)&&t>=0,elapsed) && all(isfinite,X) || throw(ArgumentError("invalid times or observations"))
    all(s->length(s.mean)==d&&s.rank==R,states) || throw(ArgumentError("batch sketches must share dimension and rank"))
    mass=k.array([s.mass for s in states]);mu=k.array(hcat([s.mean for s in states]...));diag=k.array(hcat([s.diagonal for s in states]...))
    uh=zeros(d,R,n);lh=zeros(R,n);nr=Int[]
    for (j,s) in enumerate(states)
        r=length(s.eigenvalues);push!(nr,r);uh[:,1:r,j]=s.U;lh[1:r,j]=s.eigenvalues
    end
    U=k.array(uh);lam=k.array(lh);ranks=k.array(nr)
    Q=k.array(zeros(d,R+1,n));C=k.array(zeros(R+1,R+1,n))
    launch(update_kernel!,mass,mu,diag,U,lam,ranks,k.array(Matrix{Float64}(X)),k.array(Float64.(elapsed)),Float64(h),
        k.array(zeros(d,n)),Q,C,k.array(zeros(R+1,R+1,n)),k.array(zeros(R+1,n)),k.array(zeros(d,n)),k.eigensolver===nothing;ndrange=n)
    if k.eigensolver!==nothing && R>0
        values,vectors=k.eigensolver(C)
        launch(apply_eigen_kernel!,U,lam,ranks,Q,values,vectors;ndrange=n)
    end
    mh=Array(mass);mus=Array(mu);ds=Array(diag);us=Array(U);ls=Array(lam);rs=Array(ranks)
    for (j,s) in enumerate(states)
        s.mass=mh[j];s.mean=mus[:,j];s.diagonal=ds[:,j];s.U=us[:,1:rs[j],j];s.eigenvalues=ls[1:rs[j],j]
    end
    states
end
@kernel function decay_kernel!(values,factor)
    i=@index(Global,Linear);values[i]*=factor
end
function kernel_decay!(k,s,elapsed,h)
    isfinite(elapsed)&&elapsed>=0&&h>0&&!isnan(h) || throw(ArgumentError("invalid decay time"))
    v=k.array(vcat(s.mass,s.diagonal,s.eigenvalues));launch(decay_kernel!,v,exp2(-elapsed/h);ndrange=length(v));a=Array(v)
    s.mass=a[1];d=length(s.mean);s.diagonal=a[2:d+1];s.eigenvalues=a[d+2:end];s
end
@kernel function reservoir_kernel!(out,R,x,mode,slot)
    i,j=@index(Global,NTuple)
    if mode==1
        out[i,j]=j==size(out,2) ? x[i] : R[i,j]
    elseif mode==2
        out[i,j]=j==size(out,2) ? x[i] : R[i,j+1]
    else
        out[i,j]=j==slot ? x[i] : R[i,j]
    end
end
function kernel_reservoir(k,n,x,h,rng,cap)
    R=n.reservoir;count=size(R,2)
    mode=count<cap ? 1 : isfinite(h) ? 2 : 3
    slot=mode==3 ? rand(rng,1:n.seen) : 0
    out=k.array(zeros(Float32,length(x),min(cap,count+1)))
    launch(reservoir_kernel!,out,k.array(R),k.array(Float32.(x)),mode,slot;ndrange=size(out));Array(out)
end
@kernel function widths_kernel!(widths,R)
    a=@index(Global,Linear);lo=R[a,1];hi=lo
    for j in 2:size(R,2);lo=min(lo,R[a,j]);hi=max(hi,R[a,j]);end
    widths[a]=hi-lo
end
@kernel function partition_kernel!(mask,R,axis,cut)
    j=@index(Global,Linear);mask[j]=R[axis,j]<cut
end
@kernel function sorted_axis_kernel!(values,R,axis)
    # One work item per independent cut; insertion sort of a small reservoir.
    l=@index(Global,Linear)
    for j in 1:size(R,2)
        x=R[axis,j];v=j
        while v>1 && values[v-1]>x;values[v]=values[v-1];v-=1;end
        values[v]=x
    end
end
function kernel_proposal(k,R,rng)
    dev=k.array(R);widths=k.array(zeros(Float32,size(R,1)));launch(widths_kernel!,widths,dev;ndrange=length(widths))
    ws=Array(widths);rate=sum(ws);rate>0 || return nothing
    u=rand(rng)*rate;axis=length(ws)
    for j in eachindex(ws);u-=ws[j];if u<0;axis=j;break;end;end
    values=k.array(zeros(Float32,size(R,2)));launch(sorted_axis_kernel!,values,dev,axis;ndrange=1)
    vs=Array(values);a=vs[max(1,length(vs)÷4)];b=vs[min(length(vs),3length(vs)÷4)];a<b || return nothing
    cut=Float64(a+rand(rng)*(b-a));mask=k.array(zeros(Int,length(vs)))
    launch(partition_kernel!,mask,dev,axis,cut;ndrange=length(vs));m=Array(mask).!=0
    any(m)&&!all(m) ? (axis,cut,m) : nothing
end
@kernel function conditional_kernel!(means,diags,mu,W,D,col,deltas,shifts)
    i,side=@index(Global,NTuple);v=D[i]
    for j in 1:size(W,2);v+=W[i,j]^2;end
    means[i,side]=mu[i]+col[i]*shifts[side]
    diags[i,side]=max(0.,v-deltas[side]*col[i]^2)
end
@kernel function conditional_operator!(out,W,D,col,Q,delta)
    i,j=@index(Global,NTuple)
    val=D[i]*Q[i,j];dotc=0.
    for b in 1:size(Q,1);dotc+=col[b]*Q[b,j];end
    val-=delta*col[i]*dotc
    for a in 1:size(W,2)
        v=0.
        for b in 1:size(Q,1);v+=W[b,a]*Q[b,j];end
        val+=W[i,a]*v
    end
    out[i,j]=val
end
function kernel_children(k,s,axis,cut,seedL,seedR)
    p=parameters(s);col=p.W*p.W[axis,:];col[axis]+=p.D[axis];variance=col[axis];sd=sqrt(variance)
    alpha=(cut-p.mean[axis])/sd;q=clamp(LT.cdf(alpha),1e-12,1-1e-12);phi=exp(-alpha^2/2)/sqrt(2pi)
    ll=phi/q;lr=phi/(1-q);dl=clamp(ll*(ll+alpha),0.,1.);dr=clamp(lr*(lr-alpha),0.,1.)
    W=k.array(p.W);D=k.array(p.D);c=k.array(col);means=k.array(zeros(length(col),2));diags=similar(means)
    launch(conditional_kernel!,means,diags,k.array(p.mean),W,D,c,k.array([dl/variance,dr/variance]),k.array([-ll/sd,lr/sd]);ndrange=size(means))
    mh=Array(means);dh=Array(diags)
    function op(Q,delta)
        out=k.array(zeros(size(Q)));launch(conditional_operator!,out,W,D,c,k.array(Q),delta/variance;ndrange=size(Q));Array(out)
    end
    # Rare split reorthogonalization stays on CPU; all covariance-operator products use kernels.
    l=LT.compress_conditional(s,s.mass*q,mh[:,1],dh[:,1],Q->op(Q,dl),seedL)
    r=LT.compress_conditional(s,s.mass*(1-q),mh[:,2],dh[:,2],Q->op(Q,dr),seedR)
    parentdiag=vec(sum(abs2,p.W;dims=2))+p.D
    exact=q*(dh[:,1]+(l.mean-p.mean).^2)+(1-q)*(dh[:,2]+(r.mean-p.mean).^2)
    lp=parameters(l);rp=parameters(r)
    approx=q*(vec(sum(abs2,lp.W;dims=2))+lp.D+(l.mean-p.mean).^2)+(1-q)*(vec(sum(abs2,rp.W;dims=2))+rp.D+(r.mean-p.mean).^2)
    l,r,(mass_error=abs(l.mass+r.mass-s.mass),mean_error=norm(q*l.mean+(1-q)*r.mean-p.mean),conditional_diagonal_error=norm(exact-parentdiag),compressed_diagonal_error=norm(approx-parentdiag))
end
# Parallel weighted moment combination and factor assembly. Only the thin
# d×(2r+1) SVD stays on the host, as do the rare split reorthogonalizations.
@kernel function merge_moments_kernel!(out,amu,bmu,adiag,bdiag,aU,bU,alam,blam,ratio,between)
    i=@index(Global,Linear)
    delta=bmu[i]-amu[i]
    out[i,1]=amu[i]+ratio*delta
    out[i,2]=adiag[i]+bdiag[i]+between*delta*delta
    for j in 1:length(alam);out[i,2+j]=aU[i,j]*sqrt(max(alam[j],0.));end
    for j in 1:length(blam);out[i,2+length(alam)+j]=bU[i,j]*sqrt(max(blam[j],0.));end
    out[i,size(out,2)]=sqrt(between)*delta
end
function kernel_merge_sketches(k,a,b)
    a.kind==b.kind && a.rank==b.rank && length(a.mean)==length(b.mean) ||
        throw(ArgumentError("incompatible sibling sketches"))
    total=a.mass+b.mass;d=length(a.mean)
    total==0 && return Sketch(a.kind,a.rank,d)
    ratio=b.mass/total;between=a.mass*ratio
    out=k.array(zeros(d,3+length(a.eigenvalues)+length(b.eigenvalues)))
    launch(merge_moments_kernel!,out,k.array(a.mean),k.array(b.mean),k.array(a.diagonal),k.array(b.diagonal),
        k.array(a.U),k.array(b.U),k.array(a.eigenvalues),k.array(b.eigenvalues),ratio,between;ndrange=d)
    values=Array(out)
    LT.compress_merged(a,total,values[:,1],values[:,2],values[:,3:end])
end
kernel_merge_tree!(k,f;kwargs...)=merge_tree!(f;operations=kernel_operations(k),kwargs...)
@kernel function leaf_mass_kernel!(mass,raw,last,t,h,axes)
    i=@index(Global,Linear)
    mass[i]=axes[i]==0 ? raw[i]*exp2(-(t-last[i])/h) : 0.
end
@kernel function aggregate_kernel!(mass,axes,left,right,depth,level)
    i=@index(Global,Linear)
    if depth[i]==level && axes[i]!=0;mass[i]=mass[left[i]]+mass[right[i]];end
end
@kernel function probability_kernel!(prob,mass,axes,left,right,depth,priors,level,kappa)
    i=@index(Global,Linear)
    if depth[i]==level && axes[i]!=0
        q=mass[i]+kappa>0 ? (mass[left[i]]+kappa*priors[i])/(mass[i]+kappa) : priors[i]
        prob[left[i]]=prob[i]*q;prob[right[i]]=prob[i]*(1-q);prob[i]=0.
    end
end
function kernel_probabilities(k,f;time=f.time,config=1,kappa=2.)
    isfinite(time)&&time>=f.time&&isfinite(kappa)&&kappa>=0 || throw(ArgumentError("invalid query time/prior"))
    ns=f.nodes;N=length(ns);axes=k.array([n.axis for n in ns]);left=k.array([n.left for n in ns]);right=k.array([n.right for n in ns]);depth=k.array([n.depth for n in ns])
    raw=k.array([n.axis==0 ? n.states[config].mass : 0. for n in ns]);mass=k.array(zeros(N))
    launch(leaf_mass_kernel!,mass,raw,k.array(f.last_times),Float64(time),f.half_life,axes;ndrange=N)
    for level in maximum(n.depth for n in ns):-1:0;launch(aggregate_kernel!,mass,axes,left,right,depth,level;ndrange=N);end
    priors=k.array([n.axis==0 ? 0. : (n.cut-n.lo[n.axis])/(n.hi[n.axis]-n.lo[n.axis]) for n in ns]);prob=k.array(vcat(1.,zeros(N-1)))
    for level in 0:maximum(n.depth for n in ns);launch(probability_kernel!,prob,mass,axes,left,right,depth,priors,level,Float64(kappa);ndrange=N);end
    (probabilities=Array(prob),masses=Array(mass))
end
@kernel function sample_kernel!(out,valid,mu,W,D,noise,lo,hi)
    j=@index(Global,Linear);r=size(W,2);inside=true
    for i in 1:length(mu)
        x=mu[i]+sqrt(D[i])*noise[r+i,j]
        for a in 1:r;x+=W[i,a]*noise[a,j];end
        out[i,j]=x;inside=inside && lo[i]<=x<hi[i]
    end
    valid[j]=inside
end
function kernel_sample_leaf(k,rng,node,s,count;max_draws=4096)
    count>=0 && max_draws>=0 || throw(ArgumentError("negative sample count/budget"))
    count==0 && return Vector{Vector{Float64}}(),0
    p=parameters(s);d=length(p.mean);r=size(p.W,2)
    # Generate the same normal draws, in the same order as reference sampling.
    # The entire attempt budget is consumed from rng; use separate RNGs per request.
    noise=zeros(r+d,max_draws)
    for j in 1:max_draws;noise[1:r,j]=randn(rng,r);noise[r+1:end,j]=randn(rng,d);end
    out=k.array(zeros(d,max_draws));valid=k.array(zeros(Int,max_draws))
    max_draws>0 && launch(sample_kernel!,out,valid,k.array(p.mean),k.array(p.W),k.array(p.D),k.array(noise),k.array(node.lo),k.array(node.hi);ndrange=max_draws)
    accepted=findall(!iszero,Array(valid));take=min(count,length(accepted));ids=accepted[1:take];points=Array(out)
    [points[:,j] for j in ids],take==count ? ids[end] : max_draws
end
function kernel_operations(k)
    (route=(nodes,x)->only(kernel_route(k,nodes,reshape(x,:,1))),
     advance=(s,x,elapsed,h)->update_sketches!(k,[s],reshape(x,:,1),[elapsed],h),
     decay=(s,elapsed,h)->kernel_decay!(k,s,elapsed,h),
     reservoir=(n,x,h,rng,cap)->kernel_reservoir(k,n,x,h,rng,cap),
     proposal=(R,rng)->kernel_proposal(k,R,rng),
     children=(s,axis,cut,L,R)->kernel_children(k,s,axis,cut,L,R),
     merge=(a,b)->kernel_merge_sketches(k,a,b))
end
include("ScheduledLeafKernels.jl")
include("CentroidPostKernels.jl")
function fit_kernel_tree(k,X;execution=(k.eigensolver===nothing ? :scheduled : :event),kwargs...)
    execution==:event && return fit_leaf_tree(X;operations=kernel_operations(k),kwargs...)
    execution==:scheduled || throw(ArgumentError("execution must be :event or :scheduled"))
    k.eigensolver===nothing || throw(ArgumentError("scheduled execution uses the fused portable eigensolver; construct KernelTreeBackend(array) without an external solver"))
    fit_scheduled_tree(k,X;kwargs...)
end
end
