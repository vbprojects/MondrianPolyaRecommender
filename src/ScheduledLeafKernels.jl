# Included inside LeafTreeKernels. Topology depends on observations/reservoirs,
# not PPCA values. Plan identical cuts, then evaluate the causal moment DAG.
@kernel function segment_kernel!(mass,mu,diag,U,lam,nrank,X,ts,factors,last,offsets,rows,ids,h,V,Q,C,E,z,work)
    item=@index(Global,Linear);l=ids[item]
    @inbounds for at in offsets[l]:offsets[l+1]-1
        row=rows[at];elapsed=ts[row]-last[l]
        update_one!(mass,mu,diag,U,lam,nrank,X,row,elapsed,h,V,Q,C,E,z,work,true,l,factors[row])
        last[l]=ts[row]
    end
end
# One workgroup owns a region; its lanes cooperate across embedding coordinates.
@kernel unsafe_indices=true function cooperative_segment_kernel!(mass,mu,diag,U,lam,nrank,X,ts,factors,last,offsets,rows,ids,h,::Val{D},::Val{R}) where {D,R}
    group=@index(Group,Linear);lane=@index(Local,Linear)
    l=ids[group]
    v=@localmem Float64 (D,)
    residual=@localmem Float64 (D,)
    Q=@localmem Float64 (D,R+1)
    C=@localmem Float64 (R+1,R+1)
    E=@localmem Float64 (R+1,R+1)
    partial=@localmem Float64 (32,R+1)
    z=@localmem Float64 (R+1,)
    scalars=@localmem Float64 (4,)
    meta=@localmem Int (3,)
    selected=@localmem Int (R+1,)
    @inbounds for at in offsets[l]:offsets[l+1]-1
        row=rows[at]
        if lane==1
            factor=factors[row];old=mass[l]*factor;total=old+1
            scalars[1]=factor;scalars[2]=total;scalars[3]=sqrt(old/total)
            meta[1]=nrank[l];mass[l]=total;last[l]=ts[row]
        end
        @synchronize
        r=meta[1]
        for a in lane:32:D
            delta=X[a,row]-mu[a,l];v[a]=scalars[3]*delta
            mu[a,l]+=delta/scalars[2];diag[a,l]=diag[a,l]*scalars[1]+v[a]^2
        end
        for a in lane:32:r;lam[a,l]*=scalars[1];end
        @synchronize
        for a in 1:r
            dotv=0.
            for b in lane:32:D;dotv+=basis_value(U,b,a,l,D)*v[b];Q[b,a]=basis_value(U,b,a,l,D);end
            partial[lane,a]=dotv
        end
        @synchronize
        for a in lane:32:r
            dotv=0.
            for b in 1:32;dotv+=partial[b,a];end
            z[a]=dotv
        end
        @synchronize
        norm2=0.
        for b in lane:32:D
            value=v[b]
            for a in 1:r;value-=Q[b,a]*z[a];end
            residual[b]=value;norm2+=value*value
        end
        partial[lane,R+1]=norm2
        @synchronize
        if lane==1
            norm2=0.
            for b in 1:32;norm2+=partial[b,R+1];end
            normres=sqrt(norm2);scalars[4]=normres
            m=r
            if normres>1e-12 && R>0;m=r+1;z[m]=normres;end
            meta[2]=m
        end
        @synchronize
        m=meta[2]
        if m>r
            for b in lane:32:D;Q[b,m]=residual[b]/scalars[4];end
        end
        if lane==1 && m>0 && R>0
            for a in 1:m,b in 1:m
                C[a,b]=z[a]*z[b]+(a==b && a<=r ? lam[a,l] : 0.)
                E[a,b]=a==b ? 1. : 0.
            end
            for sweep in 1:40
                off=0.;scale=0.
                for a in 1:m;scale=max(scale,abs(C[a,a]));end
                for a in 1:m-1,b in a+1:m
                    ab=C[a,b];off=max(off,abs(ab))
                    if abs(ab)>max(1e-30,1e-15*scale)
                        tau=(C[b,b]-C[a,a])/(2ab)
                        t=ifelse(tau>=0,1.,-1.)/(abs(tau)+sqrt(1+tau*tau))
                        c=inv(sqrt(1+t*t));ss=t*c
                        aa=C[a,a];bb=C[b,b]
                        C[a,a]=aa-t*ab;C[b,b]=bb+t*ab;C[a,b]=0.;C[b,a]=0.
                        for j in 1:m
                            if j!=a && j!=b
                                ja=C[j,a];jb=C[j,b]
                                C[j,a]=c*ja-ss*jb;C[a,j]=C[j,a]
                                C[j,b]=ss*ja+c*jb;C[b,j]=C[j,b]
                            end
                            ea=E[j,a];eb=E[j,b];E[j,a]=c*ea-ss*eb;E[j,b]=ss*ea+c*eb
                        end
                    end
                end
                off<=max(1e-30,1e-15*scale) && break
            end
            keep=min(R,m)
            for a in 1:keep
                best=1
                for b in 2:m;C[b,b]>C[best,best] && (best=b);end
                selected[a]=best;lam[a,l]=max(0.,C[best,best]);C[best,best]=-Inf
            end
            nrank[l]=keep
        end
        @synchronize
        keep=Int(nrank[l])
        if m>0 && R>0
            for a in 1:keep,b in lane:32:D
                value=0.
                for j in 1:m;value+=Q[b,j]*E[j,selected[a]];end
                basis_set!(U,value,b,a,l,D)
            end
        end
        @synchronize
    end
end
# The 2D/rank-1 recurrence fits in registers, including its 2×2 eigensolve.
@kernel function segment_2d_kernel!(mass,mu,diag,U,lam,nrank,X,ts,factors,last,offsets,rows,ids,h)
    item=@index(Global,Linear);l=ids[item]
    @inbounds begin
        m=mass[l];m1=mu[1,l];m2=mu[2,l];d1=diag[1,l];d2=diag[2,l]
        u1=basis_value(U,1,1,l,2);u2=basis_value(U,2,1,l,2);lambda=lam[1,l];r=nrank[l];clock=last[l]
        for at in offsets[l]:offsets[l+1]-1
            row=rows[at];f=factors[row];clock=ts[row]
            old=m*f;m=old+1;coef=sqrt(old/m)
            delta1=X[1,row]-m1;delta2=X[2,row]-m2
            v1=coef*delta1;v2=coef*delta2;m1+=delta1/m;m2+=delta2/m
            d1=d1*f+v1*v1;d2=d2*f+v2*v2;lambda*=f
            a=u1*v1+u2*v2
            b1=v1-u1*a;b2=v2-u2*a;n=sqrt(b1*b1+b2*b2)
            if r==0
                if n>1e-12;u1=b1/n;u2=b2/n;lambda=n*n;r=1;end
            elseif n>1e-12
                aa=lambda+a*a;bb=n*n;ab=a*n
                largest=(aa+bb+sqrt((aa-bb)^2+4ab*ab))/2
                x=aa>=bb ? largest-bb : ab;y=aa>=bb ? ab : largest-aa
                normxy=sqrt(x*x+y*y)
                if normxy>0
                    x/=normxy;y/=normxy
                    u1=u1*x+(b1/n)*y;u2=u2*x+(b2/n)*y
                end
                lambda=max(0.,largest)
            else
                lambda=max(0.,lambda+a*a)
            end
        end
        mass[l]=m;mu[1,l]=m1;mu[2,l]=m2;diag[1,l]=d1;diag[2,l]=d2
        basis_set!(U,u1,1,1,l,2);basis_set!(U,u2,2,1,l,2);lam[1,l]=lambda;nrank[l]=r;last[l]=clock
    end
end
function packed_views(buffer,d,R,K)
    mass=vec(view(buffer,1:1,:));mu=view(buffer,2:d+1,:);diag=view(buffer,d+2:2d+1,:)
    U=view(buffer,2d+2:2d+1+d*R,:)
    lam=view(buffer,2d+2+d*R:2d+1+d*R+R,:);nr=vec(view(buffer,size(buffer,1):size(buffer,1),:))
    mass,mu,diag,U,lam,nr
end
function fit_scheduled_tree(k,X;max_leaves=64,split_after=512,reservoir_cap=64,seed=918,
        configs=[(:diagonal,0),(:ppca,4),(:ppca,8),(:lowrank_diagonal,4),(:lowrank_diagonal,8)],
        max_depth=16,observer=nothing,observe_every=10000,on_split=nothing,timestamps=nothing,
        half_life=Inf,on_checkpoint=nothing,merge_below=0.,merge_interval=half_life,on_merge=nothing,
        _initial=nothing)
    # Intermediate callbacks require materialized prefix states. Keep their exact
    # established semantics through the event driver rather than expose placeholders.
    if observer!==nothing || on_split!==nothing || on_checkpoint!==nothing || on_merge!==nothing
        return fit_leaf_tree(X;max_leaves,split_after,reservoir_cap,seed,configs,max_depth,observer,
            observe_every,on_split,timestamps,half_life,on_checkpoint,merge_below,merge_interval,on_merge,
            _initial,operations=kernel_operations(k))
    end
    isfinite(merge_below) && 0<=merge_below<split_after || throw(ArgumentError("merge_below must be nonnegative and below split_after"))
    if merge_below>0
        return fit_merging_scheduled_tree(k,X;max_leaves,split_after,reservoir_cap,seed,configs,max_depth,
            timestamps,half_life,merge_below,merge_interval,_initial)
    end
    # Snapshot numeric state before planning mutates topology/reservoirs. Existing
    # internal nodes have no segment to execute in this epoch.
    initial_states=_initial===nothing ? Dict{Int,Vector{Sketch}}() :
        Dict(i=>deepcopy(n.states) for (i,n) in enumerate(_initial.nodes) if n.axis==0)
    initial_times=_initial===nothing ? Float64[] : copy(_initial.last_times)
    old_internal=_initial===nothing ? Set{Int}() : Set(i for (i,n) in enumerate(_initial.nodes) if n.axis!=0)
    d,N=size(X);assignments=Int[];sizehint!(assignments,N)
    transfers=Tuple[]
    function route(nodes,x)
        i=leaf_index(nodes,x);push!(assignments,i);i
    end
    # The reservoir path is bit-for-bit the reference policy, including RNG order.
    function reservoir(n,x,h,rng,cap)
        R=n.reservoir;count=size(R,2)
        if count<cap;return hcat(R,Float32.(x));end
        if isfinite(h)
            for j in 1:count-1,a in 1:d;@inbounds R[a,j]=R[a,j+1];end
            R[:,end].=x
        else
            j=rand(rng,1:n.seen);j<=cap && (R[:,j].=x)
        end
        R
    end
    function children(s,axis,cut,L,R)
        push!(transfers,(axis,cut,L,R))
        (Sketch(s.kind,s.rank,d),Sketch(s.kind,s.rank,d),
            (mass_error=0.,mean_error=0.,conditional_diagonal_error=0.,compressed_diagonal_error=0.))
    end
    ops=(route=route,advance=nothing,decay=nothing,reservoir=reservoir,
        proposal=LT.cut_proposal,children=children)
    planned=fit_leaf_tree(X;max_leaves,split_after,reservoir_cap,seed,configs,max_depth,timestamps,half_life,_initial,operations=ops)
    ns=planned.nodes;K=length(ns);ts=timestamps===nothing ? zeros(N) : Float64.(timestamps)
    counts=zeros(Int,K)
    for i in assignments;counts[i]+=1;end
    offsets=vcat(1,1 .+cumsum(counts));rows=zeros(Int,N);cursor=copy(offsets[1:end-1])
    for row in 1:N;i=assignments[row];rows[cursor[i]]=row;cursor[i]+=1;end
    births=fill(ts[1],K)
    for (i,_) in initial_states;births[i]=initial_times[i];end
    splits=[i for (i,n) in enumerate(ns) if n.axis!=0 && !(i in old_internal)]
    # Creation-order IDs are not split-event order; recover the plan from child IDs.
    sort!(splits;by=i->ns[i].left)
    transfer_at=zeros(Int,K)
    for (event,i) in enumerate(splits)
        transfer_at[i]=event
        t=ts[rows[offsets[i+1]-1]];births[ns[i].left]=t;births[ns[i].right]=t
    end
    layers=[[i for (i,n) in enumerate(ns) if n.depth==depth && !(i in old_internal)] for depth in 0:maximum(n.depth for n in ns)]
    filter!(!isempty,layers)
    factors=ones(N)
    if isfinite(half_life)
        for i in 1:K
            previous=births[i]
            for at in offsets[i]:offsets[i+1]-1
                row=rows[at];factors[row]=exp2(-(ts[row]-previous)/half_life);previous=ts[row]
            end
        end
    end
    dx=k.array(X);dt=k.array(ts);df=k.array(factors);dr=k.array(rows);doff=k.array(offsets)
    errors=[Vector{NamedTuple}(undef,length(splits)) for _ in configs];times=zeros(length(configs));split_times=zeros(length(configs))
    for (config,(kind,R)) in enumerate(configs)
        packed=zeros(2+2d+d*R+R,K)
        mass,mu,diag,U,lam,nr=packed_views(packed,d,R,K)
        for (i,states) in initial_states
            child=states[config];mass[i]=child.mass;mu[:,i]=child.mean;diag[:,i]=child.diagonal
            U[1:d*length(child.eigenvalues),i]=vec(child.U);lam[1:length(child.eigenvalues),i]=child.eigenvalues;nr[i]=length(child.eigenvalues)
        end
        device_packed=k.array(packed);dm,du,dd,dU,dl,dn=packed_views(device_packed,d,R,K)
        last=copy(births);dlast=k.array(last)
        if get_backend(dm) isa KernelAbstractions.CPU
            V=zeros(d,K);Q=zeros(d,R+1,K);C=zeros(R+1,R+1,K);E=similar(C);z=zeros(R+1,K);work=similar(V)
        else
            V=Q=C=E=z=work=nothing # cooperative kernels use register/shared memory
        end
        for ids in layers
            times[config]+=@elapsed begin
                backend=get_backend(dm)
                if d==2 && R==1
                    # Regions have unequal stream lengths. One GPU block per region
                    # avoids serializing every region behind the longest lane.
                    group_size=backend isa KernelAbstractions.CPU ? 64 : 1
                    segment_2d_kernel!(backend,group_size)(dm,du,dd,dU,dl,dn,dx,dt,df,dlast,doff,dr,k.array(ids),Float64(half_life);ndrange=length(ids))
                    KernelAbstractions.synchronize(backend)
                elseif backend isa KernelAbstractions.CPU
                    launch(segment_kernel!,dm,du,dd,dU,dl,dn,dx,dt,df,dlast,doff,dr,k.array(ids),Float64(half_life),V,Q,C,E,z,work;ndrange=length(ids))
                elseif 8*(d*(R+3)+2*(R+1)^2+34*(R+1)+7) > 48*1024
                    # Conservative shared-memory cap; retain functionality for larger ranks/dimensions.
                    # Allocate global scratch only for this fallback.
                    v=k.array(zeros(d,K));q=k.array(zeros(d,R+1,K));c=k.array(zeros(R+1,R+1,K))
                    e=similar(c);zlarge=k.array(zeros(R+1,K));w=similar(v)
                    segment_kernel!(backend,1)(dm,du,dd,dU,dl,dn,dx,dt,df,dlast,doff,dr,k.array(ids),Float64(half_life),v,q,c,e,zlarge,w;ndrange=length(ids))
                    KernelAbstractions.synchronize(backend)
                else
                    cooperative_segment_kernel!(backend,32)(dm,du,dd,dU,dl,dn,dx,dt,df,dlast,doff,dr,k.array(ids),Float64(half_life),Val(d),Val(R);ndrange=32length(ids))
                    KernelAbstractions.synchronize(backend)
                end
            end
            # A level barrier transfers compact moment state, never individual posts.
            packed=Array(device_packed);mass,mu,diag,U,lam,nr=packed_views(packed,d,R,K)
            for i in ids
                s=Sketch(kind,R,mass[i],mu[:,i],diag[:,i],reshape(U[1:d*Int(nr[i]),i],d,Int(nr[i])),lam[1:Int(nr[i]),i])
                if ns[i].axis==0
                    leaf_time=offsets[i]<offsets[i+1] ? ts[rows[offsets[i+1]-1]] : births[i]
                    decay!(s,ts[end]-leaf_time,half_life);ns[i].states[config]=s
                else
                    axis,cut,Lseed,Rseed=transfers[(transfer_at[i]-1)*length(configs)+config]
                    result=@timed conditional_children(s,axis,cut,Lseed,Rseed)
                    split_times[config]+=result.time;l,r,err=result.value;errors[config][transfer_at[i]]=err
                    for (j,child) in ((ns[i].left,l),(ns[i].right,r))
                        mass[j]=child.mass;mu[:,j]=child.mean;diag[:,j]=child.diagonal
                        U[1:d*length(child.eigenvalues),j]=vec(child.U);lam[1:length(child.eigenvalues),j]=child.eigenvalues;nr[j]=length(child.eigenvalues)
                    end
                end
            end
            copyto!(device_packed,packed)
        end
    end
    merge(planned,(errors=errors,update_seconds=times,split_seconds=split_times))
end

# A merge decision depends on decayed numeric state. Plan only up to the next
# maintenance boundary, execute those segments, merge, then plan the next epoch.
# This is causal and keeps the fused CPU/GPU update kernels active with merging.
function fit_merging_scheduled_tree(k,X;max_leaves,split_after,reservoir_cap,seed,configs,max_depth,
        timestamps,half_life,merge_below,merge_interval,_initial=nothing)
    isfinite(half_life) && half_life>0 && isfinite(merge_interval) && merge_interval>0 ||
        throw(ArgumentError("merging requires a finite half-life and positive finite merge_interval"))
    timestamps===nothing && throw(ArgumentError("decay requires timestamps"))
    N=size(X,2);ts=Float64.(timestamps)
    N>0 && length(ts)==N && all(isfinite,ts) && issorted(ts) || throw(ArgumentError("invalid chronological timestamps"))
    all(isfinite,X) && all(x->abs(x)<=1,X) || throw(ArgumentError("expected normalized finite embeddings"))
    _initial!==nothing && ts[1]<_initial.time && throw(ArgumentError("continuation precedes fitted stream"))
    f=_initial;first=1;clock=f===nothing ? ts[1] : f.last_merge_time;epochs=0
    errors=[NamedTuple[] for _ in configs];times=zeros(length(configs));split_times=similar(times);fill!(split_times,0.)
    maintenance_seconds=0.
    while first<=N
        boundary=findnext(t->t-clock>=merge_interval,ts,first)
        stop=boundary===nothing ? N : boundary-1
        if stop>=first
            f=fit_scheduled_tree(k,view(X,:,first:stop);max_leaves,split_after,reservoir_cap,seed,configs,max_depth,
                timestamps=view(ts,first:stop),half_life,_initial=f)
            epochs+=1
            for c in eachindex(configs)
                append!(errors[c],f.errors[c]);times[c]+=f.update_seconds[c];split_times[c]+=f.split_seconds[c]
            end
        end
        if boundary===nothing;break;end
        maintenance_seconds+=@elapsed f=kernel_merge_tree!(k,f;time=ts[boundary],merge_below)
        clock=ts[boundary];first=boundary
    end
    maintenance_seconds+=@elapsed f=kernel_merge_tree!(k,f;time=ts[end],merge_below)
    merge(f,(errors=errors,update_seconds=times,split_seconds=split_times,
        merge_below=Float64(merge_below),merge_interval=Float64(merge_interval),
        schedule_epochs=epochs,maintenance_seconds=maintenance_seconds))
end
