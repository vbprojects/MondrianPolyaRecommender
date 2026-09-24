using LinearAlgebra, Random, Statistics, UUIDs, Printf, Serialization
include("../src/LeafLatentTree.jl")
include("../src/LatentPostCodes.jl")
include("export_utils.jl")
using .LeafLatentTrees, .LatentPostCodes
const LT=LeafLatentTrees
mutable struct Policy
    name::String
    mode::Symbol
    T::DataType
    jitter::Float64
    rng::MersenneTwister
    buckets::Dict{UInt64,PostCodes}
    smin::Vector{Float64}
    smax::Vector{Float64}
    logcond::Vector{Float64}
    equivalence::Float64
    departure::Float64
    wrong::Int
    routed::Int
    migrations::Int
end
function policies(initial,seed)
    specs=[("Sequential Float64",:sequential,Float64,0.),("Sequential Float32",:sequential,Float32,0.),
        ("Sequential Float16",:sequential,Float16,0.),("Jitter 0.1%",:sequential,Float64,.001),
        ("Jitter 1%",:sequential,Float64,.01),("Restore variance with noise",:restore,Float64,0.),
        ("Remap from initial code",:anchor,Float64,0.),("Refresh originals every 10 cycles",:refresh,Float64,0.),
        ("Original-vector oracle",:oracle,Float64,0.),("Frozen initial decoder",:frozen,Float64,0.),
        ("Map singular-value floor 0.1",:floor,Float64,0.),("Orthogonal transport",:orthogonal,Float64,0.)]
    [Policy(name,mode,T,jitter,MersenneTwister(seed+1000i),Dict(initial.decoder.id=>PostCodes(initial.ids,initial.decoder,T.(initial.z))),
        Float64[],Float64[],Float64[],0.,0.,0,0,0) for (i,(name,mode,T,jitter)) in enumerate(specs)]
end
function move_codes!(p,c,target,state,truth,lookup,anchor)
    isempty(c.ids) && return PostCodes(c.ids,target,zeros(p.T,size(target.U,2),0))
    p.mode==:frozen && return c
    indices=[lookup[id] for id in c.ids]
    source=p.mode==:anchor ? subset(anchor,indices) : c
    map=coordinate_map(source.decoder,target);sigma=svdvals(map.A)
    push!(p.smin,minimum(sigma));push!(p.smax,maximum(sigma))
    push!(p.logcond,minimum(sigma)>0 ? log10(maximum(sigma)/minimum(sigma)) : Inf)
    @assert maximum(sigma)<=1+1e-8
    result=migrate_posts(source,target;T=Float64)
    direct=target.U'*(decode_posts(source).-target.mean)
    gap=maximum(abs.(result.z-direct);init=0.);p.equivalence=max(p.equivalence,gap)
    @assert gap<1e-10
    if p.mode in (:floor,:orthogonal)
        e=svd(map.A);values=p.mode==:floor ? max.(e.S,.1) : ones(length(e.S))
        result=PostCodes(c.ids,target,e.U*(values.*(e.Vt*Float64.(source.z))).+map.b)
        p.smin[end]=minimum(values);p.smax[end]=maximum(values);p.logcond[end]=log10(maximum(values)/minimum(values))
    elseif p.mode==:oracle
        result=encode_posts(target,c.ids,truth[:,indices];T=Float64)
    elseif p.mode==:restore && length(c.ids)>size(result.z,1)
        # Non-oracle noise: fill positive covariance deficit using the live model.
        # This can restore population spread, but cannot identify a post's lost residual.
        centered=result.z.-mean(result.z;dims=2)
        C=centered*centered'/size(centered,2)
        targetC=Diagonal(state.eigenvalues/max(state.mass,1e-12))
        e=eigen(Symmetric(targetC-C));W=e.vectors*Diagonal(sqrt.(max.(e.values,0.)))
        result=PostCodes(c.ids,target,result.z+W*randn(p.rng,size(result.z)...))
    elseif p.jitter>0
        scale=sqrt(sum(state.eigenvalues)/max(state.mass,1e-12)/size(target.U,2))
        result=PostCodes(c.ids,target,result.z+p.jitter*scale*randn(p.rng,size(result.z)...))
    end
    p.departure=max(p.departure,maximum(abs.(result.z-direct);init=0.))
    p.migrations+=length(c.ids)
    PostCodes(c.ids,target,p.T.(result.z))
end
function synthetic_spec(name,d,r,seed)
    rng=MersenneTwister(seed);Q=Matrix(qr(randn(rng,d,d)).Q);mu=.03Q[:,end]
    function draw(n,cycle)
        if name=="Full-rank 2D control"
            frame=Q;scales=[.15,.08];noise=0.
        elseif name=="Stable rank-four manifold"
            frame=Q[:,1:r];scales=[.15,.12,.08,.04];noise=1e-5
        elseif name=="Nearly equal spectrum"
            frame=Q[:,1:8];scales=collect(range(.1,.1001;length=8));noise=.001
        elseif name=="Very thin direction"
            frame=Q[:,1:r];scales=[.15,.1,.03,1e-5];noise=1e-7
        else
            theta=cycle*pi/40
            frame=cos(theta)*Q[:,1:r]+sin(theta)*Q[:,r+1:2r]
            scales=[.15,.12,.08,.04];noise=.001
        end
        X=mu.+frame*(scales.*randn(rng,length(scales),n))+noise*randn(rng,d,n)
        @assert maximum(abs,X)<1
        X
    end
    draw,Q
end
function neighbors(Y,queries,queryids;k=10)
    D=sum(abs2,Y;dims=1).+sum(abs2,queries;dims=1)'.-2queries'*Y
    for (j,i) in enumerate(queryids);D[j,i]=Inf;end
    [partialsortperm(vec(D[j,:]),1:k) for j in axes(D,1)]
end
function evaluate(p,X,lookup,initialY,queryids,true_neighbors,case,seed,cycle,phase,counts)
    d,N=size(X);Y=zeros(d,N);seen=Set{UUID}();lc=Float64[];ranks=Float64[];singular=0;eligible=0
    orth=0.
    for c in values(p.buckets)
        @assert isempty(intersect(seen,Set(c.ids)));union!(seen,c.ids)
        ids=[lookup[id] for id in c.ids];Y[:,ids]=decode_posts(c)
        r=size(c.z,1);orth=max(orth,norm(c.decoder.U'*c.decoder.U-I))
        if length(ids)>r && r>0
            eligible+=1;Z=Float64.(c.z);Z .-= mean(Z;dims=2);C=Z*Z'/size(Z,2)
            e=max.(eigvals(Symmetric(C)),0.);top=maximum(e);bottom=minimum(e)
            singular+=bottom<=max(1e-30,top*1e-12)
            push!(lc,bottom>0 ? log10(top/bottom) : Inf)
            v=e/max(sum(e),1e-300);push!(ranks,exp(-sum(x>0 ? x*log(x) : 0. for x in v)))
        end
    end
    @assert length(seen)==N && all(isfinite,Y)
    centered=X.-mean(X;dims=2);den=sum(abs2,centered)
    recs=neighbors(Y,X[:,queryids],queryids)
    recall=mean(length(intersect(a,b))/10 for (a,b) in zip(recs,true_neighbors))
    rec=(case=case,seed=seed,cycle=cycle,phase=phase,method=p.name,d=d,n=N,
        nrmse=sqrt(sum(abs2,Y-X)/den),migration_nrmse=sqrt(sum(abs2,Y-initialY)/den),
        variance_ratio=sum(abs2,Y.-mean(Y;dims=2))/den,recall10=recall,
        code_log10_condition=maximum(lc;init=0.),code_effective_rank=isempty(ranks) ? 0. : mean(ranks),
        singular_code_fraction=singular/max(eligible,1),basis_orthogonality_error=orth,
        min_map_singular=minimum(p.smin;init=1.),max_map_singular=maximum(p.smax;init=0.),
        max_map_log10_condition=maximum(p.logcond;init=0.),map_equivalence_error=p.equivalence,
        max_transport_projection_departure=p.departure,
        routing_error=p.wrong/max(p.routed,1),routed=p.routed,migrations=p.migrations,
        splits=counts.splits,merges=counts.merges,active_regions=length(p.buckets),ids_preserved=true)
    rec,Y
end
function run_case(case,d,r,seed;cycles=80,N=384,batch=256,artifact_path=nothing)
    draw,Q=synthetic_spec(case,d,r,seed);train=draw(512,0);X=draw(N,0)
    ids=[uuid4(MersenneTwister(seed*100000+i)) for i in 1:N];lookup=Dict(id=>i for (i,id) in enumerate(ids))
    f=fit_leaf_tree(train;timestamps=zeros(512),half_life=1.,configs=[(:ppca,r)],max_leaves=1,reservoir_cap=32,seed=seed+91)
    root=f.nodes[1].states[1];decoder=Decoder(1,root.mean,root.U);anchor=encode_posts(decoder,ids,X;T=Float64)
    initialY=decode_posts(anchor);ps=policies(anchor,seed)
    journal=NamedTuple[]
    regions=IdDict(root=>UInt64(1));nextid=Ref(UInt64(1));splits=Ref(0);merges=Ref(0)
    newdecoder(s)=begin;nextid[]+=1;Decoder(nextid[],s.mean,s.U);end
    function children(s,axis,cut,L,R)
        left,right,err=conditional_children(s,axis,cut,L,R)
        oldid=pop!(regions,s);ld=newdecoder(left);rd=newdecoder(right)
        regions[left]=ld.id;regions[right]=rd.id;splits[]+=1
        for p in ps
            c=pop!(p.buckets,oldid);positions=c.decoder.mean[axis].+vec(c.decoder.U[axis:axis,:]*Float64.(c.z));mask=positions.<cut
            truthmask=[X[axis,lookup[id]]<cut for id in c.ids]
            p.wrong+=sum(mask.!=truthmask);p.routed+=length(mask)
            p.buckets[ld.id]=move_codes!(p,subset(c,findall(mask)),ld,left,X,lookup,anchor)
            p.buckets[rd.id]=move_codes!(p,subset(c,findall(.!mask)),rd,right,X,lookup,anchor)
            artifact_path!==nothing && p.name=="Sequential Float64" && push!(journal,
                (kind="split",old=c,left=p.buckets[ld.id],right=p.buckets[rd.id],axis=axis,cut=cut))
        end
        left,right,err
    end
    function combine(a,b)
        parent=merge_sketches(a,b);lid=pop!(regions,a);rid=pop!(regions,b);pd=newdecoder(parent);regions[parent]=pd.id;merges[]+=1
        for p in ps
            oldleft=p.buckets[lid];oldright=p.buckets[rid]
            l=move_codes!(p,pop!(p.buckets,lid),pd,parent,X,lookup,anchor)
            rr=move_codes!(p,pop!(p.buckets,rid),pd,parent,X,lookup,anchor)
            # Frozen codes all share the initial decoder, including empty partitions.
            if p.mode==:frozen
                allids=vcat(l.ids,rr.ids);p.buckets[pd.id]=subset(anchor,[lookup[id] for id in allids])
            else
                p.buckets[pd.id]=PostCodes(vcat(l.ids,rr.ids),pd,hcat(l.z,rr.z))
            end
            artifact_path!==nothing && p.name=="Sequential Float64" && push!(journal,
                (kind="merge",left=oldleft,right=oldright,parent=p.buckets[pd.id]))
        end
        parent
    end
    function reservoir(n,x,h,rng,cap)
        R=n.reservoir
        if size(R,2)<cap;return hcat(R,Float32.(x));end
        R[:,1:end-1].=R[:,2:end];R[:,end].=x;R
    end
    ops=(route=leaf_index,advance=LT.advance_state!,decay=decay!,reservoir=reservoir,proposal=LT.cut_proposal,children=children,merge=combine)
    records=NamedTuple[];frames=NamedTuple[];queryids=randperm(MersenneTwister(seed+52),N)[1:24]
    true_neighbors=neighbors(X,X[:,queryids],queryids)
    function snapshot(cycle,phase)
        for p in ps
            rec,Y=evaluate(p,X,lookup,initialY,queryids,true_neighbors,case,seed,cycle,phase,(splits=splits[],merges=merges[]))
            push!(records,rec)
            if seed==1 && phase=="merged" && case=="Rotating 256D manifold"
                projected=Q[:,1:2]'*Y;truth=Q[:,1:2]'*X
                push!(frames,(cycle=cycle,method=p.name,metrics=rec,points=[collect(projected[:,j]) for j in 1:128],truth=[collect(truth[:,j]) for j in 1:128]))
            end
        end
    end
    snapshot(0,"merged");checkpoints=Set((1,2,5,9,10,19,20,39,40,59,60,79,80))
    for cycle in 1:cycles
        fresh=draw(batch,case=="Rotating 256D manifold" ? cycle : 0)
        times=f.time.+collect(range(.001,.25;length=batch))
        f=fit_leaf_tree(fresh;timestamps=times,half_life=1.,configs=[(:ppca,r)],max_leaves=8,split_after=32,
            reservoir_cap=32,operations=ops,_initial=f)
        @assert length(regions)==count(n->n.axis==0,f.nodes)
        cycle in checkpoints && snapshot(cycle,"split")
        f=merge_tree!(f;time=f.time+12.,merge_below=3.,operations=ops)
        @assert length(f.nodes)==1 && length(regions)==1
        for p in ps
            @assert length(p.buckets)==1
            if p.mode==:refresh && cycle%10==0
                key=only(keys(p.buckets));c=p.buckets[key];idx=[lookup[id] for id in c.ids]
                p.buckets[key]=encode_posts(c.decoder,c.ids,X[:,idx];T=p.T)
            end
        end
        cycle in checkpoints && snapshot(cycle,"merged")
    end
    @assert splits[]==merges[] && splits[]>cycles
    println(case," seed ",seed,": ",splits[]," splits / ",merges[]," merges; all ",N," UUIDs preserved");flush(stdout)
    if artifact_path!==nothing
        serialize(artifact_path,(case=case,seed=seed,truth=X,ids=ids,anchor=anchor,journal=journal,
            final=Dict(p.name=>only(values(p.buckets)) for p in ps)))
    end
    records,frames
end
function main_migration()
    BLAS.set_num_threads(1);out=normpath(joinpath(@__DIR__,"../output/latent_migration"));mkpath(out)
    records=NamedTuple[];frames=NamedTuple[]
    cases=[("Full-rank 2D control",2,2),("Stable rank-four manifold",32,4),("Nearly equal spectrum",32,4),
        ("Very thin direction",32,4),("Rotating 256D manifold",256,4)]
    elapsed=@elapsed for (case,d,r) in cases,seed in 1:3
        artifact=case=="Rotating 256D manifold" && seed==1 ? joinpath(out,"transition_audit.jls") : nothing
        a,b=run_case(case,d,r,seed;artifact_path=artifact);append!(records,a);append!(frames,b)
        write(joinpath(out,"results.json"),sprint(io->json(io,(records=records,frames=frames))))
    end
    control=NamedTuple[]
    for degrees in (5.,20.,60.)
        theta=deg2rad(degrees);a=Decoder(1,zeros(2),reshape([1.,0.],2,1));b=Decoder(2,zeros(2),reshape([cos(theta),sin(theta)],2,1))
        c=PostCodes([UUID(UInt128(1))],a,ones(1,1))
        for _ in 1:100;c=migrate_posts(migrate_posts(c,b),a);end
        @assert isapprox(only(c.z),cos(theta)^200;rtol=1e-12)
        push!(control,(degrees=degrees,cycles=100,condition=cond(coordinate_map(a,b).A),amplitude=only(c.z),expected_amplitude=cos(theta)^200))
    end
    payload=(records=records,frames=frames,analytic_control=control,seconds=elapsed,
        configuration=(seeds=3,cycles=80,cohort=384,fresh_per_cycle=256,half_life=1.,idle_half_lives=12.,rank=4,max_leaves=8,split_after=32,merge_below=3.,
        arithmetic="Float64 throughout; Float32/Float16 policies round stored codes only.",
        isolation="All policies share a tree trained on fresh original observations; cohort codes never train the tree. Original cohort vectors are used only for audits and labeled oracle/refresh policies. The persistent cohort deliberately outlives the decay window.",
        note="Frozen and initial-code remapping controls retain the initial decoder. This fixed-cohort experiment does not measure growing decoder storage under a continuous arrival stream."))
    write(joinpath(out,"results.json"),sprint(io->json(io,payload)))
    println("Completed in ",elapsed," seconds")
end
if abspath(PROGRAM_FILE)==@__FILE__;main_migration();end
