using Test, Random, LinearAlgebra
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
const LT=LeafLatentTrees
covariance(s)=let p=parameters(s);p.W*p.W'+Diagonal(p.D);end
@testset "Sibling moment combination" begin
    for (kind,r) in ((:diagonal,0),(:ppca,4),(:lowrank_diagonal,4))
        rng=MersenneTwister(2);X=randn(rng,9,100);a=Sketch(kind,r,9);b=Sketch(kind,r,9)
        for x in eachcol(X[:,1:40]);LT.update!(a,x);end
        for x in eachcol(X[:,41:end]);LT.update!(b,x);end
        decay!(a,3.,2.);decay!(b,1.,2.)
        w=vcat(fill(exp2(-3/2),40),fill(exp2(-1/2),60));mu=X*w/sum(w)
        c=merge_sketches(a,b)
        @test c.mass≈sum(w)
        @test c.mean≈mu
        @test c.diagonal≈(X.-mu).^2*w
        @test size(c.U,2)<=r
        @test all(eigvals(Symmetric(covariance(c))).>=0)
        # The retained scatter is the optimal rank-r approximation of the input sketches.
        delta=a.mean-b.mean
        S=a.U*Diagonal(a.eigenvalues)*a.U'+b.U*Diagonal(b.eigenvalues)*b.U'+a.mass*(b.mass/c.mass)*delta*delta'
        if r>0
            ev=eigen(Symmetric(S));ids=length(ev.values):-1:length(ev.values)-r+1
            @test c.U*Diagonal(c.eigenvalues)*c.U'≈ev.vectors[:,ids]*Diagonal(ev.values[ids])*ev.vectors[:,ids]' atol=1e-10
        end
        z=Sketch(kind,r,9);d=merge_sketches(z,c)
        @test d.mass==c.mass && d.mean≈c.mean && d.diagonal≈c.diagonal
        @test all(isfinite,merge_sketches(z,z).mean)
    end
end
@testset "Decay-driven collapse and capacity reuse" begin
    rng=MersenneTwister(88);X=clamp.(.3randn(rng,6,480),-.95,.95)
    ts=vcat(zeros(160),fill(15.,160),fill(30.,160))
    opts=(timestamps=ts,half_life=1.,configs=[(:ppca,4),(:diagonal,0)],split_after=24,max_leaves=8,reservoir_cap=24,seed=44)
    changes=[]
    f=fit_leaf_tree(X;opts...,merge_below=3.,merge_interval=1.,
        on_split=(row,i,ns)->push!(changes,(:split,row,length(ns))),
        on_merge=(row,n,ns)->push!(changes,(:merge,row,length(ns))))
    @test f.merge_count>0
    @test f.split_count>opts.max_leaves-1 # recycled capacity exceeds the lifetime cut cap
    @test f.split_count-f.merge_count==count(n->n.axis!=0,f.nodes)
    @test length(f.nodes)==2count(n->n.axis==0,f.nodes)-1<=2opts.max_leaves-1
    @test any(x->x[1]==:merge && x[3]==1,changes) # silence collapses even multi-level subtrees
    @test any(x->x[1]==:split && x[2]>320,changes) # fresh traffic splits again
    w=exp2.(-(ts[end].-ts))
    for config in 1:2
        leaves=filter(n->n.axis==0,f.nodes)
        @test sum(n.states[config].mass for n in leaves)≈sum(w)
        @test sum(n.states[config].mass*n.states[config].mean for n in leaves)≈X*w atol=1e-10
    end
    for (i,n) in enumerate(f.nodes)
        if n.axis==0
            @test issorted(f.reservoir_rows[i])
            @test n.reservoir==Float32.(X[:,f.reservoir_rows[i]])
            @test size(n.reservoir,2)<=opts.reservoir_cap
            @test all(n.lo .<= n.reservoir) && all(n.reservoir .< n.hi)
        else
            @test i<n.left<=length(f.nodes) && i<n.right<=length(f.nodes)
            @test isempty(n.states) && isempty(n.reservoir)
        end
    end
    @test sum(leaf_probabilities(f).probabilities)≈1
    before=Base.summarysize(f.nodes);oldmass=sum(n.states[1].mass for n in f.nodes if n.axis==0)
    f=merge_tree!(f;time=50.)
    @test length(f.nodes)==1
    @test Base.summarysize(f.nodes)<before
    @test f.nodes[1].states[1].mass≈oldmass*exp2(-20.)
    @test f.effective_arrivals==[0.]
    @test f.last_times==[50.] && f.time==50.
    @test leaf_probabilities(f).probabilities==[1.]
    @test all(i->leaf_index(f.nodes,X[:,i])==1,1:480)
    @test_throws ArgumentError merge_tree!(f;time=49.)
    @test_throws ArgumentError fit_leaf_tree(X;opts...,merge_below=24.)
    @test_throws ArgumentError fit_leaf_tree(X;opts...,merge_below=-1.)
    @test_throws ArgumentError fit_leaf_tree(X;opts...,merge_below=3.,merge_interval=0.)
    @test_throws ArgumentError fit_leaf_tree(X;merge_below=3.)
    # Numerical extinction remains finite, and zero-threshold maintenance can collapse empty leaves.
    f=fit_leaf_tree(X;opts...)
    f=merge_tree!(f;time=1e8,merge_below=0.)
    @test length(f.nodes)==1 && f.nodes[1].states[1].mass==0.
    @test all(isfinite,covariance(f.nodes[1].states[1]))
end
@testset "Busy siblings are retained; disabled merging is unchanged" begin
    X=.2randn(MersenneTwister(9),2,120);opts=(configs=[(:ppca,1)],max_leaves=2,split_after=24,timestamps=zeros(120),half_life=1.)
    a=fit_leaf_tree(X;opts...);b=fit_leaf_tree(X;opts...,merge_below=2.)
    @test a.merge_count==b.merge_count==0
    @test [(n.axis,n.cut,n.left,n.right) for n in a.nodes]==[(n.axis,n.cut,n.left,n.right) for n in b.nodes]
    @test leaf_probabilities(a).probabilities≈leaf_probabilities(b).probabilities
    @test all(a.nodes[i].states[1].diagonal≈b.nodes[i].states[1].diagonal for i in (2,3))
end
