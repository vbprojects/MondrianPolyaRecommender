using Test, Random, LinearAlgebra
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
@testset "Exponential leaf decay" begin
    rng=MersenneTwister(7);X=.2randn(rng,8,400);ts=collect(0.:399.);h=80.
    f=fit_leaf_tree(X;timestamps=ts,half_life=h,max_leaves=1,configs=[(:ppca,4)])
    w=exp2.(-(ts[end].-ts)./h);mu=X*w/sum(w);s=f.nodes[1].states[1]
    @test s.mass≈sum(w)
    @test s.mean≈mu
    @test s.diagonal≈(X.-mu).^2*w
    p=parameters(s);decay!(s,h,h);p2=parameters(s)
    @test s.mass≈sum(w)/2
    @test p.mean≈p2.mean
    @test p.W*p.W'+Diagonal(p.D)≈p2.W*p2.W'+Diagonal(p2.D)
    a=deepcopy(s);b=deepcopy(s);decay!(a,10.,h);decay!(a,30.,h);decay!(b,40.,h)
    @test a.diagonal≈b.diagonal
    g=fit_leaf_tree(X;timestamps=ts,half_life=h,split_after=24,max_leaves=8,configs=[(:ppca,4)])
    leaves=filter(n->n.axis==0,g.nodes)
    @test length(leaves)>1
    @test sum(n.states[1].mass for n in leaves)≈sum(w)
    @test sum(n.states[1].mass*n.states[1].mean for n in leaves)≈X*w
    now=leaf_probabilities(g);later=leaf_probabilities(g;time=ts[end]+h)
    @test sum(now.probabilities)≈1
    @test sum(later.probabilities)≈1
    @test later.masses≈now.masses/2
    @test all(n.axis==0 || isempty(n.states) for n in g.nodes)
    @test_throws ArgumentError leaf_probabilities(g;time=0.)
    @test_throws ArgumentError fit_leaf_tree(X;half_life=2.)
    @test_throws ArgumentError fit_leaf_tree(X;timestamps=reverse(ts),half_life=h)
    @test_throws ArgumentError decay!(s,-1.,h)
    @test_throws ArgumentError decay!(s,1.,0.)
    sparse=fit_leaf_tree(X;timestamps=ts.*100,half_life=1.,split_after=24,configs=[(:ppca,4)])
    @test length(sparse.nodes)==1 # lifetime arrivals alone cannot force a split
    same=fit_leaf_tree(X;timestamps=zeros(400),half_life=h,max_leaves=1,configs=[(:ppca,4)])
    @test same.nodes[1].states[1].mass==400
    @test same.nodes[1].states[1].mean≈vec(sum(X;dims=2))/400
    @test leaf_probabilities(g;time=1e8).masses==zeros(length(g.nodes))
    @test sum(leaf_probabilities(g;time=1e8).probabilities)≈1
    # After numerical extinction, a new arrival starts a fresh moment estimate.
    decay!(s,1e8,h);LeafLatentTrees.update!(s,X[:,1])
    @test s.mass==1 && s.mean≈X[:,1]
    @test all(isfinite,s.diagonal)
end
