using Test, Random, LinearAlgebra
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
const LT=LeafLatentTrees
@testset "Leaf-only latent Gaussian transfer" begin
    rng=MersenneTwister(56);X=clamp.(.2randn(rng,12,1600),-.9,.9)
    fit=fit_leaf_tree(X;max_leaves=8,split_after=100,reservoir_cap=32,configs=[(:diagonal,0),(:ppca,2),(:lowrank_diagonal,2)])
    nodes=fit.nodes;leaves=findall(n->n.axis==0,nodes)
    @test length(leaves)==8
    @test all(n.axis==0 || (isempty(n.states)&&isempty(n.reservoir)) for n in nodes)
    for k in 1:3
        @test sum(nodes[i].states[k].mass for i in leaves)≈1600
        @test sum(nodes[i].states[k].mass*nodes[i].states[k].mean for i in leaves)≈vec(sum(X;dims=2))
        @test maximum(e.mean_error for e in fit.errors[k])<1e-10
        @test maximum(e.conditional_diagonal_error for e in fit.errors[k])<1e-10
        for i in leaves
            n=nodes[i];s=n.states[k];p=parameters(s)
            @test all(p.D.>0)
            @test size(p.W,2)<=s.rank
            points,tries=sample_leaf(rng,n,s,5;max_draws=512)
            @test tries<=512
            @test all(leaf_index(nodes,x)==i for x in points)
            if k!=2
                @test vec(sum(abs2,p.W;dims=2))+p.D≈max.(s.diagonal/s.mass,1e-7)
            end
        end
    end
    # Analytic standard-normal halfspace: means ±sqrt(2/pi), var 1-2/pi.
    s=Sketch(:diagonal,0,100.,zeros(3),fill(100.,3),zeros(3,0),Float64[])
    l,r,e=conditional_children(s,1,0.,zeros(3,0),zeros(3,0))
    @test l.mass≈50 && r.mass≈50
    @test l.mean[1]≈-sqrt(2/pi)
    @test r.mean[1]≈sqrt(2/pi)
    @test l.diagonal[1]/l.mass≈1-2/pi
    @test l.diagonal[2]/l.mass≈1
end
