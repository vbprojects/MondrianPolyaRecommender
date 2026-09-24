using Test, Random
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
@testset "Tracking does not change tree fitting" begin
    X=rand(MersenneTwister(91),Float32,12,200).-0.5f0
    history=[];cuts=[]
    a=fit_leaf_tree(X;configs=[(:ppca,4)],split_after=32,max_leaves=8)
    b=fit_leaf_tree(X;configs=[(:ppca,4)],split_after=32,max_leaves=8,observe_every=20,observer=(row,nodes)->push!(history,(row,length(nodes))),on_split=(row,i,nodes)->push!(cuts,(row,i)))
    @test length(history)==10
    @test last(history)[1]==200
    @test length(cuts)==count(n->n.axis!=0,b.nodes)
    @test [(n.axis,n.cut,n.left,n.right) for n in a.nodes]==[(n.axis,n.cut,n.left,n.right) for n in b.nodes]
    @test [s.mass for n in a.nodes for s in n.states]==[s.mass for n in b.nodes for s in n.states]
end
