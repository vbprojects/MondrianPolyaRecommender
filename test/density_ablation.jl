using Test, LinearAlgebra, Random
include("../src/LeafLatentTree.jl")
include("../src/DensityAblation.jl")
const LT=LeafLatentTrees
const DA=DensityAblation
BLAS.set_num_threads(1)
@testset "Density ablation: density, geometry and inference" begin
    rng=MersenneTwister(22);X=.12randn(rng,5,512)
    f=LT.fit_leaf_tree(X;max_leaves=4,split_after=64,configs=[(:diagonal,0),(:ppca,4)])
    leaves=findall(n->n.axis==0,f.nodes);counts=zeros(length(f.nodes))
    for x in eachcol(X);counts[LT.leaf_index(f.nodes,x)]+=1;end
    p=DA.polyaprobs(f.nodes,counts)
    @test sum(p)≈1
    @test all(p[i]==0 for (i,n) in enumerate(f.nodes) if n.axis!=0)
    for i in leaves
        n=f.nodes[i]
        for state in n.states
            g=DA.gaussian_leaf(state,n,rng;draws=8192)
            C=g.W*g.W'+Diagonal(g.D);points=.1randn(rng,5,17)
            calculated=DA.gaussian_logpdf(g,points)
            exact=[-.5*(5log(2pi)+logdet(C)+dot(x-g.mean,C\(x-g.mean))) for x in eachcol(points)]
            @test calculated≈exact atol=1e-9
            Y,attempts=DA.draw_box(rng,n,g,50)
            @test attempts>=50
            @test all(n.lo .<= Y) && all(Y .< n.hi)
            @test 0<g.acceptance<=1
        end
        U,attempts=DA.draw_box(rng,n,nothing,100)
        @test attempts==100 && all(n.lo .<= U) && all(U .< n.hi)
    end
    uniform=DA.density_logs(f.nodes,p,nothing,X)
    @test all(isfinite,uniform)
    @test sum(exp(uniform[findfirst(j->LT.leaf_index(f.nodes,X[:,j])==i,1:512)])*prod(f.nodes[i].hi-f.nodes[i].lo) for i in leaves)≈1
    cost,center=DA.density_cost([-1000.,0.,1000.],2)
    @test cost==[4.,1.,.25]
    @test DA.density_cost([-995.,5.,1005.],2)[1]==cost
    P=.2randn(rng,3,20);D=DA.distance_matrix(P,P);edges=DA.knn_edges(D;k=3)
    weights=[D[a,b] for (a,b) in edges];paths=DA.shortest_from(20,edges,weights,1:20)
    @test all(isfinite,paths)
    @test paths≈paths' atol=1e-12
    @test maximum(abs,diag(paths))==0
    floyd=fill(Inf,20,20);floyd[diagind(floyd)].=0
    for ((a,b),w) in zip(edges,weights);floyd[a,b]=w;floyd[b,a]=w;end
    for k in 1:20,i in 1:20,j in 1:20;floyd[i,j]=min(floyd[i,j],floyd[i,k]+floyd[k,j]);end
    @test paths≈floyd atol=1e-12
    geometry=DA.graph_geometry(P,ones(20),edges,6)
    attached=DA.attach_distances(P,P,ones(20),geometry,[[i] for i in 1:20])
    @test attached≈geometry.paths/geometry.scale
    @test all(.05-1e-15 .<= DA.likelihood(attached) .<= .95+1e-15)
    @test DA.information(fill(.25,4),fill(.5,3,4))≈zeros(3) atol=1e-14
    @test DA.information([.5,.5],[.95 .05])[1]>.7
    # First query is independent of the hidden target. Matching random numbers
    # produce identical proposal and inference histories until responses differ.
    H=X[:,1:12];states,hleaves=DA.branch_states(f.nodes,H)
    banks=Dict(i=>collect(1:40) for i in leaves);L=DA.likelihood(DA.distance_matrix(X[:,101:140],H)/.3)
    HH=DA.distance_matrix(H,H)/.3;u=rand(rng,8)
    a=DA.trial(f.nodes,states,hleaves,banks,L,L,HH,HH,1,u;budget=8,history=true)
    b=DA.trial(f.nodes,states,hleaves,banks,L,L,HH,HH,12,u;budget=8,history=true)
    @test a.history[1].query==b.history[1].query
    @test all(isapprox(sum(r.belief),1.;atol=1e-12) && all(r.belief .>=0) for r in a.history)
    @test length(unique(r.query for r in a.history))==8
    @test a.used==(a.stop==0 ? 8 : a.stop)
    @test a.capped_correct==(a.correct ? a.stop : 9)
    # Force a confident false stop: likelihood inversion cannot be counted as
    # efficient localization merely because it consumes one response.
    H2=X[:,1:2];ss,ll=DA.branch_states(f.nodes,H2);bb=Dict(i=>collect(1:4) for i in leaves)
    LI=repeat([.999 .001],4,1);TR=1 .-LI;balls=[0. 1.;1. 0.]
    wrong=DA.trial(f.nodes,ss,ll,bb,LI,TR,balls,balls,1,fill(.5,2);budget=2,confidence=.9)
    @test wrong.stop==1 && wrong.false_stop && !wrong.correct && wrong.capped_correct==3
    continued=DA.trial(f.nodes,ss,ll,bb,LI,TR,balls,balls,1,fill(.5,2);budget=2,confidence=.9,history=true)
    @test continued.used==wrong.used && continued.error==wrong.error && continued.brier==wrong.brier
end
