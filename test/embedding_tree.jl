using Test, Random, MondrianPolyaToy
const ET= MondrianPolyaToy
function terminal_ids(t,m=nothing)
    result=Int[]
    function visit(i)
        n=t.nodes[i]
        if n.axis==0 || (m!==nothing && !m.active[i]); push!(result,i)
        else; visit(n.left); visit(n.right); end
    end
    visit(1); result
end
@testset "Embedding tree and compression" begin
    X=randn(MersenneTwister(43),12,1500)
    X[12,:].=0 # constant coordinate must have finite padded volume
    t=fit_partition(X;max_nodes=127,min_count=30)
    @test length(t.nodes)<=127
    @test t.nodes[1].count==1500
    @test all(isfinite(ET.logvolume(n)) for n in t.nodes)
    @test all(n.axis==0 || n.count==t.nodes[n.left].count+t.nodes[n.right].count for n in t.nodes)
    @test sum(embedding_probabilities(t)[terminal_ids(t)])≈1
    assignments=[embedding_leaf(t,view(X,:,i)) for i in axes(X,2)]
    @test all(count(==(i),assignments)==t.nodes[i].count for i in terminal_ids(t))
    for kind in (:residual,:microclusters)
        m=fit_compression(t,X,kind;split_count=30,cluster_cap=3)
        ids=terminal_ids(t,m); p=embedding_probabilities(t,m)
        @test m.counts[1]==1500
        @test sum(p[ids])≈1
        @test all(isfinite(embedding_logdensity(t,view(X,:,i),m;probabilities=p)) for i in 1:20)
        if kind==:residual
            @test sum(m.residual)==1500
            @test m.active[1] && m.residual[1]==30
            @test any(m.residual[i]>0 && m.active[i] for i in eachindex(m.residual))
        else
            cs=reduce(vcat,m.clusters)
            @test sum(c.count for c in cs)==1500
            @test all(length(c)<=3 for c in m.clusters)
            @test all(isempty(m.clusters[i]) for i in eachindex(m.active) if m.active[i])
            @test sum(c.count*c.mean for c in cs)≈vec(sum(X;dims=2))
            @test sum(c.m2+c.count*c.mean.^2 for c in cs)≈vec(sum(abs2,X;dims=2))
        end
    end
    singleton=fit_partition(zeros(256,1))
    @test length(singleton.nodes)==1
    @test embedding_probabilities(singleton)==[1.]
    @test_throws ArgumentError fit_partition(fill(NaN,2,2))
    @test_throws ArgumentError embedding_leaf(t,zeros(2))
    @test_throws ArgumentError embedding_leaf(t,fill(1e9,12))
end
