using Test
include("../examples/bluesky_mppca.jl")
@testset "High-dimensional online PPCA" begin
    rng=MersenneTwister(21); X=randn(rng,12,400); weights=.1 .+rand(rng,400)
    c=LocalPPCA(X[:,1])
    for j in axes(X,2);update!(c,X[:,j],weights[j],4);end
    total=sum(weights); mu=X*weights/total
    @test c.mass≈total
    @test c.mean≈mu
    @test c.trace≈sum(weights[j]*sum(abs2,X[:,j]-mu) for j in axes(X,2))
    @test size(c.basis)==(12,4)
    @test c.basis'*c.basis≈Matrix{Float64}(I,4,4)
    @test sum(c.scatter)<=c.trace+1e-8
    @test all(isfinite(logscore(c,X[:,j])) for j in 1:10)
    # With full available rank, incremental covariance reproduces the batch result.
    exact=LocalPPCA(X[:,1])
    for j in axes(X,2);update!(exact,X[:,j],weights[j],12);end
    centered=X.-mu
    @test exact.basis*diagm(exact.scatter)*exact.basis'≈centered*diagm(weights)*centered'
    t=fit_partition(X;max_nodes=31,min_count=30)
    m=fit_local(t,X;split_count=30)
    @test count(!isempty,m.mixtures)>1
    @test !isempty(m.mixtures[1]) && m.active[1]
    @test all(length(cs)<=2 for cs in m.mixtures)
    counts,p=publish(t,m;samples=16)
    # Allocation must conserve mass in all ancestors of each mixture owner,
    # including samples clamped next to an upper cut boundary.
    for (owner,cs) in enumerate(m.mixtures)
        isempty(cs) && continue
        one=(mixtures=[i==owner ? cs : LocalPPCA[] for i in eachindex(t.nodes)],active=m.active)
        localcounts,_=publish(t,one;samples=8)
        @test localcounts[owner]≈sum(c.mass for c in cs)
    end
    @test counts[1]≈400
    @test sum(p[leafids(t)])≈1
    @test all(n.axis==0 || counts[n.left]+counts[n.right]≈counts[i] for (i,n) in enumerate(t.nodes))
end
