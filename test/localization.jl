@testset "Static localization: graph, feedback and Bayesian accounting" begin
    land=static_landscape(grid_size=7,nposts=500)
    @test sum(land.prior)≈1
    @test all(isfinite,land.distances)
    @test land.distances≈transpose(land.distances)
    @test all(land.distances[i,i]==0 for i in eachindex(land.prior))
    @test all(.05 .< land.likelihood .<= .95+1e-12)
    @test all(i in land.candidates for i in land.representatives)
    @test response_probability(0.,1.)≈.95
    @test response_probability(2.,1.)<response_probability(1.,1.)
    b=[.2,.8];posterior_update!(b,[.9,.1],true)
    @test b≈[.18/.26,.08/.26]
    @test query_information([.5,.5],[.5,.5])≈0
    @test query_information([.5,.5],[.99,.01])>query_information([.5,.5],[.6,.4])
    for i in 1:7,j in 8:14,k in 15:21
        @test land.distances[i,k]<=land.distances[i,j]+land.distances[j,k]+1e-12
    end
    original=copy(land.prior)
    for policy in (:random,:greedy,:information,:tree_information)
        r=localization_trial(land,policy;seed=9,budget=12,history=true)
        @test 1<=r.used<=12
        @test length(r.history)==13
        @test all(isapprox(sum(s.belief),1.) for s in r.history)
        @test all(s.query in land.candidates for s in r.history[2:end])
        if r.stop>0
            @test r.history[r.stop+1].confidence>=.9
            @test r.correct==(r.history[r.stop+1].error<=land.radius)
            @test all(s.confidence<.9 for s in r.history[2:r.stop])
        end
    end
    @test land.prior==original
    a=localization_trial(land,:information;target=1,budget=1,history=true)
    b=localization_trial(land,:information;target=49,budget=1,history=true)
    @test a.history[2].query==b.history[2].query
end

@testset "Tree-guided Fermat proposals and accounting" begin
    land=static_landscape(grid_size=9,nposts=700)
    for (a,b) in ((1,81),(8,60),(25,25),(42,9))
        route=M.fermat_path(land,a,b)
        @test first(route)==a && last(route)==b
        @test length(unique(route))==length(route)
        @test sum((land.distances[u,v] for (u,v) in zip(route[1:end-1],route[2:end]));init=0.)≈land.distances[a,b]
        @test M.path_fraction(land,route,0.)==a
        @test M.path_fraction(land,route,1.)==b
        for f in (.2,.35,.5,.65,.8)
            q=M.path_fraction(land,route,f)
            @test abs(land.distances[a,q]-f*land.distances[a,b])≈minimum(abs(land.distances[a,v]-f*land.distances[a,b]) for v in route)
        end
    end
    for policy in (:tree_path_information,:tree_midpoint)
        choice=M.query_choice(MersenneTwister(19),land.prior,land,policy)
        @test choice.query in land.candidates
        @test all(q in land.candidates for q in choice.proposals)
        @test choice.evaluated==length(unique(choice.proposals))
        @test 1<=choice.evaluated<=(policy==:tree_midpoint ? 4 : 28)
        score=q->query_information(land.prior,view(land.likelihood,q,:))
        @test score(choice.query)≈maximum(score(q) for q in choice.proposals)
        @test score(choice.query)<=maximum(score(q) for q in land.candidates)+1e-12
        r=localization_trial(land,policy;budget=15,history=true)
        @test r.evaluated==sum(s.evaluated for s in r.history[2:r.used+1])
        @test r.path_edges==sum(sum(length(p)-1 for p in s.routes) for s in r.history[2:r.used+1])
        @test r.selection_seconds>=0
        single=zeros(length(land.prior));single[20]=1.
        fallback=M.query_choice(MersenneTwister(1),single,land,policy)
        @test fallback.query==land.nearest_post[20]
        a=localization_trial(land,policy;target=1,budget=1,history=true)
        b=localization_trial(land,policy;target=81,budget=1,history=true)
        @test a.history[2].query==b.history[2].query
    end
end
