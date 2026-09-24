using Test, Random, LinearAlgebra
include("../src/LeafLatentTree.jl")
include("../src/LeafTreeKernels.jl")
using .LeafLatentTrees, .LeafTreeKernels
function check_scheduled(k)
    @testset "Scheduled stream equivalence" begin
        for seed in 1:3,(d,r) in ((2,1),(16,4),(16,8)),half in (Inf,.2,2.)
            rng=MersenneTwister(seed);X=clamp.(.2randn(rng,d,750),-.95,.95)
            X[:,401:end].+=.25;X=clamp.(X,-.95,.95)
            t=cumsum(rand(rng,750)./50);t[300:end].+=3
            configs=[(:ppca,r),(:diagonal,0),(:lowrank_diagonal,r)]
            kwargs=(timestamps=t,half_life=half,configs=configs,split_after=24,max_leaves=20,reservoir_cap=24,seed=seed)
            f=fit_kernel_tree(k,X;kwargs...);ref=fit_leaf_tree(X;kwargs...)
            @test [(n.axis,n.cut,n.left,n.right) for n in f.nodes]==[(n.axis,n.cut,n.left,n.right) for n in ref.nodes]
            @test f.last_times==ref.last_times
            @test f.effective_arrivals==ref.effective_arrivals
            @test all(n.reservoir==a.reservoir for (n,a) in zip(f.nodes,ref.nodes))
            for config in eachindex(configs)
                a=leaf_probabilities(f;config);b=leaf_probabilities(ref;config)
                @test a.probabilities≈b.probabilities atol=1e-8
                @test a.masses≈b.masses atol=1e-8
                for (n,refn) in zip(f.nodes,ref.nodes)
                    n.axis==0 || continue
                    s=n.states[config];rs=refn.states[config];p=parameters(s);q=parameters(rs)
                    @test s.mean≈rs.mean atol=1e-8
                    @test s.diagonal≈rs.diagonal atol=1e-8
                    @test p.W*p.W'+Diagonal(p.D)≈q.W*q.W'+Diagonal(q.D) atol=1e-8
                end
            end
        end
        for X in (zeros(2,100),fill(.4,2,100)),half in (Inf,.1)
            kwargs=(configs=[(:ppca,1)],timestamps=collect(1.:100.),half_life=half,split_after=24)
            a=fit_kernel_tree(k,X;kwargs...);b=fit_leaf_tree(X;kwargs...)
            @test length(a.nodes)==length(b.nodes)==1
            @test a.nodes[1].states[1].mass≈b.nodes[1].states[1].mass
        end
        # Exercise the global-scratch fallback above the cooperative shared-memory cap.
        X=.01randn(MersenneTwister(77),1024,60)
        opts=(configs=[(:ppca,4)],max_leaves=2,split_after=24,timestamps=collect(1.:60.),half_life=100.)
        large=fit_kernel_tree(k,X;opts...);baseline=fit_leaf_tree(X;opts...)
        @test leaf_probabilities(large).probabilities≈leaf_probabilities(baseline).probabilities atol=1e-8
        for (n,a) in zip(large.nodes,baseline.nodes)
            n.axis==0 || continue
            p=parameters(n.states[1]);q=parameters(a.states[1])
            @test p.mean≈q.mean atol=1e-8
            @test p.W*p.W'+Diagonal(p.D)≈q.W*q.W'+Diagonal(q.D) atol=1e-8
        end
        # Callback execution must receive real prefix states, never plan placeholders.
        seen=[];X=.1randn(MersenneTwister(3),2,100)
        f=fit_kernel_tree(k,X;configs=[(:ppca,1)],max_leaves=1,on_checkpoint=(i,p)->push!(seen,(i,sum(n.states[1].mass for n in p.nodes))),observe_every=25)
        @test seen==[(25,25.),(50,50.),(75,75.),(100,100.)]
    end
end
BLAS.set_num_threads(1)
check_scheduled(KernelTreeBackend())
if "--gpu" in ARGS
    using CUDA;CUDA.functional() || error("CUDA unavailable");CUDA.allowscalar(false)
    check_scheduled(KernelTreeBackend(CUDA.CuArray))
end
