using Test, Random, LinearAlgebra, KernelAbstractions
include("../src/LeafLatentTree.jl")
include("../src/LeafTreeKernels.jl")
using .LeafLatentTrees, .LeafTreeKernels
const LT=LeafLatentTrees
function covariance(s)
    p=parameters(s);p.W*p.W'+Diagonal(p.D)
end
function check_backend(k;gpu=false)
    @testset "Kernel numerical operations" begin
        rng=MersenneTwister(51)
        for (d,r) in ((2,1),(12,4),(256,4),(3,0))
            reference=[Sketch(r==0 ? :diagonal : :ppca,r,d) for _ in 1:8];states=deepcopy(reference)
            for step in 1:35
                X=.1randn(rng,d,8);elapsed=fill(step==20 ? 1000. : .2,8)
                for j in 1:8;LT.advance_state!(reference[j],X[:,j],elapsed[j],2.);end
                update_sketches!(k,states,X,elapsed,2.)
                for j in 1:8
                    @test states[j].mass≈reference[j].mass
                    @test states[j].mean≈reference[j].mean atol=1e-12
                    @test states[j].diagonal≈reference[j].diagonal atol=1e-12
                    @test covariance(states[j])≈covariance(reference[j]) atol=2e-9 rtol=2e-7
                end
            end
        end
    end
    @testset "Full adaptive tree parity" begin
        rng=MersenneTwister(16);X=clamp.(.24randn(rng,2,1200),-.95,.95);X[:,601:end].+=.3
        ts=collect(1:1200)./100;ts[801:end].+=5
        for half in (Inf,1.)
            kwargs=(timestamps=ts,half_life=half,configs=[(:ppca,1)],max_leaves=16,split_after=24,reservoir_cap=24,seed=19)
            ref=fit_leaf_tree(X;kwargs...);f=fit_kernel_tree(k,X;kwargs...)
            @test [(n.axis,n.cut,n.left,n.right) for n in f.nodes]==[(n.axis,n.cut,n.left,n.right) for n in ref.nodes]
            @test f.last_times==ref.last_times
            @test kernel_route(k,f.nodes,X)==[leaf_index(ref.nodes,x) for x in eachcol(X)]
            for (n,a) in zip(f.nodes,ref.nodes)
                @test n.reservoir==a.reservoir
                if n.axis==0
                    @test n.states[1].mass≈a.states[1].mass rtol=2e-7 atol=1e-10
                    @test n.states[1].mean≈a.states[1].mean atol=2e-7
                    @test covariance(n.states[1])≈covariance(a.states[1]) atol=2e-7
                    samples,attempts=kernel_sample_leaf(k,MersenneTwister(82),n,n.states[1],5;max_draws=128)
                    actual,ntries=sample_leaf(MersenneTwister(82),n,n.states[1],5;max_draws=128)
                    @test attempts==ntries
                    @test length(samples)==length(actual)
                    @test all(isapprox(s,a;atol=1e-12) for (s,a) in zip(samples,actual))
                    @test all(all(n.lo.<=s)&&all(s.<n.hi) for s in samples)
                end
            end
            for time in (f.time,f.time+1000),prior in (0.,2.)
                q=kernel_probabilities(k,f;time=time,kappa=prior);a=leaf_probabilities(ref;time=time,kappa=prior)
                @test q.masses≈a.masses atol=1e-7 rtol=2e-7
                @test q.probabilities≈a.probabilities atol=1e-7
                @test sum(q.probabilities)≈1
            end
        end
    end
end
BLAS.set_num_threads(1)
check_backend(KernelTreeBackend())
check_backend(KernelTreeBackend(Array;eigensolver=lapack_eigen_batch))
if "--gpu" in ARGS
    using CUDA
    CUDA.functional() || error("CUDA unavailable; GPU verification requested")
    CUDA.allowscalar(false)
    check_backend(KernelTreeBackend(CUDA.CuArray);gpu=true)
    check_backend(KernelTreeBackend(CUDA.CuArray;eigensolver=C->CUDA.cuSOLVER.syevjBatched!('V','U',C));gpu=true)
end
