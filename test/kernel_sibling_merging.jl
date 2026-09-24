using Test, Random, LinearAlgebra
include("../src/LeafLatentTree.jl")
include("../src/LeafTreeKernels.jl")
using .LeafLatentTrees, .LeafTreeKernels
function compare(a,b)
    @test [(n.axis,n.cut,n.left,n.right) for n in a.nodes]==[(n.axis,n.cut,n.left,n.right) for n in b.nodes]
    @test a.merge_count==b.merge_count
    @test a.split_count==b.split_count
    @test a.reservoir_rows==b.reservoir_rows
    @test a.time==b.time && a.last_times==b.last_times
    @test a.effective_arrivals≈b.effective_arrivals
    for c in eachindex(a.configs)
        @test leaf_probabilities(a;config=c).probabilities≈leaf_probabilities(b;config=c).probabilities atol=1e-8
        @test leaf_probabilities(a;config=c).masses≈leaf_probabilities(b;config=c).masses atol=1e-8
        for (n,m) in zip(a.nodes,b.nodes)
            @test n.reservoir==m.reservoir
            n.axis==0 || continue
            s=n.states[c];t=m.states[c]
            @test s.mass≈t.mass atol=1e-8
            @test s.mean≈t.mean atol=1e-8
            @test s.diagonal≈t.diagonal atol=1e-8
            p=parameters(s);q=parameters(t)
            @test p.W*p.W'+Diagonal(p.D)≈q.W*q.W'+Diagonal(q.D) atol=1e-8
        end
    end
end
function checks(k;external=false)
    modes=external ? (:event,) : (:event,:scheduled)
    @testset "Sibling kernels: $(typeof(k))" begin
        for execution in modes, (d,r) in ((2,1),(12,4)), seed in 1:2
            X=clamp.(.2randn(MersenneTwister(seed),d,240),-.95,.95)
            # Exercise real-time maintenance, unequal child clocks, collapse, and regrowth.
            t=vcat(collect(range(0.,.4;length=80)),collect(range(15.,15.4;length=80)),collect(range(30.,30.4;length=80)))
            opts=(timestamps=t,half_life=1.,configs=[(:ppca,r),(:diagonal,0),(:lowrank_diagonal,r)],
                split_after=24,max_leaves=4,reservoir_cap=24,seed=seed,merge_below=3.,merge_interval=.2)
            ref=fit_leaf_tree(X;opts...);f=fit_kernel_tree(k,X;execution,opts...)
            @test f.merge_count>0 && f.split_count>3
            execution==:scheduled && @test f.schedule_epochs>=3
            compare(f,ref)
            @test kernel_route(k,f.nodes,X)==[leaf_index(ref.nodes,x) for x in eachcol(X)]
            for c in 1:3
                @test kernel_probabilities(k,f;config=c).probabilities≈leaf_probabilities(ref;config=c).probabilities atol=1e-8
            end
            ref=merge_tree!(ref;time=60.)
            f=kernel_merge_tree!(k,f;time=60.)
            @test length(f.nodes)==1
            compare(f,ref)
            points,_=kernel_sample_leaf(k,MersenneTwister(31),f.nodes[1],f.nodes[1].states[1],10)
            @test length(points)==10 && all(x->all(-1 .<= x .< 1),points)
            # All evidence can vanish without NaNs.
            f=kernel_merge_tree!(k,f;time=1e8)
            @test f.nodes[1].states[1].mass==0
            @test all(isfinite,f.nodes[1].states[1].diagonal)
        end
        if !external
            # No cut can be proposed, but checkpoints must still execute.
            seen=Int[];events=Int[]
            X=zeros(2,60)
            f=fit_kernel_tree(k,X;timestamps=collect(1.:60.),half_life=100.,configs=[(:ppca,1)],split_after=16,
                merge_below=2.,observe_every=10,on_checkpoint=(row,f)->push!(seen,row),on_merge=(row,n,ns)->push!(events,n))
            @test seen==collect(10:10:60) && isempty(events)
        end
    end
end
BLAS.set_num_threads(1)
checks(KernelTreeBackend())
checks(KernelTreeBackend(Array;eigensolver=lapack_eigen_batch);external=true)
if "--gpu" in ARGS
    using CUDA
    CUDA.functional() || error("CUDA unavailable");CUDA.allowscalar(false)
    checks(KernelTreeBackend(CUDA.CuArray))
    checks(KernelTreeBackend(CUDA.CuArray;eigensolver=C->CUDA.cuSOLVER.syevjBatched!('V','U',C));external=true)
end
