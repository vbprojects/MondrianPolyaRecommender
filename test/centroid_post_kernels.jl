using Test, Random, LinearAlgebra, Statistics
include("../src/LeafLatentTree.jl")
include("../src/LeafTreeKernels.jl")
using .LeafLatentTrees, .LeafTreeKernels
BLAS.set_num_threads(1)

function checks(k)
    @testset "Immutable centroid codes: $(typeof(k))" begin
        for (d,r) in ((2,1),(32,4),(256,4)), seed in 1:2
            X=clamp.(.2randn(MersenneTwister(seed),Float32,d,160),-.95f0,.95f0)
            c=encode_centroid(k,X);ref=encode_centroid(KernelTreeBackend(),X)
            @test c.center==ref.center && c.scales==ref.scales && c.residuals==ref.residuals
            Y=decode_centroid(k,c)
            @test Y==decode_centroid(KernelTreeBackend(),ref)
            @test all(abs.(Float64.(Y).-X) .<= reshape(Float64.(c.scales)./2 .+ 1e-7,1,:))
            @test centroid_distances(k,c,X[:,1])≈vec(sum(abs2,Float64.(Y).-X[:,1];dims=1)) atol=1e-12
            @test decode_centroid(k,c;indices=[3,7])==Y[:,[3,7]]
            opts=(half_life=1.,configs=[(:ppca,r)],split_after=24,reservoir_cap=24,max_leaves=8,merge_below=2.,merge_interval=.15)
            t=collect(range(0.,.3;length=160))
            f=fit_kernel_tree(k,X;timestamps=t,opts...)
            @test route_centroid(k,f.nodes,c)==[leaf_index(f.nodes,y) for y in eachcol(Y)]
            initialbytes=copy(c.residuals);initialcenter=copy(c.center)
            f=kernel_merge_tree!(k,f;time=20.)
            @test length(f.nodes)==1 && all(==(1),route_centroid(k,f.nodes,c))
            @test c.residuals==initialbytes && c.center==initialcenter
            # Regression: a scheduled merging continuation previously discarded f.
            f=fit_kernel_tree(k,X;timestamps=t.+21,opts...,_initial=f)
            @test f.processed==320 && f.split_count>7 && f.merge_count>0
            @test route_centroid(k,f.nodes,c)==[leaf_index(f.nodes,y) for y in eachcol(Y)]
            @test c.residuals==initialbytes && decode_centroid(k,c)==Y
            # Weighted recurrence is checked against direct weighted moments.
            w=rand(MersenneTwister(81),160);w[1:5].=0
            groups=repeat(1:2,80);states=[Sketch(:ppca,r,d) for _ in 1:2]
            kernel_weighted_update!(k,states,X,groups,w)
            for i in 1:2
                ids=findall(==(i),groups);W=w[ids];A=Float64.(X[:,ids]);mu=A*W/sum(W)
                @test states[i].mass≈sum(W)
                @test states[i].mean≈mu atol=1e-12
                @test states[i].diagonal≈(abs2.(A.-mu)*W) atol=1e-12
                @test length(states[i].eigenvalues)<=r
            end
            cpu=[Sketch(:ppca,r,d) for _ in 1:2];kernel_weighted_update!(KernelTreeBackend(),cpu,X,groups,w)
            for i in 1:2
                p=parameters(cpu[i]);q=parameters(states[i])
                @test p.W*p.W'+Diagonal(p.D)≈q.W*q.W'+Diagonal(q.D) atol=1e-10
            end
        end
        c=encode_centroid(k,fill(.25f0,4,20));@test decode_centroid(k,c)==fill(.25f0,4,20)
        @test isempty(decode_centroid(k,c;indices=Int[]))
        @test_throws BoundsError decode_centroid(k,c;indices=[21])
        @test_throws ArgumentError encode_centroid(k,[NaN;;])
    end
end
checks(KernelTreeBackend())
if "--gpu" in ARGS
    using CUDA
    CUDA.functional() || error("CUDA unavailable");CUDA.allowscalar(false)
    checks(KernelTreeBackend(CUDA.CuArray))
end
