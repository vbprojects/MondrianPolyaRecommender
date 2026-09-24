using Test, LinearAlgebra, Random
include("../src/EmbeddingClustering.jl")
using .EmbeddingClustering

@testset "Matched embedding clustering metrics" begin
    @test adjusted_rand([1,1,2,2],[8,8,9,9]) == 1
    @test adjusted_rand([1,1,2,2],[1,2,1,2]) ≈ -0.5
    @test adjusted_rand([1,1,1],[2,2,2]) == 1
    @test label_metrics([1,1,2,2],[7,7,8,8]).nmi ≈ 1
    @test label_metrics([1,2,1,2],[7,7,8,8]).nmi ≈ 0
    @test topic_metrics([1,1,2,2],[1,1,2,2],[1,2],[1,2]).balanced_accuracy == 1
    # Uniform fallback for an unseen partition uses training-majority ties consistently.
    @test topic_metrics([1,1,2,2],[1,1,2,2],[3,3],[1,2]).balanced_accuracy == .5
    X=Float32[1 1 -1 -1;0 0 0 0]
    @test silhouette(X,[1,1,2,2]) ≈ 1
    @test silhouette(X,[1,1,1,1]) == 0
    @test silhouette(X,[1,2,3,4]) == 0
    @test group_metrics([1,1,2,2]).effective_groups ≈ 2
    rng=MersenneTwister(42)
    Y=hcat(randn(rng,Float32,6,100).*.01 .+ Float32[1,0,0,0,0,0],
           randn(rng,Float32,6,100).*.01 .- Float32[1,0,0,0,0,0])
    Y./=sqrt.(sum(abs2,Y;dims=1))
    f=spherical_kmeans(Y,2)
    @test adjusted_rand(f.assignments,vcat(fill(1,100),fill(2,100))) == 1
    @test all(isfinite,f.centers)
    @test assign_centers(f.centers,Y)==f.assignments
    @test spherical_kmeans(Y,2).assignments==f.assignments
    @test_throws ArgumentError spherical_kmeans(Y.*2,2)
end
