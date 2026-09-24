using Test
include("../examples/mnist_pca.jl")
@testset "PCA plane clipping" begin
    box=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)]
    p=clip_poly(box,1.,0.,.5)
    @test length(p)==4
    @test all(v->0<=v[1]<=.5 && 0<=v[2]<=1,p)
    @test isempty(clip_poly(box,1.,0.,-1.))
    @test clip_poly(box,0.,0.,1.)==box
    @test isempty(clip_poly(box,0.,0.,-1.))
    tri=clip_poly(box,1.,1.,1.)
    @test all(v->v[1]+v[2]<=1+1e-12,tri)
end
