using Test
include("../examples/embedding_comparison.jl")

@testset "Embedding comparison protocol helpers" begin
    a=word_shingles("Choose twenty books that have stayed with you or influenced you. Day 10. @alice https://example.org/a")
    b=word_shingles("Choose twenty books that have stayed with you or influenced you. Day 20. @bob https://example.org/b")
    @test template_overlap(a,b)
    @test !template_overlap(a,word_shingles("The red fox climbed over the old wooden fence near the river"))
    @test !template_overlap(word_shingles("Hello how are you"),word_shingles("Hello how are you"))
    @test isempty(word_shingles("Hi"))
    @test macro_average([1.,0.,0.],[1,2,2])==.5
    X=Float32[-3 3 0 0;0 0 -1 1]
    p=projection(X,X)
    @test p.explained≈1
    @test sum(x[1]^2 for x in p.points)>sum(x[2]^2 for x in p.points)
end
