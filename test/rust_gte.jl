using Test, JSON3
include("../src/EmbeddingBridge.jl")
const EB=EmbeddingBridge

@testset "Native GTE INT8 bridge" begin
    # This must work even when no Python executable is available.
    w=EB.Worker(model=EB.GTE_INT8_MODEL,python="/no/python")
    try
        @test w.model==EB.GTE_INT8_MODEL && w.dimension==384
        X=EB.embed(w,["Julia and Rust.","日本語 café 🦋",""])
        @test size(X)==(384,3)
        @test all(isfinite,X)
        @test vec(sum(abs2,X;dims=1))≈ones(3) atol=2e-6
        @test X==EB.embed(w,["Julia and Rust.","日本語 café 🦋",""])
        @test size(EB.embed(w,String[]))==(384,0)
        @test EB.verify_nostr(w,[Dict()])==[false]
        @test isempty(EB.verify_nostr(w,[]))
        @test_throws ErrorException EB.embed(w,["x" for _ in 1:1025])
        @test size(EB.embed(w,["Still usable."]))==(384,1)
    finally
        close(w)
    end
    @test EB.GTE_INT8_MODEL!=EB.GTE_MODEL
    @test_throws ArgumentError EB.Worker(model=EB.GTE_INT8_MODEL,backend="python")
    @test_throws ArgumentError EB.Worker(model=EB.GTE_MODEL,backend="rust")
    @test_throws ArgumentError EB.Worker(model=EB.GTE_INT8_MODEL,device="cuda")
    @test_throws ArgumentError EB.Worker(model=EB.GTE_INT8_MODEL,batch_size=33)
    @test_throws ArgumentError EB.Worker(model="unknown")
end
