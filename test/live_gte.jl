using Test, JSON3, LinearAlgebra
include("../examples/disk_live_modules.jl")
include("../src/EmbeddingBridge.jl")
const DS=DiskTreeStore
const EB=EmbeddingBridge
using .PostRecords
@testset "GTE Julia bridge and Float16 persistence" begin
    w=EB.Worker(model=EB.GTE_MODEL)
    try
        @test w.dimension==384 && w.model==EB.GTE_MODEL
        X=EB.embed(w,["Space telescopes reveal distant stars.","Gardening and plants.","#Julia numerical computing"])
        @test size(X)==(384,3)
        @test all(isfinite,X)
        @test vec(sum(abs2,X;dims=1))≈ones(3) atol=1e-5
        @test size(EB.embed(w,String[]))==(384,0)
        mktempdir() do root
            s=DS.open_store(root;create_config=DS.config(dimension=384,model=EB.GTE_MODEL),budget=32*1024^2)
            try
                ps=[PostRecord(PostKey(:threads,"gte-$i");text="test $i",observed_at=100.,created_at=100.) for i in 1:3]
                DS.append_batch!(s,ps,X;time=100.)
                @test maximum(abs.(DS.fetch_vectors(s,collect(1:3)).-X))<0.001
                @test DS.audit!(s)!==nothing
            finally;DS.close_store(s);end
            s=DS.open_store(root)
            try
                @test s.config.model==EB.GTE_MODEL && s.config.dimension==384
                @test size(DS.fetch_vectors(s,collect(1:3)))==(384,3)
            finally;DS.close_store(s);end
        end
    finally;close(w);end
    @test_throws ArgumentError EB.Worker(model="unrecognized")
end

@testset "Configured encoder cannot resume a different store" begin
    mktempdir() do root
        s=DS.open_store(root;create_config=DS.config(dimension=256,model=EB.QWEN_MODEL))
        before=DS.one(s,"SELECT value FROM meta WHERE key='runtime'").value
        DS.close_store(s)
        project=dirname(Base.active_project())
        config=normpath(joinpath(@__DIR__,"../examples/live_sources.toml"))
        entry=normpath(joinpath(@__DIR__,"../examples/ingest_live.jl"))
        output=IOBuffer()
        p=run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$project $entry --storage disk --config $config --store $root --rounds 1`);stdout=output,stderr=output))
        @test p.exitcode!=0
        @test occursin("stored configuration/model mismatch",String(take!(output)))
        s=DS.open_store(root)
        try
            @test DS.one(s,"SELECT value FROM meta WHERE key='runtime'").value==before
            @test !DS.keyword_enabled(s)
        finally;DS.close_store(s);end
    end
end
