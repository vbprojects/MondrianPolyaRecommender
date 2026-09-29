using Test, Random
include("../bin/rebuild_keywords.jl")
using .PostRecords
@testset "Hashtag configuration and resumable offline migration" begin
    @test KE.configured_extractor(Dict("method"=>"hashtags")) isa KE.HashtagsOnlyExtractor
    @test KE.configured_extractor(Dict()) isa KE.JuliaYakeExtractor
    @test_throws ArgumentError KE.configured_extractor(Dict("method"=>"typo"))
    xs=[KE.KeywordInput("one","no unmarked phrases #Space #SPACE #Café",["space","julia"],"fr"),
        KE.KeywordInput("two","no hashtags here",String[],"en")]
    ks=KE.prepare_keywords(KE.HashtagsOnlyExtractor(),xs)
    @test ks[1].terms==["space","julia","café"]
    @test isempty(ks[2].terms)
    mktempdir() do root
        path=joinpath(root,"store");cfg=joinpath(root,"config.toml")
        write(cfg,"[keywords]\nmethod = \"hashtags\"\n")
        s=DS.open_store(path;create_config=DS.config(dimension=4,split_after=16,max_leaves=4))
        ps=[PostRecord(PostKey(:threads,string(i));text="Distant galaxies and stars #Space",tags=["SPACE"],observed_at=100.,created_at=100.) for i in 1:160]
        DS.enable_keywords!(s,KE.JuliaYakeExtractor())
        DS.append_batch!(s,ps,rand(MersenneTwister(5),Float32,4,160);time=100.,
            keywords=KE.prepare_keywords(KE.JuliaYakeExtractor(),KE.keyword_input.(ps)))
        before=DS.tree_signature(s);X=DS.fetch_vectors(s,collect(1:160));DS.close_store(s)
        main(["--store",path,"--config",cfg,"--rebuild","--max-batches","1"])
        s=DS.open_store(path)
        @test DS.keyword_stats(s).covered==128
        DS.close_store(s)
        main(["--store",path,"--config",cfg])
        s=DS.open_store(path)
        try
            @test DS.keyword_stats(s).coverage==1
            @test DS.tree_signature(s)==before
            @test DS.fetch_vectors(s,collect(1:160))==X
            @test [r.term for r in DS.rows(s,"SELECT term FROM kw_terms")]==["space"]
            r=DS.one(s,"SELECT n,mass FROM kw_counts WHERE leaf=0 AND term>0")
            @test r.n==160 && r.mass≈160
            @test DS.audit!(s)!==nothing
        finally;DS.close_store(s);end
    end
end
