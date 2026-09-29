using Test
include("../src/KeywordExtraction.jl")
include("../examples/disk_ablation_common.jl")
const KE=KeywordExtraction
@testset "Native extractor contract and disk integration" begin
    e=KE.JuliaYakeExtractor()
    xs=[
        KE.KeywordInput("1","Space telescopes observe distant galaxies. #NASA",["NASA"],"en-US"),
        KE.KeywordInput("2","Bonjour le monde #Paris",String[],"fr"),
        KE.KeywordInput("3","https://example.test @person",String[],""),
        KE.KeywordInput("4",repeat("galaxy ",2000),String[],"")]
    a=KE.prepare_keywords(e,xs)
    @test !isempty(a[1].terms) && "nasa" in a[1].terms
    @test length(a[1].terms)==length(unique(a[1].terms))
    @test a[2].terms==["paris"]
    @test !a[2].eligible
    @test isempty(a[3].terms) && a[3].assumed_english
    @test a[4].truncated
    @test all(k->k.fingerprint==KE.fingerprint(e),a)
    @test KE.fingerprint(e)!=KE.fingerprint(KE.HashtagsOnlyExtractor())
    @test isempty(KE.prepare_keywords(e,KE.KeywordInput[]))
    @test_throws ErrorException KE.extract_keywords(e,fill(xs[1],1025))
    @test_throws ErrorException KE.extractor_id(KE.JuliaYakeExtractor(KE.Yake.Extractor(n=1)))
    # Every document uses independent statistics; safe to reuse a read-only engine.
    out=Vector{Any}(undef,16)
    Threads.@threads for i in 1:16
        out[i]=KE.prepare_keywords(e,xs[1:3])
    end
    @test all(v->[x.terms for x in v]==[x.terms for x in a[1:3]],out)
    mktempdir() do root
        s=DS.open_store(root;budget=32*1024^2,create_config=DS.config(dimension=2,split_after=16,max_leaves=8,reservoir_cap=16))
        ps=[PostRecord(PostKey(:threads,"julia$i");text=xs[1].text,created_at=0.,observed_at=0.) for i in 1:24]
        X=rand(MersenneTwister(7),Float32,2,24).*1.8f0.-.9f0
        DS.enable_keywords!(s,e)
        DS.append_batch!(s,ps,X;keywords=KE.prepare_keywords(e,KE.keyword_input.(ps)))
        @test DS.one(s,"SELECT COUNT(*) AS n FROM kw_posts").n==24
        n=DS.one(s,"SELECT n FROM kw_counts WHERE leaf=0 AND term=(SELECT id FROM kw_terms WHERE term='nasa')").n
        @test n==24
        DS.clear_caches!(s);DS.close_store(s)
        s=DS.open_store(root;budget=32*1024^2)
        DS.enable_keywords!(s,e)
        @test_throws ErrorException DS.enable_keywords!(s,KE.HashtagsOnlyExtractor())
        DS.enable_keywords!(s,KE.HashtagsOnlyExtractor();rebuild=true)
        DS.backfill_keywords!(s,KE.HashtagsOnlyExtractor())
        DS.enable_keywords!(s,e;rebuild=true)
        @test DS.backfill_keywords!(s,e;batch_size=7,max_batches=1)==7
        DS.close_store(s);s=DS.open_store(root)
        @test DS.backfill_keywords!(s,e;batch_size=7)==17
        @test DS.one(s,"SELECT n FROM kw_counts WHERE leaf=0 AND term=(SELECT id FROM kw_terms WHERE term='nasa')").n==24
        DS.close_store(s)
    end
end
