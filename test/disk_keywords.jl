using Test, Random, LinearAlgebra, JSON3
include("../src/KeywordExtraction.jl")
include("../examples/disk_ablation_common.jl")
const KE=DS.KeywordExtraction
using .DiskTreeStore.KeywordExtraction
struct TestExtractor <: AbstractKeywordExtractor end
KE.extractor_id(::TestExtractor)="deterministic-test-v1"
KE.extract_keywords(e::TestExtractor,xs)=[KeywordResult([Candidate(t,i,nothing) for (i,t) in enumerate(split(x.text))],extractor_id(e),false) for x in xs]
prepared(ps,e=TestExtractor())=prepare_keywords(e,keyword_input.(ps))
function ps(n;t=0.)
    [PostRecord(PostKey(:threads,"p$i");text=i%2==0 ? "space telescope stars" : "garden plants soil",tags=["Topic$(i%3)"],created_at=t,observed_at=t) for i in 1:n]
end
function exact(s)
    @test DS.one(s,"SELECT COUNT(*) AS n FROM kw_posts k LEFT JOIN posts p ON p.id=k.id WHERE p.id IS NULL OR p.status!=1").n==0
    expected=Dict{Tuple{Int,Int},Tuple{Int,Float64}}()
    for p in DS.rows(s,"SELECT p.id,p.leaf,p.density_at,k.terms,k.eligible FROM posts p JOIN kw_posts k ON k.id=p.id WHERE p.status=1")
        terms=vcat([-1],p.eligible==1 ? [0] : Int[],DS.deserialize_bytes(p.terms))
        w=exp2(-(s.clock-p.density_at)/s.keyword_config.half_life)
        for scope in (0,p.leaf),term in terms
            n,m=get(expected,(scope,term),(0,0.));expected[(scope,term)]=(n+1,m+w)
        end
    end
    actual=DS.rows(s,"SELECT * FROM kw_counts")
    @test length(actual)==length(expected)
    for r in actual
        n,m=expected[(r.leaf,r.term)]
        @test r.n==n
        @test isapprox(r.mass*exp2(-(s.clock-r.at)/s.keyword_config.half_life),m;atol=1e-9,rtol=1e-9)
    end
end
@testset "Extractor contract and canonicalization" begin
    @test Main.KeywordExtraction === DS.KeywordExtraction
    x=KeywordInput("1","SPACE #Space #Café https://example.test @person",["SPACE","cafe\u0301"],"")
    k=prepare_keywords(TestExtractor(),[x])[1]
    @test length(k.terms)==length(unique(k.terms));@test "café" in k.terms
    @test k.assumed_english
    @test count(==("space"),k.terms)==1 && !("#space" in k.terms)
    nonen=KeywordInput("2","bonjour monde #Paris",String[],"fr")
    @test prepare_keywords(TestExtractor(),[nonen])[1].terms==["paris"]
    e=PythonYakeExtractor()
    try
        xs=[KeywordInput("3","Space telescopes observe distant galaxies and stars.",String[],"en"),nonen,KeywordInput("4","https://example.test @person",String[],"en"),KeywordInput("5",repeat("galaxy ",2000),String[],"")]
        a=prepare_keywords(e,xs)
        @test !isempty(a[1].terms);@test a[2].terms==["paris"];@test isempty(a[3].terms);@test a[4].truncated
        @test prepare_keywords(e,xs)[1].terms==a[1].terms
        @test fingerprint(e)==a[1].fingerprint
    finally;close(e);end
end
@testset "Disk lifecycle and exact statistics" begin
    mktempdir() do root
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2,create_config=DS.config(dimension=2,split_after=16,max_leaves=8,reservoir_cap=16,half_life=10.,retention_seconds=100.))
        rs=ps(160);X=rand(MersenneTwister(55),Float32,2,160).*1.8f0.-0.9f0
        DS.enable_keywords!(s,TestExtractor())
        @test_throws ErrorException DS.append_batch!(s,rs,X)
        @test_throws ErrorException DS.append_batch!(s,rs,X;keywords=reverse(prepared(rs)))
        DS.append_batch!(s,rs,X;keywords=prepared(rs));exact(s)
        before=DS.tree_signature(s)
        DS.append_batch!(s,rs,X;keywords=prepared(rs),time=2.);exact(s)
        @test DS.tree_signature(s)==before
        for leaf in sort([n.id for n in values(s.nodes) if n.axis==0])
            summary=DS.leaf_keywords(s,leaf;ranking=:common)
            @test summary.coverage==1
            @test DS.leaf_keywords(s,leaf;ranking=:common)==summary
        end
        p=PostRecord(rs[1].key;text=rs[1].text,tags=["NewTag"],created_at=0.,observed_at=3.,revision_at=3.)
        density_at=DS.lookup(s,p.key).density_at
        DS.keyword_update!(s,[p],prepared([p]);time=3.)
        @test DS.lookup(s,p.key).density_at==density_at
        terms=DS.deserialize_bytes(DS.one(s,"SELECT terms FROM kw_posts WHERE id=1").terms)
        @test DS.one(s,"SELECT id FROM kw_terms WHERE term='newtag'").id in terms
        exact(s)
        edit=PostRecord(rs[2].key;text="new physics topic",created_at=0.,observed_at=4.,revision_at=4.)
        DS.append_batch!(s,[edit],X[:,2:2];keywords=prepared([edit]),time=4.);exact(s)
        DS.delete_post!(s,rs[3].key;revision_at=5.,time=5.);exact(s)
        DS.clear_caches!(s);DS.close_store(s);s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2);exact(s)
        # Repeated deterministic forced splits / merges stress contributions independently of PPCA.
        for cycle in 1:3
            DS.maintain!(s;time=60. + cycle)
            exact(s)
            leaf=first(sort([n.id for n in values(s.nodes) if n.axis==0]))
            cut=median(DS.get_state(s,leaf).reservoir[1,:])
            DS.commit!(s) do;DS.split_leaf!(s,leaf,1,cut);end
            exact(s)
        end
        DS.close_store(s);s=DS.open_store(root)
        leaf=first([n.id for n in values(s.nodes) if n.axis==0])
        @test DS.leaf_keywords(s,leaf;time=101.).total==0
        @test DS.keyword_stats(s).covered==0;exact(s)
        DS.collect_garbage!(s)
        @test DS.keyword_stats(s).terms==0
        DS.close_store(s)
    end
end
@testset "Backfill, rebuild and rollback" begin
    mktempdir() do root
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=DS.config(dimension=2))
        rs=ps(24);X=fill(0.2f0,2,24);DS.append_batch!(s,rs,X)
        DS.enable_keywords!(s,TestExtractor())
        @test DS.backfill_keywords!(s,TestExtractor();batch_size=7,max_batches=1)==7
        @test DS.keyword_stats(s).coverage==7/24
        DS.close_store(s);s=DS.open_store(root)
        @test DS.backfill_keywords!(s,TestExtractor();batch_size=7)==17
        exact(s)
        @test_throws ErrorException DS.enable_keywords!(s,HashtagsOnlyExtractor())
        p=PostRecord(rs[1].key;text=rs[1].text,tags=["rollback"],created_at=0.,observed_at=1.,revision_at=1.)
        @test_throws ErrorException DS.keyword_update!(s,[p],prepared([p]);time=1.,fault=:before_commit)
        DS.close_store(s);s=DS.open_store(root);exact(s)
        @test DS.lookup(s,p.key).revision==0.
        DS.enable_keywords!(s,HashtagsOnlyExtractor();rebuild=true)
        @test DS.keyword_stats(s).covered==0
        DS.backfill_keywords!(s,HashtagsOnlyExtractor());exact(s)
        DS.close_store(s)
    end
end
@testset "Forced eviction and empty/language coverage" begin
    mktempdir() do root
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2,create_config=DS.config(dimension=2,max_leaves=8,split_after=16,reservoir_cap=16))
        s.block_cache.capacity=2048;s.state_cache.capacity=8192
        DS.enable_keywords!(s,HashtagsOnlyExtractor())
        rng=MersenneTwister(61)
        for batch in 1:20
            rs=[PostRecord(PostKey(:threads,"b$(batch)p$i");text=i%2==0 ? "#space #space" : "plain",metadata=i%3==0 ? "{\"language\":\"fr\"}" : "{}",created_at=0.,observed_at=0.) for i in 1:32]
            DS.append_batch!(s,rs,rand(rng,Float32,2,32).*1.8f0.-0.9f0;keywords=prepare_keywords(HashtagsOnlyExtractor(),keyword_input.(rs)))
        end
        DS.fetch_vectors(s,collect(1:100));exact(s)
        @test s.block_cache.evictions>0
        @test DS.kw_count(s,0,-1,s.clock).n==640
        @test DS.kw_count(s,0,0,s.clock).n==440
        DS.close_store(s)
    end
end

struct BrokenExtractor <: AbstractKeywordExtractor end
KE.extractor_id(::BrokenExtractor)="broken-test"
KE.extract_keywords(::BrokenExtractor, inputs)=error("injected extractor failure")
struct MisalignedExtractor <: AbstractKeywordExtractor end
KE.extractor_id(::MisalignedExtractor)="misaligned-test"
KE.extract_keywords(::MisalignedExtractor, inputs)=KeywordResult[]
@testset "Extraction failure is not empty success" begin
    inputs=keyword_input.(ps(2))
    @test_throws ErrorException prepare_keywords(BrokenExtractor(),inputs)
    @test_throws ErrorException prepare_keywords(MisalignedExtractor(),inputs)
    @test isempty(prepare_keywords(HashtagsOnlyExtractor(),KeywordInput[]))
    @test fingerprint(TestExtractor())!=fingerprint(HashtagsOnlyExtractor())
end
