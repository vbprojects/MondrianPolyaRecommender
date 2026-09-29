using Test, Random, JSON3
include("../examples/disk_live_modules.jl")
include("../src/SourceAdapters.jl")
include("../src/DiskIngestionRunner.jl")
using .PostRecords
const DS=DiskTreeStore
const DR=DiskIngestionRunner
const KE=KeywordExtraction
record(id,text="Space telescopes galaxies #NASA";at=100.,revision=at,tags=["NASA"],aliases=PostKey[])=
    PostRecord(PostKey(:threads,string(id));text,created_at=100.,observed_at=at,revision_at=revision,tags,aliases)
batch(events,cursor="1")=(events=PostEvent[events...],cursor=cursor,received=length(events),complete=true)
function exact(s)
    expected=Dict{Tuple{Int,Int},Tuple{Int,Float64}}()
    for p in DS.rows(s,"SELECT p.leaf,p.density_at,k.terms,k.eligible FROM posts p JOIN kw_posts k ON k.id=p.id WHERE p.status=1")
        terms=vcat([-1],p.eligible==1 ? [0] : Int[],DS.deserialize_bytes(p.terms))
        w=exp2(-(s.clock-p.density_at)/s.keyword_config.half_life)
        for leaf in (0,p.leaf),term in terms
            n,m=get(expected,(leaf,term),(0,0.));expected[(leaf,term)]=(n+1,m+w)
        end
    end
    actual=DS.rows(s,"SELECT * FROM kw_counts")
    @test length(actual)==length(expected)
    for r in actual
        n,m=expected[(r.leaf,r.term)]
        @test r.n==n
        @test isapprox(r.mass*exp2(-(s.clock-r.at)/s.keyword_config.half_life),m;atol=1e-9)
    end
    @test DS.audit!(s)!==nothing
end
@testset "Disk live publication, keywords, replay and maintenance" begin
    mktempdir() do root
        cfg=DS.config(dimension=4,split_after=16,reservoir_cap=16,max_leaves=8,half_life=30.,retention_seconds=1000.)
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2,create_config=cfg)
        e=KE.JuliaYakeExtractor();DS.enable_keywords!(s,e)
        calls=Ref(0)
        embed=texts->begin calls[]+=length(texts);rand(MersenneTwister(12),Float32,4,length(texts)).*1.8f0.-.9f0 end
        ps=[record(i,isodd(i) ? "Space telescopes galaxies #NASA" : "Garden plants flowers #Plants") for i in 1:64]
        r=DR.commit_batch!(s,"test",batch(PostEvent.(ps)),embed;now=100.)
        @test calls[]==64 && r.counts.encoded==64
        @test DR.source_cursor(s,"test")=="1"
        @test length(s.nodes)>1
        exact(s)
        DR.commit_batch!(s,"test",batch(PostEvent.(ps),"2"),x->error("replay embedded");now=101.)
        @test DR.source_cursor(s,"test")=="2"
        original=DS.lookup(s,ps[1].key)
        tags=record(1;at=102.,tags=["Physics"])
        DR.commit_batch!(s,"test",batch([PostEvent(tags)],"tags"),x->error("tag edit embedded");now=102.)
        @test DS.lookup(s,tags.key).density_at==original.density_at
        @test DS.lookup(s,tags.key).block==original.block
        @test DS.one(s,"SELECT n FROM kw_counts WHERE leaf=0 AND term=(SELECT id FROM kw_terms WHERE term='physics')").n==1
        exact(s)
        changed=record(1,"Ocean marine fish";at=103.,tags=String[])
        DR.commit_batch!(s,"test",batch([PostEvent(changed)],"edit"),embed;now=103.)
        @test calls[]==65 && DS.lookup(s,changed.key).density_at==103.
        deletion=PostEvent(ps[2].key;observed_at=104.)
        DR.commit_batch!(s,"test",batch([deletion,PostEvent(ps[2])],"delete"),embed;now=104.)
        @test DS.lookup(s,ps[2].key).status==2
        @test_throws KeyError DS.fetch_records(s,[DS.lookup(s,ps[2].key).id])
        exact(s)
        # Retained vectors and text remain identical after physical relocation.
        ids=[p.id for p in DS.rows(s,"SELECT id FROM posts WHERE status=1 ORDER BY id")]
        X=DS.fetch_vectors(s,ids);texts=JSON3.write(DS.fetch_records(s,ids))
        @test DR.reclaim_daily!(s)
        @test !DR.reclaim_daily!(s)
        @test DS.fetch_vectors(s,ids)==X
        @test JSON3.write(DS.fetch_records(s,ids))==texts
        @test get(s.metrics,"text_compaction_bytes",0)>0
        @test DS.one(s,"SELECT COUNT(*) AS n FROM files f WHERE f.kind='texts' AND f.bytes!=(SELECT COALESCE(SUM(textlen),0) FROM posts p WHERE p.textfile=f.name AND p.status=1)").n==0
        # Commit fault leaves cursor, posts and keywords at the prior generation.
        generation=s.generation
        @test_throws ErrorException DR.commit_batch!(s,"test",batch([PostEvent(record(100;at=105.))],"bad"),embed;now=105.,fault=:before_commit)
        DS.close_store(s);s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2)
        @test s.generation==generation && DR.source_cursor(s,"test")=="delete"
        @test DS.lookup(s,PostKey(:threads,"100"))===nothing
        exact(s)
        # Preparation failures never enter publication.
        @test_throws ErrorException DR.commit_batch!(s,"test",batch([PostEvent(record(100;at=105.))],"bad"),x->error("inference failed");now=105.)
        @test !s.poisoned && s.generation==generation
        # All keyword evidence expires before summaries/status advance.
        DR.commit_batch!(s,"test",batch(PostEvent[],"idle"),embed;now=1101.)
        @test DR.status(s).active==0 && DS.keyword_stats(s).covered==0
        @test isempty(DS.rows(s,"SELECT * FROM kw_counts"))
        DR.reclaim_daily!(s;force=true)
        @test DS.storage_stats(s).text_files==0
        @test DS.storage_stats(s).vector_files==0
        DS.clear_caches!(s);DS.close_store(s)
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2)
        @test DR.source_cursor(s,"test")=="idle"
        @test DR.status(s).active==0
        DS.close_store(s)
    end
end
@testset "Alias identity and Nostr ownership" begin
    mktempdir() do root
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=DS.config(dimension=4))
        DS.enable_keywords!(s,KE.JuliaYakeExtractor());embed=ts->fill(.2f0,4,length(ts))
        alias=PostKey(:threads,"alias")
        p=record("primary";aliases=[alias])
        DR.commit_batch!(s,"test",batch([PostEvent(p)]),embed;now=100.)
        id=DS.lookup(s,p.key).id
        q=PostRecord(alias;text="New content",created_at=100.,observed_at=101.)
        # Explicit revision distinguishes the content edit.
        q=PostRecord(alias;text=q.text,created_at=100.,observed_at=101.,revision_at=101.)
        DR.commit_batch!(s,"test",batch([PostEvent(q)],"2"),embed;now=101.)
        @test DS.lookup(s,alias).id==id && DS.lookup(s,p.key).id==id
        @test DR.status(s).active==1
        key=PostKey(:nostr,repeat("a",64))
        note=PostRecord(key;text="nostr science",author="owner",created_at=102.,observed_at=102.)
        bad=PostEvent(key;author="stranger",observed_at=103.)
        DR.commit_batch!(s,"test",batch([PostEvent(note),bad],"3"),embed;now=103.)
        @test DS.lookup(s,key).status==1
        good=PostEvent(key;author="owner",observed_at=104.)
        DR.commit_batch!(s,"test",batch([good],"4"),embed;now=104.)
        @test DS.lookup(s,key).status==2
        # Alias introduced only by a note deleted in the same batch is retained.
        p=record("doomed";at=105.,aliases=[PostKey(:threads,"doomed-alias")])
        d=PostEvent(p.key;observed_at=106.)
        DR.commit_batch!(s,"test",batch([PostEvent(p),d],"5"),embed;now=106.)
        @test DS.lookup(s,PostKey(:threads,"doomed-alias")).status==2
        exact(s);DS.close_store(s)
    end
end
