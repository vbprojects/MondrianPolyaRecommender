using Test, Random, LinearAlgebra, JSON3
include("../src/MondrianServer.jl")
const DS=DiskTreeStore
const RM=DS.RocksMetadata
const KE=KeywordExtraction
using .PostRecords
BLAS.set_num_threads(1)
function makepost(i,t;text="post $i")
    PostRecord(PostKey(:threads,string(i));text,tags=["topic$(i%5)","common"],created_at=t,observed_at=t,revision_at=t)
end
if !isempty(ARGS)
    root,phase=ARGS
    if phase=="recover"
        DS.open_store(root;recovery_fault=:exit_during_recovery)
    else
        s=DS.open_store(root);s.db.buffer_limit=8192
        phase=="commit" && (s.db.commit_fault=:exit_after_marker)
        DS.commit!(s;fault=phase=="rollback" ? :exit_before_commit : nothing) do
            for i in 1:500
                DS.exec(s,"INSERT OR REPLACE INTO cursors VALUES (?,?)",("cursor$i",repeat("new",700)))
            end
        end
    end
    error("fault failed to terminate process")
end
@testset "Native RocksDB recovery can itself be interrupted" begin
    for phase in ("rollback","commit")
        mktempdir() do root
            s=DS.open_store(root;backend=:rocksdb,create_config=DS.config(dimension=4),budget=32*1024^2)
            DS.commit!(s) do
                for i in 1:500;DS.exec(s,"INSERT OR REPLACE INTO cursors VALUES (?,?)",("cursor$i","old"));end
            end
            generation=s.generation;DS.close_store(s)
            command=`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $(@__FILE__) $root $phase`
            @test run(ignorestatus(command)).exitcode==(phase=="commit" ? 95 : 93)
            recover=`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $(@__FILE__) $root recover`
            @test run(ignorestatus(recover)).exitcode==94
            s=DS.open_store(root;budget=32*1024^2)
            @test s.generation==generation+Int(phase=="commit")
            @test all(r.cursor==(phase=="commit" ? repeat("new",700) : "old") for r in DS.rows(s,"SELECT * FROM cursors"))
            @test RM.rawget(s.db,"!transaction")===nothing
            @test DS.audit!(s)==0
            @test !isfile(joinpath(root,"index.sqlite"))
            DS.close_store(s)
        end
    end
end
@testset "SQLite/RocksDB tree and keyword parity, explicit import" begin
    mktempdir() do root
        cfg=DS.config(dimension=12,split_after=24,reservoir_cap=24,max_leaves=8,half_life=5.,retention_seconds=60.,merge_below=1.)
        stores=[DS.open_store(joinpath(root,b);backend=Symbol(b),create_config=cfg,budget=32*1024^2) for b in ("sqlite","rocksdb")]
        e=KE.HashtagsOnlyExtractor();foreach(s->DS.enable_keywords!(s,e),stores)
        rng=MersenneTwister(18)
        function compare()
            a,b=stores
            @test DS.tree_signature(a)==DS.tree_signature(b)
            @test a.generation==b.generation
            ids=[r.id for r in DS.rows(a,"SELECT id FROM posts WHERE status=1 ORDER BY id")]
            @test ids==[r.id for r in DS.rows(b,"SELECT id FROM posts WHERE status=1 ORDER BY id")]
            for start in 1:128:length(ids)
                page=ids[start:min(start+127,end)]
                @test DS.fetch_vectors(a,page)==DS.fetch_vectors(b,page)
            end
            @test DS.polya_weights(a)==DS.polya_weights(b)
            @test DS.rows(a,"SELECT * FROM kw_counts ORDER BY leaf,term")==DS.rows(b,"SELECT * FROM kw_counts ORDER BY leaf,term")
            @test DS.audit!(a)==DS.audit!(b)==length(ids)
        end
        for round in 0:2
            t=round*50.
            rs=[makepost(round*512+i,t) for i in 1:512]
            X=randn(rng,Float32,12,512);X./=sqrt.(sum(abs2,X;dims=1))
            keywords=KE.prepare_keywords(e,KE.keyword_input.(rs))
            for s in stores
                for first in 1:128:512
                    ids=first:first+127
                    DS.append_batch!(s,rs[ids],X[:,ids];time=t,keywords=keywords[ids],cursor=("source",string(last(ids)+round*512)))
                end
                DS.delete_post!(s,rs[1].key;revision_at=t+1,time=t+1)
                # Explicit semantic edit and keyword-only edit preserve parity.
                edit=makepost(round*512+2,t+2;text="edited text")
                DS.append_batch!(s,[edit],X[:,2:2];time=t+2,keywords=KE.prepare_keywords(e,KE.keyword_input.([edit])))
                meta=PostRecord(rs[3].key;text=rs[3].text,tags=["changed"],created_at=t,observed_at=t+3,revision_at=t+3)
                DS.keyword_update!(s,[meta],KE.prepare_keywords(e,KE.keyword_input.([meta]));time=t+3)
                DS.expire!(s;time=t+40);DS.maintain!(s;time=t+40)
                DS.compact!(s);DS.compact_texts!(s);DS.clear_caches!(s)
            end
            compare()
        end
        for _ in 1:1100;DS.rows(stores[2],"SELECT * FROM nodes ORDER BY id");end
        @test length(stores[2].db.native.children)<=1024
        @test stores[2].db.peak_buffer_bytes<stores[2].db.buffer_limit+2*1024^2
        @test DS.storage_stats(stores[2]).sqlite_bytes==0
        generation=stores[2].generation
        RM.rewrite_tables!(stores[2].db)
        @test stores[2].generation==generation
        @test DS.audit!(stores[2])==DS.audit!(stores[1])
        foreach(DS.close_store,stores)
        destination=joinpath(root,"imported")
        report=DS.import_sqlite_to_rocks(joinpath(root,"sqlite"),destination;budget=32*1024^2,page_size=47)
        @test report.active>0
        stores[1]=DS.open_store(joinpath(root,"sqlite"));stores[2]=DS.open_store(destination)
        compare()
        @test DS.metadata_backend(stores[2])==:rocksdb
        @test_throws ErrorException DS.open_store(destination;backend=:sqlite)
        @test_throws ErrorException DS.import_sqlite_to_rocks(joinpath(root,"sqlite"),destination)
        foreach(DS.close_store,stores)
    end
end
