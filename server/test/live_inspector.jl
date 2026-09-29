using Test, Random, HTTP, JSON3, SQLite
include("../src/LiveInspector.jl")
const LI=LiveInspector
const DS=DiskTreeStore
const KE=KeywordExtraction
using .PostRecords

@testset "Read-only live inspector alongside an open writer" begin
    mktempdir() do root
        s=DS.open_store(root;create_config=DS.config(dimension=4,split_after=16,max_leaves=8,reservoir_cap=16,half_life=10.,merge_below=0.),budget=32*1024^2)
        ins=LI.Inspector(root)
        try
            empty=LI.snapshot(ins)
            @test empty.active==0 && empty.leaves==1
            @test isempty(LI.leaf_detail(ins,1).posts)
            @test !LI.leaf_detail(ins,1).keywords.enabled
            ex=KE.HashtagsOnlyExtractor();DS.enable_keywords!(s,ex)
            ps=[PostRecord(PostKey(:threads,"inspector-$i");text="Post $i <script>alert(1)</script>",tags=[i%2==0 ? "space" : "garden"],observed_at=10.,created_at=10.) for i in 1:128]
            X=rand(MersenneTwister(7),Float32,4,128).*1.8f0.-.9f0
            DS.append_batch!(s,ps,X;keywords=KE.prepare_keywords(ex,KE.keyword_input.(ps)),time=10.)
            before=DS.one(s,"SELECT value FROM meta WHERE key='runtime'").value
            snap=LI.snapshot(ins);weights=DS.polya_weights(s)
            @test snap.active==128
            @test snap.leaves==count(n->n.axis==0,values(s.nodes))
            @test sum(n.probability for n in snap.nodes if n.leaf)≈1
            for n in snap.nodes
                @test n.mass≈weights.masses[parse(Int,n.id)]
                @test n.leaf || n.count==sum(c.count for c in snap.nodes if c.parent==n.id)
                n.leaf || continue
                id=parse(Int,n.id)
                @test n.probability≈weights.probabilities[id]
                d=LI.leaf_detail(ins,id;generation=snap.generation)
                @test d.count==n.count
                @test d.ppca.rank<=4
                @test length(d.posts)==min(6,n.count)
                @test length(unique(p.id for p in d.posts))==length(d.posts)
                @test d.unavailable==0
                for ranking in (:common,:distinctive)
                    expected=DS.leaf_keywords(s,id;ranking)
                    @test getproperty(d.keywords,ranking)==expected.terms
                    @test d.keywords.covered==expected.covered
                end
            end
            @test DS.one(s,"SELECT value FROM meta WHERE key='runtime'").value==before
            @test_throws SQLite.SQLiteException LI.read_snapshot(ins) do db,rt,cfg
                LI.q(db,"INSERT INTO meta(key,value) VALUES('illegal','write')")
            end
            @test_throws SQLite.SQLiteException LI.read_snapshot(ins;seconds=0.000001) do db,rt,cfg
                LI.q(db,"WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<10000000) SELECT SUM(n) FROM x")
            end
            id=parse(Int,first(n.id for n in snap.nodes if n.leaf && n.count>0))
            ins.cache=snap;ins.cached_at=time()
            @test LI.handler(ins,HTTP.Request("GET","/api/snapshot")).status==200
            DS.delete_post!(s,ps[1].key;revision_at=11.,time=11.)
            @test LI.handler(ins,HTTP.Request("GET","/api/leaf?id=$id&generation=$(snap.generation)")).status==409
            @test ins.cache===nothing
            current=LI.snapshot(ins)
            @test current.active==127 && current.generation!=snap.generation
            @test LI.handler(ins,HTTP.Request("GET","/api/leaf?id=-1")).status==400
            @test LI.handler(ins,HTTP.Request("GET","/api/leaf?id=999999")).status==409
            @test LI.handler(ins,HTTP.Request("POST","/api/snapshot")).status==405
            @test LI.handler(ins,HTTP.Request("GET","/nope")).status==404
            @test LI.handler(ins,HTTP.Request("GET","/")).status==200
            @test LI.handler(ins,HTTP.Request("GET","/inspector.js")).status==200
            # An immutable file can be reclaimed after the metadata snapshot.
            file=DS.one(s,"SELECT textfile FROM posts WHERE status=1 AND leaf=? AND slot=1",(id,)).textfile
            path=joinpath(root,file);mv(path,path*".held")
            try
                @test LI.leaf_detail(ins,id).unavailable>0
            finally;mv(path*".held",path);end
            DS.audit!(s)
        finally;DS.close_store(s);end
        # Restarted inspector needs no full corpus or writer handle.
        @test LI.snapshot(LI.Inspector(root)).active==127
    end
end
