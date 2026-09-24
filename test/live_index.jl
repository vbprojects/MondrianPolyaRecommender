using Test, Random, LinearAlgebra, SHA
include("../examples/live_modules.jl")
using .PostRecords, .LeafLatentTrees, .LeafTreeKernels, .LiveIndex, .ArrowPostStore
BLAS.set_num_threads(1)
function records(first,last;time=10.,offset=0.)
    [PostRecord(PostKey(:bluesky,"at://did:plc:example/app.bsky.feed.post/$i");
        text="post $(i+offset) with + 🦋",author="did:plc:example",created_at=time,observed_at=time,revision_at=time,
        tags=["julia"],metadata="{\"likes\":0}") for i in first:last]
end
# Deterministic synthetic embedding callback, shared across restarts/edits.
function embed(texts)
    hcat([clamp.(.25randn(MersenneTwister(reinterpret(UInt32,sha256(t)[1:4])[1]),Float32,12),-.9f0,.9f0) for t in texts]...)
end
config=IndexConfig(dimension=12,rank=4,half_life=1.,max_leaves=8,split_after=24,
    reservoir_cap=24,merge_below=2.,merge_interval=.2,codec_batch=32,retention_seconds=100.)
@testset "Canonical identities" begin
    for kind in (:activitypub,:threads,:nostr,:bluesky)
        key=kind==:activitypub ? PostKey(kind,"https://x.example/a+b?x=%2F") :
            kind==:threads ? PostKey(kind,"00123456789012345678901234567890") :
            kind==:nostr ? PostKey(kind,repeat("Ab",32)) : PostKey(kind,"at://did:plc:example/app.bsky.feed.post/r:key")
        @test key_from_bytes(key.kind,native_bytes(key))==key
        @test Dict(key=>1)[key_from_bytes(key.kind,native_bytes(key))]==1
    end
    @test length(native_bytes(PostKey(:nostr,repeat("f0",32))))==32
    @test_throws ArgumentError PostKey(:bluesky,"at://handle.example/app.bsky.feed.post/x")
    @test_throws ArgumentError PostKey(:nostr,"note1invalid")
    @test parse_time("2025-01-01T01:30:00+01:30")==parse_time("2025-01-01T00:00:00Z")
    @test parse_time("2025-01-01T00:00:00.123456Z")≈parse_time("2025-01-01T00:00:00Z")+.123456
    @test_throws ArgumentError parse_time("2025-01-01T00:00:00")
end
function invariant(s)
    a=audit(s);@test a.active==length(s.locations)
    if s.fitted!==nothing
        p=kernel_probabilities(s.backend,s.fitted)
        @test sum(p.probabilities)≈1
        @test all(isfinite,p.masses)
    end
end
function signatures(s)
    Dict(id=>(copy(s.blocks[bi].codes.center),s.blocks[bi].codes.scales[j],copy(s.blocks[bi].codes.residuals[:,j]))
        for (id,(bi,j)) in s.locations)
end
function compare(a,b)
    @test audit(a)==audit(b)
    @test a.identities==b.identities
    @test a.next_id==b.next_id && a.next_block==b.next_block
    @test signatures(a)==signatures(b)
    if a.fitted!==nothing
        @test [(n.axis,n.cut,n.left,n.right) for n in a.fitted.nodes]==[(n.axis,n.cut,n.left,n.right) for n in b.fitted.nodes]
        @test a.fitted.reservoir_rows==b.fitted.reservoir_rows
        for (x,y) in zip(a.fitted.nodes,b.fitted.nodes)
            x.axis==0 || continue
            p=parameters(only(x.states));q=parameters(only(y.states))
            @test p.mean≈q.mean atol=1e-12
            @test p.W*p.W'+Diagonal(p.D)≈q.W*q.W'+Diagonal(q.D) atol=1e-12
        end
    end
end
@testset "Live density and immutable post identity" begin
    s=Index(config);rs=records(1,160)
    result=ingest!(s,PostEvent.(rs),embed);@test result.inserted==160
    @test s.fitted.processed==160;invariant(s)
    ids=Dict(p.key=>lookup(s,p.key).id for p in rs);frozen=signatures(s)
    ingest!(s,PostEvent.(rs),texts->error("duplicates must not be embedded"))
    @test s.fitted.processed==160 && signatures(s)==frozen
    maintain!(s;time=30.);@test length(s.fitted.nodes)==1 && s.fitted.merge_count>0
    ingest!(s,PostEvent.(records(161,320;time=31.)),embed)
    @test s.fitted.processed==320 && s.fitted.split_count>7
    @test all(signatures(s)[id]==sig for (id,sig) in frozen);invariant(s)
    # Edits preserve post ID, replace only their own code and remove old density.
    edit=PostRecord(rs[1].key;text="edited post",created_at=10.,observed_at=32.,revision_at=32.)
    ingest!(s,[PostEvent(edit)],embed)
    @test lookup(s,rs[1].key).id==ids[rs[1].key]
    @test all(signatures(s)[id]==sig for (id,sig) in frozen if id!=ids[rs[1].key])
    ingest!(s,[PostEvent(rs[2].key;observed_at=33.)],texts->error("delete cannot embed"))
    @test lookup(s,rs[2].key).status==2 && !haskey(s.locations,ids[rs[2].key])
    ingest!(s,[PostEvent(rs[2])],texts->error("old replay cannot embed"))
    @test lookup(s,rs[2].key).status==2;invariant(s)
    # Embedding failure/validation leaves persistent identities and codes untouched.
    oldgen=s.generation;before=signatures(s)
    @test_throws ArgumentError ingest!(s,PostEvent.(records(321,322;time=34.)),t->fill(NaN,12,2))
    @test s.generation==oldgen && signatures(s)==before
    # Real metadata identity from four namespaces, including aliases and literal '+'.
    other=[PostRecord(PostKey(:activitypub,"https://one.example/posts/a+b");text="AP 🦋",created_at=34.,observed_at=34.,aliases=[PostKey(:mastodon_local,"https://mirror.example/api/v1/statuses/12")]),
        PostRecord(PostKey(:threads,"001234");text="Threads",created_at=34.,observed_at=34.),
        PostRecord(PostKey(:nostr,repeat("ab",32));text="Nostr",author="alice",created_at=34.,observed_at=34.)]
    ingest!(s,PostEvent.(other),embed)
    @test lookup(s,other[1].key).id==lookup(s,only(other[1].aliases)).id
    ingest!(s,[PostEvent(other[3].key;observed_at=35.,author="mallory")],t->error("no embedding"))
    @test lookup(s,other[3].key).status==1
    @test length(candidates(s,zeros(12);limit=20,leaf_budget=8))==20
    mktempdir() do root
        s.cursors["bluesky:test"]="123456";checkpoint!(s,root);restored=restore(root)
        compare(s,restored);@test restored.cursors==s.cursors
        @test lookup(restored,other[1].key).record.text=="AP 🦋"
        @test lookup(restored,other[2].key).key.native=="001234"
        # Continue through a fresh merge epoch and compare RNG-driven cuts exactly.
        next=PostEvent.(records(401,560;time=55.))
        ingest!(s,next,embed);ingest!(restored,next,embed);compare(s,restored)
        committed=checkpoint_path(root)
        @test_throws ErrorException checkpoint!(s,root;fault_after="codes")
        @test checkpoint_path(root)==committed
        @test_throws ErrorException checkpoint!(s,root;fault_after="publish")
        @test checkpoint_path(root)==committed
        checkpoint!(s,root);compare(s,restore(root))
        @test prune_checkpoints!(root;keep=1)>=1
        @test length(filter(n->startswith(n,"generation-"),readdir(root)))==1
        path=checkpoint_path(root)
        open(joinpath(path,"codes.arrow"),"a") do io;write(io,0x00);end
        @test_throws ErrorException restore(root)
    end
    expire!(s;before=56.,time=56.);@test isempty(s.locations)
    @test all(p->p.record===nothing,values(s.posts));invariant(s)
    mktempdir() do root;checkpoint!(s,root);compare(s,restore(root));end
    mktempdir() do root
        empty=Index(config);checkpoint!(empty,root);compare(empty,restore(root))
    end
    previous=s.next_id;@test gc_tombstones!(s;before=-45.)==0
    maintain!(s;time=1000.);@test gc_tombstones!(s;before=800.)>0
    @test isempty(s.posts) && isempty(s.identities) && s.next_id==previous
end
@testset "Deletion ordering and aliases within a batch" begin
    for bad in (false,true)
        s=Index(config);key=PostKey(:nostr,repeat("aa",32))
        note=PostRecord(key;author="alice",text="note",created_at=10.,observed_at=10.)
        deletion=PostEvent(key;observed_at=11.,author=bad ? "mallory" : "alice")
        ingest!(s,[PostEvent(note),deletion],embed)
        @test lookup(s,key).status==(bad ? 1 : 2)
        if !bad
            ingest!(s,[PostEvent(note)],t->error("deleted note must not return"))
            @test lookup(s,key).status==2
        end
        invariant(s)
    end
    s=Index(config);key=PostKey(:activitypub,"https://original.example/posts/a+b")
    alias=PostKey(:mastodon_local,"https://mirror.example/api/v1/statuses/1")
    note=PostRecord(key;text="aliased note",created_at=10.,observed_at=10.,aliases=[alias])
    ingest!(s,[PostEvent(note),PostEvent(alias;observed_at=11.)],t->error("same-batch deleted note must not be embedded"))
    @test lookup(s,key).status==2 && lookup(s,alias).id==lookup(s,key).id
    ingest!(s,[PostEvent(note)],t->error("old aliased replay must be ignored"))
    @test isempty(s.locations);invariant(s)
    s=Index(config)
    latest=PostRecord(key;text="same text",created_at=10.,observed_at=20.,metadata="{\"likes\":5}")
    stale=PostRecord(key;text="same text",created_at=10.,observed_at=15.,metadata="{\"likes\":2}")
    ingest!(s,[PostEvent(latest)],embed)
    ingest!(s,[PostEvent(stale)],t->error("stale metadata must not embed"))
    @test lookup(s,key).record.metadata==latest.metadata
    s=Index(config);ingest!(s,[PostEvent(stale),PostEvent(latest)],embed)
    @test lookup(s,key).record.metadata==latest.metadata && s.fitted.processed==1
end
