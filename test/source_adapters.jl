using Test, JSON3
include("../src/PostRecords.jl")
include("../src/SourceAdapters.jl")
include("../src/EmbeddingBridge.jl")
using .PostRecords, .SourceAdapters, .EmbeddingBridge
@testset "Platform normalization" begin
    at="2025-01-01T00:00:00Z";time=parse_time(at)
    record=Dict("text"=>"hello + 🦋","createdAt"=>at,"facets"=>[Dict("features"=>[Dict("\$type"=>"app.bsky.richtext.facet#tag","tag"=>"julia")])])
    legacy=Dict("did"=>"did:plc:example","time_us"=>Int(time*1e6),"kind"=>"commit",
        "commit"=>Dict("operation"=>"create","collection"=>"app.bsky.feed.post","rkey"=>"123","record"=>record))
    e=only(bluesky_event(legacy;observed_at=time));@test e.record.text=="hello + 🦋" && e.record.tags==["julia"]
    current=Dict("payload"=>Dict("did"=>"did:plc:example","time"=>at,"seq"=>1234,"cursor"=>"1234",
        "operation"=>"create","collection"=>"app.bsky.feed.post","rkey"=>"123","record"=>record))
    @test only(bluesky_event(current;observed_at=time)).key==e.key
    @test jetstream_cursor(current)=="1234"
    current["payload"]["operation"]="delete";delete!(current["payload"],"record")
    @test only(bluesky_event(current;observed_at=time)).operation==:delete
    mast=Dict("id"=>"12","uri"=>"https://origin.example/object/a+b","content"=>"<p>A &amp; B</p><p>Julia</p>",
        "visibility"=>"public","created_at"=>at,"account"=>Dict("url"=>"https://origin.example/@u"),"tags"=>[Dict("name"=>"julia")])
    e=only(mastodon_status(mast;origin="https://mirror.example",observed_at=time))
    @test occursin("A & B",e.record.text) && occursin("Julia",e.record.text)
    @test e.key.native==mast["uri"] && only(e.record.aliases).native=="https://mirror.example/api/v1/statuses/12"
    deletion=only(mastodon_event(Dict("event"=>"delete","payload"=>"12");origin="https://mirror.example",observed_at=time+1))
    @test deletion.key==only(e.record.aliases)
    mast["visibility"]="private";@test only(mastodon_status(mast;origin="https://mirror.example",observed_at=time)).operation==:delete
    ap=Dict("type"=>"Create","actor"=>"https://origin.example/users/u","object"=>Dict("type"=>"Note",
        "id"=>"https://origin.example/objects/a+b","attributedTo"=>"https://origin.example/users/u","published"=>at,
        "content"=>"<p>Julia &amp; Arrow</p>","to"=>["https://www.w3.org/ns/activitystreams#Public"]))
    e=only(activitypub_event(ap;observed_at=time,expected_actor=ap["actor"]))
    @test e.record.text=="Julia & Arrow"
    @test_throws ArgumentError activitypub_event(ap;observed_at=time,expected_actor="https://other.example/u")
    ap["type"]="Delete";ap["object"]=e.key.native
    @test only(activitypub_event(ap;observed_at=time)).key==e.key
    th=Dict("id"=>"00123","text"=>"Threads + Julia","timestamp"=>"2025-01-01T00:00:00+0000","shortcode"=>"abc",
        "owner"=>Dict("id"=>"99"),"permalink"=>"https://www.threads.net/@u/post/abc")
    e=only(threads_post(th;observed_at=time));@test e.key.native=="00123" && e.record.author=="99"
    th["text"]="";@test only(threads_post(th;observed_at=time+1)).operation==:delete
    current["payload"]["operation"]="update";current["payload"]["record"]=Dict("text"=>"","createdAt"=>at)
    @test only(bluesky_event(current;observed_at=time+1)).operation==:delete
end
@testset "Verified Nostr and pinned embedding bridge" begin
    fixtures=JSON3.read(read(joinpath(@__DIR__,"fixtures/nostr_signed.json"),String))
    worker=Worker()
    try
        @test verify_nostr(worker,[fixtures.note,fixtures.delete])==[true,true]
        changed=JSON3.read(JSON3.write(fixtures.note),Dict{String,Any});changed["content"]="tampered"
        @test verify_nostr(worker,[changed])==[false]
        verifier=e->only(verify_nostr(worker,[e]))
        note=only(nostr_events(fixtures.note;observed_at=1735689700.,verify=verifier))
        deletion=only(nostr_events(fixtures.delete;observed_at=1735689700.,verify=verifier))
        @test note.key==deletion.key && deletion.operation==:delete
        @test note.record.tags==["julia"]
        @test_throws ArgumentError nostr_events(changed;observed_at=1735689700.,verify=verifier)
        if "--embed" in ARGS
            X=EmbeddingBridge.embed(worker,["Julia numerical computing","Bluesky social posts"])
            @test size(X)==(256,2) && all(isfinite,X)
            @test vec(sum(abs2,X;dims=1))≈ones(2) atol=1e-5
        end
    finally close(worker) end
end
