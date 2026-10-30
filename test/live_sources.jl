using Test, JSON3, HTTP
include("../src/PostRecords.jl")
include("../src/SourceAdapters.jl")
include("../src/LiveSources.jl")
using .PostRecords, .LiveSources
fixtures=[JSON3.read(line,Dict{String,Any}) for line in eachline(joinpath(@__DIR__,"fixtures/live_sources.jsonl"))]
now=1735689800.
@testset "Bounded transports and cursors" begin
    sent=Ref{Any}();blue=fixtures[1]["payload"]
    stream=function (consume,url;kwargs...)
        sent[]=(url=url,kwargs=kwargs)
        consume(blue)
    end
    batch=collect_source(Dict("kind"=>"bluesky"),"17";stream,now)
    @test batch.cursor=="1" && only(batch.events).record.text==blue["payload"]["record"]["text"]
    @test sent[].kwargs[:subprotocols]==["xrpc.v1.json"]
    @test ("cursor"=>"17") in sent[].kwargs[:query]
    @test_throws SourceError collect_source(Dict("kind"=>"bluesky");stream=(f,u;kw...)->f(Dict("\$type"=>"error")))
    # Signed Nostr normalization is tested separately with the real verifier.
    signed=JSON3.read(read(joinpath(@__DIR__,"fixtures/nostr_signed.json"),String),Dict{String,Any})
    relay=function (consume,url;kwargs...)
        sent[]=kwargs[:initial]
        consume(["EVENT","mondrian-live",signed["note"]]);consume(["EOSE","mondrian-live"])
    end
    batch=collect_source(Dict("kind"=>"nostr","url"=>"ws://relay"),"1735689700";stream=relay,verify=x->true,now)
    @test batch.cursor==string(Int(now)) && length(batch.events)==1
    @test sent[][3]["since"]==1735689698 && sent[][3]["kinds"]==[1,5]
    @test_throws ArgumentError collect_source(Dict("kind"=>"nostr","url"=>"ws://relay");now)
    batch=collect_source(Dict("kind"=>"nostr","url"=>"ws://relay"),"9";
        stream=(f,u;kw...)->f(["EVENT","mondrian-live",signed["note"]]),verify=x->true,now)
    @test batch.cursor=="9" && !batch.complete
    origin="https://mastodon.example"
    mast=(id)->Dict("id"=>id,"uri"=>origin*"/objects/"*id,"visibility"=>"public","content"=>"<p>Post</p>","created_at"=>"2025-01-01T00:00:00Z")
    getter=function(url;query,headers,seconds)
        sent[]=query
        [mast("13"),mast("12")]
    end
    batch=collect_source(Dict("kind"=>"mastodon","origin"=>origin),"11";getjson=getter,limit=2,now)
    @test batch.cursor=="13" && first(batch.events).key.native==origin*"/objects/12"
    @test ("min_id"=>"11") in sent[]
    # Threads cursors retain a fixed time window and never follow tokenized next URLs.
    withenv("MONDRIAN_TEST_TOKEN"=>"dummy-token") do
        getter=function(url;query,headers,seconds)
            @test headers==["Authorization"=>"Bearer dummy-token"]
            sent[]=query
            Dict("data"=>[fixtures[66]["payload"]],"paging"=>Dict("next"=>"https://untrusted.example/?access_token=no","cursors"=>Dict("after"=>"page2")))
        end
        source=Dict("kind"=>"threads","query"=>"julia","token_env"=>"MONDRIAN_TEST_TOKEN")
        batch=collect_source(source;getjson=getter,now)
        state=JSON3.read(batch.cursor);@test state.after=="page2" && !state.done
        @test only(batch.events).key.native=="001234567890"
        second=function(url;query,headers,seconds)
            @test ("after"=>"page2") in query
            @test ("until"=>string(Int(now))) in query
            Dict("data"=>[],"paging"=>Dict())
        end
        @test JSON3.read(collect_source(source,batch.cursor;getjson=second,now=now+10).cursor).done
    end
    @test_throws SourceError collect_source(Dict("kind"=>"threads","query"=>"julia","token_env"=>"MISSING_MONDRIAN_TOKEN");now)
    ap=fixtures[65];actor=ap["actor"]
    pages=Dict(actor=>Dict("id"=>actor,"outbox"=>actor*"/outbox"),actor*"/outbox"=>Dict("first"=>actor*"/page1"),
        actor*"/page1"=>Dict("orderedItems"=>[ap["payload"],ap["payload"]]))
    getter=(url;kwargs...)->pages[url]
    batch=collect_source(Dict("kind"=>"activitypub","actor"=>actor);getjson=getter,limit=1,now)
    @test !batch.complete && JSON3.read(batch.cursor).offset==1
    batch=collect_source(Dict("kind"=>"activitypub","actor"=>actor),batch.cursor;getjson=getter,limit=1,now)
    @test batch.complete && batch.cursor=="" && length(batch.events)==1
end
@testset "Actual HTTP and WebSocket transport" begin
    server=HTTP.serve!(request->HTTP.Response(200,["Content-Type"=>"application/json"],
        request.target=="/array" ? "[{\"id\":\"12\"}]" : request.target=="/empty" ? "[]" : JSON3.write((ok=true,)));listenany=true)
    try
        @test json_get("http://"*HTTP.server_addr(server))["ok"]
        @test only(json_get("http://"*HTTP.server_addr(server)*"/array"))["id"]=="12"
        @test isempty(json_get("http://"*HTTP.server_addr(server)*"/empty"))
    finally close(server) end
    blue=fixtures[1]["payload"]
    server=HTTP.WebSockets.listen!("127.0.0.1",0;listenany=true,subprotocols=["xrpc.v1.json"]) do ws
        HTTP.WebSockets.send(ws,JSON3.write(blue))
    end
    try
        batch=collect_source(Dict("kind"=>"bluesky","url"=>"ws://"*HTTP.WebSockets.server_addr(server));limit=1,seconds=2.,now)
        @test length(batch.events)==1 && batch.cursor=="1"
    finally close(server) end
end
