"""Bounded read-only source readers. Cursors are returned, never committed here.
Polling coverage is source-dependent; these are not universal archive crawlers.
"""
module LiveSources
using HTTP, JSON3
using ..PostRecords, ..SourceAdapters
export SourceBatch, collect_source, json_get, stream_frames, SourceError
struct SourceError <: Exception
    message::String
end
Base.showerror(io::IO,e::SourceError)=print(io,e.message)
struct SourceBatch
    events::Vector{PostEvent}
    cursor::String
    received::Int
    complete::Bool
end
getv(x,k,d=nothing)=get(x,k,d)
function auth_headers(c;required=false)
    name=String(getv(c,"token_env",""));token=isempty(name) ? "" : get(ENV,name,"")
    required && isempty(token) && throw(SourceError("source requires its configured token_env"))
    isempty(token) ? Pair{String,String}[] : ["Authorization"=>"Bearer "*token]
end
function json_get(url;query=Pair{String,String}[],headers=Pair{String,String}[],seconds=20.)
    # Limit decoded payload size and elapsed time. Never follow an authenticated
    # pagination URL supplied by an upstream response; reconstruct our own query.
    io=IOBuffer(maxsize=8*1024*1024)
    response=HTTP.get(url;query,headers,response_stream=io,redirect=false,retry=false,
        status_exception=false,connect_timeout=min(10.,seconds),request_timeout=seconds,
        max_decompressed_size=8*1024*1024)
    response.status==200 || throw(SourceError("source HTTP status $(response.status)"))
    # Mastodon timelines are arrays; Threads and ActivityPub return objects.
    JSON3.read(take!(io),Any)
end
"""Call consume for each bounded frame. Returning false stops the connection.
isready polling also ends quiet streams without relying on incoming traffic.
"""
function stream_frames(consume,url;query=Pair{String,String}[],headers=Pair{String,String}[],
        subprotocols=String[],seconds=10.,limit=256,initial=nothing)
    HTTP.WebSockets.open(url;query,headers,subprotocols,redirect=false,
            maxframesize=1024*1024,maxfragmentation=8,connect_timeout=min(10.,seconds),
            response_header_timeout=min(10.,seconds),read_idle_timeout=max(2.,seconds+1)) do ws
        initial!==nothing && HTTP.WebSockets.send(ws,JSON3.write(initial))
        deadline=time()+seconds;count=0
        while time()<deadline && count<limit
            if isready(ws)
                frame=HTTP.WebSockets.receive(ws);count+=1
                consume(JSON3.read(frame,Any))===false && break
            elseif HTTP.WebSockets.isclosed(ws)
                break
            else
                sleep(.01)
            end
        end
    end
    nothing
end

function collect_source(c,cursor="";limit=256,seconds=10.,verify=nothing,
        getjson=json_get,stream=stream_frames,now=time())
    1<=limit<=1024 && isfinite(seconds) && seconds>0 || throw(ArgumentError("invalid collection budget"))
    kind=String(c["kind"]);events=PostEvent[];received=0;next=String(cursor);complete=true
    if kind=="bluesky"
        legacy=getv(c,"legacy",false)
        url=getv(c,"url",legacy ? "wss://jetstream1.us-west.bsky.network/subscribe" :
            "wss://jetstream.us-east.bsky.network/xrpc/network.bsky.jetstream.subscribeEvents")
        query=legacy ? ["wantedCollections"=>"app.bsky.feed.post"] : ["collections"=>"app.bsky.feed.post","kinds"=>"commit"]
        !isempty(cursor) && push!(query,"cursor"=>cursor)
        stream(url;query,seconds,limit,subprotocols=legacy ? String[] : ["xrpc.v1.json"]) do raw
            getv(raw,"\$type","") in ("error","info") && throw(SourceError("Jetstream returned a control/error frame; cursor unchanged"))
            append!(events,bluesky_event(raw;observed_at=now));received+=1
            candidate=jetstream_cursor(raw);!isempty(candidate) && (next=candidate)
            true
        end
    elseif kind=="nostr"
        verify===nothing && throw(ArgumentError("Nostr signature verifier is required"))
        since=isempty(cursor) ? floor(Int,now-getv(c,"lookback_seconds",60)) : max(0,parse(Int,cursor)-2)
        sub="mondrian-live";eose=false
        initial=["REQ",sub,Dict("kinds"=>[1,5],"since"=>since,"limit"=>max(1,limit÷2))]
        stream(String(c["url"]);seconds,limit=limit+16,initial) do raw
            raw isa AbstractVector && !isempty(raw) || throw(SourceError("invalid relay message"))
            if raw[1]=="EVENT" && length(raw)>=3 && raw[2]==sub
                append!(events,nostr_events(raw[3];observed_at=now,verify));received+=1
            elseif raw[1]=="EOSE" && length(raw)>=2 && raw[2]==sub
                eose=true
            elseif raw[1] in ("CLOSED","AUTH")
                throw(SourceError("Nostr relay refused subscription or requires authentication"))
            end
            received<limit
        end
        # Before EOSE, never pretend a bounded historical response is complete.
        # Relays can cap/filter results: this is a best-effort recent feed.
        next=eose ? string(floor(Int,now)) : cursor;complete=eose
    elseif kind=="mastodon_stream"
        origin=rstrip(String(c["origin"]),'/');headers=auth_headers(c;required=true)
        url=String(c["url"]);query=["stream"=>String(getv(c,"stream","public"))]
        stream(url;query,headers,seconds,limit) do raw
            append!(events,mastodon_event(raw;origin,observed_at=now));received+=1;true
        end
        # Mastodon streaming has no durable replay cursor.
        next="";complete=false
    elseif kind=="mastodon"
        origin=rstrip(String(c["origin"]),'/');headers=auth_headers(c)
        # min_id pages forward immediately after the cursor (since_id can skip
        # backlog). Never parse or compare opaque server-local IDs numerically.
        deadline=time()+seconds
        while received<limit && time()<deadline
            n=min(40,limit-received);query=["limit"=>string(n)]
            !isempty(next) && push!(query,"min_id"=>next)
            getv(c,"local",false) && push!(query,"local"=>"true")
            rows=getjson(origin*"/api/v1/timelines/public";query,headers,seconds=max(.1,deadline-time()))
            isempty(rows) && break
            length(rows)<=n || throw(SourceError("Mastodon exceeded requested page size"))
            for row in reverse(rows);append!(events,mastodon_status(row;origin,observed_at=now));end
            received+=length(rows);candidate=String(first(rows)["id"])
            candidate==next && throw(SourceError("Mastodon cursor did not advance"))
            next=candidate;length(rows)<n && break
        end
        complete=received<limit
    elseif kind=="threads"
        headers=auth_headers(c;required=true);url=getv(c,"url","https://graph.threads.net/keyword_search")
        state=isempty(cursor) ? Dict("since"=>floor(Int,now-getv(c,"lookback_seconds",3600)),"until"=>floor(Int,now),"after"=>"") : JSON3.read(cursor,Dict{String,Any})
        if getv(state,"done",false)
            state=Dict("since"=>max(0,Int(state["until"])-60),"until"=>floor(Int,now),"after"=>"")
        end
        query=["q"=>String(c["query"]),"search_type"=>"RECENT","limit"=>string(min(100,limit)),
            "since"=>string(state["since"]),"until"=>string(state["until"]),
            "fields"=>"id,text,timestamp,permalink,owner,username,shortcode,media_type,topic_tag"]
        !isempty(state["after"]) && push!(query,"after"=>String(state["after"]))
        response=getjson(url;query,headers,seconds);rows=getv(response,"data",[])
        length(rows)<=limit || throw(SourceError("Threads exceeded requested page size"))
        for row in rows;append!(events,threads_post(row;observed_at=now));end
        received=length(rows);paging=getv(response,"paging",Dict())
        after=String(getv(getv(paging,"cursors",Dict()),"after",""))
        # `after` may exist on the last page; presence of `next` indicates more.
        complete=isempty(after) || !haskey(paging,"next")
        !complete && after==state["after"] && throw(SourceError("Threads cursor did not advance"))
        state["after"]=after;state["done"]=complete;next=JSON3.write(state)
    elseif kind=="activitypub"
        actor=String(c["actor"]);headers=["Accept"=>"application/activity+json"]
        profile=getjson(actor;headers,seconds)
        getv(profile,"id","")==actor || throw(SourceError("actor document identity mismatch"))
        state=isempty(cursor) ? Dict("url"=>String(profile["outbox"]),"offset"=>0) : JSON3.read(cursor,Dict{String,Any})
        url=String(state["url"]);offset=Int(state["offset"]);deadline=time()+seconds;pages=0
        while received<limit && pages<8 && time()<deadline
            page=getjson(url;headers,seconds=max(.1,deadline-time()));pages+=1
            if !haskey(page,"orderedItems") && !haskey(page,"items") && haskey(page,"first")
                firstpage=page["first"]
                if firstpage isa AbstractString;url=String(firstpage);offset=0;continue
                else;page=firstpage;end
            end
            rows=getv(page,"orderedItems",getv(page,"items",[]))
            for i in offset+1:length(rows)
                item=rows[i];item isa AbstractString && (item=getjson(item;headers,seconds=max(.1,deadline-time())))
                if getv(item,"type","") in ("Create","Update") && getv(item,"object",nothing) isa AbstractString
                    item=copy(item);item["object"]=getjson(item["object"];headers,seconds=max(.1,deadline-time()))
                end
                append!(events,activitypub_event(item;observed_at=now,expected_actor=actor))
                received+=1;offset=i
                received>=limit && break
            end
            if offset<length(rows);complete=false;break;end
            link=getv(page,"next",nothing)
            if link===nothing
                next="";return SourceBatch(events,next,received,true)
            end
            newurl=link isa AbstractString ? String(link) : String(link["id"])
            newurl==url && throw(SourceError("ActivityPub pagination did not advance"))
            url=newurl;offset=0;complete=false
        end
        next=JSON3.write((url=url,offset=offset))
    else
        throw(ArgumentError("unsupported source kind: $kind"))
    end
    SourceBatch(events,String(next),received,complete)
end
end
