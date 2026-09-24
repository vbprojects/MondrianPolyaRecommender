"""Normalize source payloads. Transports establish provenance; this is not an AP inbox
and does not authenticate arbitrary HTTP submissions. Nostr requires signature verification.
"""
module SourceAdapters
using JSON3, Gumbo
using ..PostRecords
export bluesky_event, mastodon_status, mastodon_event, activitypub_event, threads_post,
       nostr_events, plain_text, jetstream_cursor
getv(x,k,default=nothing)=get(x,k,default)
str(x)=x===nothing ? "" : string(x)
function plain_text(html)
    # Gumbo decodes entities and parses malformed HTML. Insert boundaries so adjacent
    # paragraphs/line breaks do not concatenate words in the embedding input.
    expanded=replace(String(html),r"(?i)<(?:br\s*/?|/p|/div|/li)>"=>"\n")
    strip(Gumbo.text(Gumbo.parsehtml(expanded).root))
end
validtext(s)=!isempty(strip(s)) && ncodeunits(s)<=65536
function jetstream_cursor(raw)
    x=getv(raw,"payload",raw)
    str(getv(x,"cursor",getv(x,"seq",getv(raw,"time_us",""))))
end
function bluesky_event(raw;observed_at)
    x=getv(raw,"payload",raw);commit=getv(x,"commit",x)
    getv(commit,"collection","")=="app.bsky.feed.post" || return PostEvent[]
    operation=getv(commit,"operation","")
    operation in ("create","update","delete") || return PostEvent[]
    did=String(x["did"]);key=PostKey(:bluesky,"at://"*did*"/app.bsky.feed.post/"*String(commit["rkey"]))
    revision=haskey(x,"time") ? parse_time(x["time"]) : Float64(x["time_us"])/1e6
    operation=="delete" && return [PostEvent(key;observed_at,revision_at=revision,author=did)]
    r=commit["record"];text=str(getv(r,"text",""))
    # An edit removing text must retire any previously indexed text/vector.
    validtext(text) || return operation=="update" ? [PostEvent(key;observed_at,revision_at=revision,author=did)] : PostEvent[]
    tags=String[]
    for facet in getv(r,"facets",[]),feature in getv(facet,"features",[])
        getv(feature,"\$type","")=="app.bsky.richtext.facet#tag" && push!(tags,String(feature["tag"]))
    end
    append!(tags,String.(getv(r,"tags",[])))
    reply=str(getv(getv(getv(r,"reply",Dict()),"parent",Dict()),"uri",""))
    metadata=JSON3.write((cid=getv(commit,"cid",nothing),rev=getv(commit,"rev",nothing),langs=getv(r,"langs",[]),
        labels=getv(r,"labels",nothing),embed=getv(r,"embed",nothing)))
    [PostEvent(PostRecord(key;author=did,text,created_at=parse_time(r["createdAt"]),observed_at,revision_at=revision,
        permalink="https://bsky.app/profile/"*did*"/post/"*String(commit["rkey"]),reply_to=reply,tags,metadata))]
end
function mastodon_status(x;origin,observed_at)
    getv(x,"reblog",nothing)!==nothing && return mastodon_status(x["reblog"];origin,observed_at)
    key=PostKey(:activitypub,String(x["uri"]));account=getv(x,"account",Dict())
    created=parse_time(x["created_at"]);edited=getv(x,"edited_at",nothing)
    revision=edited===nothing ? created : parse_time(edited)
    text=plain_text(str(getv(x,"content","")))
    if getv(x,"visibility","")!="public" || !validtext(text)
        return [PostEvent(key;observed_at,revision_at=revision)]
    end
    alias=PostKey(:mastodon_local,rstrip(origin,'/')*"/api/v1/statuses/"*String(x["id"]))
    metadata=JSON3.write((api_id=x["id"],origin=origin,language=getv(x,"language",nothing),
        likes=getv(x,"favourites_count",nothing),replies=getv(x,"replies_count",nothing),
        reposts=getv(x,"reblogs_count",nothing),engagement_observed_at=observed_at,
        sensitive=getv(x,"sensitive",false),spoiler_text=getv(x,"spoiler_text","")))
    reply=getv(x,"in_reply_to_id",nothing)
    [PostEvent(PostRecord(key;platform="mastodon",author=str(getv(account,"uri",getv(account,"url",""))),text,
        created_at=created,observed_at,revision_at=revision,permalink=str(getv(x,"url","")),
        reply_to=reply===nothing ? "" : rstrip(origin,'/')*"/api/v1/statuses/"*String(reply),
        tags=[String(t["name"]) for t in getv(x,"tags",[])],metadata,aliases=[alias]))]
end
function mastodon_event(x;origin,observed_at)
    kind=getv(x,"event","");payload=x["payload"]
    if kind=="delete"
        key=PostKey(:mastodon_local,rstrip(origin,'/')*"/api/v1/statuses/"*String(payload))
        return [PostEvent(key;observed_at)]
    elseif kind in ("update","status.update")
        return mastodon_status(payload isa AbstractString ? JSON3.read(payload) : payload;origin,observed_at)
    end
    PostEvent[]
end
function strings(x)
    x===nothing && return String[]
    x isa AbstractString ? [String(x)] : String.(x)
end
function activitypub_event(x;observed_at,expected_actor=nothing)
    kind=getv(x,"type","");object=kind in ("Create","Update","Delete") ? getv(x,"object",nothing) : x
    object===nothing && return PostEvent[]
    actor=str(getv(x,"actor",getv(object isa AbstractString ? Dict() : object,"attributedTo","")))
    expected_actor!==nothing && actor!=expected_actor && throw(ArgumentError("ActivityPub actor does not match the configured outbox"))
    if kind=="Delete"
        id=object isa AbstractString ? object : object["id"]
        return [PostEvent(PostKey(:activitypub,String(id));observed_at,
            revision_at=parse_time(getv(x,"published",observed_at)),author=actor)]
    end
    object isa AbstractString && throw(ArgumentError("dereference an ActivityPub object before normalization"))
    getv(object,"type","") in ("Note","Article","Page") || return PostEvent[]
    author=str(getv(object,"attributedTo",actor));actor==author || throw(ArgumentError("ActivityPub actor/author mismatch"))
    audience=vcat(strings(getv(object,"to",[])),strings(getv(object,"cc",[])),strings(getv(x,"to",[])),strings(getv(x,"cc",[])))
    created=parse_time(object["published"]);revision=parse_time(getv(object,"updated",created))
    content=str(getv(object,"content",""));text=plain_text(content)
    if !("https://www.w3.org/ns/activitystreams#Public" in audience) || !validtext(text)
        return kind=="Update" ? [PostEvent(PostKey(:activitypub,String(object["id"]));observed_at,revision_at=revision,author)] : PostEvent[]
    end
    tags=[replace(str(getv(t,"name","")),r"^#"=>"") for t in getv(object,"tag",[]) if getv(t,"type","")=="Hashtag"]
    url=getv(object,"url",object["id"]);url isa AbstractString || (url=object["id"])
    metadata=JSON3.write((summary=getv(object,"summary",nothing),sensitive=getv(object,"sensitive",false)))
    [PostEvent(PostRecord(PostKey(:activitypub,String(object["id"]));author,text,created_at=created,
        observed_at,revision_at=revision,permalink=url,reply_to=str(getv(object,"inReplyTo","")),tags,metadata))]
end
function threads_post(x;observed_at)
    text=str(getv(x,"text",""));validtext(text) || return [PostEvent(PostKey(:threads,String(x["id"]));observed_at)]
    # Polling snapshots have no source edit sequence; observation time is their
    # revision clock. Identical text only refreshes metadata, never PPCA evidence.
    metadata=JSON3.write((shortcode=getv(x,"shortcode",nothing),media_type=getv(x,"media_type",nothing),
        is_quote_post=getv(x,"is_quote_post",false),has_replies=getv(x,"has_replies",nothing)))
    owner=getv(x,"owner",Dict());tag=str(getv(x,"topic_tag",""))
    [PostEvent(PostRecord(PostKey(:threads,String(x["id"]));author=str(getv(owner,"id",getv(x,"username",""))),
        text,created_at=parse_time(x["timestamp"]),observed_at,revision_at=observed_at,
        permalink=str(getv(x,"permalink","")),tags=isempty(tag) ? String[] : [tag],metadata))]
end
"""Ordinary notes (kind 1) and deletion requests (kind 5). Verifier must check both
the canonical event hash and BIP-340 signature; no trust-the-relay bypass is provided.
"""
function nostr_events(x;observed_at,verify)
    verify(x) || throw(ArgumentError("invalid Nostr event hash or signature"))
    kind=x["kind"];author=String(x["pubkey"]);revision=Float64(x["created_at"])
    if kind==5
        return [PostEvent(PostKey(:nostr,String(t[2]));observed_at,revision_at=revision,author)
            for t in x["tags"] if length(t)>=2 && t[1]=="e"]
    end
    kind==1 || return PostEvent[]
    text=String(x["content"]);validtext(text) || return PostEvent[]
    tags=[String(t[2]) for t in x["tags"] if length(t)>=2 && t[1]=="t"]
    replies=[String(t[2]) for t in x["tags"] if length(t)>=2 && t[1]=="e"]
    [PostEvent(PostRecord(PostKey(:nostr,String(x["id"]));author,text,created_at=revision,observed_at,revision_at=revision,
        reply_to=isempty(replies) ? "" : last(replies),tags,metadata=JSON3.write((kind=kind,signature=x["sig"]))))]
end
end
