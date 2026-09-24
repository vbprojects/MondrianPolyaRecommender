"""Canonical post identities and normalized ingestion events. No tree/node IDs here."""
module PostRecords
using Dates, SHA
export PostKey, PostRecord, PostEvent, platform_name, native_bytes, key_from_bytes,
       parse_time, content_digest

const KIND=Dict(:activitypub=>UInt8(1),:threads=>UInt8(2),:nostr=>UInt8(3),:bluesky=>UInt8(4),:mastodon_local=>UInt8(5))
struct PostKey
    kind::UInt8
    native::String
    function PostKey(kind::UInt8,native::AbstractString)
        kind in values(KIND) || throw(ArgumentError("unknown identifier namespace"))
        s=String(native)
        !isempty(s) && ncodeunits(s)<=8192 && !occursin(r"[\x00-\x20\x7f]",s) || throw(ArgumentError("invalid native identifier"))
        if kind==KIND[:nostr]
            occursin(r"^[0-9a-fA-F]{64}$",s) || throw(ArgumentError("Nostr key requires a 32-byte event ID, not a relay or bech32 wrapper"))
            s=lowercase(s)
        elseif kind==KIND[:bluesky]
            m=match(r"^at://(did:[a-z0-9]+:[^/?#]+)/([A-Za-z0-9.-]+)/([A-Za-z0-9._:~\-]{1,512})$",s)
            m!==nothing && !(m[3] in (".","..")) && m[2]=="app.bsky.feed.post" || throw(ArgumentError("expected a DID-based Bluesky post AT-URI"))
        elseif kind in (KIND[:activitypub],KIND[:mastodon_local])
            occursin(r"^(?i:https?)://[^/?#]+(?:[/?#].*)?$",s) || throw(ArgumentError("expected an absolute HTTP(S) object identifier"))
        end
        new(kind,s)
    end
end
PostKey(kind::Symbol,s)=PostKey(get(KIND,kind,UInt8(0)),s)
Base.:(==)(a::PostKey,b::PostKey)=a.kind==b.kind && a.native==b.native
Base.isequal(a::PostKey,b::PostKey)=a==b
Base.hash(k::PostKey,h::UInt)=hash(k.native,hash(k.kind,h))
platform_name(k::PostKey)=String(only(name for (name,value) in KIND if value==k.kind))
native_bytes(k::PostKey)=k.kind==KIND[:nostr] ? hex2bytes(k.native) : collect(codeunits(k.native))
key_from_bytes(kind,bytes)=PostKey(UInt8(kind),kind==KIND[:nostr] ? bytes2hex(bytes) : String(Vector{UInt8}(bytes)))

function parse_time(t::Real)
    isfinite(t) || throw(ArgumentError("nonfinite timestamp"));Float64(t)
end
function parse_time(t::AbstractString)
    m=match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})$",t)
    m===nothing && throw(ArgumentError("timestamp must include an explicit UTC offset"))
    value=datetime2unix(DateTime(m[1],dateformat"yyyy-mm-ddTHH:MM:SS"))
    m[2]!==nothing && (value+=parse(Float64,"0."*m[2]))
    if m[3]!="Z"
        zone=replace(m[3],":"=>"");hour=parse(Int,zone[2:3]);minute=parse(Int,zone[4:5])
        hour<=23 && minute<=59 || throw(ArgumentError("invalid timezone offset"))
        value-=(zone[1]=='+' ? 1 : -1)*(3600hour+60minute)
    end
    value
end
struct PostRecord
    key::PostKey
    platform::String
    author::String
    text::String
    created_at::Float64
    observed_at::Float64
    revision_at::Float64
    permalink::String
    reply_to::String
    tags::Vector{String}
    metadata::String
    aliases::Vector{PostKey}
end
function PostRecord(key::PostKey;platform=platform_name(key),author="",text,
        created_at,observed_at,revision_at=created_at,permalink="",reply_to="",
        tags=String[],metadata="{}",aliases=PostKey[])
    !isempty(strip(text)) || throw(ArgumentError("empty post text"))
    PostRecord(key,String(platform),String(author),String(text),parse_time(created_at),parse_time(observed_at),
        parse_time(revision_at),String(permalink),String(reply_to),unique(String.(tags)),String(metadata),unique(PostKey[aliases...]))
end
content_digest(p::PostRecord)=bytes2hex(sha256(codeunits(p.text)))
struct PostEvent
    operation::Symbol
    key::PostKey
    record::Union{Nothing,PostRecord}
    observed_at::Float64
    revision_at::Float64
    author::String
    function PostEvent(operation,key,record,observed_at,revision_at,author)
        operation in (:upsert,:delete) || throw(ArgumentError("unsupported event operation"))
        operation==:upsert && (record===nothing || record.key!=key) && throw(ArgumentError("upsert requires a matching post"))
        operation==:delete && record!==nothing && throw(ArgumentError("delete must not carry a post"))
        new(operation,key,record,parse_time(observed_at),parse_time(revision_at),String(author))
    end
end
PostEvent(p::PostRecord)=PostEvent(:upsert,p.key,p,p.observed_at,p.revision_at,p.author)
PostEvent(key::PostKey;observed_at,revision_at=observed_at,author="")=PostEvent(:delete,key,nothing,observed_at,revision_at,author)
end
