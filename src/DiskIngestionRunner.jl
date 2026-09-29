"""Live source batches backed by one SQLite publication, immutable Float16 files,
and extractor-independent keyword contributions. No full post catalog is resident."""
module DiskIngestionRunner
using JSON3
using ..PostRecords, ..DiskTreeStore, ..KeywordExtraction, ..SourceAdapters
const DS=DiskTreeStore
const KE=KeywordExtraction
export commit_batch!, maintenance!, reclaim_daily!, source_cursor, status, normalize_fixture
source_cursor(s,name)=let r=DS.rows(s,"SELECT cursor FROM cursors WHERE source=?",(name,))
    isempty(r) ? "" : String(only(r).cursor)
end
function set_cursor!(s,name,value)
    DS.exec(s,"INSERT OR REPLACE INTO cursors VALUES (?,?)",(name,String(value)))
end
function status(s)
    # Bounded topology and aggregate counts, avoiding a catalog-wide audit per batch.
    (generation=s.generation,clock=s.clock,
     active=DS.one(s,"SELECT COALESCE(SUM(count),0) AS n FROM leaves").n,
     leaves=count(n->n.axis==0,values(s.nodes)),keywords=DS.keyword_enabled(s),
     cache_bytes=s.state_cache.used+s.block_cache.used,managed_budget=s.budget)
end
function canonical_record(p,key)
    PostRecord(key;platform=p.platform,author=p.author,text=p.text,created_at=p.created_at,
        observed_at=p.observed_at,revision_at=p.revision_at,permalink=p.permalink,
        reply_to=p.reply_to,tags=p.tags,metadata=p.metadata,
        aliases=filter(!=(key),unique(vcat([p.key],p.aliases))))
end
function existing(s,e)
    found=DS.lookup(s,e.key)
    if e.operation==:upsert
        for key in e.record.aliases
            p=DS.lookup(s,key)
            p===nothing && continue
            found!==nothing && found.id!=p.id && error("conflicting identity aliases")
            found=p
        end
    end
    found
end
function select_events(s,events,now)
    length(events)<=1024 || error("live batch exceeds 1024 normalized events")
    primary=Dict{PostKey,PostKey}();authors=Dict{PostKey,String}()
    for e in events
        e.operation==:upsert || continue
        old=existing(s,e);key=old===nothing ? e.key : key_from_bytes(old.kind,old.native)
        for alias in vcat([e.key],e.record.aliases)
            get(primary,alias,key)==key || error("conflicting batch aliases")
            primary[alias]=key
        end
        key.kind==3 && (authors[key]=e.record.author)
    end
    selected=Dict{PostKey,PostEvent}();order=PostKey[]
    for e in events
        old=existing(s,e)
        key=old===nothing ? get(primary,e.key,e.key) : key_from_bytes(old.kind,old.native)
        if e.operation==:delete && key.kind==3
            author=old!==nothing && old.status==1 ?
                String(only(DS.fetch_records(s,[old.id])).author) : get(authors,key,nothing)
            (author===nothing || author!=e.author) && continue
        end
        if !haskey(selected,key)
            push!(order,key);selected[key]=e
        elseif e.revision_at>selected[key].revision_at ||
            (e.revision_at==selected[key].revision_at && (e.operation==:delete ||
                (selected[key].operation==:upsert && e.observed_at>selected[key].observed_at)))
            selected[key]=e
        end
    end
    records=PostRecord[];changed=Bool[];deletes=PostEvent[]
    ignored=length(events)-length(order)
    for key in order
        e=selected[key];old=DS.lookup(s,key)
        if old!==nothing && (e.revision_at<old.revision ||
                (e.revision_at==old.revision && (old.status!=1 ||
                  (e.operation==:upsert && e.observed_at<old.observed))))
            ignored+=1;continue
        end
        if e.operation==:delete
            push!(deletes,PostEvent(key;observed_at=e.observed_at,revision_at=e.revision_at,author=e.author));continue
        end
        p=canonical_record(e.record,key)
        if p.created_at<now-s.config.retention_seconds
            ignored+=1;continue
        end
        digest=content_digest(p)
        old!==nothing && old.status==1 && p.revision_at==old.revision && digest!=old.digest &&
            error("conflicting content at identical source revision")
        # Exact replay needs neither model inference, extraction, nor a text rewrite.
        aliases_known=all(k->DS.lookup(s,k)!==nothing,p.aliases)
        if old!==nothing && old.status==1 && old.digest==digest &&
                p.revision_at==old.revision && p.observed_at==old.observed && aliases_known
            ignored+=1;continue
        end
        push!(records,p);push!(changed,old===nothing || old.status!=1 || old.digest!=digest)
    end
    (;records,changed,deletes,primary,ignored)
end
function prepare_batch(s,e,records)
    DS.keyword_enabled(s) || return nothing
    e!==nothing && KE.fingerprint(e)==s.keyword_config.fingerprint || error("keyword extractor mismatch")
    output=Vector{KE.PreparedKeywords}(undef,length(records));missing=Int[];inputs=KE.keyword_input.(records)
    for (i,input) in enumerate(inputs)
        old=DS.lookup(s,records[i].key)
        cached=old===nothing ? [] : DS.rows(s,"SELECT * FROM kw_posts WHERE id=?",(old.id,))
        if !isempty(cached) && only(cached).digest==KE.input_digest(input)
            k=only(cached)
            terms=[String(DS.one(s,"SELECT term FROM kw_terms WHERE id=?",(id,)).term) for id in DS.deserialize_bytes(k.terms)]
            output[i]=KE.PreparedKeywords(input.identity,k.digest,KE.fingerprint(e),terms,k.eligible==1,k.assumed==1,k.truncated==1)
        else
            push!(missing,i)
        end
    end
    if !isempty(missing)
        result=KE.prepare_keywords(e,inputs[missing])
        for (i,k) in zip(missing,result);output[i]=k;end
    end
    output
end
function maintenance!(s;now=max(s.clock,time()))
    isfinite(now) && now>=s.clock || error("invalid maintenance clock")
    DS.mutation!(s) do
        DS.expire!(s;time=now) # Logical expiry before publishing keyword summaries.
        DS.maintain!(s;time=now)
    end
    s
end
function commit_batch!(s,source,batch,embed;extractor=KE.JuliaYakeExtractor(),now=max(s.clock,time()),fault=nothing)
    !startswith(source,"__") || error("reserved source name")
    isfinite(now) && now>=s.clock || error("invalid ingestion clock")
    plan=select_events(s,batch.events,now)
    # All potentially failing preparation finishes before opening the write transaction.
    started=time_ns()
    keywords=prepare_batch(s,extractor,plan.records)
    DS.metric!(s,"keyword_prepare_seconds",(time_ns()-started)/1e9)
    X=zeros(Float32,s.config.dimension,length(plan.records))
    cols=findall(plan.changed)
    if !isempty(cols)
        started=time_ns();embedded=Float32.(embed([plan.records[i].text for i in cols]))
        size(embedded)==(s.config.dimension,length(cols)) && all(isfinite,embedded) && all(x->abs(x)<=1,embedded) ||
            error("embedding callback returned invalid coordinates")
        X[:,cols]=embedded;DS.metric!(s,"embedding_seconds",(time_ns()-started)/1e9)
    end
    result=DS.commit!(s;fault) do
        deleted=0
        for e in plan.deletes
            deleted+=DS.delete_post!(s,e.key;revision_at=e.revision_at,time=now)
            id=DS.lookup(s,e.key).id
            for (alias,key) in plan.primary
                key==e.key || continue
                prior=DS.lookup(s,alias)
                prior!==nothing && prior.id!=id && error("identity alias conflict")
                DS.exec(s,"INSERT OR REPLACE INTO identities VALUES (?,?,?)",(Int(alias.kind),native_bytes(alias),id))
            end
        end
        counts=isempty(plan.records) ? (encoded=0,ignored=0,metadata=0) :
            DS.append_batch!(s,plan.records,X;time=now,keywords)
        maintenance!(s;now)
        set_cursor!(s,source,batch.cursor)
        (encoded=counts.encoded,metadata=counts.metadata,deleted=deleted,ignored=counts.ignored+plan.ignored)
    end
    (counts=result,index=status(s))
end
"""Physical reclamation is separate from the source publication; interruption
retries on restart. Frozen vectors are moved without re-encoding."""
function reclaim_daily!(s;force=false)
    day=string(floor(Int,s.clock/86400))
    !force && source_cursor(s,"__reclaimed_utc_day")==day && return false
    DS.compact!(s)
    DS.compact_texts!(s)
    DS.commit!(s) do;set_cursor!(s,"__reclaimed_utc_day",day);end
    true
end
function normalize_fixture(row;verify)
    now=Float64(row["observed_at"]);kind=row["kind"];payload=row["payload"]
    events=kind=="bluesky" ? bluesky_event(payload;observed_at=now) :
        kind=="mastodon" ? mastodon_status(payload;origin=row["origin"],observed_at=now) :
        kind=="mastodon_stream" ? mastodon_event(payload;origin=row["origin"],observed_at=now) :
        kind=="activitypub" ? activitypub_event(payload;observed_at=now,expected_actor=row["actor"]) :
        kind=="threads" ? threads_post(payload;observed_at=now) :
        kind=="nostr" ? nostr_events(payload;observed_at=now,verify) : error("unknown replay kind")
    events,now
end
end
