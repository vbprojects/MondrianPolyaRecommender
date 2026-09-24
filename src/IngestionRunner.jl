"""Single-writer orchestration. Arrow generation = posts + tree + source cursors.
An exception during commit is fatal to this in-memory instance; restart from CURRENT.
"""
module IngestionRunner
using JSON3
using ..PostRecords, ..SourceAdapters, ..LiveIndex, ..ArrowPostStore
export with_store_lock, commit_batch!, daily_maintenance!, normalize_fixture
function with_store_lock(f,root)
    Sys.isunix() || throw(ArgumentError("the CLI writer lock currently requires POSIX flock"))
    mkpath(root)
    open(joinpath(root,".writer.lock"),"a+") do io
        ccall(:flock,Cint,(Cint,Cint),Base.fd(io),2|4)==0 || error("another ingestion writer owns this Arrow store")
        try f()
        finally ccall(:flock,Cint,(Cint,Cint),Base.fd(io),8) end
    end
end
function daily_maintenance!(index;now)
    last=get(index.cursors,"__maintenance_utc_day","");day=floor(Int,now/86400)
    if last!=string(day)
        # At each UTC day boundary, remove posts older than the retention window.
        expire!(index;before=now-index.config.retention_seconds,time=max(now,index.clock))
        gc_tombstones!(index;before=index.clock-max(7*86400.,index.config.retention_seconds))
        index.cursors["__maintenance_utc_day"]=string(day)
    else
        maintain!(index;time=max(now,index.clock))
    end
    index
end
function commit_batch!(index,root,source,batch,embed;now=max(index.clock,time()),keep=2,checkpoint_fault=nothing)
    # Do not catch and continue after a partially mutated kernel/storage operation.
    # The durable old cursor is retained unless checkpoint publication succeeds.
    result=ingest!(index,batch.events,embed;time=now)
    daily_maintenance!(index;now)
    index.cursors[source]=batch.cursor
    path=checkpoint!(index,root;fault_after=checkpoint_fault)
    prune_checkpoints!(root;keep)
    (counts=result,path=path,audit=audit(index))
end
function normalize_fixture(row;verify)
    now=Float64(row["observed_at"]);kind=row["kind"];payload=row["payload"]
    events=kind=="bluesky" ? bluesky_event(payload;observed_at=now) :
        kind=="mastodon" ? mastodon_status(payload;origin=row["origin"],observed_at=now) :
        kind=="mastodon_stream" ? mastodon_event(payload;origin=row["origin"],observed_at=now) :
        kind=="activitypub" ? activitypub_event(payload;observed_at=now,expected_actor=row["actor"]) :
        kind=="threads" ? threads_post(payload;observed_at=now) :
        kind=="nostr" ? nostr_events(payload;observed_at=now,verify) : throw(ArgumentError("unknown replay kind"))
    events,now
end
end
