# Included inside DiskTreeStore. All mutations share the store's publication transaction.
if !isdefined(parentmodule(@__MODULE__), :KeywordExtraction)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "KeywordExtraction.jl"))
end
using ..KeywordExtraction
export enable_keywords!, backfill_keywords!, leaf_keywords, keyword_stats, keyword_update!, KeywordExtraction
keyword_enabled(s)=s.keyword_config!==nothing
function load_keywords!(s)
    r=rows(s,"SELECT value FROM meta WHERE key='keywords'")
    s.keyword_config=isempty(r) ? nothing : JSON3.read(String(only(r).value))
end
function enable_keywords!(s,extractor::AbstractKeywordExtractor;half_life=s.config.half_life,rebuild=false)
    isfinite(half_life) && half_life>0 || error("invalid keyword half-life")
    cfg=(fingerprint=fingerprint(extractor),extractor=extractor_id(extractor),half_life=Float64(half_life),policy=KeywordExtraction.POLICY)
    if keyword_enabled(s) && !rebuild
        s.keyword_config.fingerprint==cfg.fingerprint && s.keyword_config.half_life==half_life || error("explicit rebuild required for new keyword configuration")
        return s
    end
    commit!(s) do
        for sql in [
            "CREATE TABLE IF NOT EXISTS kw_terms(id INTEGER PRIMARY KEY,term TEXT UNIQUE NOT NULL)",
            "CREATE TABLE IF NOT EXISTS kw_posts(id INTEGER PRIMARY KEY,digest TEXT NOT NULL,eligible INTEGER,assumed INTEGER,truncated INTEGER,terms BLOB NOT NULL)",
            "CREATE TABLE IF NOT EXISTS kw_counts(leaf INTEGER,term INTEGER,n INTEGER,mass REAL,at REAL,PRIMARY KEY(leaf,term)) WITHOUT ROWID"
        ];exec(s,sql);end
        if rebuild
            for table in ("kw_counts","kw_posts","kw_terms");exec(s,"DELETE FROM $table");end
        end
        exec(s,"INSERT OR REPLACE INTO meta VALUES ('keywords',?)",(collect(codeunits(JSON3.write(cfg))),))
        exec(s,"INSERT OR REPLACE INTO meta VALUES ('keyword_backfill_after',?)",(serialize_bytes(Int64(0)),))
        s.keyword_config=cfg
    end
    s
end
# term -1 counts covered posts (including non-English); term 0 counts English-eligible posts.
# Hashtags also use the covered population: term frequencies must never exceed denominators.
function kw_delta!(s,leaf,term,n,w)
    r=rows(s,"SELECT n,mass,at FROM kw_counts WHERE leaf=? AND term=?",(leaf,term))
    old=isempty(r) ? nothing : only(r)
    count=n+(old===nothing ? 0 : old.n)
    count>=0 || error("negative keyword count")
    mass=w+(old===nothing ? 0. : old.mass*exp2(-(s.clock-old.at)/s.keyword_config.half_life))
    if count==0
        exec(s,"DELETE FROM kw_counts WHERE leaf=? AND term=?",(leaf,term))
    else
        mass>=-1e-8 || error("negative keyword mass")
        exec(s,"INSERT OR REPLACE INTO kw_counts VALUES (?,?,?,?,?)",(leaf,term,count,max(0.,mass),s.clock))
    end
end
function kw_contribute!(s,id,leaf,sign;global_scope=false)
    keyword_enabled(s) || return
    rs=rows(s,"SELECT k.*,p.density_at FROM kw_posts k JOIN posts p ON p.id=k.id WHERE k.id=?",(id,))
    isempty(rs) && return
    p=only(rs);terms=deserialize_bytes(p.terms)
    w=sign*exp2(-(s.clock-p.density_at)/s.keyword_config.half_life)
    for scope in (global_scope ? (leaf,0) : (leaf,))
        kw_delta!(s,scope,-1,sign,w)
        p.eligible==1 && kw_delta!(s,scope,0,sign,w)
        for term in terms;kw_delta!(s,scope,term,sign,w);end
    end
end
function kw_forget!(s,id)
    keyword_enabled(s) || return
    # remove_slot! already removed the leaf contribution.
    kw_contribute!(s,id,0,-1)
    exec(s,"DELETE FROM kw_posts WHERE id=?",(id,))
end
function validate_keywords(s,records,keywords)
    if !keyword_enabled(s)
        keywords===nothing || error("enable keywords before supplying a keyword batch")
        return
    end
    keywords!==nothing && length(keywords)==length(records) || error("keyword-enabled ingestion requires aligned keyword results")
    for (p,k) in zip(records,keywords)
        x=keyword_input(p)
        k isa PreparedKeywords && k.identity==x.identity && k.digest==input_digest(x) && k.fingerprint==s.keyword_config.fingerprint || error("keyword identity/content/configuration mismatch")
        length(k.terms)<=12 && length(unique(k.terms))==length(k.terms) && all(t->!isempty(t) && ncodeunits(t)<=256 && t==KeywordExtraction.canonical(t),k.terms) || error("invalid bounded keyword terms")
    end
end
function kw_attach!(s,id,k)
    p=lookup(s,id);p.status==1 || error("keywords require active post")
    old=rows(s,"SELECT digest FROM kw_posts WHERE id=?",(id,))
    !isempty(old) && only(old).digest==k.digest && return
    if !isempty(old);kw_contribute!(s,id,p.leaf,-1;global_scope=true);end
    ids=Int64[]
    for term in k.terms
        exec(s,"INSERT OR IGNORE INTO kw_terms(term) VALUES (?)",(term,))
        push!(ids,one(s,"SELECT id FROM kw_terms WHERE term=?",(term,)).id)
    end
    exec(s,"INSERT OR REPLACE INTO kw_posts VALUES (?,?,?,?,?,?)",(id,k.digest,Int(k.eligible),Int(k.assumed_english),Int(k.truncated),serialize_bytes(ids)))
    kw_contribute!(s,id,p.leaf,1;global_scope=true)
end
"""Metadata-only updates without providing or computing embeddings. Older versions cannot win."""
function keyword_update!(s,records,keywords;time=s.clock,fault=nothing)
    keyword_enabled(s) || error("keywords disabled")
    length(records)<=1024 || error("unbounded keyword update")
    validate_keywords(s,records,keywords)
    isfinite(time) && time>=s.clock || error("invalid clock")
    # Verify text before transaction; this path cannot replace semantic content.
    for p in records
        old=lookup(s,p.key)
        old!==nothing && old.status==1 && old.digest==content_digest(p) || error("metadata update requires unchanged active text")
    end
    commit!(s;fault) do
        s.clock=Float64(time)
        for (p,k) in zip(records,keywords)
            old=lookup(s,p.key)
            (p.revision_at<old.revision || (p.revision_at==old.revision && p.observed_at<old.observed)) && continue
            kw_attach!(s,old.id,k)
            file,off,len=write_text!(s,p)
            exec(s,"UPDATE posts SET revision=?,observed=?,textfile=?,textoff=?,textlen=? WHERE id=?",(p.revision_at,p.observed_at,file,off,len,old.id))
        end
    end
end
function stored_keyword_record(s,p)
    q=only(fetch_records(s,[p.id]))
    PostRecord(key_from_bytes(p.kind,p.native);text=q.text,tags=String.(q.tags),metadata=String(q.metadata),created_at=p.created,observed_at=p.observed,revision_at=p.revision)
end
"""Each committed page is resumable. No full catalog or text corpus is loaded."""
function backfill_keywords!(s,e::AbstractKeywordExtractor;batch_size=128,max_batches=typemax(Int))
    keyword_enabled(s) && fingerprint(e)==s.keyword_config.fingerprint || error("extractor mismatch")
    0<batch_size<=1024 || error("invalid batch size")
    expire_keywords_if_due!(s,s.clock)
    done=0
    cursor=rows(s,"SELECT value FROM meta WHERE key='keyword_backfill_after'")
    after=isempty(cursor) ? 0 : deserialize_bytes(only(cursor).value)
    for _ in 1:max_batches
        ps=rows(s,"SELECT p.* FROM posts p WHERE p.status=1 AND p.id>? AND NOT EXISTS(SELECT 1 FROM kw_posts k WHERE k.id=p.id) ORDER BY p.id LIMIT ?",(after,batch_size))
        isempty(ps) && break
        records=[stored_keyword_record(s,p) for p in ps]
        prepared=prepare_keywords(e,keyword_input.(records));validate_keywords(s,records,prepared)
        commit!(s) do
            for (p,k) in zip(ps,prepared);kw_attach!(s,p.id,k);end
            exec(s,"INSERT OR REPLACE INTO meta VALUES ('keyword_backfill_after',?)",(serialize_bytes(Int64(last(ps).id)),))
        end
        after=last(ps).id
        done+=length(ps)
    end
    done
end
function expire_keywords_if_due!(s,time)
    isfinite(time) && time>=s.clock || error("invalid clock")
    due=rows(s,"SELECT id FROM posts WHERE status=1 AND created<? LIMIT 1",(time-s.config.retention_seconds,))
    !isempty(due) && expire!(s;time)
end
function kw_count(s,leaf,term,time)
    r=rows(s,"SELECT n,mass,at FROM kw_counts WHERE leaf=? AND term=?",(leaf,term))
    isempty(r) ? (n=0,mass=0.) : (n=only(r).n,mass=only(r).mass*exp2(-(time-only(r).at)/s.keyword_config.half_life))
end
function keyword_page(s,leaf,after;distinctive=true)
    if !(s.db isa SQLite.DB)
        return RocksMetadata.keyword_page(s.db,leaf,after;distinctive)
    end
    sql=distinctive ?
        "SELECT c.term,c.n,c.mass,c.at,t.term AS label,g.mass AS gm,g.at AS ga FROM kw_counts c JOIN kw_terms t ON t.id=c.term JOIN kw_counts g ON g.leaf=0 AND g.term=c.term WHERE c.leaf=? AND c.term>? ORDER BY c.term LIMIT 256" :
        "SELECT c.term,c.n,c.mass,c.at,t.term AS label FROM kw_counts c JOIN kw_terms t ON t.id=c.term WHERE c.leaf=? AND c.term>? ORDER BY c.term LIMIT 256"
    rows(s,sql,(leaf,after))
end
function leaf_keywords(s,leaf_id;limit=10,ranking=:distinctive,time=s.clock)
    s.poisoned && error("reopen after a failed commit")
    keyword_enabled(s) || error("keywords disabled")
    1<=limit<=100 && ranking in (:common,:distinctive) || error("invalid summary request")
    expire_keywords_if_due!(s,time)
    haskey(s.nodes,leaf_id) && s.nodes[leaf_id].axis==0 || error("not a current leaf")
    key=(s.generation,Int(leaf_id),limit,ranking,Float64(time))
    s.keyword_cache!==nothing && first(s.keyword_cache)==key && return deepcopy(last(s.keyword_cache))
    covered=kw_count(s,leaf_id,-1,time);eligible=kw_count(s,leaf_id,0,time)
    global_count=ranking==:distinctive ? kw_count(s,0,-1,time) : nothing
    best=NamedTuple[];after=0
    while true
        page=keyword_page(s,leaf_id,after;distinctive=ranking==:distinctive)
        isempty(page) && break
        for r in page
            mass=r.mass*exp2(-(time-r.at)/s.keyword_config.half_life)
            score=mass
            if ranking==:distinctive
                gm=r.gm*exp2(-(time-r.ga)/s.keyword_config.half_life)
                lift=log(((mass+1)/(covered.mass+2))/((gm+1)/(global_count.mass+2)))
                score=mass*max(0.,lift)
            end
            ranking==:distinctive && (r.n<3 || score<=0) && continue
            push!(best,(term=String(r.label),count=r.n,support=mass,score=score))
            sort!(best;by=x->(-x.score,x.term));length(best)>limit && pop!(best)
        end
        after=last(page).term
    end
    total=one(s,"SELECT count FROM leaves WHERE id=?",(leaf_id,)).count
    result=(leaf=leaf_id,terms=best,covered=covered.n,eligible=eligible.n,total=total,coverage=total==0 ? 1. : covered.n/total,
        decayed_covered=covered.mass,extractor=s.keyword_config.extractor,generation=s.generation,at=Float64(time),ranking=String(ranking))
    # A single <=100-term result fits the existing 16 MiB bounded work reservation.
    s.keyword_cache=(key,deepcopy(result));result
end
function keyword_stats(s)
    s.poisoned && error("reopen after a failed commit")
    keyword_enabled(s) || return (enabled=false,)
    covered=kw_count(s,0,-1,s.clock)
    total=one(s,"SELECT COALESCE(SUM(count),0) AS n FROM leaves").n
    vocabulary=one(s,"SELECT COUNT(*) AS active_terms,COALESCE(SUM(n),0) AS assignments,COALESCE(SUM(n=1),0) AS singletons FROM kw_counts WHERE leaf=0 AND term>0")
    (enabled=true,covered=covered.n,total=total,coverage=total==0 ? 1. : covered.n/total,
     active_terms=vocabulary.active_terms,assignments=vocabulary.assignments,singleton_terms=vocabulary.singletons,
     terms=one(s,"SELECT COUNT(*) AS n FROM kw_terms").n,config=s.keyword_config)
end

# SQLite pages become reusable; shrinking the database file is a separate operation.
function keyword_gc!(s)
    after=0
    while true
        unused=rows(s,"SELECT id FROM kw_terms t WHERE t.id>? AND NOT EXISTS(SELECT 1 FROM kw_counts c WHERE c.leaf=0 AND c.term=t.id) ORDER BY t.id LIMIT 256",(after,))
        isempty(unused) && break
        commit!(s) do
            for r in unused;exec(s,"DELETE FROM kw_terms WHERE id=?",(r.id,));end
        end
        after=last(unused).id
    end
end
