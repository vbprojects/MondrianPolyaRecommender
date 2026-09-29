#!/usr/bin/env julia
for file in ("PostRecords.jl","LeafLatentTree.jl","DiskCodecs.jl","KeywordExtraction.jl","DiskTreeStore.jl")
    include(joinpath(@__DIR__,"../../src",file))
end
using SQLite, DBInterface, Tables, JSON3, SHA, Statistics, LinearAlgebra
using .PostRecords
const DS=DiskTreeStore
const DC=DiskCodecs
const KE=KeywordExtraction
rq(db,sql,args=())=Tables.rowtable(DBInterface.execute(db,sql,args))

function replay_tree(source,target,report;split_after=32,max_leaves=1024,max_depth=64,merge_below=2.)
    source=realpath(source);target=abspath(target)
    !ispath(target) || error("destination must be a new directory")
    isfile(joinpath(source,"index.sqlite")) || error("SQLite source required")
    # Exclude a writer, while the inspector may continue using read snapshots.
    guard=open(joinpath(source,".writer.lock"),"r+")
    ccall(:flock,Cint,(Cint,Cint),Base.fd(guard),2|4)==0 || (close(guard);error("stop source ingestion first"))
    db=nothing;out=nothing
    try
        uri="file:"*replace(joinpath(source,"index.sqlite"),"%"=>"%25","?"=>"%3F","#"=>"%23")*"?mode=ro"
        db=SQLite.DB(uri);SQLite.C.sqlite3_db_readonly(db.handle,"main")==1 || error("read-only open failed")
        rq(db,"PRAGMA query_only=ON");rq(db,"PRAGMA cache_size=-8192");rq(db,"BEGIN")
        runtime_before=only(rq(db,"SELECT value FROM meta WHERE key='runtime'")).value
        rt=DS.deserialize_bytes(runtime_before);rt.julia_version==string(VERSION) || error("Julia version mismatch")
        cfg=DS.config(;Dict(Symbol(k)=>v for (k,v) in JSON3.read(String(only(rq(db,"SELECT value FROM meta WHERE key='config'")).value)))...)
        cfg.mode==:f16 || error("exact replay currently requires Float16 source vectors")
        kw=JSON3.read(String(only(rq(db,"SELECT value FROM meta WHERE key='keywords'")).value))
        ex=KE.HashtagsOnlyExtractor();kw.fingerprint==KE.fingerprint(ex) || error("hashtag-only source required")
        settings=merge(cfg,(split_after=Int(split_after),max_leaves=Int(max_leaves),max_depth=Int(max_depth),merge_below=Float64(merge_below)))
        function leafstats(counts,depth)
            (leaves=length(counts),posts=sum(counts),minimum=minimum(counts),median=median(counts),
                mean=mean(counts),p95=quantile(counts,.95),maximum=maximum(counts),depth=depth)
        end
        before=leafstats([r.count for r in rq(db,"SELECT count FROM leaves")],only(rq(db,"SELECT MAX(depth) AS d FROM nodes")).d)
        out=DS.open_store(target;create_config=DS.config(;settings...),budget=256*1024^2)
        DS.enable_keywords!(out,ex;half_life=kw.half_life)
        frames=Dict{Int,Matrix{Float32}}();fifo=Int[]
        function vector(p)
            if !haskey(frames,p.block)
                raw=open(joinpath(source,p.vectorfile)) do io;seek(io,p.vectoroff);read(io,p.vectorbytes);end
                bytes2hex(sha256(raw))==p.vectorsha || error("source frame checksum mismatch")
                f=DC.read_frame(raw);length(fifo)>=16 && delete!(frames,popfirst!(fifo))
                frames[p.block]=DC.decode(f);push!(fifo,p.block)
            end
            frames[p.block][:,p.row]
        end
        function record(p)
            raw=open(joinpath(source,p.textfile)) do io;seek(io,p.textoff);read(io,p.textlen);end
            length(raw)==p.textlen || error("short source text")
            text=JSON3.read(String(raw));key=key_from_bytes(p.kind,p.native)
            PostRecord(key;platform=text.platform,author=text.author,text=text.text,created_at=p.created,
                observed_at=p.observed,revision_at=p.revision,permalink=text.permalink,reply_to=text.reply_to,
                tags=String.(text.tags),metadata=String(text.metadata))
        end
        count=0;after=0;started=time();trajectory=Any[]
        src_digest=SHA.SHA2_256_CTX();dst_digest=SHA.SHA2_256_CTX()
        while true
            page=rq(db,"SELECT p.*,b.file AS vectorfile,b.offset AS vectoroff,b.bytes AS vectorbytes,b.sha AS vectorsha FROM posts p JOIN blocks b ON b.id=p.block WHERE p.status=1 AND p.sequence>? ORDER BY p.sequence,p.id LIMIT 1024",(after,))
            isempty(page) && break
            a=1
            while a<=length(page)
                b=a
                while b<length(page) && page[b+1].density_at==page[a].density_at;b+=1;end
                ps=page[a:b];t=Float64(first(ps).density_at);t>=out.clock || error("nonmonotonic content ages")
                records=record.(ps);X=reduce(hcat,vector.(ps))
                prepared=KE.prepare_keywords(ex,KE.keyword_input.(records))
                result=DS.commit!(out) do
                    r=DS.append_batch!(out,records,X;time=t,keywords=prepared)
                    DS.expire!(out;time=t);DS.maintain!(out;time=t);r
                end
                result.encoded==length(ps) || error("replay dropped records")
                ids=[DS.lookup(out,p.key).id for p in records];Y=DS.fetch_vectors(out,ids)
                X==Y || error("Float16 values changed during replay")
                for (i,p) in enumerate(records)
                    got=DS.lookup(out,p.key)
                    got.density_at==ps[i].density_at && got.digest==ps[i].digest || error("content or age mismatch")
                    # Canonical identities, version metadata and decoded vectors.
                    row=(p.key.kind,p.key.native,p.created_at,p.observed_at,p.revision_at,ps[i].density_at,p.text,p.tags,p.metadata)
                    SHA.update!(src_digest,DS.serialize_bytes((row,X[:,i])))
                    SHA.update!(dst_digest,DS.serialize_bytes((row,Y[:,i])))
                end
                count+=length(ps);a=b+1
            end
            after=last(page).sequence
            row=(posts=count,leaves=count_leaves(out),seconds=time()-started,splits=get(out.metrics,"splits",0.),merges=get(out.metrics,"merges",0.))
            push!(trajectory,row)
            length(trajectory)%10==0 && (println(JSON3.write(row));flush(stdout))
        end
        DS.commit!(out) do;DS.expire!(out;time=rt.clock);DS.maintain!(out;time=rt.clock);end
        # Scan identity aliases once; the source has no index on identities.id.
        DS.commit!(out) do
            sql="SELECT i.kind,i.native,p.kind AS pk,p.native AS pn FROM identities i JOIN posts p ON p.id=i.id WHERE p.status=1 AND NOT(i.kind=p.kind AND i.native=p.native)"
            for r in DBInterface.execute(db,sql)
                id=DS.lookup(out,key_from_bytes(r.pk,r.pn)).id
                DS.exec(out,"INSERT INTO identities VALUES (?,?,?)",(r.kind,r.native,id))
            end
        end
        checked=DS.audit!(out);checked==count==before.posts || error("post count mismatch")
        # Compare global keyword counts and decayed support at the same clock.
        actual=Dict(r.term=>r for r in DS.rows(out,"SELECT t.term,c.n,c.mass,c.at FROM kw_counts c JOIN kw_terms t ON t.id=c.term WHERE c.leaf=0 AND c.term>0"))
        for r in rq(db,"SELECT t.term,c.n,c.mass,c.at FROM kw_counts c JOIN kw_terms t ON t.id=c.term WHERE c.leaf=0 AND c.term>0")
            a=pop!(actual,r.term);a.n==r.n || error("hashtag count mismatch")
            isapprox(a.mass*exp2(-(rt.clock-a.at)/kw.half_life),r.mass*exp2(-(rt.clock-r.at)/kw.half_life);rtol=1e-8,atol=1e-8) || error("hashtag decay mismatch")
        end
        isempty(actual) || error("unexpected hashtag terms")
        only(rq(db,"SELECT value FROM meta WHERE key='runtime'")).value==runtime_before || error("source runtime changed")
        h1=bytes2hex(SHA.digest!(src_digest));h2=bytes2hex(SHA.digest!(dst_digest));h1==h2 || error("replay digest mismatch")
        counts=[r.count for r in DS.rows(out,"SELECT count FROM leaves")]
        result=(source,target,source_generation=rt.generation,clock=rt.clock,settings,
            scope="Retained active content versions in sequence order at original density_at times; historical deletes/edits and empty collection rounds are not replayed. Saved Float16 vectors reused; internal numeric IDs are new.",
            before,after=leafstats(counts,maximum(n.depth for n in values(out.nodes))),elapsed_seconds=time()-started,
            splits=get(out.metrics,"splits",0.),merges=get(out.metrics,"merges",0.),verified_posts=checked,
            decoded_vectors_identical=true,global_hashtags_identical=true,identity_vector_sha256=h1,
            storage=DS.storage_stats(out),
            replay_memory=(state_cache_capacity=out.state_cache.capacity,state_cache_bytes=out.state_cache.used,
                state_cache_hits=out.state_cache.hits,state_cache_misses=out.state_cache.misses,
                state_evictions=out.state_cache.evictions,peak_process_rss_bytes=Sys.maxrss()),trajectory)
        mkpath(dirname(abspath(report)));write(report,JSON3.write(result));println(JSON3.write((complete=true,report,after=result.after)));flush(stdout)
        result
    finally
        out!==nothing && DS.close_store(out)
        db!==nothing && SQLite.close(db)
        close(guard)
    end
end
count_leaves(s)=count(n->n.axis==0,values(s.nodes))
if abspath(PROGRAM_FILE)==(@__FILE__)
    (length(ARGS)==3 || (length(ARGS)==5 && ARGS[4]=="--max-leaves")) ||
        error("usage: replay_tree.jl SOURCE NEW_DESTINATION REPORT.json [--max-leaves N]")
    BLAS.set_num_threads(1)
    replay_tree(ARGS[1:3]...;max_leaves=length(ARGS)==5 ? parse(Int,ARGS[5]) : 1024)
end
