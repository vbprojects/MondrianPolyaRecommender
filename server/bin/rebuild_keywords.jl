#!/usr/bin/env julia
# Offline, resumable keyword-only migration. Never starts embedding or source workers.
for file in ("PostRecords.jl","LeafLatentTree.jl","DiskCodecs.jl","KeywordExtraction.jl","DiskTreeStore.jl")
    include(joinpath(@__DIR__,"../../src",file))
end
using TOML, JSON3, SHA
const DS=DiskTreeStore
const KE=KeywordExtraction
function main(args)
    opts=Dict{String,String}();rebuild=false;i=1
    while i<=length(args)
        key=args[i]
        if key in ("--help","-h")
            println("julia --project=server server/bin/rebuild_keywords.jl --store PATH [--config examples/live_sources.toml] [--rebuild] [--max-batches N]")
            return
        elseif key=="--rebuild";rebuild=true
        else
            key in ("--store","--config","--max-batches") && i<length(args) || error("invalid argument")
            i+=1;opts[key]=args[i]
        end
        i+=1
    end
    cfg=TOML.parsefile(get(opts,"--config","examples/live_sources.toml"))
    kw=get(cfg,"keywords",Dict());extractor=KE.configured_extractor(kw)
    limit=parse(Int,get(opts,"--max-batches",string(typemax(Int))));limit>0 || error("positive batch limit required")
    root=get(opts,"--store",get(cfg,"store","data/live_disk_gte"))
    s=DS.open_store(root;budget=64*1024^2,require_existing=true)
    try
        DS.metadata_backend(s)==:sqlite || error("this offline migration verifier currently requires SQLite")
        # Fingerprint non-keyword state with a streaming digest: no full catalog in RAM.
        function invariant()
            ctx=SHA.SHA2_256_CTX()
            for sql in ("SELECT * FROM posts ORDER BY id","SELECT * FROM states ORDER BY id",
                        "SELECT * FROM nodes ORDER BY id","SELECT * FROM leaves ORDER BY id",
                        "SELECT * FROM blocks ORDER BY id","SELECT * FROM files ORDER BY name",
                        "SELECT * FROM cursors ORDER BY source")
                for row in DS.DBInterface.execute(s.db,sql)
                    SHA.update!(ctx,DS.serialize_bytes(NamedTuple(row)))
                end
            end
            bytes2hex(SHA.digest!(ctx))
        end
        before=invariant();started=time()
        DS.enable_keywords!(s,extractor;half_life=get(kw,"half_life",s.config.half_life),rebuild)
        total=0
        for batch in 1:limit
            n=DS.backfill_keywords!(s,extractor;batch_size=128,max_batches=1);total+=n
            if batch==1 || batch%100==0 || n==0
                println(JSON3.write((backfilled=total,elapsed_seconds=time()-started,keywords=DS.keyword_stats(s))));flush(stdout)
            end
            n==0 && break
        end
        after=invariant();before==after || error("non-keyword state changed during migration")
        println(JSON3.write((complete=DS.keyword_stats(s).coverage==1,extractor=KE.extractor_id(extractor),
            preserved_state_sha256=after,elapsed_seconds=time()-started,keywords=DS.keyword_stats(s))));flush(stdout)
    finally;DS.close_store(s);end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main(ARGS)
