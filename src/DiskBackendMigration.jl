# Included in DiskTreeStore. An offline copy into an unpublished destination.
"""Copy an existing SQLite store to a NEW RocksDB directory, preserving IDs,
encoded files, moments, keywords, generations and source cursors. Holds the source
writer lock, streams rows, validates the result, then atomically publishes it.
No encoder, extractor, quantization or density rebuild runs during this copy.
"""
function import_sqlite_to_rocks(source,destination;budget=256MiB,page_size=256)
    isdefined(@__MODULE__,:RocksMetadata) || error("RocksDB.jl is required")
    1<=page_size<=1024 || error("invalid import page size")
    source=abspath(source);destination=abspath(destination)
    detect_backend(source)==:sqlite || error("source must be an existing SQLite disk store")
    ispath(destination) && error("destination already exists; choose a new directory")
    parent=dirname(destination);mkpath(parent)
    # A destination-specific lock prevents two importers publishing over each other.
    publish_lock=open(destination*".import.lock","a+")
    if ccall(:flock,Cint,(Cint,Cint),Base.fd(publish_lock),2|4)!=0
        close(publish_lock);error("destination import already running")
    end
    original=nothing;db=nothing;stage=mktempdir(parent;prefix=".rocks-import-")
    try
        ispath(destination) && error("destination already exists")
        original=open_store(source;backend=:sqlite,budget,require_existing=true)
        db=RocksMetadata.open_db(joinpath(stage,"metadata.rocks"),min(64MiB,(budget-16MiB)÷4))
        totals=Dict{String,Int}()
        existing=Set(String(r.name) for r in rows(original,"SELECT name FROM sqlite_master WHERE type='table'"))
        for table in sort(collect(keys(RocksMetadata.SCHEMA)))
            table in existing || continue
            n=0;RocksMetadata.begin!(db)
            # Stream from the source DB; statement owns at most one row at a time.
            for r in DBInterface.execute(original.db,"SELECT * FROM $table")
                nt=NamedTuple{RocksMetadata.SCHEMA[table]}(Tuple(r[c] for c in RocksMetadata.SCHEMA[table]))
                RocksMetadata.putrow!(db,table,nt);n+=1
                if n%page_size==0;RocksMetadata.commit!(db);RocksMetadata.begin!(db);end
            end
            RocksMetadata.commit!(db);totals[table]=n
        end
        RocksMetadata.close_db(db);db=nothing
        copied=0
        for r in DBInterface.execute(original.db,"SELECT name,bytes FROM files ORDER BY name")
            name=String(r.name)
            basename(name)==name || error("invalid stored payload path")
            input=joinpath(source,name);output=joinpath(stage,name)
            # Independent files: source reclamation cannot affect this store.
            cp(input,output)
            open(output,"r+") do io
                ccall(:fsync,Cint,(Cint,),Base.fd(io))==0 || error("copied payload fsync failed")
            end
            filesize(output)==r.bytes || error("copied payload size mismatch")
            open(sha256,input)==open(sha256,output) || error("copied payload hash mismatch")
            copied+=r.bytes
        end
        syncdir(stage)
        check=open_store(stage;backend=:rocksdb,budget)
        audited=try
            check.generation==original.generation && tree_signature(check)==tree_signature(original) || error("import topology/generation mismatch")
            audit!(check;check_vectors=false)
        finally close_store(check) end
        ispath(destination) && error("destination appeared during import")
        mv(stage,destination);syncdir(parent)
        (source,destination,backend="rocksdb",tables=totals,active=audited,payload_bytes=copied,generation=original.generation)
    finally
        db!==nothing && RocksMetadata.close_db(db)
        original!==nothing && close_store(original)
        # Only this invocation's private staging directory may be removed.
        isdir(stage) && rm(stage;recursive=true)
        close(publish_lock)
    end
end
