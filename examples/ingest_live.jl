#!/usr/bin/env julia
# Disk storage is the default for new stores. Existing Arrow roots remain readable.
using TOML
function dispatch(args)
    value(key,default)=let i=findfirst(==(key),args)
        i===nothing ? default : (i<length(args) ? args[i+1] : error("missing value for $key"))
    end
    cfgpath=value("--config","")
    cfg=isempty(cfgpath) ? Dict{String,Any}() : TOML.parsefile(cfgpath)
    root=value("--store",get(cfg,"store","data/live_disk"))
    kind=value("--storage",get(cfg,"storage",isfile(joinpath(root,"CURRENT")) ? "arrow" : "disk"))
    kind in ("disk","arrow") || error("--storage must be disk or arrow")
    forwarded=copy(args)
    i=findfirst(==("--storage"),forwarded)
    i!==nothing && deleteat!(forwarded,i:i+1)
    include(kind=="disk" ? "ingest_disk_live.jl" : "ingest_arrow_live.jl")
    Base.invokelatest(() -> main(forwarded))
end
abspath(PROGRAM_FILE)==(@__FILE__) && dispatch(ARGS)
