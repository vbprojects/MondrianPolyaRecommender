#!/usr/bin/env julia
include("../src/MondrianServer.jl")
using JSON3
function main(args)
    if isempty(args) || "--help" in args
        println("julia --project=server server/bin/import_sqlite.jl --source SQLITE_STORE --destination NEW_ROCKS_STORE [--cache-mib 256]")
        return
    end
    iseven(length(args)) || error("expected option/value pairs")
    opts=Dict(args[i]=>args[i+1] for i in 1:2:length(args))
    all(k->k in ("--source","--destination","--cache-mib"),keys(opts)) || error("unknown option")
    result=DiskTreeStore.import_sqlite_to_rocks(opts["--source"],opts["--destination"];budget=parse(Int,get(opts,"--cache-mib","256"))*1024^2)
    println(JSON3.write(result))
end
main(ARGS)
