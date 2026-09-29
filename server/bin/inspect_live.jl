#!/usr/bin/env julia
include("../src/LiveInspector.jl")
using HTTP, JSON3
Base.exit_on_sigint(false)
function main(args)
    root="data/live_disk_gte";port=8091;run_dir="data/live_runs/latest";i=1
    while i<=length(args)
        if args[i] in ("--help","-h")
            println("julia --project=server server/bin/inspect_live.jl [--store data/live_disk_gte] [--port 8091] [--run-dir data/live_runs/latest]")
            return
        end
        i<length(args) || error("missing argument")
        if args[i]=="--store";root=args[i+1]
        elseif args[i]=="--port";port=parse(Int,args[i+1])
        elseif args[i]=="--run-dir";run_dir=args[i+1]
        else;error("unknown option");end
        i+=2
    end
    ins=LiveInspector.Inspector(root;run_dir)
    # Fail before binding if the committed store cannot be read safely.
    ins.cache=LiveInspector.snapshot(ins);ins.cached_at=time()
    server=HTTP.serve!(req->LiveInspector.handler(ins,req),"127.0.0.1",port;
        read_timeout=10,write_timeout=10,max_concurrent_streams=4,max_header_bytes=8192)
    println(JSON3.write((inspector="ready",url="http://127.0.0.1:$port",store=ins.root)));flush(stdout)
    try
        while true;sleep(1);end
    catch e
        e isa InterruptException || rethrow()
    finally HTTP.forceclose(server) end
end
main(ARGS)
