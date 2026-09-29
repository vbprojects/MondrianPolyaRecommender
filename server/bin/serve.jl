#!/usr/bin/env julia
include("../src/MondrianServer.jl")
using .MondrianServer
# Container STOPSIGNAL is SIGINT. Let run_config's finally block drain the owner,
# stop background tasks and release the writer lock before exiting.
Base.exit_on_sigint(false)
function main(args)
    config=normpath(joinpath(@__DIR__,"../config.example.toml"));seconds=Inf;i=1
    while i<=length(args)
        if args[i] in ("--help","-h")
            println("julia --threads=4,1 --project=server server/bin/serve.jl [--config FILE] [--seconds N]")
            return
        end
        i<length(args) || error("missing argument")
        if args[i]=="--config";config=args[i+1]
        elseif args[i]=="--seconds";seconds=parse(Float64,args[i+1]);isfinite(seconds)&&seconds>0 || error("invalid duration")
        else;error("unknown option");end
        i+=2
    end
    try MondrianServer.run_config(config;seconds)
    catch e
        e isa InterruptException || rethrow()
    end
end
main(ARGS)
