# The stock precompile pass runs before the script hook, so replay the trace
# after loading Main.*. These names are part of the on-disk Serialization format.
include("load_runtime.jl")
using JSON3, TOML, HTTP, SQLite, DBInterface, Tables, CodecZlib, Gumbo
using Base64, Dates, LinearAlgebra, Printf, Random, Serialization, SHA, Statistics, UUIDs, Unicode, Sockets
using MondrianCLI
let success=0,skipped=0
    for line in eachline(ENV["MONDRIAN_BUILD_TRACE"])
        startswith(line,"precompile(") || continue
        try
            Core.eval(Main,Meta.parse(line));success+=1
        catch
            # Trace files may contain process-local closures or REPL/script names.
            skipped+=1
        end
    end
    println("Application precompile statements: ",success," loaded, ",skipped," unavailable")
end
