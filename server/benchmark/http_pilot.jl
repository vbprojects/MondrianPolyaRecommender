# One bounded HTTP pilot on a private copy of the 100k keyword fixture.
using JSON3, HTTP, Statistics, Random
include("../src/MondrianServer.jl")
const MS=MondrianServer
const DS=MS.DS
const ROOT=normpath(joinpath(@__DIR__,"../.."))
root=joinpath(ROOT,get(ENV,"MONDRIAN_SERVER_PILOT_STORE","data/server_http_pilot"))
trials=parse(Int,get(ENV,"MONDRIAN_SERVER_PILOT_REQUESTS","32"))
output=joinpath(ROOT,get(ENV,"MONDRIAN_SERVER_PILOT_OUTPUT","output/server/http-pilot.json"))
if !isdir(root)
    haskey(ENV,"MONDRIAN_SERVER_PILOT_STORE") && error("pilot store must already exist")
    source=joinpath(ROOT,"data/keyword_experiment/yake")
    # Obtain the existing writer lock while taking this offline fixture copy.
    original=DS.open_store(source)
    try
        DS.rows(original,"PRAGMA wal_checkpoint(TRUNCATE)")
        mkpath(root)
        for f in readdir(source)
            endswith(f,".bin") && hardlink(joinpath(source,f),joinpath(root,f))
        end
        cp(joinpath(source,"index.sqlite"),joinpath(root,"index.sqlite"))
    finally DS.close_store(original) end
end
s=DS.open_store(root;budget=32*1024^2)
clock=s.clock
ids=[r.id for r in DS.rows(s,"SELECT id FROM posts WHERE status=1 ORDER BY id LIMIT 32")]
X=DS.fetch_vectors(s,ids);DS.clear_caches!(s)
svc=MS.start_server(s;port=0,maintenance=false,recover=false,clock=()->clock,
    settings=MS.Settings(timeout=60.))
base="http://127.0.0.1:$(HTTP.port(svc.http))"
space=MS.submit(ctx->ctx.identity.embedding_space_id,svc.owner)
gettree(view="partitions")=HTTP.get(base*"/v1/tree?view="*view;headers=["Connection"=>"close"],proxy=HTTP.ProxyConfig(),status_exception=false)
function query(endpoint,j)
    req=Dict("embedding_space_id"=>space,"vector"=>X[:,mod1(j,size(X,2))])
    HTTP.post(base*endpoint;body=JSON3.write(req),headers=["Content-Type"=>"application/json","Connection"=>"close"],proxy=HTTP.ProxyConfig(),status_exception=false)
end
function memory()
    Dict(first(split(line,':'))=>parse(Int,split(line)[2])*1024 for line in eachline("/proc/self/status") if startswith(line,"VmRSS:") || startswith(line,"VmHWM:"))
end
process_io()=Dict(first(split(line,':'))=>parse(Int,last(split(line))) for line in eachline("/proc/self/io"))
try
    # Warm compilation and hot OS caches explicitly; no cold-disk claim.
    for (name,request) in (("tree",()->gettree()),("ppca",()->gettree("ppca")),("nearby",()->query("/v1/nearby",1)),("terms",()->query("/v1/terms",1)))
        println("warming ",name);flush(stdout)
        MS.submit(svc.owner;timeout=60.) do ctx
            args=Dict("embedding_space_id"=>space,"vector"=>X[:,1])
            name=="tree" ? MS.tree_pack(ctx;time=clock) : name=="ppca" ? MS.tree_pack(ctx;time=clock,view=:ppca) : name=="nearby" ? MS.nearby(ctx,args;time=clock) : MS.terms(ctx,args;time=clock)
        end
        r=request()
        r.status==200 || error("warmup failed: $(r.status) $(String(r.body))")
    end
    gen=s.generation;series=Dict{String,Any}();io_before=process_io()
    for name in ("tree","nearby","terms")
        println("measuring ",name," (",trials," requests)");flush(stdout)
        lat=Float64[];sizes=Int[];returned=Int[];blocks=Int[];readbytes=Int[];misses=Int[]
        for j in 1:trials
            r=nothing;t=@elapsed r=name=="tree" ? gettree() : query("/v1/"*name,j)
            r.status==200 || error("HTTP failure $(r.status): $(String(r.body))")
            body=String(r.body);push!(lat,1000t);push!(sizes,ncodeunits(body))
            if name=="nearby"
                data=JSON3.read(body);push!(returned,data.returned)
                push!(blocks,data.vector_blocks);push!(readbytes,data.vector_file_bytes_read)
                push!(misses,data.block_cache_misses)
            end
        end
        series[name]=(;milliseconds=lat,p50_ms=median(lat),p95_ms=quantile(lat,.95),p99_ms=quantile(lat,.99),
            response_bytes=sizes,returned,blocks,vector_file_bytes_read=readbytes,block_cache_misses=misses)
    end
    io_after=process_io();io_delta=Dict(k=>io_after[k]-v for (k,v) in io_before)
    @assert s.generation==gen
    pack=JSON3.read(String(gettree().body));ppca=gettree("ppca")
    results=(posts=DS.one(s,"SELECT COUNT(*) AS n FROM posts WHERE status=1").n,
        leaves=length(pack.leaves),dimension=s.config.dimension,model=s.config.model,
        storage_format=s.config.mode,cache_condition="warm OS cache; bounded 32 MiB managed store cache",
        backend=string(DS.metadata_backend(s)),connection_policy="fresh connection per request",requests_per_endpoint=trials,tree_bytes=ncodeunits(JSON3.write(pack)),ppca_bytes=ncodeunits(String(ppca.body)),
        storage_stats=DS.storage_stats(s),server_budget=svc.settings.server_budget,
        process_memory=memory(),process_io_during_requests=io_delta,timings=series,read_generation_unchanged=true,
        limitations=["single local client; no simultaneous inference", "term scan timeout raised to 60 seconds for compilation", "not a cold-device or weekly-scale benchmark"])
    mkpath(dirname(output));write(output,JSON3.write(results))
    println(JSON3.write(results))
finally MS.stop_server(svc) end
