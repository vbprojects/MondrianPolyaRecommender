mutable struct Service
    owner::Owner
    http::Any
    scheduler::Any
    stopped::Threads.Atomic{Bool}
    settings::Settings
    clock::Function
    ingest_task::Any
    error::Base.RefValue{Any}
    http_gate::ReentrantLock
    active_http::Int
end
function parse_input(req)
    try
        x=JSON3.read(String(req.body),Dict{String,Any})
        x isa Dict || fail("expected_object");x
    catch e
        e isa APIError && rethrow()
        fail("invalid_json")
    end
end
function response(status,body;headers=Pair{String,String}[])
    HTTP.Response(status;headers=vcat(["Content-Type"=>"application/json","Cache-Control"=>"no-store"],headers),body)
end
function handler(service,req)
    c=service.settings
    origin=HTTP.header(req,"Origin","")
    cors=Pair{String,String}[]
    if !isempty(origin) && origin in c.cors
        append!(cors,["Access-Control-Allow-Origin"=>origin,"Vary"=>"Origin"])
    end
    try
        service.stopped[] && fail("server_stopping",503)
        ncodeunits(String(req.target))<=4096 || fail("target_too_large",413)
        length(req.body)<=c.body_limit || fail("body_too_large",413)
        target=HTTP.URI(req.target);path=target.path
        if req.method=="OPTIONS"
            !isempty(origin) && !(origin in c.cors) && fail("origin_not_allowed",403)
            return HTTP.Response(204;headers=vcat(cors,["Access-Control-Allow-Methods"=>"GET, POST, OPTIONS",
                "Access-Control-Allow-Headers"=>"Content-Type, If-None-Match"]),body="")
        end
        if path=="/healthz" && req.method=="GET"
            return response(200,JSON3.write((status="ok",));headers=cors)
        elseif path=="/readyz" && req.method=="GET"
            service.error[]!==nothing && fail("background_failed",503)
            payload=submit(service.owner;timeout=c.timeout) do ctx
                fresh(ctx,max(service.clock(),ctx.store.clock));merge(envelope(ctx,ctx.store.clock),(status="ready",))
            end
            return response(200,JSON3.write(payload);headers=cors)
        elseif path=="/v1/tree" && req.method=="GET"
            query=HTTP.queryparams(target)
            all(k->k=="view",keys(query)) || fail("unknown_query_parameter")
            view=Symbol(get(query,"view","partitions"));view in (:partitions,:ppca) || fail("invalid_view")
            p=submit(service.owner;timeout=c.timeout) do ctx
                cached_pack(ctx,view,max(service.clock(),ctx.store.clock))
            end
            h=vcat(cors,["Content-Type"=>"application/json","ETag"=>p.etag,"Cache-Control"=>"public, max-age=0, must-revalidate","Vary"=>"Accept-Encoding"])
            if HTTP.header(req,"If-None-Match","")==p.etag
                return HTTP.Response(304;headers=h,body="")
            end
            if accepts_gzip(HTTP.header(req,"Accept-Encoding",""))
                push!(h,"Content-Encoding"=>"gzip");return HTTP.Response(200;headers=h,body=copy(p.gzip))
            end
            return HTTP.Response(200;headers=h,body=p.body)
        elseif path in ("/v1/nearby","/v1/terms") && req.method=="POST"
            input=parse_input(req)
            payload=submit(service.owner;timeout=c.timeout) do ctx
                t=max(service.clock(),ctx.store.clock)
                path=="/v1/nearby" ? nearby(ctx,input;time=t) : terms(ctx,input;time=t)
            end
            body=JSON3.write(payload)
            ncodeunits(body)<=c.server_budget÷32 || fail("response_too_large",503)
            return response(200,body;headers=cors)
        elseif path in ("/v1/tree","/v1/nearby","/v1/terms","/healthz","/readyz")
            fail("method_not_allowed",405)
        else
            fail("not_found",404)
        end
    catch e
        status=e isa APIError ? e.status : 503
        code=e isa APIError ? e.code : "storage_unavailable"
        h=status in (429,503) ? vcat(cors,["Retry-After"=>"1"]) : cors
        # No payload, vectors, tokens, or raw internal exceptions in responses/logs.
        return response(status,JSON3.write((error=(code=code,),));headers=h)
    end
end
function accepts_gzip(header)
    for item in split(lowercase(header),',')
        parts=strip.(split(item,';'))
        first(parts)=="gzip" || continue
        q=1.
        for param in parts[2:end]
            startswith(param,"q=") || continue
            parsed=tryparse(Float64,param[3:end]);q=parsed===nothing ? 0. : parsed
        end
        return 0<q<=1
    end
    false
end
function stream_handler(service,stream)
    admitted=lock(service.http_gate) do
        if service.active_http>=min(8,service.settings.queue_capacity)
            false
        else
            service.active_http+=1;true
        end
    end
    try
        raw=HTTP.startread(stream)
        resp=if !admitted
            response(429,JSON3.write((error=(code="http_capacity",),));headers=["Connection"=>"close","Retry-After"=>"1"])
        else
            declared=tryparse(Int,HTTP.header(raw,"Content-Length","0"))
            if declared!==nothing && declared>service.settings.body_limit
                response(413,JSON3.write((error=(code="body_too_large",),));headers=["Connection"=>"close"])
            else
                # Read at most limit+1 even for chunked requests. Admission remains
                # held until the bounded response finishes writing or times out.
                data=read(stream,service.settings.body_limit+1)
                if length(data)>service.settings.body_limit
                    response(413,JSON3.write((error=(code="body_too_large",),));headers=["Connection"=>"close"])
                else
                    req=HTTP.Request(raw.method,raw.target;headers=raw.headers,body=data)
                    handler(service,req)
                end
            end
        end
        stream.response=resp
        bytes=String(resp.body)
        !isempty(bytes) && write(stream,bytes)
        closewrite(stream)
        HTTP.closeread(stream)
    finally
        admitted && lock(service.http_gate) do;service.active_http-=1;end
    end
end
function start_server(s;settings=Settings(),host="127.0.0.1",port=8080,descriptor=nothing,
        clock=time,maintenance=true,reclaim=true,recover=true)
    ctx=Context(s,settings;descriptor)
    # Single startup recovery, before HTTP is exposed, also removes crash orphans.
    recover && DS.collect_garbage!(s)
    maintenance && DR.maintenance!(s;now=max(clock(),s.clock))
    service=Service(Owner(ctx),nothing,nothing,Threads.Atomic{Bool}(false),settings,clock,nothing,Ref{Any}(nothing),ReentrantLock(),0)
    try
        service.http=HTTP.listen!(stream->stream_handler(service,stream),host,port;listenany=port==0,
            read_header_timeout=5,read_timeout=10,write_timeout=10,idle_timeout=15,
            max_header_bytes=16384,max_concurrent_streams=8)
        if maintenance
            service.scheduler=Threads.@spawn begin
                while !service.stopped[]
                    sleep(settings.maintenance_interval)
                    service.stopped[] && break
                    try
                        submit(service.owner;timeout=Inf) do ctx
                            DR.maintenance!(ctx.store;now=max(clock(),ctx.store.clock))
                            reclaim && reclamation_step!(ctx)
                            # Cached packs are rebuilt lazily but always generation-checked.
                            nothing
                        end
                    catch e
                        service.error[]=typeof(e);break
                    end
                end
            end
        end
        service
    catch
        stop_owner!(service.owner);rethrow()
    end
end
function stop_server(service;close_store=true)
    service.stopped[]=true
    service.http!==nothing && HTTP.forceclose(service.http)
    service.scheduler!==nothing && wait(service.scheduler)
    # Source reads have bounded deadlines; let an in-flight durable batch finish.
    service.ingest_task!==nothing && wait(service.ingest_task)
    stop_owner!(service.owner)
    close_store && DS.close_store(service.owner.store)
    nothing
end
