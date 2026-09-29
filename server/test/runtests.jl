using Test, Random, JSON3, HTTP, LinearAlgebra, SHA
include("../src/MondrianServer.jl")
const MS=MondrianServer
const DS=MS.DS
const KE=MS.KE
using .MondrianServer.PostRecords
post(id,text="space galaxies #Space";at=100.,tags=["Space"])=PostRecord(PostKey(:threads,string(id));text,tags,created_at=100.,observed_at=at,revision_at=at)
batch(ps,cursor)=(events=MS.PostRecords.PostEvent.(ps),cursor=cursor,received=length(ps),complete=true)
desc=(model="fixture",dimension=4,preparation="fixture-v1",normalization="none",coordinate_dtype="float32")
function fixture(root)
    s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2,create_config=DS.config(dimension=4,model="fixture",retention_seconds=1000.,half_life=30.,split_after=16,reservoir_cap=16,max_leaves=8))
    DS.enable_keywords!(s,KE.JuliaYakeExtractor())
    ps=[post(i,isodd(i) ? "space galaxies #Space" : "plants garden #Garden") for i in 1:80]
    X=rand(MersenneTwister(4),Float32,4,80).*1.8f0.-.9f0
    DS.append_batch!(s,ps,X;time=100.,keywords=KE.prepare_keywords(KE.JuliaYakeExtractor(),KE.keyword_input.(ps)))
    s
end
function payload(s,vector=[.1,.2,.3,.4];kwargs...)
    merge(Dict{String,Any}("embedding_space_id"=>s.identity.embedding_space_id,"vector"=>vector),Dict(string(k)=>v for (k,v) in kwargs))
end
@testset "Pure API geometry, offsets, terms and read invariants" begin
    mktempdir() do root
        s=fixture(root);ctx=MS.Context(s;descriptor=desc)
        gen=s.generation;rng=DS.serialize_bytes(s.rng);basis=DS.serialize_bytes(s.basisrng)
        p=MS.tree_pack(ctx;view=:ppca,time=101.)
        @test sum(l.probability for l in p.leaves)≈1
        @test sum(exp(l.log_density+l.log_volume) for l in p.leaves)≈1
        @test all(l->l.ppca.support_normalization=="untruncated_gaussian",p.leaves)
        @test all(n->n.kind=="leaf" || 0<=n.axis<4,p.nodes)
        mapping=Dict(n.id=>n for n in p.nodes)
        route=x->begin
            n=mapping[p.root.id]
            while n.kind=="split";n=mapping[x[n.axis+1]<n.cut ? n.left : n.right];end
            n.id
        end
        for x in eachcol(rand(MersenneTwister(5),4,50).*2 .-1)
            @test route(x)==string(DS.route(s,x))
        end
        for n in values(s.nodes)
            n.axis==0 && continue
            x=zeros(4);x[n.axis]=n.cut
            @test route(x)==string(DS.route(s,x))
        end
        q=payload(ctx;candidate_budget=1024,leaf_budget=16)
        a=MS.nearby(ctx,q;time=101.)
        @test length(a.items)==25 && a.approximate
        @test a.candidates_scored==80
        @test length(unique(x.post_id for x in a.items))==25
        ids=[r.id for r in DS.rows(s,"SELECT id FROM posts WHERE status=1")]
        X=DS.fetch_vectors(s,ids);query=q["vector"]
        exact=sort([(clamp(1-dot(query,x)/(norm(query)*norm(x)),0.,2.),id) for (id,x) in zip(ids,eachcol(X))])
        @test [x.post_id for x in a.items]==string.([x[2] for x in exact[1:25]])
        for item in a.items
            x=DS.fetch_vectors(s,[parse(Int,item.post_id)])[:,1]
            @test isapprox(query.+item.delta,x;atol=1e-7)
            @test item.identity.native_id==DS.key_from_bytes(DS.lookup(s,parse(Int,item.post_id)).kind,DS.lookup(s,parse(Int,item.post_id)).native).native
        end
        result=MS.terms(ctx,payload(ctx;limit=1);time=101.)
        labels=String[x.term for x in result.terms]
        while result.next_cursor!==nothing
            result=MS.terms(ctx,payload(ctx;limit=1,page_cursor=result.next_cursor);time=101.)
            append!(labels,[x.term for x in result.terms])
        end
        reference=DS.leaf_keywords(s,DS.route(s,query);limit=100,ranking=:common,time=101.)
        @test labels==[x.term for x in reference.terms]
        @test length(unique(labels))==length(labels)
        @test s.generation==gen && DS.serialize_bytes(s.rng)==rng && DS.serialize_bytes(s.basisrng)==basis
        @test_throws MS.APIError MS.nearby(ctx,payload(ctx,zeros(4)))
        @test_throws MS.APIError MS.nearby(ctx,payload(ctx,[NaN,0.,0.,0.]))
        @test_throws MS.APIError MS.nearby(ctx,Dict("embedding_space_id"=>"wrong","vector"=>query))
        @test_throws MS.APIError MS.terms(ctx,payload(ctx;page_cursor="invalid"))
        @test_throws MS.APIError MS.tree_pack(ctx;time=1101.)
        @test isempty(MS.nearby(ctx,q;time=1101.).items)
        @test s.generation==gen
        oldcursor=MS.terms(ctx,payload(ctx;limit=1)).next_cursor
        MS.DR.maintenance!(s;now=102.)
        @test_throws MS.APIError MS.terms(ctx,payload(ctx;page_cursor=oldcursor))
        identity=ctx.identity
        DS.close_store(s);s=DS.open_store(root);ctx=MS.Context(s;descriptor=desc)
        @test ctx.identity.store_id==identity.store_id
        @test ctx.identity.embedding_space_id==identity.embedding_space_id
        DS.close_store(s)
    end
end
@testset "HTTP, owner, background preparation and restartable compaction" begin
    mktempdir() do root
        s=fixture(root);clock=Ref(101.)
        svc=MS.start_server(s;descriptor=desc,port=0,maintenance=false,recover=false,clock=()->clock[])
        url="http://127.0.0.1:$(HTTP.port(svc.http))"
        get(path;kwargs...)=HTTP.get(url*path;proxy=HTTP.ProxyConfig(),status_exception=false,kwargs...)
        send(path,obj)=HTTP.post(url*path;body=JSON3.write(obj),headers=["Content-Type"=>"application/json"],proxy=HTTP.ProxyConfig(),status_exception=false)
        try
            r=get("/v1/tree")
            @test r.status==200
            p=JSON3.read(String(r.body));space=String(p.embedding_space_id)
            req=Dict("embedding_space_id"=>space,"vector"=>[.1,.2,.3,.4])
            @test get("/healthz").status==200
            @test get("/readyz").status==200
            @test get("/v1/tree";headers=["If-None-Match"=>HTTP.header(r,"ETag")]).status==304
            @test get("/v1/tree?view=ppca").status==200
            for _ in 1:3
                zipped=get("/v1/tree";headers=["Accept-Encoding"=>"gzip"],decompress=false)
                @test zipped.status==200
                @test HTTP.header(zipped,"Content-Encoding")=="gzip"
                @test JSON3.read(String(MS.transcode(MS.GzipDecompressor,Vector{UInt8}(zipped.body)))).store_id==p.store_id
            end
            @test HTTP.header(get("/v1/tree";headers=["Accept-Encoding"=>"gzip;q=0"]),"Content-Encoding","")==""
            @test send("/v1/nearby",req).status==200
            @test send("/v1/terms",req).status==200
            @test send("/v1/nearby",merge(req,Dict("limit"=>26))).status==400
            @test send("/v1/nearby",merge(req,Dict("vector"=>[1,2]))).status==422
            @test get("/v1/nearby").status==405
            @test get("/unknown").status==404
            @test HTTP.post(url*"/v1/terms";body="{",status_exception=false,proxy=HTTP.ProxyConfig()).status==400
            @test HTTP.post(url*"/v1/terms";body=repeat("x",70000),status_exception=false,proxy=HTTP.ProxyConfig()).status==413
            # Inference sleeps outside StoreOwner: queries still complete meanwhile.
            gate=Channel{Bool}(1);go=Channel{Bool}(1)
            work=Threads.@spawn MS.ingest_batch!(svc,"fixture",batch([post(100;at=102.)],"1"),
                ts->begin put!(gate,true);take!(go);fill(.3f0,4,length(ts));end;now=102.)
            take!(gate)
            @test get("/v1/tree").status==200
            put!(go,true);fetch(work)
            @test get("/v1/tree";headers=["If-None-Match"=>HTTP.header(r,"ETag")]).status==200
            # Concurrent reads are serialized safely; inference/cursors remain correct.
            tasks=[Threads.@spawn send(isodd(i) ? "/v1/nearby" : "/v1/terms",req) for i in 1:12]
            @test all(t->fetch(t).status in (200,429),tasks)
            @test MS.submit(ctx->MS.DR.source_cursor(ctx.store,"fixture"),svc.owner)=="1"
            # Delete one post then drive bounded daily reclamation to completion.
            MS.submit(svc.owner) do ctx;DS.delete_post!(ctx.store,PostKey(:threads,"2");revision_at=103.,time=103.);end
            before=MS.submit(ctx->DS.fetch_vectors(ctx.store,[1,3,4]),svc.owner)
            for i in 1:100
                done=MS.submit(ctx->MS.reclamation_step!(ctx),svc.owner)
                !done && break
                @test i<100
            end
            after=MS.submit(ctx->DS.fetch_vectors(ctx.store,[1,3,4]),svc.owner)
            @test before==after
            @test MS.submit(ctx->DS.audit!(ctx.store)!==nothing,svc.owner)
            clock[]=1101.
            @test get("/v1/tree").status==503 # requests do not run expiry
            @test isempty(JSON3.read(String(send("/v1/nearby",req).body)).items)
            MS.submit(ctx->MS.DR.maintenance!(ctx.store;now=clock[]),svc.owner)
            @test get("/v1/tree").status==200
        finally
            MS.stop_server(svc)
        end
        s=DS.open_store(root)
        @test MS.DR.source_cursor(s,"fixture")=="1"
        @test DS.keyword_stats(s).covered==0
        DS.close_store(s)
    end
end
@testset "Owner overload, skipped deadlines and preparation failure" begin
    mktempdir() do root
        s=fixture(root);ctx=MS.Context(s,MS.Settings(queue_capacity=2);descriptor=desc)
        owner=MS.Owner(ctx);entered=Channel{Bool}(1);release=Channel{Bool}(1);ran=Ref(false)
        blocker=Threads.@spawn MS.submit(owner;timeout=Inf) do _
            put!(entered,true);take!(release)
        end
        take!(entered)
        @test_throws MS.APIError MS.submit(_->(ran[]=true),owner;timeout=.02)
        @test_throws MS.APIError MS.submit(_->nothing,owner;timeout=.02)
        put!(release,true);fetch(blocker)
        timedwait(()->owner.pending==0,2.;pollint=.01)
        @test !ran[]
        @test MS.submit(_->42,owner)==42
        MS.stop_owner!(owner)
        svc=MS.start_server(s;descriptor=desc,port=0,maintenance=false,recover=false,clock=()->101.)
        try
            generation=s.generation
            @test_throws ErrorException MS.ingest_batch!(svc,"failing",batch([post(101;at=102.)],"bad"),ts->error("test inference failure");now=102.)
            @test MS.submit(c->c.store.generation,svc.owner)==generation
            @test MS.submit(c->MS.DR.source_cursor(c.store,"failing"),svc.owner)==""
        finally MS.stop_server(svc) end
        # Shutdown must drain a full queue without enqueueing another owner job.
        s=DS.open_store(root)
        svc=MS.start_server(s;settings=MS.Settings(queue_capacity=1),descriptor=desc,
            port=0,maintenance=false,recover=false,clock=()->101.)
        entered=Channel{Bool}(1);release=Channel{Bool}(1)
        busy=Threads.@spawn MS.submit(svc.owner;timeout=Inf) do _
            put!(entered,true);take!(release)
        end
        take!(entered)
        shutdown=Threads.@spawn MS.stop_server(svc)
        put!(release,true);fetch(busy)
        @test fetch(shutdown)===nothing
    end
end
