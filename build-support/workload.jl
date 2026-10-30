include("load_runtime.jl")
using MondrianCLI, HTTP, JSON3
MondrianCLI.configure_runtime!()
MondrianCLI.server(["--help"])
redirect_stdout(devnull) do
    MondrianCLI.tools(["demo"])
end
feed=HTTP.serve!(_->HTTP.Response(200,["Content-Type"=>"application/json"],"[]");listenany=true)
try
    @assert isempty(Main.LiveSources.json_get("http://"*HTTP.server_addr(feed)))
finally
    close(feed)
end
mktempdir() do base
    root=MondrianCLI.fixture(base;embed=get(ENV,"MONDRIAN_BUILD_ENCODER","false")=="true")
    MondrianCLI.tools(["audit",root])
    DS=Main.DiskTreeStore;S=Main.MondrianServer
    s=DS.open_store(root;require_existing=true,budget=32*1024^2)
    service=S.start_server(s;host="127.0.0.1",port=0)
    try
        url="http://127.0.0.1:$(HTTP.port(service.http))"
        @assert HTTP.get(url*"/readyz").status==200
        tree=JSON3.read(HTTP.get(url*"/v1/tree").body)
        query=JSON3.write((embedding_space_id=tree.embedding_space_id,vector=vcat(1.,zeros(s.config.dimension-1))))
        for endpoint in ("/v1/nearby","/v1/terms")
            @assert HTTP.post(url*endpoint,["Content-Type"=>"application/json"],query).status==200
        end
        @assert HTTP.get(url*"/v1/tree?view=ppca").status==200
    finally
        S.stop_server(service)
    end
end
