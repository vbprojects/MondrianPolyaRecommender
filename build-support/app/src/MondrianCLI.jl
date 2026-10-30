module MondrianCLI
using JSON3, TOML, Random, LinearAlgebra

# The image build loads the existing modules into Main. Keeping their original
# names preserves the Serialization contract of indexes created from scripts.
runtime() = isdefined(Main, :MondrianServer) ? Main.MondrianServer :
    error("server modules are missing; build this application with build-support/build.jl")
app_root() = normpath(joinpath(Sys.BINDIR, ".."))

function configure_runtime!()
    root=app_root()
    get!(ENV,"MONDRIAN_EMBEDDING_RUST",joinpath(root,"libexec","mondrian-gte-worker"))
    get!(ENV,"MONDRIAN_GTE_BUNDLE",joinpath(root,"share","mondrian","gte"))
    BLAS.set_num_threads(1)
    Base.exit_on_sigint(false)
end

function server(args)
    if isempty(args) || any(x->x in ("--help","-h"),args)
        println("mondrian-server --config FILE [--seconds N] [--julia-args --threads=2,1]")
        println("SQLite by default; combined mode uses the bundled Rust INT8 encoder.")
        return
    end
    config=nothing;seconds=Inf;i=1
    while i<=length(args)
        i<length(args) || error("missing option value")
        if args[i]=="--config"
            config=abspath(args[i+1])
        elseif args[i]=="--seconds"
            seconds=parse(Float64,args[i+1]);isfinite(seconds)&&seconds>0 || error("invalid duration")
        else
            error("unknown option: $(args[i])")
        end
        i+=2
    end
    config===nothing && error("--config is required")
    srv=runtime();cfg=srv.load_config(config)
    if get(cfg,"mode","combined")=="combined"
        ingestion=TOML.parsefile(cfg["ingestion_config"])
        get(get(ingestion,"index",Dict()),"embedding_model",nothing)==Main.EmbeddingBridge.GTE_INT8_MODEL ||
            error("compiled combined mode requires the Rust GTE INT8 model; use serve mode for an existing FP32/Qwen store")
        get(cfg,"embedding_device","cpu") in ("cpu","auto") || error("bundled encoder supports CPU only")
    end
    srv.run_config(config;seconds)
end

"""Create a new offline fixture; never overwrite a user's store."""
function fixture(base;embed=false)
    root=joinpath(abspath(base),"store")
    ispath(root) && error("fixture refuses an existing store")
    mkpath(base)
    DS=Main.DiskTreeStore;PR=Main.PostRecords;KE=Main.KeywordExtraction;EB=Main.EmbeddingBridge
    texts=["Julia trees and public data example $i." for i in 1:32]
    model=embed ? EB.GTE_INT8_MODEL : EB.QWEN_MODEL
    d=embed ? 384 : 256
    X=if embed
        w=EB.Worker(model=model)
        try EB.embed(w,texts) finally close(w) end
    else
        x=randn(MersenneTwister(82),Float32,d,32);x./=sqrt.(sum(abs2,x;dims=1));x
    end
    t=time()
    s=DS.open_store(root;backend=:sqlite,budget=32*1024^2,
        create_config=DS.config(;dimension=d,model,max_leaves=1))
    try
        records=[PR.PostRecord(PR.PostKey(:threads,"compiled-$i");text=texts[i],tags=["Julia"],
            created_at=t,observed_at=t) for i in 1:32]
        e=KE.JuliaYakeExtractor();DS.enable_keywords!(s,e)
        DS.append_batch!(s,records,X;time=t,keywords=KE.prepare_keywords(e,KE.keyword_input.(records)))
        @assert DS.audit!(s)==32
    finally
        DS.close_store(s)
    end
    # All config paths are relative to the config, not to the build machine.
    open(joinpath(base,"sources.toml"),"w") do io
        TOML.print(io,Dict("index"=>Dict("embedding_model"=>model,"dimension"=>d)))
    end
    open(joinpath(base,"serve.toml"),"w") do io
        TOML.print(io,Dict("mode"=>"serve","store"=>"store","ingestion_config"=>"sources.toml",
            "http"=>Dict("host"=>"127.0.0.1","port"=>8080),
            "storage"=>Dict("backend"=>"sqlite","managed_budget_mib"=>32)))
    end
    println(JSON3.write((fixture=root,posts=32,dimension=d)))
    root
end

function tools(args)
    if isempty(args) || first(args) in ("--help","-h")
        println("mondrian-tool embed TEXT [TEXT ...]")
        println("mondrian-tool fixture DIRECTORY [--embed]")
        println("mondrian-tool audit STORE")
        println("mondrian-tool demo")
        return
    end
    runtime()
    if first(args)=="embed"
        length(args)>=2 || error("at least one text is required")
        EB=Main.EmbeddingBridge;w=EB.Worker(model=EB.GTE_INT8_MODEL)
        try
            x=EB.embed(w,args[2:end])
            println(JSON3.write((model=w.model,dimension=w.dimension,vectors=[collect(c) for c in eachcol(x)])))
        finally
            close(w)
        end
    elseif first(args)=="fixture"
        length(args) in (2,3) || error("expected a directory and optional --embed")
        length(args)==3 && args[3]!="--embed" && error("unknown fixture option")
        fixture(args[2];embed=length(args)==3)
    elseif first(args)=="audit"
        length(args)==2 || error("expected a store directory")
        DS=Main.DiskTreeStore;s=DS.open_store(abspath(args[2]);require_existing=true,budget=32*1024^2)
        try println(JSON3.write((posts=DS.audit!(s),model=s.config.model,dimension=s.config.dimension)))
        finally DS.close_store(s) end
    elseif first(args)=="demo"
        length(args)==1 || error("demo takes no arguments")
        M=Main.MondrianPolyaToy;m=M.ReferenceModel()
        M.observe!(m,1,(.2,.3);now=1.);M.observe!(m,2,(.7,.8);now=2.)
        println(JSON3.write((integral=M.integral(m.tree),cells=M.cells(m.tree),observations=length(m.records))))
    else
        error("unknown command: $(first(args))")
    end
end

function entry(f)::Cint
    try
        configure_runtime!()
        f(ARGS)
        return 0
    catch e
        e isa InterruptException && return 0
        showerror(stderr,e);println(stderr)
        return 1
    end
end
server_main()::Cint=entry(server)
tools_main()::Cint=entry(tools)
end
