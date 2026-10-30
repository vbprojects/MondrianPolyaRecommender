using Pkg, TOML, SHA, UUIDs
using PackageCompiler
const ROOT=normpath(joinpath(@__DIR__,".."))
function main(args)
    output=joinpath(ROOT,"output","mondrian-app");encoder=true;rocks=false;force=false;cpu="generic"
    i=1
    while i<=length(args)
        a=args[i]
        if a=="--without-encoder";encoder=false
        elseif a=="--with-rocksdb";rocks=true
        elseif a=="--force";force=true
        elseif a in ("--output","--cpu-target")
            i<length(args) || error("missing value for $a");i+=1
            if a=="--output";output=abspath(args[i]);else;cpu=args[i];end
        elseif a in ("--help","-h")
            println("build.jl [--output DIR] [--without-encoder] [--with-rocksdb] [--cpu-target TARGET] [--force]")
            return
        else;error("unknown option $a");end
        i+=1
    end
    Sys.islinux() || error("This build recipe is currently validated on Linux")
    VERSION>=v"1.13" || error("Use Julia 1.13 or newer for this pinned recipe")
    output=abspath(output)
    (output==dirname(output) || output==ROOT || startswith(ROOT,output*"/") || output==homedir()) && error("unsafe output directory")
    ispath(output) && !force && error("output exists; choose a new directory or pass --force")
    mkpath(joinpath(ROOT,"output"))
    stage=mktempdir(joinpath(ROOT,"output");prefix="mondrian-build-",cleanup=false)
    runtime=joinpath(stage,"runtime");mkpath(joinpath(runtime,"server"));mkpath(joinpath(runtime,"julia-yake"))
    cp(joinpath(ROOT,"src"),joinpath(runtime,"src"))
    cp(joinpath(ROOT,"server","src"),joinpath(runtime,"server","src"))
    for folder in ("src","data")
        cp(joinpath(ROOT,"julia-yake",folder),joinpath(runtime,"julia-yake",folder))
    end
    app=joinpath(stage,"app");mkpath(app)
    cp(joinpath(@__DIR__,"app","src"),joinpath(app,"src"))
    project=TOML.parsefile(joinpath(ROOT,"server","Project.toml"))
    project["name"]="MondrianCLI";project["uuid"]="43e5223b-e5c4-4e38-8247-6beb89d481f8";project["version"]="0.1.0"
    if !rocks
        delete!(project["deps"],"RocksDB");delete!(project["compat"],"RocksDB")
    end
    manifest=TOML.parsefile(joinpath(ROOT,"server","Manifest.toml"))
    for name in ("Base64","Dates","LinearAlgebra","Printf","Random","Serialization","SHA","Statistics","TOML","UUIDs","Unicode","Sockets")
        id=haskey(manifest["deps"],name) ? manifest["deps"][name][1]["uuid"] : string(Base.identify_package(name).uuid)
        project["deps"][name]=id
    end
    open(joinpath(app,"Project.toml"),"w") do io;TOML.print(io,project);end
    cp(joinpath(ROOT,"server","Manifest.toml"),joinpath(app,"Manifest.toml"))
    Pkg.activate(app);Pkg.resolve();Pkg.instantiate()
    native=joinpath(ROOT,"native","gte-worker","target","release","mondrian-gte-worker")
    bundle=joinpath(ROOT,"output","gte_tiny_rust")
    if encoder
        run(`cargo build --release --locked --manifest-path $(joinpath(ROOT,"native","gte-worker","Cargo.toml"))`)
        for name in ("model_quantized.onnx","tokenizer.json","libonnxruntime.so")
            isfile(joinpath(bundle,name)) || error("Missing encoder asset $name; run native/gte-worker/prepare.py first")
        end
    end
    trace=joinpath(stage,"precompile.jl")
    env=Dict("MONDRIAN_BUILD_RUNTIME"=>runtime,"MONDRIAN_BUILD_TRACE"=>trace,
        "MONDRIAN_BUILD_ENCODER"=>string(encoder),"MONDRIAN_EMBEDDING_RUST"=>native,
        "MONDRIAN_GTE_BUNDLE"=>bundle,"JULIA_NUM_PRECOMPILE_TASKS"=>"2",
        "JULIA_NUM_THREADS"=>"2,1","OPENBLAS_NUM_THREADS"=>"1")
    withenv(env...) do
        run(`$(Base.julia_cmd()) --startup-file=no --project=$app --threads=2,1 --trace-compile=$trace $(joinpath(@__DIR__,"workload.jl"))`)
        PackageCompiler.create_app(app,output;
            executables=["mondrian-server"=>"server_main","mondrian-tool"=>"tools_main"],
            cpu_target=cpu,incremental=false,filter_stdlibs=true,include_lazy_artifacts=true,
            compress_sysimage=true,force,script=joinpath(@__DIR__,"image.jl"),
            precompile_statements_file=trace)
    end
    share=joinpath(output,"share","mondrian");mkpath(share)
    if encoder
        mkpath(joinpath(output,"libexec"));cp(native,joinpath(output,"libexec","mondrian-gte-worker"))
        mkpath(joinpath(share,"gte"))
        for name in ("model_quantized.onnx","tokenizer.json","libonnxruntime.so")
            cp(joinpath(bundle,name),joinpath(share,"gte",name))
        end
    end
    cp(joinpath(ROOT,"examples","live_sources.rust.toml"),joinpath(share,"sources.toml"))
    cfg=Dict("mode"=>encoder ? "combined" : "serve","store"=>"../../../var/mondrian","ingestion_config"=>"sources.toml",
        "embedding_device"=>"cpu","max_events"=>32,"source_seconds"=>5,
        "http"=>Dict("host"=>"127.0.0.1","port"=>8080),
        "storage"=>Dict("backend"=>rocks ? "auto" : "sqlite","managed_budget_mib"=>128,"server_budget_mib"=>32))
    open(joinpath(share,"server.toml"),"w") do io;TOML.print(io,cfg);end
    mkpath(joinpath(share,"notices"))
    for name in ("LICENSE","NOTICE.md")
        cp(joinpath(ROOT,"julia-yake",name),joinpath(share,"notices","julia-yake-"*name))
    end
    cp(joinpath(app,"Project.toml"),joinpath(share,"AppProject.toml"))
    cp(joinpath(app,"Manifest.toml"),joinpath(share,"AppManifest.toml"))
    open(joinpath(share,"build.toml"),"w") do io
        TOML.print(io,Dict("julia"=>string(VERSION),"packagecompiler"=>string(pkgversion(PackageCompiler)),
            "cpu_target"=>cpu,"encoder"=>encoder,"rocksdb"=>rocks,"source_stage"=>stage,
            "server_source_sha256"=>bytes2hex(sha256(read(joinpath(ROOT,"server","src","MondrianServer.jl")))),
            "cli_source_sha256"=>bytes2hex(sha256(read(joinpath(app,"src","MondrianCLI.jl"))))))
    end
    println("Built ",output,"; build inputs retained at ",stage)
end
main(ARGS)
