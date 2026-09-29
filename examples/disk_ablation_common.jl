using JSON3, Random, LinearAlgebra, Statistics, SHA
include("../src/LeafLatentTree.jl")
include("../src/PostRecords.jl")
include("../src/DiskCodecs.jl")
include("../src/DiskTreeStore.jl")
using .PostRecords
const DS=DiskTreeStore
const DC=DiskCodecs
const LT=LeafLatentTrees
const ROOT=normpath(joinpath(@__DIR__,".."))
const INPUT=get(ENV,"MONDRIAN_DISK_INPUT",joinpath(ROOT,"data/gte_disk_ablation"))
const OUT=get(ENV,"MONDRIAN_DISK_OUTPUT",joinpath(ROOT,"output/disk_ablation"))
const STORE=get(ENV,"MONDRIAN_DISK_STORE",joinpath(ROOT,"data/disk_ablation"))
BLAS.set_num_threads(1)
function savejson(path,x)
    mkpath(dirname(path));tmp=path*".tmp";open(io->JSON3.pretty(io,x),tmp,"w");mv(tmp,path;force=true)
end
function osinfo()
    io=Dict{String,Int}()
    for line in eachline("/proc/self/io")
        k,v=split(line,':');io[k]=parse(Int,strip(v))
    end
    for line in eachline("/proc/self/status")
        startswith(line,"VmRSS:") && (io["rss"]=parse(Int,split(line)[2])*1024)
        startswith(line,"VmHWM:") && (io["peak_rss"]=parse(Int,split(line)[2])*1024)
    end
    io
end
delta(a,b,k)=get(b,k,0)-get(a,k,0)
metric(s,k)=get(s.metrics,k,0.)
function fixture_vectors(seed,n=48)
    rng=MersenneTwister(seed+910);N=Int(JSON3.read(read(joinpath(INPUT,"manifest.json"),String)).contract.count)
    ids=rand(rng,1:N,n);X=zeros(Float32,384,n)
    open(joinpath(INPUT,"embeddings.f32")) do io
        for (j,id) in enumerate(ids);seek(io,(id-1)*384*4);read!(io,view(X,:,j));end
    end
    X,ids
end
function readbatch(textio,vecio,first,n;shift=0.,prefix="")
    X=read!(vecio,Matrix{Float32}(undef,384,n));records=PostRecord[]
    for i in first:first+n-1
        p=JSON3.read(readline(textio));t=shift+i/100
        key=isempty(prefix) ? PostKey(:bluesky,p.uri) : PostKey(:threads,"$prefix$i")
        push!(records,PostRecord(key;text=p.text,author=p.author,created_at=t,observed_at=t,
            metadata=JSON3.write((source_created_at=p.source_created_at,))))
    end
    records,X
end
function fingerprints(s)
    ctx=SHA.SHA2_256_CTX();after=0
    while true
        ps=DS.rows(s,"SELECT id,block,row FROM posts WHERE status=1 AND id>? ORDER BY id LIMIT 256",(after,))
        isempty(ps) && break
        for p in ps;SHA.update!(ctx,codeunits("$(p.id):"*DC.signature(DS.frame_for(s,p.block),p.row)));end
        after=last(ps).id
    end
    bytes2hex(SHA.digest!(ctx))
end
function db_breakdown(s)
    # SQLite dbstat reports actual table/index B-tree pages, including slack.
    Dict(r.name=>r.bytes for r in DS.rows(s,"SELECT name,SUM(pgsize) AS bytes FROM dbstat GROUP BY name"))
end
function warmup()
    mktempdir() do root
        s=DS.open_store(root;create_config=DS.config(dimension=12,max_leaves=4,split_after=16),budget=32*1024^2)
        rng=MersenneTwister(3);X=randn(rng,Float32,12,100);X./=sqrt.(sum(abs2,X;dims=1))
        ps=[PostRecord(PostKey(:threads,"warm$i");text="warm",created_at=0.,observed_at=0.) for i in 1:100]
        DS.append_batch!(s,ps,X);DS.sample_ids(s,X[:,1]);DS.fetch_vectors(s,[1]);DS.fetch_records(s,[1]);DS.maintain!(s;time=1e6)
        DS.compact!(s);DS.close_store(s)
    end
    GC.gc()
end
