using Random, LinearAlgebra, Serialization, Mmap, Statistics
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
include("export_utils.jl")
function rss(field="VmRSS")
    s=read("/proc/self/status",String);m=match(Regex(field*":\\s+(\\d+)"),s)
    isnothing(m) ? 0 : parse(Int,m.captures[1])*1024
end
function main()
    BLAS.set_num_threads(1)
    root=normpath(joinpath(@__DIR__,".."));input=joinpath(root,"data/bluesky_1000000")
    out=joinpath(root,"output/bluesky_million");mkpath(out)
    N,d=parse.(Int,split(read(joinpath(input,"shape.txt"),String)));@assert N==1000000 && d==256
    X=open(joinpath(input,"embeddings.f32")) do io;Mmap.mmap(io,Matrix{Float32},(d,N));end
    order=randperm(MersenneTwister(192),N);stream=view(X,:,order)
    fit_leaf_tree(view(stream,:,1:1024);max_leaves=2,configs=[(:ppca,4)])
    GC.gc();records=NamedTuple[];cuts=NamedTuple[];start=time_ns()
    cutfile=open(joinpath(out,"cuts.csv"),"w");println(cutfile,"post,node,axis,cut,depth,left,right,left_mass,right_mass")
    trackfile=open(joinpath(out,"memory.csv"),"w");println(trackfile,"posts,seconds,nodes,leaves,cuts,max_depth,tree_bytes,states_bytes,reservoir_bytes,rss_bytes,peak_rss_bytes")
    function observe(row,nodes)
        leaf=findall(n->n.axis==0,nodes)
        rec=(posts=row,seconds=(time_ns()-start)/1e9,nodes=length(nodes),leaves=length(leaf),cuts=length(nodes)-length(leaf),max_depth=maximum(n.depth for n in nodes),tree_bytes=Base.summarysize(nodes),states_bytes=Base.summarysize([nodes[i].states for i in leaf]),reservoir_bytes=Base.summarysize([nodes[i].reservoir for i in leaf]),rss_bytes=rss(),peak_rss_bytes=rss("VmHWM"))
        push!(records,rec);println(trackfile,join(values(rec),","));flush(trackfile)
    end
    function cut(row,i,nodes)
        n=nodes[i];rec=(post=row,node=i,axis=n.axis,cut=n.cut,depth=n.depth,left=n.left,right=n.right,left_mass=nodes[n.left].states[1].mass,right_mass=nodes[n.right].states[1].mass)
        push!(cuts,rec);println(cutfile,join(values(rec),","));flush(cutfile)
    end
    # A high safety ceiling; actual growth is controlled by arrivals and depth.
    fitted=fit_leaf_tree(stream;max_leaves=10000,max_depth=24,split_after=512,reservoir_cap=64,configs=[(:ppca,4)],observer=observe,on_split=cut)
    fit_seconds=(time_ns()-start)/1e9;close(cutfile);close(trackfile)
    nodes=fitted.nodes;leaves=findall(n->n.axis==0,nodes);mass=sum(nodes[i].states[1].mass for i in leaves)
    @assert isapprox(mass,N;atol=1e-6)
    @assert all(n.axis==0 || (isempty(n.states)&&isempty(n.reservoir)) for n in nodes)
    @assert length(cuts)==length(leaves)-1
    @assert all(all(isfinite,nodes[i].states[1].mean) for i in leaves)
    exact=zeros(Int,length(nodes));for j in 1:N;exact[leaf_index(nodes,view(X,:,j))]+=1;end
    tv=.5sum(abs(nodes[i].states[1].mass/N-exact[i]/N) for i in leaves)
    serialize(joinpath(out,"models.jls"),(fitted=fitted,seed=192,input=input))
    payload=(count=N,dimensions=d,rank=4,seed=192,max_leaves=10000,max_depth_limit=24,split_after=512,reservoir_cap=64,fit_seconds=fit_seconds,update_seconds=fitted.update_seconds[1],split_seconds=fitted.split_seconds[1],mass=mass,train_leaf_mass_tv=tv,embedding_bytes=filesize(joinpath(input,"embeddings.f32")),model_file_bytes=filesize(joinpath(out,"models.jls")),final=last(records),tracking=records,cuts=cuts)
    write(joinpath(out,"results.json"),sprint(io->json(io,payload)));println(last(records));println("Fit seconds: ",fit_seconds,"; mass TV: ",tv)
end
main()
