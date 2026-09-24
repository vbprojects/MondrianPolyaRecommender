using MondrianPolyaToy, Serialization, Printf
include("load_bluesky.jl")
const MP = MondrianPolyaToy
function leaves(t,m=nothing)
    ids=Int[]
    function visit(i)
        n=t.nodes[i]
        if n.axis==0 || (m!==nothing && !m.active[i]); push!(ids,i)
        else; visit(n.left); visit(n.right); end
    end
    visit(1); ids
end
function main()
    input=length(ARGS)>0 ? abspath(ARGS[1]) : abspath(joinpath(@__DIR__,"../data/bluesky_100000"))
    out=length(ARGS)>1 ? abspath(ARGS[2]) : abspath(joinpath(@__DIR__,"../output/bluesky_tree"))
    mkpath(out); X=load_bluesky_embeddings(input)
    println("Loaded ",size(X),"; fitting full-dimensional adaptive partition.")
    # Warm compilation on a small disposable problem.
    w=fit_partition(X[:,1:256];max_nodes=15)
    for kind in (:residual,:microclusters); fit_compression(w,X[:,1:256],kind); end
    fit=@timed fit_partition(X;max_nodes=2047,min_count=128,max_depth=24,seed=7)
    t=fit.value
    println("Reference: ",length(t.nodes)," nodes in ",fit.time," seconds")
    timings=Dict("reference"=>fit.time)
    models=Dict{String,EmbeddingCompression}()
    for kind in (:residual,:microclusters)
        result=@timed fit_compression(t,X,kind;split_count=128,cluster_cap=4,radius=.2)
        models[string(kind)]=result.value; timings[string(kind)]=result.time
        println(kind,": ",result.time," seconds, mass ",result.value.counts[1])
    end
    serialize(joinpath(out,"models.jls"),(partition=t,models=models,input=input,
        config=(seed=7,max_nodes=2047,min_count=128,max_depth=24,split_count=128,cluster_cap=4,radius=.2),
        input_manifest=read(joinpath(input,"manifest.json"),String)))
    # Portable geometry; bounds are stored separately with one row per node/axis.
    open(joinpath(out,"nodes.csv"),"w") do io
        println(io,"node,left,right,axis,cut,count,depth,cut_time,log_volume")
        for (i,n) in enumerate(t.nodes)
            println(io,join((i,n.left,n.right,n.axis,n.cut,n.count,n.depth,n.tau,MP.logvolume(n)),','))
        end
    end
    open(joinpath(out,"bounds.csv"),"w") do io
        println(io,"node,axis,lower,upper")
        for (i,n) in enumerate(t.nodes), j in 1:t.dimensions
            println(io,join((i,j,n.lo[j],n.hi[j]),','))
        end
    end
    names=["reference","residual","microclusters"]
    ps=[embedding_probabilities(t,get(models,name,nothing)) for name in names]
    leafsets=[leaves(t,get(models,name,nothing)) for name in names]
    summary=["100,000-post static embedding tree experiment", "Input: $input", "Dimensions: $(size(X,1)); posts: $(size(X,2)); no projection or time decay.",
        "Adaptive Mondrian-style scaffold: observed-range random cuts, mass-priority scheduling, max 2047 nodes, min split occupancy 128, depth cap 24.",
        "This is an occupancy-adapted heuristic, not an exact/projective Mondrian process.",
        "Pólya branch prior kappa=2, centered on split-axis volume ratio.",
        "Residuals: prospective activation, old mass remains at ancestor; uniform-component mixture (not Beta branch inference).",
        "Microclusters: max 4 per active leaf, Euclidean radius 0.2, count/centroid/diagonal M2; whole-cluster centroid routing on split.",
        "Microcluster moments are exact global sufficient statistics; spatial branch counts after centroid routing are approximate. Not mixture PPCA.",
        "Input order is seeded random sample order, not chronological. All 100k rows also determine the scaffold; no held-out evaluation.", "",
        "method,leaves,summaries,represented_mass,reachable_model_bytes,fit_seconds,normalization_error,L1_vs_reference"]
    # All methods share a nested partition: integrate differences on finest leaves.
    fine=leaves(t); ref=ps[1]
    for (k,name) in enumerate(names)
        m=get(models,name,nothing); ids=leafsets[k]; p=ps[k]
        z=sum(p[ids]); @assert isapprox(z,1.;atol=1e-10)
        total=m===nothing ? t.nodes[1].count : m.counts[1]
        @assert total==size(X,2)
        err=0.
        for i in fine
            n=t.nodes[i]; x=(n.lo+n.hi)/2
            j=embedding_leaf(t,x,m===nothing ? nothing : m.active)
            approx=p[j]*exp(MP.logvolume(n)-MP.logvolume(t.nodes[j]))
            err+=abs(ref[i]-approx)
        end
        summaries=m===nothing ? length(ids) : name=="residual" ? count(>(0),m.residual) : sum(length,m.clusters)
        bytes=Base.summarysize(m===nothing ? t : m)
        push!(summary,join((name,length(ids),summaries,total,bytes,timings[name],abs(z-1),err),','))
        open(joinpath(out,name*"_leaves.csv"),"w") do io
            println(io,"node,probability,log_density,subtree_count,depth")
            for i in ids
                println(io,join((i,p[i],log(p[i])-MP.logvolume(t.nodes[i]),m===nothing ? t.nodes[i].count : m.counts[i],t.nodes[i].depth),','))
            end
        end
    end
    open(joinpath(out,"assignments.csv"),"w") do io
        println(io,"embedding_row,reference_leaf,residual_leaf,microclusters_leaf")
        for row in axes(X,2)
            println(io,join((row-1,(embedding_leaf(t,view(X,:,row),get(models,name,nothing)===nothing ? nothing : models[name].active) for name in names)...),','))
        end
    end
    open(joinpath(out,"residuals.csv"),"w") do io
        println(io,"node,residual_mass,active_split")
        m=models["residual"]
        for i in eachindex(t.nodes); println(io,join((i,m.residual[i],m.active[i]),',')); end
    end
    open(joinpath(out,"microclusters.csv"),"w") do io
        println(io,"node,cluster,count,axis,mean,central_sum_squares")
        for (i,cs) in enumerate(models["microclusters"].clusters), (k,c) in enumerate(cs), j in eachindex(c.mean)
            println(io,join((i,k,c.count,j,c.mean[j],c.m2[j]),','))
        end
    end
    push!(summary,"", "Memory includes scaffold geometry and model statistics, excludes external posts/embeddings and exports. Reference is already a count summary, not a raw replay buffer; compressors need not be smaller than this baseline.",
        "L1 is exact over common nested box support up to floating-point error. It measures approximation of the reference density, not recommendation quality.",
        "models.jls is Julia Serialization (load only trusted artifacts with compatible code/Julia). Portable CSV exports retain geometry and statistics.")
    write(joinpath(out,"summary.txt"),join(summary,'\n')*"\n")
    println(join(summary,'\n')); println("Saved to ",out)
end
main()
