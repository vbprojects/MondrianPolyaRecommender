using Random, LinearAlgebra, CUDA
include("leaf_decay_stress.jl") # defines the identical synthetic stream, guarded main
include("../src/LeafTreeKernels.jl")
using .LeafTreeKernels
include("load_bluesky.jl")
function compare_run(label,X,ts,h,out)
    options=(timestamps=ts,half_life=h,configs=[(:ppca,min(4,size(X,1)-1))],max_leaves=128,split_after=24,reservoir_cap=24,seed=82)
    ref=nothing;results=NamedTuple[]
    for (name,k) in (("Reference Julia",nothing),("KA CPU + LAPACK",KernelTreeBackend(Array;eigensolver=lapack_eigen_batch)),("KA NVIDIA + cuSOLVER",KernelTreeBackend(CUDA.CuArray;eigensolver=C->CUDA.cuSOLVER.syevjBatched!('V','U',C))))
        # Warm compilation on a separate short stream; excluded from measured time.
        warmopts=merge(options,(timestamps=ts[1:100],))
        k===nothing ? fit_leaf_tree(X[:,1:100];warmopts...) : fit_kernel_tree(k,X[:,1:100];warmopts...)
        GC.gc();elapsed=@elapsed f=k===nothing ? fit_leaf_tree(X;options...) : fit_kernel_tree(k,X;options...)
        ref===nothing && (ref=f)
        topology=[(n.axis,n.cut,n.left,n.right) for n in f.nodes]==[(n.axis,n.cut,n.left,n.right) for n in ref.nodes]
        @assert topology
        q=k===nothing ? leaf_probabilities(f) : kernel_probabilities(k,f);qr=leaf_probabilities(ref)
        leaves=findall(n->n.axis==0,f.nodes)
        meanerr=maximum(norm(f.nodes[i].states[1].mean-ref.nodes[i].states[1].mean) for i in leaves)
        coverr=maximum(begin
            a=parameters(f.nodes[i].states[1]);b=parameters(ref.nodes[i].states[1])
            norm(a.W*a.W'+Diagonal(a.D)-b.W*b.W'-Diagonal(b.D))
        end for i in leaves)
        tv=.5sum(abs.(q.probabilities-qr.probabilities))
        @assert meanerr<1e-6 && coverr<1e-6 && tv<1e-6
        row=(case=label,method=name,points=size(X,2),dimensions=size(X,1),leaves=length(leaves),seconds=elapsed,identical_topology=topology,probability_tv=tv,max_leaf_mean_error=meanerr,max_covariance_error=coverr)
        println(row);push!(results,row)
    end
    results
end
function main_comparison()
    BLAS.set_num_threads(1);CUDA.functional() || error("CUDA required for this comparison");CUDA.allowscalar(false)
    out=normpath(joinpath(@__DIR__,"../output/kernel_tree"));mkpath(out);records=NamedTuple[]
    X,ts=synthetic(31;rate=100)
    for (label,h) in (("2D static",Inf),("2D five-minute decay",5.),("2D one-minute decay",1.))
        append!(records,compare_run(label,X,ts,h,out))
    end
    real=load_bluesky_embeddings(joinpath(@__DIR__,"../data/bluesky_100000"))[:,1:2000]
    append!(records,compare_run("256D Bluesky static",real,zeros(size(real,2)),Inf,out))
    write(joinpath(out,"results.json"),sprint(io->json(io,records)))
end
main_comparison()
