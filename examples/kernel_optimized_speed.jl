using Random, LinearAlgebra, Statistics, Printf, CUDA
include("leaf_decay_stress.jl")
include("../src/LeafTreeKernels.jl")
using .LeafTreeKernels
include("load_bluesky.jl")
function audit_fit(f,ref)
    @assert [(n.axis,n.cut,n.left,n.right) for n in f.nodes]==[(n.axis,n.cut,n.left,n.right) for n in ref.nodes]
    q=leaf_probabilities(f);qr=leaf_probabilities(ref);tv=.5sum(abs.(q.probabilities-qr.probabilities))
    ids=findall(n->n.axis==0,f.nodes)
    meanerr=maximum(norm(f.nodes[i].states[1].mean-ref.nodes[i].states[1].mean) for i in ids)
    coverr=maximum(begin
        a=parameters(f.nodes[i].states[1]);b=parameters(ref.nodes[i].states[1]);norm(a.W*a.W'+Diagonal(a.D)-b.W*b.W'-Diagonal(b.D))
    end for i in ids)
    @assert max(tv,meanerr,coverr)<1e-6
    @assert all(f.nodes[i].reservoir==ref.nodes[i].reservoir for i in ids)
    (probability_tv=tv,max_mean_error=meanerr,max_covariance_error=coverr)
end
function main_optimized()
    BLAS.set_num_threads(1);CUDA.allowscalar(false);CUDA.functional() || error("CUDA required")
    out=normpath(joinpath(@__DIR__,"../output/kernel_optimized"));mkpath(out)
    syn,t=synthetic(31;rate=100)
    allX=load_bluesky_embeddings(joinpath(@__DIR__,"../data/bluesky_100000"))
    cs=split.(readlines(joinpath(@__DIR__,"../output/leaf_decay/chronology.csv")),',')
    chronological=allX[:,[parse(Int,c[1]) for c in cs]];ts=[parse(Float64,c[2]) for c in cs]
    records=NamedTuple[]
    hardware=(cpu=Sys.cpu_info()[1].model,gpu=CUDA.name(CUDA.device()),julia=string(VERSION),threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),precision="Float64",timing="Whole supplied-batch fit, including topology planning and transfers. Warmup/data loading/audits excluded. Trial order rotated; no concurrent benchmark jobs.")
    cases=[("2D one-minute decay",syn,t,1.,24,256,5),
        ("2D static",syn,t,Inf,24,256,5),
        ("256D 10k six-hour decay",chronological[:,1:10000],ts[1:10000],21600.,128,256,5),
        ("256D 89,798 six-hour decay",chronological,ts,21600.,128,1024,3),
        ("256D 100k static",allX,zeros(size(allX,2)),Inf,128,1024,3)]
    methods=[("Reference Julia",nothing),("Scheduled KA CPU",KernelTreeBackend()),("Scheduled KA GPU",KernelTreeBackend(CUDA.CuArray))]
    for (label,A,times,h,threshold,cap,repeats) in cases
        kwargs=(timestamps=times,half_life=h,configs=[(:ppca,min(4,size(A,1)-1))],max_leaves=cap,split_after=threshold,reservoir_cap=32,seed=82)
        ref=fit_leaf_tree(A;kwargs...)
        runners=[()->k===nothing ? fit_leaf_tree(A;kwargs...) : fit_kernel_tree(k,A;kwargs...) for (_,k) in methods]
        println("Warmup: ",label);flush(stdout)
        for run in runners;audit_fit(run(),ref);end
        samples=[NamedTuple[] for _ in methods]
        for trial in 1:repeats,j in circshift(collect(1:3),trial-1)
            GC.gc();CUDA.synchronize()
            run=runners[j]
            result=@timed begin;f=run();CUDA.synchronize();f;end
            errors=audit_fit(result.value,ref)
            push!(samples[j],merge((seconds=result.time,allocated_bytes=result.bytes,gc_seconds=result.gctime),errors))
            @printf("%s / %s / %d: %.5fs, %.0f posts/s, TV %.3g\n",label,methods[j][1],trial,result.time,size(A,2)/result.time,errors.probability_tv);flush(stdout)
        end
        baseline=median(s.seconds for s in samples[1])
        for j in 1:3
            median_time=median(s.seconds for s in samples[j])
            push!(records,(case=label,method=methods[j][1],n=size(A,2),d=size(A,1),rank=kwargs.configs[1][2],half_life=isinf(h) ? "disabled" : string(h),split_after=threshold,max_leaves=cap,reservoir_cap=32,seed=82,leaves=count(n->n.axis==0,ref.nodes),median_seconds=median_time,posts_per_second=size(A,2)/median_time,speedup=baseline/median_time,samples=samples[j]))
        end
        write(joinpath(out,"results.json"),sprint(io->json(io,(hardware=hardware,records=records))))
    end
end
if abspath(PROGRAM_FILE)==@__FILE__;main_optimized();end
