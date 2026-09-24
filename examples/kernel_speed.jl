# Live, repeated speed comparison of the current tree drivers; no implementation changes.
using Random, LinearAlgebra, Statistics, Printf, CUDA
include("leaf_decay_stress.jl")
include("../src/LeafTreeKernels.jl")
using .LeafTreeKernels
include("load_bluesky.jl")
const SPEED_OUT=normpath(joinpath(@__DIR__,"../output/kernel_speed"))
function median_record(case,method,units,samples;extra=NamedTuple())
    seconds=[s.seconds for s in samples]
    merge((case=case,method=method,units=units,repeats=length(samples),median_seconds=median(seconds),min_seconds=minimum(seconds),max_seconds=maximum(seconds),units_per_second=units/median(seconds),samples=samples),extra)
end
function measure(f)
    GC.gc();CUDA.synchronize()
    result=@timed begin
        value=f();CUDA.synchronize();value
    end
    (seconds=result.time,host_allocated_bytes=result.bytes,gc_seconds=result.gctime),result.value
end
function main_speed()
    BLAS.set_num_threads(1);CUDA.functional() || error("CUDA required");CUDA.allowscalar(false)
    mkpath(SPEED_OUT)
    hardware=(cpu=Sys.cpu_info()[1].model,gpu=CUDA.name(CUDA.device()),julia=string(VERSION),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),precision="Float64 model statistics; source embeddings Float32",timing="warmup excluded; full driver includes validation, allocations, copies and synchronization; no embedding inference or disk loading")
    backends=[("Reference Julia",nothing),("KA CPU + LAPACK",KernelTreeBackend(Array;eigensolver=lapack_eigen_batch)),("KA GPU + cuSOLVER",KernelTreeBackend(CUDA.CuArray;eigensolver=C->CUDA.cuSOLVER.syevjBatched!('V','U',C)))]
    records=NamedTuple[]
    save()=write(joinpath(SPEED_OUT,"results.json"),sprint(io->json(io,(hardware=hardware,records=records))))
    syntheticX,syntheticT=synthetic(31;rate=100)
    allX=load_bluesky_embeddings(joinpath(@__DIR__,"../data/bluesky_100000"))
    # Reuse the actual chronological row references from the decay experiment.
    chronology=split.(readlines(joinpath(@__DIR__,"../output/leaf_decay/chronology.csv")),',')
    selected=chronology[1:10000];ids=[parse(Int,e[1]) for e in selected];realT=[parse(Float64,e[2]) for e in selected]
    realX=allX[:,ids]
    for (label,X,ts,h,threshold) in (("2D moving topics / 1-minute decay",syntheticX,syntheticT,1.,24),("256D Bluesky / 6-hour decay",realX,realT,21600.,128))
        options=(timestamps=ts,half_life=h,configs=[(:ppca,min(4,size(X,1)-1))],max_leaves=256,split_after=threshold,reservoir_cap=32,seed=82)
        reference=nothing;refprob=nothing
        for (name,k) in backends
            println("Warming ",label," / ",name);flush(stdout)
            warmn=min(1000,size(X,2));warmopts=merge(options,(timestamps=ts[1:warmn],))
            k===nothing ? fit_leaf_tree(X[:,1:warmn];warmopts...) : fit_kernel_tree(k,X[:,1:warmn];warmopts...)
            samples=NamedTuple[];worsttv=0.;same=true;leaves=0;bytes=0
            for trial in 1:3
                sample,f=measure(()->k===nothing ? fit_leaf_tree(X;options...) : fit_kernel_tree(k,X;options...))
                reference===nothing && (reference=f;refprob=leaf_probabilities(f))
                q=leaf_probabilities(f)
                identical=[(n.axis,n.cut,n.left,n.right) for n in f.nodes]==[(n.axis,n.cut,n.left,n.right) for n in reference.nodes]
                tv=.5sum(abs.(q.probabilities-refprob.probabilities));@assert identical && tv<1e-6
                same &= identical;worsttv=max(worsttv,tv);leaves=count(n->n.axis==0,f.nodes);bytes=Base.summarysize(f.nodes)
                push!(samples,sample)
                @printf("  %d/3: %.4f s; %.0f posts/s; %d leaves; probability TV %.3g\n",trial,sample.seconds,size(X,2)/sample.seconds,leaves,tv);flush(stdout)
            end
            push!(records,median_record(label,name,size(X,2),samples;extra=(operation="full tree fit",leaves=leaves,tree_bytes=bytes,identical_topology=same,max_probability_tv=worsttv)));save()
        end
        # Routing benchmark on a fixed final tree, including packing/transfers.
        queries=label=="256D Bluesky / 6-hour decay" ? allX : repeat(X,1,10)
        expected=[leaf_index(reference.nodes,x) for x in eachcol(queries)]
        for (name,k) in backends
            f=()->k===nothing ? [leaf_index(reference.nodes,x) for x in eachcol(queries)] : kernel_route(k,reference.nodes,queries)
            @assert f()==expected
            samples=NamedTuple[]
            for _ in 1:5
                sample,routes=measure(f);@assert routes==expected;push!(samples,sample)
            end
            push!(records,median_record(label,name,size(queries,2),samples;extra=(operation="batched routing",leaves=count(n->n.axis==0,reference.nodes),tree_bytes=Base.summarysize(reference.nodes),identical_topology=true,max_probability_tv=0.)));save()
            @printf("  Routing %s: %.5f s / %d queries\n",name,median(s.seconds for s in samples),size(queries,2));flush(stdout)
        end
    end
    println("Saved ",joinpath(SPEED_OUT,"results.json"))
end
main_speed()
