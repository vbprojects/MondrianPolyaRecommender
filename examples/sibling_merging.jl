using Random, LinearAlgebra, Statistics, Printf, CUDA
include("leaf_decay_stress.jl")
include("../src/LeafTreeKernels.jl")
using .LeafTreeKernels
include("load_bluesky.jl")
modelbytes(f)=Base.summarysize((f.nodes,f.reservoir_rows,f.last_times,f.effective_arrivals))
function audit_merging(f,ref)
    @assert [(n.axis,n.cut,n.left,n.right) for n in f.nodes]==[(n.axis,n.cut,n.left,n.right) for n in ref.nodes]
    @assert f.merge_count==ref.merge_count && f.split_count==ref.split_count
    @assert f.reservoir_rows==ref.reservoir_rows
    @assert all(n.reservoir==a.reservoir for (n,a) in zip(f.nodes,ref.nodes))
    tv=.5sum(abs.(leaf_probabilities(f).probabilities-leaf_probabilities(ref).probabilities))
    ids=findall(n->n.axis==0,f.nodes)
    meanerr=maximum(norm(f.nodes[i].states[1].mean-ref.nodes[i].states[1].mean) for i in ids)
    coverr=maximum(begin
        p=parameters(f.nodes[i].states[1]);q=parameters(ref.nodes[i].states[1]);norm(p.W*p.W'+Diagonal(p.D)-q.W*q.W'-Diagonal(q.D))
    end for i in ids)
    @assert max(tv,meanerr,coverr)<1e-6
    (probability_tv=tv,mean_error=meanerr,covariance_error=coverr)
end
function main_merging()
    BLAS.set_num_threads(1);CUDA.allowscalar(false);CUDA.functional() || error("CUDA required")
    out=normpath(joinpath(@__DIR__,"../output/sibling_merging"));mkpath(out)
    X,t=synthetic(31;rate=100)
    settings=(timestamps=t,half_life=1.,configs=[(:ppca,1)],split_after=24,max_leaves=128,reservoir_cap=32,seed=82)
    history=NamedTuple[];frames=NamedTuple[]
    for threshold in (0.,3.)
        label=threshold==0 ? "Decay only" : "Decay + sibling merging"
        function checkpoint(row,f)
            ids=findall(n->n.axis==0,f.nodes);q=leaf_probabilities(f);w=exp2.(-(f.time.-view(t,1:row)))
            total=sum(w);mu=view(X,:,1:row)*w/total
            mean_est=sum(q.masses[i]*f.nodes[i].states[1].mean for i in ids)/sum(q.masses[ids])
            @assert abs(sum(q.masses[ids])-total)<1e-8 && norm(mean_est-mu)<1e-8
            rec=(method=label,time=f.time,leaves=length(ids),nodes=length(f.nodes),tree_bytes=Base.summarysize(f.nodes),splits=f.split_count,merges=f.merge_count)
            push!(history,rec)
            cells=[(lo=f.nodes[i].lo,hi=f.nodes[i].hi,mass=q.masses[i]/total) for i in ids]
            push!(frames,(metrics=rec,cells=cells,points=[collect(X[:,j]) for j in max(1,row-149):row]))
        end
        fit_leaf_tree(X;settings...,merge_below=threshold,merge_interval=1.,on_checkpoint=checkpoint,observe_every=100)
    end
    # Time full fits separately, without visualization callbacks (which use the event driver).
    raw=load_bluesky_embeddings(joinpath(@__DIR__,"../data/bluesky_100000"))
    real=raw[:,1:12000]
    # Actual embeddings with a deliberately synthetic burst/silence clock.
    real_t=vcat(collect(range(0.,1.;length=4000)),collect(range(15.,16.;length=4000)),collect(range(30.,31.;length=4000)))
    cases=[("2D moving topics",X,settings,3.),("256D Bluesky embeddings / synthetic clock",real,
        (timestamps=real_t,half_life=1.,configs=[(:ppca,4)],split_after=128,max_leaves=128,reservoir_cap=32,seed=82),16.)]
    records=NamedTuple[]
    for (label,A,opts,threshold) in cases
        kwargs=merge(opts,(merge_below=threshold,merge_interval=1.))
        ref=fit_leaf_tree(A;kwargs...)
        methods=[("Reference Julia",()->fit_leaf_tree(A;kwargs...)),
            ("Scheduled KA CPU",()->fit_kernel_tree(KernelTreeBackend(),A;kwargs...)),
            ("Scheduled KA GPU",()->fit_kernel_tree(KernelTreeBackend(CUDA.CuArray),A;kwargs...))]
        println("Warmup: ",label);flush(stdout)
        for (_,run) in methods;audit_merging(run(),ref);end
        samples=[NamedTuple[] for _ in methods]
        for trial in 1:3, j in circshift(collect(1:3),trial-1)
            GC.gc();CUDA.synchronize()
            result=@timed begin;f=methods[j][2]();CUDA.synchronize();f;end
            errors=audit_merging(result.value,ref)
            push!(samples[j],merge((seconds=result.time,allocated_bytes=result.bytes),errors))
            @printf("%s / %s / %d: %.4fs, %d splits, %d merges, %d live leaves\n",label,methods[j][1],trial,result.time,ref.split_count,ref.merge_count,count(n->n.axis==0,ref.nodes));flush(stdout)
        end
        # Explicit maintenance during silence verifies real reclamation with no new posts.
        before=modelbytes(ref);beforeleaves=count(n->n.axis==0,ref.nodes)
        ref=merge_tree!(ref;time=ref.time+30.)
        @assert length(ref.nodes)==1 && modelbytes(ref)<before
        for j in 1:3
            push!(records,(case=label,method=methods[j][1],n=size(A,2),d=size(A,1),
                median_seconds=median(s.seconds for s in samples[j]),samples=samples[j],
                leaves_before_idle=beforeleaves,bytes_before_idle=before,bytes_after_idle=modelbytes(ref),
                final_leaves=length(ref.nodes),split_count=ref.split_count,merge_count_after_idle=ref.merge_count))
        end
    end
    payload=(hardware=(cpu=Sys.cpu_info()[1].model,gpu=CUDA.name(CUDA.device()),julia=string(VERSION)),
        description="2D moving clusters: 10,500 arrivals across 120 minutes, one-minute half-life, 15-minute silence. Real 256D embeddings: 12,000 existing Bluesky posts with synthetic burst/silence timestamps; this clock is not their historical posting time. Three warmed timed fits per backend; callbacks and audits excluded from timing.",
        records=records,history=history,frames=frames)
    write(joinpath(out,"results.json"),sprint(io->json(io,payload)))
end
if abspath(PROGRAM_FILE)==@__FILE__;main_merging();end
