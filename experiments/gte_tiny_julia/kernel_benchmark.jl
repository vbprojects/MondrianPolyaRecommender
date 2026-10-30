include("KernelGte.jl")
using .GteTiny, JSON3, Statistics, SHA
const WITH_GPU=!("--cpu-only" in ARGS)
WITH_GPU && include("GPUKernels.jl")
artifact=normpath(joinpath(@__DIR__,"../../output/gte_tiny_julia"))
readjson(name)=JSON3.read(read(joinpath(artifact,name),String))
columns(rows)=hcat((Float32.(r) for r in rows)...)
hash(path)=bytes2hex(sha256(read(path)))
validation=readjson("kernel-validation.json")
validation.passed || error("Kernel validation has not passed")
WITH_GPU && !validation.gpu_tested && error("Validate the GPU before benchmarking it")
for (file,checksum) in pairs(validation.sources)
    hash(joinpath(@__DIR__,String(file)))==checksum || error("Source changed; rerun kernel_validate.jl")
end
for (file,checksum) in pairs(validation.fixtures)
    hash(joinpath(artifact,String(file)))==checksum || error("Inputs changed; rerun kernel_validate.jl")
end
reference=readjson("kernel-reference.json")
reference.threads==Threads.nthreads() || error("Match the reference's CPU thread count")
reference.panel_sha256==hash(joinpath(artifact,"panel.json")) || error("Reference timing panel mismatch")
model=GteTiny.Model(joinpath(artifact,"model"))
native=KernelGte.KernelModel(model)
poly=KernelGte.KernelModel(model;polynomial_math=true)
backends=Any[("julia_reference_cpu",model),("ka_cpu",native),("ka_cpu_polynomial",poly)]
WITH_GPU && push!(backends,("ka_cuda",GPUKernels.GPUModel(model)))
WITH_GPU && GPUKernels.CUDA.allowscalar(false)
function run(name,m,texts,batch)
    name=="julia_reference_cpu" ? GteTiny.encode(m,texts;batch_size=batch) :
    name=="ka_cuda" ? GPUKernels.encode(m,texts;batch_size=batch) : KernelGte.encode(m,texts;batch_size=batch)
end
panel=String.(readjson("panel.json"))
timings=Any[collect(reference.timings)...]
for batch in (1,8)
    samples=Dict(name=>Any[] for (name,_) in backends)
    for (name,m) in backends
        run(name,m,panel,batch)
    end
    for round in 1:5
        for i in circshift(collect(eachindex(backends)),round-1)
            name,m=backends[i]
            GC.gc()
            trial=@timed run(name,m,panel,batch)
            push!(samples[name],Dict("seconds"=>trial.time,"allocated_bytes"=>trial.bytes,
                                    "gc_seconds"=>trial.gctime))
        end
    end
    for (name,_) in backends
        rate=length(panel)/median([s["seconds"] for s in samples[name]])
        push!(timings,Dict("backend"=>name,"batch"=>batch,"samples"=>samples[name],"texts_per_second"=>rate))
        println(name," batch ",batch,": ",round(rate;digits=2)," texts/s");flush(stdout)
    end
end

function ndcg(q,d,screen)
    result=Float64[]
    for (i,qid) in enumerate(screen.qids)
        relevant=Set(String.(screen.relevant[Symbol(qid)]))
        scores=[sum(view(q,:,i).*view(d,:,j)) for j in axes(d,2)]
        order=sortperm(eachindex(scores);by=j->(-scores[j],String(screen.cids[j])))
        dcg=sum((screen.cids[order[r]] in relevant)/log2(r+1) for r in 1:min(10,length(order)))
        ideal=sum(1/log2(r+1) for r in 1:min(10,length(relevant)))
        push!(result,dcg/ideal)
    end
    result
end
cosines(a,b)=[sum(a[:,i].*b[:,i])/sqrt(sum(abs2,a[:,i])*sum(abs2,b[:,i])) for i in axes(a,2)]
retrieval=Any[]
vectorsdir=joinpath(artifact,"kernel-vectors");mkpath(vectorsdir)
for screen in readjson("retrieval.json")
    reference_q=nothing;reference_d=nothing
    oq=columns(screen.onnx_int8.queries);od=columns(screen.onnx_int8.docs)
    for (name,m) in backends
        elapsed=@elapsed begin
            q=run(name,m,String.(screen.queries),screen.batch)
            d=run(name,m,String.(screen.docs),screen.batch)
        end
        if name=="julia_reference_cpu"
            reference_q=q;reference_d=d
        end
        scores=ndcg(q,d,screen)
        c_ref=min(minimum(cosines(q,reference_q)),minimum(cosines(d,reference_d)))
        c_ort=min(minimum(cosines(q,oq)),minimum(cosines(d,od)))
        exact=q==reference_q && d==reference_d
        name=="ka_cpu" && !exact && error("Strict CPU kernel changed reference vectors")
        push!(retrieval,Dict("dataset"=>screen.name,"backend"=>name,"queries"=>length(screen.queries),
            "documents"=>length(screen.docs),"ndcg10"=>mean(scores),"per_query"=>scores,
            "seconds"=>elapsed,"reference_cosine_min"=>c_ref,"onnx_int8_cosine_min"=>c_ort,
            "bitwise_reference"=>exact))
        for (suffix,mat) in (("queries",q),("docs",d))
            write(joinpath(vectorsdir,"$(screen.name)-$name-$suffix.f32"),mat)
        end
        println(screen.name," ",name,": ",round(mean(scores);digits=6)," cosine/reference=",c_ref," exact=",exact);flush(stdout)
    end
    for (name,vectors) in (("onnx_int8_cpu",screen.onnx_int8),("onnx_fp32_cpu",screen.onnx_fp32))
        scores=ndcg(columns(vectors.queries),columns(vectors.docs),screen)
        push!(retrieval,Dict("dataset"=>screen.name,"backend"=>name,"ndcg10"=>mean(scores),"per_query"=>scores))
    end
end
stats=Dict("julia"=>string(VERSION),"cpu"=>Sys.CPU_NAME,"threads"=>Threads.nthreads(),
    "kernelabstractions"=>string(pkgversion(KernelGte.KernelAbstractions)),"vnni"=>native.vnni,
    "gpu"=>WITH_GPU ? GPUKernels.CUDA.name(GPUKernels.CUDA.device()) : nothing,
    "panel_texts"=>length(panel),"timings"=>timings,"retrieval"=>retrieval,
    "source_hashes"=>validation.sources,"fixture_hashes"=>validation.fixtures,
    "reference_hash"=>hash(joinpath(artifact,"kernel-reference.json")),
    "benchmark_source_hash"=>hash(@__FILE__),
    "scope"=>"Reduced retrieval screen; CPU/GPU timings are separate hardware comparisons. Julia backends rotate order; ONNX ran separately.")
open(joinpath(artifact,"kernel-benchmark.json"),"w") do io
    JSON3.pretty(io,stats)
end
println("Saved kernel-benchmark.json")
