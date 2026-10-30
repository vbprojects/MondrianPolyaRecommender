include("GteTiny.jl")
using .GteTiny, JSON3, Statistics, SHA

artifact = isempty(ARGS) ? normpath(joinpath(@__DIR__,"../../output/gte_tiny_julia")) : abspath(ARGS[1])
readjson(name) = JSON3.read(read(joinpath(artifact,name),String))
ascolumns(rows) = hcat((Float32.(r) for r in rows)...)
validation = readjson("validation.json")
validation.passed && validation.source_sha256 == bytes2hex(sha256(read(joinpath(@__DIR__,"GteTiny.jl")))) ||
    error("Run validate.jl for this implementation before benchmarking")
for (file, expected) in pairs(validation.fixtures_sha256)
    bytes2hex(sha256(read(joinpath(artifact,String(file))))) == expected || error("Reference changed; rerun validation")
end
reference = readjson("reference.json")
Threads.nthreads() == reference.threads || error("Match the ONNX reference thread count")
load_time = @elapsed model = Model(joinpath(artifact,"model"))
panel = String.(readjson("panel.json"))
timings = Any[]
first_encode_seconds = @elapsed encode(model,[panel[1]])
for batch in (1,8)
    encode(model,panel;batch_size=batch) # warm the exact path
    samples = Float64[]; allocated = Int[]; gc = Float64[]
    for _ in 1:3
        GC.gc()
        trial = @timed encode(model,panel;batch_size=batch)
        push!(samples,trial.time);push!(allocated,trial.bytes);push!(gc,trial.gctime)
    end
    row = Dict("backend"=>"julia_int8","batch"=>batch,"samples_seconds"=>samples,
        "texts_per_second"=>length(panel)/median(samples),"allocated_bytes"=>allocated,"gc_seconds"=>gc)
    push!(timings,row)
    println("Julia batch $batch: ",round(row["texts_per_second"];digits=2)," texts/s");flush(stdout)
end

function ndcg_scores(q,d,screen)
    rows = Float64[]
    for (i,qid) in enumerate(screen.qids)
        rel = Set(String.(screen.relevant[Symbol(qid)]))
        scores = [sum(view(q,:,i).*view(d,:,j)) for j in axes(d,2)]
        order = sortperm(eachindex(scores);by=j->(-scores[j],String(screen.cids[j])))
        gain(id) = id in rel ? 1.0 : 0.0
        ideal = ones(length(rel))
        dcg = sum((2.0^gain(screen.cids[order[r]])-1)/log2(r+1) for r in 1:min(10,length(order)))
        idcg = sum((2.0^ideal[r]-1)/log2(r+1) for r in 1:min(10,length(ideal)))
        push!(rows,idcg==0 ? 0.0 : dcg/idcg)
    end
    rows
end

screens = Any[]
for screen in readjson("retrieval.json")
    elapsed = @elapsed begin
        q = encode(model,String.(screen.queries);batch_size=screen.batch)
        d = encode(model,String.(screen.docs);batch_size=screen.batch)
    end
    jr = ndcg_scores(q,d,screen)
    ortq = ascolumns(screen.onnx_int8.queries); ortd = ascolumns(screen.onnx_int8.docs)
    qr = ndcg_scores(ortq,ortd,screen)
    fr = ndcg_scores(ascolumns(screen.onnx_fp32.queries),ascolumns(screen.onnx_fp32.docs),screen)
    cosine(a,b) = [sum(a[:,i].*b[:,i])/sqrt(sum(abs2,a[:,i])*sum(abs2,b[:,i])) for i in axes(a,2)]
    row = Dict("dataset"=>screen.name,"queries"=>length(jr),"documents"=>length(screen.cids),
        "julia_ndcg10"=>mean(jr),"onnx_int8_ndcg10"=>mean(qr),"onnx_fp32_ndcg10"=>mean(fr),
        "julia_seconds"=>elapsed,"julia_per_query"=>jr,"onnx_int8_per_query"=>qr,"onnx_fp32_per_query"=>fr,
        "query_cosine_min"=>minimum(cosine(q,ortq)),"document_cosine_min"=>minimum(cosine(d,ortd)),
        "source_sha256"=>screen.source_sha256)
    push!(screens,row)
    println(screen.name,": Julia=",round(mean(jr);digits=5)," ONNX INT8=",round(mean(qr);digits=5),
        " FP32=",round(mean(fr);digits=5)," seconds=",round(elapsed;digits=2));flush(stdout)
end
meta = readjson("model/weights.json")
weight_bytes = sum(t.bytes for t in values(meta.tensors))
stats = Dict("model"=>reference.model,"revision"=>reference.revision,"julia"=>string(VERSION),
    "cpu"=>Sys.CPU_NAME,"threads"=>Threads.nthreads(),"ort"=>reference.ort,
    "load_seconds"=>load_time,"first_encode_seconds"=>first_encode_seconds,
    "model_object_bytes"=>Base.summarysize(model),"weight_file_bytes"=>weight_bytes,
    "panel_texts"=>length(panel),"timings"=>vcat(collect(reference.timings),timings),"retrieval"=>screens,
    "scope"=>reference.scope,"source_sha256"=>validation.source_sha256)
open(joinpath(artifact,"benchmark.json"),"w") do io
    JSON3.pretty(io,stats)
end
println("Benchmark saved to ",joinpath(artifact,"benchmark.json"))
