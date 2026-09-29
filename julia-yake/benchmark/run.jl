# Run with --project=environments/storage. Python is benchmark-only.
using JSON3, Statistics, SHA
include("../../src/KeywordExtraction.jl")
const KE=KeywordExtraction
function run()
    root=normpath(joinpath(@__DIR__,"../.."))
    posts=[JSON3.read(line) for line in eachline(joinpath(root,"data/gte_disk_ablation/posts.jsonl"))]
    inputs=[KE.KeywordInput(String(p.uri),String(p.text),String[],"") for p in posts]
    native=KE.JuliaYakeExtractor()
    KE.prepare_keywords(native,inputs[1:32]) # JIT warmup excluded
    latencies=Float64[];total=0.;overlap=0.;exact=0;top1=0;nonempty=0
    open(joinpath(root,"data/keyword_preparation/keywords.jsonl")) do reference
        for start in 1:256:length(inputs)
            batch=inputs[start:min(end,start+255)]
            elapsed=@elapsed results=KE.extract_keywords(native,batch)
            total+=elapsed;push!(latencies,elapsed)
            for (input,result) in zip(batch,results)
                py=JSON3.read(readline(reference))
                input.identity==py.identity || error("reference alignment")
                bytes2hex(sha256(input.text))==py.text_sha256 || error("reference text mismatch")
                a=[KE.canonical(c.phrase) for c in result.candidates][1:min(end,8)]
                b=[KE.canonical(c.phrase) for c in py.candidates][1:min(end,8)]
                overlap+=isempty(union(a,b)) ? 1. : length(intersect(a,b))/length(union(a,b))
                exact+=a==b
                if !isempty(a) || !isempty(b)
                    nonempty+=1;top1+=!isempty(a)&&!isempty(b)&&first(a)==first(b)
                end
            end
        end
    end
    # Identical input subset, one Julia thread / one Python worker; protocol overhead included.
    subset=inputs[1:min(10000,end)]
    timings=Dict{String,Any}()
    for (name,e) in [("julia",native),("python",KE.PythonYakeExtractor())]
        try
            KE.prepare_keywords(e,inputs[1:32])
            t=@elapsed for start in 1:256:length(subset)
                KE.prepare_keywords(e,subset[start:min(end,start+255)])
            end
            timings[name]=(;seconds=t,posts_per_second=length(subset)/t,posts=length(subset))
        finally
            name=="python" && close(e)
        end
    end
    result=(;posts=length(inputs),julia_version=string(VERSION),julia_threads=Threads.nthreads(),
        cpu=Sys.cpu_info()[1].model,extractor=KE.extractor_id(native),python_version="yake 0.6.0",
        native_extraction_seconds=total,native_posts_per_second=length(inputs)/total,
        batch_size=256,batch_latency_p50=median(latencies),batch_latency_p95=quantile(latencies,.95),
        top8_mean_jaccard=overlap/length(inputs),top8_exact_rank_fraction=exact/length(inputs),
        top1_agreement_nonempty=top1/nonempty,paired_preparation=timings,
        notes="100k extraction excludes JSON IO/reference comparison; paired 10k preparation includes canonicalization and digest, Python IPC, excludes warmup; single worker each; unknown languages treated as English.")
    write(joinpath(@__DIR__,"results.json"),JSON3.write(result))
    JSON3.pretty(result);println()
end
run()
