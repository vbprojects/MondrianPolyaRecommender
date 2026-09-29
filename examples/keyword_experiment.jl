# One isolated storage worker per method; embedding preparation excluded.
include("disk_ablation_common.jl")
const KE=DS.KeywordExtraction
using .DiskTreeStore.KeywordExtraction
method=isempty(ARGS) ? "yake" : ARGS[1]
method in ("disabled","hashtags","yake","online") || error("unknown method")
resultdir=joinpath(ROOT,"output/keywords");mkpath(resultdir)
root=joinpath(ROOT,"data/keyword_experiment",method)
isdir(root) && error("use a fresh experiment directory: $root")
# Compile storage and keyword paths on a disposable store before timing.
warmup()
mktempdir() do tmp
    s=DS.open_store(tmp;create_config=DS.config(dimension=2));e=HashtagsOnlyExtractor();DS.enable_keywords!(s,e)
    ps=[PostRecord(PostKey(:threads,"warm$i");text="warm #space",created_at=0.,observed_at=0.) for i in 1:10]
    DS.append_batch!(s,ps,fill(0.1f0,2,10);keywords=prepare_keywords(e,keyword_input.(ps)))
    DS.leaf_keywords(s,1);DS.close_store(s)
end
struct PreparedYake <: AbstractKeywordExtractor end
KE.extractor_id(::PreparedYake)="yake@0.6.0;en;n=3;dedup=0.9;window=1;top=24;maxchars=8192;clean-v1"
e=method=="online" ? PythonYakeExtractor() : method=="yake" ? PreparedYake() : HashtagsOnlyExtractor()
s=DS.open_store(root;budget=32*1024^2,create_config=DS.config(mode=:f16,layout=:append,seed=17))
method!="disabled" && DS.enable_keywords!(s,e)
N=method=="online" ? 10000 : Int(JSON3.read(read(joinpath(INPUT,"manifest.json"),String)).contract.count)
latencies=Float64[];times=Float64[];snapshots=Any[];snapshot_seconds=0.;start=time_ns();initial=osinfo()
if method=="yake"
    manifest=JSON3.read(read(joinpath(ROOT,"data/keyword_preparation/manifest.json"),String))
    manifest.contract.extractor==extractor_id(e) || error("sidecar extractor mismatch")
    bytes2hex(open(sha256,joinpath(INPUT,"posts.jsonl")))==manifest.contract.source_sha256 || error("sidecar source mismatch")
    bytes2hex(open(sha256,joinpath(ROOT,"data/keyword_preparation/keywords.jsonl")))==manifest.sidecar_sha256 || error("sidecar checksum mismatch")
end
side=method=="yake" ? open(joinpath(ROOT,"data/keyword_preparation/keywords.jsonl")) : nothing
include("keyword_snapshot.jl")

try
    open(joinpath(INPUT,"posts.jsonl")) do textio
        open(joinpath(INPUT,"embeddings.f32")) do vecio
            for first in 1:1024:N
                n=min(1024,N-first+1);records,X=readbatch(textio,vecio,first,n)
                kw=if method=="disabled"
                    nothing
                elseif method in ("hashtags","online")
                    prepare_keywords(e,keyword_input.(records))
                else
                    [begin
                        row=JSON3.read(readline(side))
                        row.identity==p.key.native && row.text_sha256==bytes2hex(sha256(p.text)) || error("sidecar alignment failure")
                        raw=KeywordResult([Candidate(String(c.phrase),Int(c.rank),Float64(c.score)) for c in row.candidates],extractor_id(e),Bool(row.truncated))
                        finalize_keywords(keyword_input(p),raw)
                    end for p in records]
                end
                t=@elapsed DS.append_batch!(s,records,X;keywords=kw,time=(first+n-1)/100)
                push!(times,t)
                if first==1 && method!="disabled"
                    global snapshot_seconds+=@elapsed push!(snapshots,snapshot(s,"After first 1,024 posts / early splits"))
                end
                if first==1 || first+1024>N || (first-1)%10240==0
                    println("$method $(first+n-1)/$N storage=$(round(n/t;digits=1)) posts/s");flush(stdout)
                end
            end
        end
    end
finally
    side!==nothing && close(side)
end
ingestwall=(time_ns()-start)/1e9-snapshot_seconds
signature=DS.tree_signature(s)
if method!="disabled"
    push!(snapshots,snapshot(s,"After $(N) posts"))
    for leaf in [n.id for n in values(s.nodes) if n.axis==0]
        DS.clear_caches!(s)
        push!(latencies,@elapsed DS.leaf_keywords(s,leaf;ranking=:distinctive))
    end
end
println("$method summaries measured; starting 72-hour decay/merge probe");flush(stdout)
stats=DS.storage_stats(s);breakdown=db_breakdown(s);memory=osinfo()
summary_times=isempty(latencies) ? nothing : (p50=quantile(latencies,.5),p95=quantile(latencies,.95),p99=quantile(latencies,.99))
# Induce strong decay and structural merges while keeping seven-day retention separate.
t=@elapsed DS.maintain!(s;time=s.clock+72*3600)
method!="disabled" && push!(snapshots,snapshot(s,"After 72 hours of decay / merges"))
metrics=copy(s.metrics)
result=(method=method,count=N,seed=17,storage_seconds=sum(times),storage_posts_per_second=N/sum(times),
    ingest_with_prepared_keywords_seconds=ingestwall,ingest_with_prepared_keywords_per_second=N/ingestwall,
    includes_extraction=method=="online",end_to_end_posts_per_second=method=="online" ? N/ingestwall : nothing,
    commit_seconds=times,summary_seconds=summary_times,merge_seconds=t,stats=stats,db_pages=breakdown,
    memory=memory,metrics=metrics,tree_signature=signature,keyword_stats=DS.keyword_stats(s),snapshots=snapshots,
    config=s.config,keyword_config=s.keyword_config,budget_mib=32,blas_threads=BLAS.get_num_threads())
savejson(joinpath(resultdir,"$method.json"),result)
DS.close_store(s)
method=="online" && close(e)
println("saved $method")
