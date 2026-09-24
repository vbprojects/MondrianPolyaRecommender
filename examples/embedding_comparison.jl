# Compare native embedding geometry and the actual adaptive PPCA tree on matched posts.
using LinearAlgebra, Random, Statistics, Printf, JSON3, SHA
include("../src/LeafLatentTree.jl")
include("../src/EmbeddingClustering.jl")
const LT = LeafLatentTrees
const EC = EmbeddingClustering
BLAS.set_num_threads(4)
const ROOT = normpath(joinpath(@__DIR__, ".."))
const INPUT = joinpath(ROOT,"data/embedding_comparison")
const OUTPUT = joinpath(ROOT,"output/embedding_comparison")
const SEEDS = [17,29,43]

macro_average(values,labels) = mean(mean(values[labels.==l]) for l in unique(labels))
function word_shingles(text)
    cleaned=replace(lowercase(String(text)),r"https?://\S+"=>" ",r"@\S+"=>" ")
    words=[m.match for m in eachmatch(r"\p{L}+",cleaned)]
    Set(Tuple(words[i:i+2]) for i in 1:length(words)-2)
end
function template_overlap(a,b)
    shared=length(intersect(a,b))
    shared>=6 && shared/max(1,length(union(a,b)))>=.5
end
function compact_post(p)
    (text=String(p.text),input_text=String(p.input_text),uri=String(p.uri),
     author=String(p.author),topic=String(p.topic),train=Bool(p.train))
end
function projection(train,X)
    mu=mean(train;dims=2);Y=Float64.(train.-mu)
    e=eigen(Symmetric(Y*Y'));Q=e.vectors[:,end:-1:end-1]
    (points=[collect(p) for p in eachcol(Q'*(X.-mu))],
     explained=sum(e.values[end-1:end])/sum(e.values))
end

function run_experiment()
    mkpath(OUTPUT)
    manifest=JSON3.read(read(joinpath(INPUT,"manifest.json"),String))
    posts=[JSON3.read(s) for s in eachline(joinpath(INPUT,"posts.jsonl"))]
    @assert bytes2hex(sha256(read(joinpath(INPUT,"posts.jsonl")))) == manifest.posts_sha256
    result=Dict{String,Any}("manifest"=>manifest,"seeds"=>SEEDS,"cohorts"=>Dict(),"runs"=>Any[],"stability"=>Any[])
    gpu=try strip(read(`nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`,String)) catch; "Unavailable" end
    result["runtime"]=(julia=string(VERSION),cpu=Sys.cpu_info()[1].model,blas_threads=BLAS.get_num_threads(),gpu=gpu)
    result["config"]=(natural_groups=64,topic_groups=16,natural_split_after=128,topic_split_after=16,
        reservoir_cap=64,ppca_rank=4,kmeans_max_iterations=200,half_life="Inf",max_depth=32,
        input_dimensions="native",silhouette_max_posts=1024,natural_neighbor_queries=512)
    # Compile both algorithms before recording fit times.
    warm=randn(MersenneTwister(1),Float32,12,100);warm./=sqrt.(sum(abs2,warm;dims=1))
    EC.spherical_kmeans(warm,4;iterations=2)
    LT.fit_leaf_tree(warm;max_leaves=3,split_after=20,reservoir_cap=20,configs=[(:ppca,4)])
    for cohort in ["natural","topics"]
        rows=findall(p->p.cohort==cohort,posts);localposts=posts[rows]
        trainids=findall(p->p.train,localposts);testids=findall(p->!p.train,localposts)
        topics=sort(unique(String(p.topic) for p in localposts))
        labels=[findfirst(==(String(p.topic)),topics) for p in localposts]
        trainlabels=labels[trainids];testlabels=labels[testids]
        k=cohort=="natural" ? 64 : 16
        rng=MersenneTwister(811)
        plotids=sort(shuffle(rng,collect(eachindex(testids)))[1:min(2000,length(testids))])
        qids=cohort=="natural" ? sort(shuffle(rng,collect(eachindex(testids)))[1:min(512,length(testids))]) : collect(eachindex(testids))
        displayq=cohort=="natural" ? qids[1:min(12,length(qids))] : [findfirst(==(l),testlabels) for l in unique(testlabels)]
        report=Dict{String,Any}("train"=>length(trainids),"test"=>length(testids),"groups"=>k,"topics"=>topics,
            "posts"=>compact_post.(localposts[testids[plotids]]),"models"=>Dict(),
            "topic_counts"=>[Dict("topic"=>topics[l],"train"=>count(==(l),trainlabels),"test"=>count(==(l),testlabels)) for l in eachindex(topics)])
        blocked=Vector{Int}[]
        if cohort=="topics"
            shingles=word_shingles.(getproperty.(localposts,:input_text))
            blocked=[findall(j->template_overlap(shingles[trainids[j]],shingles[testids[q]]),eachindex(trainids)) for q in qids]
            report["template_control"]=(rule="Exclude training posts with word-trigram Jaccard >= 0.5 and at least six shared trigrams; URLs, mentions and nonletters removed for overlap detection only.",
                mean_excluded_candidates=mean(length.(blocked)),queries_with_exclusions=count(!isempty,blocked))
        end
        result["cohorts"][cohort]=report
        for model in ["current","arctic"]
            info=manifest.models[model];d=Int(info.dimensions)
            path=joinpath(INPUT,model*".f32")
            @assert bytes2hex(sha256(read(path))) == info.embedding_sha256
            allX=open(path) do io
                reshape(read!(io,Vector{Float32}(undef,d*length(posts))),d,length(posts))
            end
            X=allX[:,rows];train=X[:,trainids];test=X[:,testids]
            @assert all(isfinite,X) && maximum(abs.(vec(sum(abs2,X;dims=1)).-1))<2e-6
            similarities=train'*test[:,qids]
            nearest=[partialsortperm(view(similarities,:,j),1:10;rev=true) for j in eachindex(qids)]
            exact_topic=cohort=="topics" ? [mean(trainlabels[nearest[j]].==testlabels[qids[j]]) for j in eachindex(qids)] : Float64[]
            modelreport=Dict{String,Any}("dimensions"=>d,"projection"=>projection(train,test[:,plotids]),"methods"=>Dict())
            report["models"][model]=modelreport
            queryexamples=Any[]
            for qi in displayq
                j=findfirst(==(qi),qids)
                push!(queryexamples,Dict("query"=>compact_post(localposts[testids[qi]]),
                    "neighbors"=>[merge(compact_post(localposts[trainids[n]]),(cosine=similarities[n,j],)) for n in nearest[j][1:5]]))
            end
            modelreport["neighbors"]=queryexamples
            if cohort=="topics"
                modelreport["exact_topic_precision10"]=macro_average(exact_topic,testlabels[qids])
                modelreport["query_precision10"]=exact_topic
                filtered_precision=Float64[];removed_top10=Int[]
                for (j,q) in enumerate(qids)
                    eligible=setdiff(collect(eachindex(trainids)),blocked[j])
                    top=eligible[partialsortperm(similarities[eligible,j],1:10;rev=true)]
                    push!(filtered_precision,mean(trainlabels[top].==testlabels[q]))
                    push!(removed_top10,length(intersect(nearest[j],blocked[j])))
                end
                modelreport["template_filtered_precision10"]=macro_average(filtered_precision,testlabels[qids])
                modelreport["template_filtered_query_precision10"]=filtered_precision
                modelreport["mean_excluded_top10"]=mean(removed_top10)
            end
            for method in ["kmeans","tree"]
                all_assignments=Vector{Int}[]
                for seed in SEEDS
                    println("Fitting ",cohort," / ",model," / ",method," / ",seed);flush(stdout)
                    if method=="kmeans"
                        seconds=@elapsed fit=EC.spherical_kmeans(train,k;seed)
                        ta=fit.assignments;ha=EC.assign_centers(fit.centers,test)
                        statebytes=Base.summarysize(fit.centers)
                        details=(leaves=k,iterations=fit.iterations,probability_sum=1.,maximum_depth=0)
                    else
                        order=randperm(MersenneTwister(seed),size(train,2))
                        seconds=@elapsed fit=LT.fit_leaf_tree(train[:,order];max_leaves=k,
                            split_after=cohort=="natural" ? 128 : 16,reservoir_cap=64,
                            max_depth=32,seed,configs=[(:ppca,4)])
                        ta=[LT.leaf_index(fit.nodes,x) for x in eachcol(train)]
                        ha=[LT.leaf_index(fit.nodes,x) for x in eachcol(test)]
                        statebytes=Base.summarysize(fit)
                        probs=LT.leaf_probabilities(fit)
                        @assert abs(sum(probs.probabilities)-1)<1e-10
                        @assert abs(probs.masses[1]-length(trainids))<1e-6
                        details=(leaves=count(n->n.axis==0,fit.nodes),iterations=0,
                            probability_sum=sum(probs.probabilities),maximum_depth=maximum(n.depth for n in fit.nodes))
                    end
                    push!(all_assignments,ha)
                    recall=[mean(ta[nearest[j]].==ha[qids[j]]) for j in eachindex(qids)]
                    candidates=[count(==(ha[q]),ta) for q in qids]
                    # Restrict to one partition and then rerank by exact cosine.
                    # Short/empty partitions incur missing-result penalties at k=10.
                    local_topic=Float64[]
                    if cohort=="topics"
                        for (j,q) in enumerate(qids)
                            ids=findall(==(ha[q]),ta)
                            top=isempty(ids) ? Int[] : ids[partialsortperm(similarities[ids,j],1:min(10,length(ids));rev=true)]
                            push!(local_topic,sum(trainlabels[top].==testlabels[q])/10)
                        end
                    end
                    silids=sort(shuffle(MersenneTwister(1),collect(eachindex(ha)))[1:min(1024,length(ha))])
                    metrics=merge(EC.group_metrics(ta),details,
                        (cohort=cohort,model=model,method=method,seed=seed,fit_seconds=seconds,
                         state_bytes=statebytes,vector_bytes=d*4*length(rows),
                         silhouette=EC.silhouette(test[:,silids],ha[silids]),
                         recall10=mean(recall),candidate_fraction=mean(candidates)/length(ta),
                         candidates=mean(candidates),empty_query_fraction=mean(candidates.==0)))
                    if cohort=="topics"
                        metrics=merge(metrics,EC.topic_metrics(ta,trainlabels,ha,testlabels),
                            (local_topic_precision10=macro_average(local_topic,testlabels[qids]),
                             exact_topic_precision10=macro_average(exact_topic,testlabels[qids])))
                    end
                    push!(result["runs"],metrics)
                    if seed==first(SEEDS)
                        representatives=Any[]
                        for g in sort(unique(ta);by=g->-count(==(g),ta))
                            ids=findall(==(g),ta);center=vec(mean(train[:,ids];dims=2));center./=max(norm(center),1e-12)
                            scores=vec(center'*train[:,ids]);best=ids[partialsortperm(scores,1:min(5,length(ids));rev=true)]
                            push!(representatives,Dict("id"=>g,"size"=>length(ids),
                                "examples"=>compact_post.(localposts[trainids[best]])))
                        end
                        modelreport["methods"][method]=Dict("assignments"=>ha[plotids],"groups"=>representatives)
                    end
                end
                stability=[EC.adjusted_rand(all_assignments[a],all_assignments[b]) for a in 1:2 for b in a+1:3]
                push!(result["stability"],(cohort=cohort,model=model,method=method,mean_ari=mean(stability),pair_ari=stability))
            end
        end
    end
    # Conditional uncertainty of the paired direct-neighbor topic probe.
    # Resample held-out authors, retaining all of each sampled author's posts.
    # This does not cover a new training corpus, topic definitions, or a live feed.
    probe=result["cohorts"]["topics"];pp=probe["posts"]
    labels=[p.topic for p in pp];authors=unique(p.author for p in pp)
    delta=probe["models"]["arctic"]["query_precision10"]-probe["models"]["current"]["query_precision10"]
    rng=MersenneTwister(719);boot=Float64[]
    while length(boot)<2000
        draw=rand(rng,authors,length(authors));weights=[count(==(p.author),draw) for p in pp]
        any(sum(weights[labels.==l])==0 for l in unique(labels)) && continue
        push!(boot,mean(sum(delta[labels.==l].*weights[labels.==l])/sum(weights[labels.==l]) for l in unique(labels)))
    end
    result["paired_topic_retrieval"]=(difference=macro_average(delta,labels),
        author_bootstrap_ci95=quantile(boot,[.025,.975]),replicates=length(boot),authors=length(authors),
        caveat="Conditional on fixed training corpus and weak hashtag labels; resamples test authors only.")
    open(joinpath(OUTPUT,"results.json"),"w") do io;JSON3.pretty(io,result);end
    # JSON is embedded as inert data; literal HTML delimiters from posts cannot close it.
    data=replace(JSON3.write(result),"<"=>"\\u003c",">"=>"\\u003e","&"=>"\\u0026")
    template=read(joinpath(@__DIR__,"embedding_comparison_viewer.html"),String)
    write(joinpath(OUTPUT,"demo.html"),replace(template,"__EXPERIMENT_DATA__"=>data))
    open(joinpath(OUTPUT,"summary.txt"),"w") do io
        println(io,"Arctic XS versus current static Model2Vec — matched historical Bluesky posts")
        for cohort in ["natural","topics"],method in ["kmeans","tree"],model in ["current","arctic"]
            runs=filter(r->r.cohort==cohort&&r.method==method&&r.model==model,result["runs"])
            @printf(io,"%s / %s / %s: silhouette %.4f; exact-neighbor recall@10 %.3f; candidate fraction %.3f; fit %.3fs\n",
                cohort,method,model,mean(r.silhouette for r in runs),mean(r.recall10 for r in runs),mean(r.candidate_fraction for r in runs),mean(r.fit_seconds for r in runs))
            if cohort=="topics"
                @printf(io,"  held-out topic ARI %.3f; NMI %.3f; balanced accuracy %.3f; same-topic P@10 direct %.3f, within partition %.3f\n",
                    mean(r.ari for r in runs),mean(r.nmi for r in runs),mean(r.balanced_accuracy for r in runs),mean(r.exact_topic_precision10 for r in runs),mean(r.local_topic_precision10 for r in runs))
            end
        end
    end
    println("Report written to ",OUTPUT)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_experiment()
end
