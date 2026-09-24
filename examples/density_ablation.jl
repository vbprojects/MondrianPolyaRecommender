using LinearAlgebra, Random, Statistics, UUIDs, Printf, SHA
include("../src/LeafLatentTree.jl")
include("../src/StreamingCodecs.jl")
include("../src/DensityAblation.jl")
include("export_utils.jl")
const LT=LeafLatentTrees
const SC=StreamingCodecs
const DA=DensityAblation
const METHODS=[("Uniform leaf",:uniform),("Diagonal Gaussian",:diagonal),("PPCA rank 4",:ppca),("Actual posts; no Gaussian",:posts)]
const HYBRIDS=[("Actual posts + diagonal density",:posts,:diagonal),("Actual posts + PPCA density",:posts,:ppca)]
const WORLDS=["Euclidean: proposals only","Fermat: proposals only","Fermat: density + proposals"]

function dataset(name,seed)
    rng=MersenneTwister(seed+612)
    if name=="Bluesky 100,000"
        X=open(joinpath(@__DIR__,"../data/bluesky_100000/embeddings.f32")) do io
            reshape(read!(io,Vector{Float32}(undef,25600000)),256,100000)
        end
        return X,80000
    end
    d=name=="Correlated 32D topics" ? 32 : 2;N=12000;X=zeros(Float32,d,N)
    Q=Matrix(qr(randn(rng,d,d)).Q);centers=[[-.45,-.35],[.4,.4],[-.3,.5]]
    for j in 1:N
        topic=rand(rng)<.45 ? 1 : rand(rng)<.64 ? 2 : 3
        if d==2
            angle=name=="Axis-aligned 2D topics" ? 0. : [pi/4,-pi/5,pi/3][topic]
            A=[cos(angle) -sin(angle);sin(angle) cos(angle)]
            x=centers[topic]+A*([.14,.035].*randn(rng,2))
            while maximum(abs,x)>=.99;x=centers[topic]+A*([.14,.035].*randn(rng,2));end
        else
            U=Q[:,(topic-1)*4+1:topic*4]
            center=.45Q[:,12+topic]
            x=center+U*([.12,.10,.065,.03].*randn(rng,4))+.007randn(rng,d)
            @assert maximum(abs,x)<1
        end
        X[:,j]=x
    end
    X,10000
end
function projection(X)
    size(X,1)==2 && return Matrix{Float64}(I,2,2)
    R=Float64.(X).-mean(X;dims=2)
    svd(R;full=false).U[:,1:2]'
end
function fit_states(nodes,leaves,X,assignments,mode)
    d=size(X,1);states=Dict{Int,LT.Sketch}()
    for leaf in leaves;states[leaf]=LT.Sketch(mode==:ppca ? :ppca : :diagonal,mode==:ppca ? min(4,d) : 0,d);end
    elapsed=@elapsed for j in axes(X,2);LT.update!(states[assignments[j]],view(X,:,j));end
    states,elapsed
end
function build_case(name,seed;bank_size=64,nh=64,nlandmarks=448)
    X,ntrain=dataset(name,seed);d=size(X,1);rng=MersenneTwister(950+seed)
    order=randperm(rng,size(X,2));trainrows=order[1:ntrain];holdrows=order[ntrain+1:end]
    train=X[:,trainrows];held=X[:,holdrows]
    nreference=min(4096,size(held,2)÷2);reference=held[:,1:nreference]
    pool=held[:,nreference+1:min(nreference+512,size(held,2))]
    costs,_=DA.reference_cost(pool,reference);eligible=findall(<=(quantile(costs,.7)),costs)
    hids=Int[]
    for i in shuffle(rng,eligible)
        all(norm(pool[:,i]-pool[:,j])>1e-5 for j in hids) && push!(hids,i)
        length(hids)==nh && break
    end
    length(hids)==nh || error("insufficient distinct dense-region hypotheses")
    H=pool[:,hids]
    # Geometry uses only a diagonal sketch to propose the common scaffold.
    # Each tested state is then trained online in the same final leaves.
    tree_time=@elapsed fitted=LT.fit_leaf_tree(train;configs=[(:diagonal,0)],max_leaves=d==256 ? 32 : 16,
        split_after=d==256 ? 256 : 96,reservoir_cap=64,seed=seed+310)
    nodes=fitted.nodes;leaves=findall(n->n.axis==0,nodes)
    assignments=[LT.leaf_index(nodes,x) for x in eachcol(train)]
    counts=zeros(length(nodes));uniform_time=@elapsed for leaf in assignments;counts[leaf]+=1;end
    probs=DA.polyaprobs(nodes,counts);@assert abs(sum(probs)-1)<1e-12
    dictionaries=Dict{Symbol,Any}(:uniform=>nothing);statebytes=Dict(:uniform=>sizeof(counts));fittimes=Dict(:uniform=>uniform_time)
    prep=Dict{Symbol,Float64}(:uniform=>0.);accepts=Dict{Symbol,Float64}(:uniform=>1.)
    for mode in (:diagonal,:ppca)
        states,elapsed=fit_states(nodes,leaves,train,assignments,mode);fittimes[mode]=elapsed
        tm=@elapsed models=Dict(leaf=>DA.gaussian_leaf(states[leaf],nodes[leaf],MersenneTwister(91000+seed+leaf)) for leaf in leaves)
        dictionaries[mode]=models;prep[mode]=tm;accepts[mode]=minimum(p.acceptance for p in values(models))
        statebytes[mode]=Base.summarysize((states,models,counts))
    end
    # A single shared centroid+8-bit store. Chunks bound decoder fitting buffers.
    store=SC.Store("Shared centroid + 8-bit",:c8,d)
    codec_seconds=@elapsed for leaf in leaves
        ids=findall(==(leaf),assignments)
        for start in 1:256:length(ids)
            selected=ids[start:min(start+255,length(ids))]
            SC.ingest!(store,UUID.(UInt128.(selected)),UInt64(leaf),1,train[:,selected],nothing)
        end
    end
    Y=zeros(Float32,size(train))
    for block in store.blocks;Y[:,[Int(id.value) for id in block.ids]]=SC.decode(block);end
    digest=bytes2hex(sha256(reinterpret(UInt8,vec(Y))));st=SC.storage(store)
    @assert st.posts==ntrain && all(isfinite,Y)
    banks=Dict{Symbol,Dict{Int,Vector{Int}}}();diagnostics=NamedTuple[]
    for (label,mode) in METHODS
        localbanks=Dict{Int,Vector{Int}}();attempts=0;gaps=Float64[];seconds=0.
        for leaf in leaves
            corpus=findall(==(leaf),assignments);gen=MersenneTwister(seed*10000+leaf)
            if mode==:posts
                localbanks[leaf]=rand(gen,corpus,bank_size)
            else
                elapsed=@elapsed begin
                    p=mode==:uniform ? nothing : dictionaries[mode][leaf]
                    proposals,draws=DA.draw_box(gen,nodes[leaf],p,bank_size);attempts+=draws
                    distances=DA.distance_matrix(proposals,Y[:,corpus]);nearest=[argmin(view(distances,i,:)) for i in 1:bank_size]
                    localbanks[leaf]=corpus[nearest]
                    append!(gaps,[distances[i,nearest[i]] for i in 1:bank_size])
                end
                seconds+=elapsed
            end
            @assert all(assignments[id]==leaf for id in localbanks[leaf])
        end
        banks[mode]=localbanks
        densitymode=mode==:posts ? :uniform : mode
        logscore=DA.density_logs(nodes,probs,dictionaries[densitymode],held[:,end-255:end])
        push!(diagnostics,(case=name,seed=seed,method=label,dimensions=d,training_posts=ntrain,leaves=length(leaves),
            density_bytes=statebytes[densitymode],density_fit_seconds=fittimes[densitymode],density_prepare_seconds=prep[densitymode],
            shared_tree_seconds=tree_time,shared_tree_bytes=Base.summarysize([(n.lo,n.hi,n.axis,n.cut,n.left,n.right) for n in nodes]),
            shared_reservoir_bytes=Base.summarysize([nodes[i].reservoir for i in leaves]),
            codec_bytes=st.packed_bytes,codec_runtime_bytes=st.runtime_bytes,codec_seconds=codec_seconds,codec_sha256=digest,
            log_density_per_dimension=mean(logscore)/d,min_truncation_acceptance=accepts[densitymode],
            requested_proposals=length(leaves)*bank_size,proposal_draws=attempts,proposal_seconds=seconds,
            mean_snap_distance=isempty(gaps) ? 0. : mean(gaps),mean_unique_bank=mean(length(unique(ids)) for ids in values(localbanks))))
    end
    postdiag=only(r for r in diagnostics if r.method=="Actual posts; no Gaussian")
    for (label,_,densitymode) in HYBRIDS
        source=only(r for r in diagnostics if r.method==(densitymode==:diagonal ? "Diagonal Gaussian" : "PPCA rank 4"))
        push!(diagnostics,merge(source,(method=label,proposal_draws=0,proposal_seconds=0.,mean_snap_distance=0.,mean_unique_bank=postdiag.mean_unique_bank)))
    end
    unionrows=sort!(unique(vcat([ids for b in values(banks) for ids in values(b)]...)))
    index=Dict(id=>i for (i,id) in enumerate(unionrows))
    for bank in values(banks),leaf in keys(bank);bank[leaf]=[index[i] for i in bank[leaf]];end
    qoriginal=train[:,unionrows];qdecoded=Y[:,unionrows]
    lm=randperm(rng,ntrain)[1:nlandmarks];truevertices=hcat(H,train[:,lm]);vertices=hcat(H,Y[:,lm])
    D=DA.distance_matrix(vertices,vertices);edges=DA.knn_edges(D)
    trcost,refscale=DA.reference_cost(truevertices,reference)
    qtruecost,_=DA.reference_cost(qoriginal,reference;center=refscale)
    # Shared connectivity and attachment choices; only edge weights change.
    qdist=DA.distance_matrix(qdecoded,vertices)
    anchors=[partialsortperm(collect(view(qdist,i,:)),1:3) for i in axes(qdist,1)]
    tgeom=DA.graph_geometry(truevertices,trcost,edges,nh)
    truefermat=DA.attach_distances(qoriginal,truevertices,qtruecost,tgeom,anchors)
    trueHH=tgeom.paths[1:nh,:]/tgeom.scale
    # Oracle density control still uses compressed post positions for inference.
    estimatedcost,_=DA.reference_cost(vertices,reference;center=refscale)
    qestimatedcost,_=DA.reference_cost(qdecoded,reference;center=refscale)
    sharedgeom=DA.graph_geometry(vertices,estimatedcost,edges,nh)
    sharedfermat=DA.attach_distances(qdecoded,vertices,qestimatedcost,sharedgeom,anchors)
    geoms=Dict{Symbol,Any}();estimated=Dict{Symbol,Matrix{Float64}}();metrics=NamedTuple[]
    for mode in (:uniform,:diagonal,:ppca)
        elapsed=@elapsed begin
            logs=DA.density_logs(nodes,probs,dictionaries[mode],vertices)
            cost,center=DA.density_cost(logs,d)
            qc,_=DA.density_cost(DA.density_logs(nodes,probs,dictionaries[mode],qdecoded),d;center)
            geom=DA.graph_geometry(vertices,cost,edges,nh);geoms[mode]=geom
            estimated[mode]=DA.attach_distances(qdecoded,vertices,qc,geom,anchors)
        end
        actualL=DA.likelihood(truefermat);predictedL=DA.likelihood(estimated[mode])
        push!(metrics,(case=name,seed=seed,method=string(mode),graph_seconds=elapsed,
            cost_correlation=std(cost)>1e-12 ? cor(cost,trcost) : 0.,
            response_probability_rmse=sqrt(mean(abs2,predictedL-actualL)),
            graph_vertices=size(vertices,2),graph_edges=length(edges),candidate_union=length(unionrows)))
    end
    euHH=DA.distance_matrix(H,H);escale=median([euHH[i,j] for i in 1:nh for j in 1:i-1]);euHH./=escale
    euTrue=DA.likelihood(DA.distance_matrix(qoriginal,H)/escale)
    euEstimated=DA.likelihood(DA.distance_matrix(qdecoded,H)/escale)
    states,hleaves=DA.branch_states(nodes,H);project=projection(hcat(H,train[:,lm]))
    background=train[:,randperm(rng,ntrain)[1:800]]
    visual=(case=name,seed=seed,hypotheses=[collect(project*H[:,j]) for j in 1:nh],
        posts=[collect(project*qoriginal[:,j]) for j in axes(qoriginal,2)],
        background=[collect(project*background[:,j]) for j in axes(background,2)],
        post_rows=trainrows[unionrows],dimensions=d,
        cells=d==2 ? [(lo=n.lo,hi=n.hi) for n in nodes if n.axis==0] : [])
    (nodes=nodes,states=states,hleaves=hleaves,banks=banks,diagnostics=diagnostics,metrics=metrics,
        euTrue=euTrue,euEstimated=euEstimated,euHH=euHH,fermatTrue=DA.likelihood(truefermat),
        fermatShared=DA.likelihood(sharedfermat),trueHH=trueHH,sharedHH=sharedgeom.paths[1:nh,:]/sharedgeom.scale,
        estimated=Dict(k=>DA.likelihood(v) for (k,v) in estimated),
        HH=Dict(k=>v.paths[1:nh,:]/v.scale for (k,v) in geoms),visual=visual)
end
function run_case(name,seed;trials=100,budget=80,radii=(.05,.10))
    println("Preparing ",name," seed ",seed);flush(stdout)
    buildseconds=@elapsed land=build_case(name,seed)
    rows=NamedTuple[];traces=NamedTuple[];timings=NamedTuple[]
    h=size(land.euTrue,2);rng=MersenneTwister(30000+seed)
    targets=rand(rng,1:h,trials);uniforms=rand(rng,budget,trials)
    fallback=unique(vcat(collect(values(land.banks[:posts]))...))
    for radius in radii,(worldid,world) in enumerate(WORLDS)
      methods=[(label,mode,mode==:posts ? :uniform : mode) for (label,mode) in METHODS]
      worldid==3 && append!(methods,HYBRIDS)
      for (method,mode,densitymode) in methods
        if worldid==1
            L=land.euEstimated;Ltruth=land.euTrue;HH=land.euHH;truthHH=land.euHH
        elseif worldid==2
            L=land.fermatShared;Ltruth=land.fermatTrue;HH=land.sharedHH;truthHH=land.trueHH
        else
            L=land.estimated[densitymode];Ltruth=land.fermatTrue;HH=land.HH[densitymode];truthHH=land.trueHH
        end
        elapsed=@elapsed for t in 1:trials
            save=seed==1 && t==1
            r=DA.trial(land.nodes,land.states,land.hleaves,land.banks[mode],L,Ltruth,HH,truthHH,targets[t],uniforms[:,t];
                seed=seed*1000000+t,budget,history=save,fallback,radius)
            row=merge((case=name,seed=seed,world=world,method=method,trial=t,radius=radius),Base.structdiff(r,(history=r.history,)))
            push!(rows,row)
            save && push!(traces,(case=name,world=world,method=method,radius=radius,target=r.target,stop=r.stop,correct=r.correct,history=r.history))
        end
        subset=rows[end-trials+1:end]
        println("  r=",radius," ",world," / ",method,": ",count(r->r.correct,subset),"/",trials," correct; ",count(r->r.false_stop,subset)," false; ",round(elapsed;digits=2)," s");flush(stdout)
        push!(timings,(case=name,seed=seed,world=world,method=method,radius=radius,seconds=elapsed))
      end
    end
    rows,traces,land.diagnostics,land.metrics,land.visual,timings,buildseconds
end
function main()
    BLAS.set_num_threads(1);out=normpath(joinpath(@__DIR__,"../output/density_ablation"));mkpath(out)
    smoke="--smoke" in ARGS
    println("Warmup on an independent disposable landscape.")
    run_case("Axis-aligned 2D topics",99;trials=2,budget=3,radii=(.05,))
    cases=smoke ? ["Correlated 2D topics"] : ["Axis-aligned 2D topics","Correlated 2D topics","Correlated 32D topics","Bluesky 100,000"]
    rows=NamedTuple[];traces=NamedTuple[];diagnostics=NamedTuple[];metrics=NamedTuple[];visuals=NamedTuple[];timings=NamedTuple[];builds=NamedTuple[]
    elapsed=@elapsed for name in cases,seed in 1:(smoke ? 1 : 3)
        a,b,c,d,e,f,g=run_case(name,seed;trials=smoke ? 8 : 100,budget=smoke ? 20 : 80)
        append!(rows,a);append!(traces,b);append!(diagnostics,c);append!(metrics,d);seed==1 && push!(visuals,e);append!(timings,f)
        push!(builds,(case=name,seed=seed,seconds=g))
        write(joinpath(out,smoke ? "smoke.json" : "results.json"),sprint(io->json(io,(trials=rows,traces=traces,diagnostics=diagnostics,geometry=metrics,visuals=visuals,timings=timings,builds=builds))))
    end
    config=(trials_per_seed=smoke ? 8 : 100,seeds=smoke ? 1 : 3,budget=smoke ? 20 : 80,hypotheses=64,
        graph_vertices=512,proposal_bank=64,proposals_per_response=32,confidence=.9,radii=[.05,.10],
        density="Common static scaffold and exact empirical Pólya masses. Each model is refitted online in final leaves; split-history mass transfer and decay are not tested here.",
        codec="One centroid+8-bit store is shared by all methods, with groups of at most 256 posts within each final leaf. No method-specific post compression.",
        reference="Response simulator uses original vectors and an independent held-out kNN density proxy. Hypotheses are 64 distinct dense-region held-out post positions; uniform prior over these hypotheses.",
        geometry="12-NN plus MST on common compressed landmark topology; 3-anchor query attachments; density multiplier p^(-1/d), median-normalized and clamped [0.25,4]. Each graph has median hypothesis distance 1.",
        proposals="Four posterior-ambiguous splits, one hypothesis from each side, four bank draws per hypothesis. Deduplicate, remove seen posts, maximize IG. Shared actual-post fallback only if all drawn candidates were already seen.",
        controls="Complete Fermat setting additionally crosses the actual-post sampler with diagonal and PPCA density, isolating density geometry from Gaussian proposal generation.",
        feedback="P(like)=0.05+0.9/(1+(distance/0.25)^2). First posterior 90% radius ball stops inference; original reference radius determines correctness. Failed/false trials get budget+1 in capped-correct score.",
        limits="Finite candidate banks, finite hypotheses and synthetic likes. Static trees; not production feed quality or a dynamic no-PPCA split/merge implementation. MC truncation normalizers use 4096 draws per PPCA leaf.")
    payload=(trials=rows,traces=traces,diagnostics=diagnostics,geometry=metrics,visuals=visuals,timings=timings,builds=builds,configuration=config,seconds=elapsed)
    write(joinpath(out,smoke ? "smoke.json" : "results.json"),sprint(io->json(io,payload)))
    println("Completed ",length(rows)," localization trials in ",elapsed," seconds.")
end
if abspath(PROGRAM_FILE)==@__FILE__;main();end
