using MondrianPolyaToy, Random, Serialization, Printf
include("load_bluesky.jl")
include("export_utils.jl")
function main()
    root=normpath(joinpath(@__DIR__,"..")); out=joinpath(root,"output","interactive")
    mkpath(out)
    saved=deserialize(joinpath(root,"output/bluesky_tree/models.jls"))
    t=saved.partition; m=saved.models["residual"]
    X=load_bluesky_embeddings(saved.input); rng=MersenneTwister(72)
    groups=Dict{Int,Vector{Int}}()
    for row in axes(X,2)
        leaf=embedding_leaf(t,view(X,:,row),m.active)
        push!(get!(groups,leaf,Int[]),row)
    end
    leaves=sort(collect(keys(groups))); K=length(leaves)
    reps=Int[]; candidates=Int[]
    for leaf in leaves
        ids=groups[leaf]
        center=zeros(size(X,1))
        for row in ids; center .+= view(X,:,row); end
        center ./= length(ids)
        scores=[sum((X[j,row]-center[j])^2 for j in eachindex(center)) for row in ids]
        rep=ids[argmin(scores)]; push!(reps,rep); push!(candidates,rep)
        others=filter(!=(rep),ids)
        append!(candidates,shuffle(rng,others)[1:min(3,length(others))])
    end
    # Avoid identical text appearing repeatedly, including across leaf boundaries.
    # Exact text deduplication is also enforced in the browser after parsing JSON.
    candidates=unique(candidates)
    chord(a,b)=sqrt(sum((Float64(X[j,a])-X[j,b])^2 for j in axes(X,1)))
    D=[chord(a,b) for a in reps,b in reps]
    p=embedding_probabilities(t,m)[leaves]; retained=sum(p); p ./= retained
    # Occupancy weighted kNN density proxy, not 256D rectangular volume density.
    k=min(12,K-1); k>0 || error("Need at least two occupied residual leaves")
    neighborhoods=[sortperm(view(D,i,:))[2:k+1] for i in 1:K]
    rho=[(p[i]+sum(p[neighborhoods[i]]))/max(D[i,neighborhoods[i][end]],1e-6)^2 for i in 1:K]
    rho ./= sort(rho)[cld(K,2)]; rho=clamp.(rho,.1,10.)
    cost(i,j)=D[i,j]/((rho[i]*rho[j])^.125)
    G=fill(Inf,K,K)
    for i in 1:K
        G[i,i]=0.
        for j in neighborhoods[i]; G[i,j]=G[j,i]=cost(i,j); end
    end
    # A Euclidean minimum spanning tree supplies explicit bridges if kNN disconnects.
    used=falses(K); best=fill(Inf,K); parent=zeros(Int,K); best[1]=0.
    for _ in 1:K
        i=argmin([used[j] ? Inf : best[j] for j in 1:K]); used[i]=true
        if parent[i]>0; j=parent[i]; G[i,j]=G[j,i]=cost(i,j); end
        for j in 1:K
            if !used[j] && D[i,j]<best[j]; best[j]=D[i,j]; parent[j]=i; end
        end
    end
    for h in 1:K, j in 1:K, i in 1:K
        G[i,j]=min(G[i,j],G[i,h]+G[h,j])
    end
    scale=sort(vec(G))[cld(length(G),2)]*.4; scale=max(scale,1e-5)
    likelihood=Vector{Vector{Float64}}()
    for row in candidates
        ds=[chord(row,rep) for rep in reps]
        anchors=sortperm(ds)[1:min(3,K)]
        distance=[minimum(ds[a]/rho[a]^.25+G[a,h] for a in anchors) for h in 1:K]
        push!(likelihood,[round(.05+.9/(1+(d/scale)^2);digits=6) for d in distance])
    end
    @assert all(isfinite,G) && isapprox(sum(p),1.)
    @assert all(v -> all(x -> .05<=x<=.95,v),likelihood)
    lines=readlines(joinpath(saved.input,"posts.jsonl"))
    payload=sprint() do io
        print(io,"{\"version\":\"residual-ig-v1\",\"prior\":")
        json(io,p); print(io,",\"leaves\":"); json(io,leaves)
        print(io,",\"likelihood\":"); json(io,likelihood)
        print(io,",\"posts\":[")
        for (j,row) in enumerate(candidates)
            j>1 && print(io,','); print(io,lines[row])
        end
        print(io,"]}")
    end
    template=read(joinpath(@__DIR__,"interactive_viewer.html"),String)
    write(joinpath(out,"demo.html"),replace(template,"__DATA__"=>replace(payload,"<"=>"\\u003c")))
    write(joinpath(out,"summary.txt"),"Residual information-gain interactive demo\nOccupied hypothesis leaves: $K\nCandidate posts before text deduplication: $(length(candidates))\nPrior mass covered by occupied leaves: $retained\nLikelihood distance scale: $scale\nGraph: symmetric 12-neighbor plus Euclidean MST; shortest paths weighted by a bounded occupancy/area density proxy. Exponent 0.25; area exponent 2 is a heuristic, not estimated intrinsic dimension.\nCandidates: representative nearest leaf mean plus up to 3 seeded random posts per occupied leaf.\nNo dimensionality reduction; no microclusters; static corpus and static single-interest hypothesis.\nLikelihood: 0.05+0.9/(1+(distance/scale)^2).\nPosterior updates tempered by exponent 0.5 and mixed with 1% prior after each rating to limit brittle lock-in.\nInformation gain uses the untempered binary likelihood; tempered updates make this a heuristic policy.\n")
    println("Built ",joinpath(out,"demo.html")," with ",K," hypotheses and ",length(candidates)," candidates")
end
main()
