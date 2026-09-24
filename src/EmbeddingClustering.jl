module EmbeddingClustering
using LinearAlgebra, Random, Statistics
export spherical_kmeans, assign_centers, adjusted_rand, label_metrics, silhouette, group_metrics, topic_metrics

function assign_centers(C, X)
    similarities = C' * X
    [argmax(view(similarities, :, i)) for i in axes(X, 2)]
end

"""Cosine k-means with k-means++ initialization on unit vectors; no labels."""
function spherical_kmeans(X, k; seed=17, iterations=200)
    d, n = size(X)
    1 <= k <= n || throw(ArgumentError("invalid cluster count"))
    all(isfinite, X) || throw(ArgumentError("nonfinite embeddings"))
    maximum(abs.(vec(sum(abs2, X; dims=1)).-1)) < 1e-3 || throw(ArgumentError("expected unit vectors"))
    rng = MersenneTwister(seed)
    C = zeros(eltype(X), d, k)
    chosen = Int[rand(rng, 1:n)]
    C[:, 1] = X[:, chosen[1]]
    distance = fill(Inf, n)
    for j in 2:k
        distance = min.(distance, max.(0., 1 .- vec(C[:, j-1]'*X)))
        distance[chosen] .= 0
        total = sum(distance)
        next = total > 1e-12 ? min(n, searchsortedfirst(cumsum(distance), rand(rng)*total)) :
            first(setdiff(1:n, chosen))
        # Zero-mass roundoff cannot select a previously chosen point.
        next in chosen && (next = first(setdiff(1:n, chosen)))
        push!(chosen, next)
        C[:, j] = X[:, next]
    end
    old = zeros(Int, n)
    completed = 0
    for it in 1:iterations
        a = assign_centers(C, X)
        completed = it
        a == old && break
        fill!(C, 0)
        counts = zeros(Int, k)
        for i in 1:n
            @views C[:, a[i]] .+= X[:, i]
            counts[a[i]] += 1
        end
        for j in 1:k
            v = norm(view(C, :, j))
            if counts[j] == 0 || v < 1e-12
                C[:, j] = X[:, rand(rng, 1:n)]
            else
                C[:, j] ./= v
            end
        end
        old = a
    end
    (centers=C, assignments=assign_centers(C, X), iterations=completed)
end

function contingency(a, b)
    length(a) == length(b) || throw(DimensionMismatch())
    ua, ub = sort(unique(a)), sort(unique(b))
    ia, ib = Dict(x=>i for (i,x) in enumerate(ua)), Dict(x=>i for (i,x) in enumerate(ub))
    table = zeros(Int, length(ua), length(ub))
    for (x,y) in zip(a,b); table[ia[x],ib[y]] += 1; end
    table
end
choose2(n) = n*(n-1)/2
function adjusted_rand(a, b)
    t = contingency(a,b); n = sum(t)
    n < 2 && return 1.
    cells = sum(choose2,t); rows = sum(choose2,sum(t;dims=2)); cols = sum(choose2,sum(t;dims=1))
    expected = rows*cols/choose2(n); denominator = (rows+cols)/2-expected
    abs(denominator) < 1e-12 ? 1. : (cells-expected)/denominator
end
entropy(counts) = -sum(p>0 ? p*log(p) : 0. for p in counts./sum(counts))
function label_metrics(groups, labels)
    t = contingency(groups,labels); n=sum(t); rows=vec(sum(t;dims=2));cols=vec(sum(t;dims=1))
    mutual = sum(t[i,j]>0 ? t[i,j]/n*log(t[i,j]*n/(rows[i]*cols[j])) : 0.
                 for i in axes(t,1), j in axes(t,2))
    hg,hl = entropy(rows),entropy(cols)
    (ari=adjusted_rand(groups,labels), nmi=hg+hl>0 ? 2mutual/(hg+hl) : 1.,
     purity=sum(maximum(t;dims=2))/n)
end

"""Exact cosine silhouette on the supplied evaluation subset, singleton = 0."""
function silhouette(X, groups)
    n=size(X,2); u=unique(groups)
    length(u)<2 && return 0.
    distances=clamp.(1 .- X'*X,0.,2.)
    members=Dict(g=>findall(==(g),groups) for g in u)
    values=zeros(n)
    for i in 1:n
        own=members[groups[i]]
        length(own)==1 && continue
        within=sum(distances[j,i] for j in own if j!=i)/(length(own)-1)
        other=minimum(mean(view(distances,ids,i)) for (g,ids) in members if g!=groups[i])
        values[i]=max(within,other)>0 ? (other-within)/max(within,other) : 0.
    end
    mean(values)
end

function group_metrics(groups)
    counts=[count(==(g),groups) for g in unique(groups)]
    (occupied_groups=length(counts), largest_fraction=maximum(counts)/sum(counts),
     effective_groups=exp(entropy(counts)), median_group_size=median(counts))
end

"""Predict labels from training group majorities; evaluate on unseen authors."""
function topic_metrics(train_groups, train_labels, test_groups, test_labels)
    topics=sort(unique(train_labels))
    counts=Dict(g=>[sum((train_groups .==g).&(train_labels .==l)) for l in topics] for g in unique(train_groups))
    global_counts=[count(==(l),train_labels) for l in topics]
    predictions=[topics[argmax(get(counts,g,global_counts))] for g in test_groups]
    balanced=mean(mean(predictions[test_labels.==l].==l) for l in unique(test_labels))
    merge(label_metrics(test_groups,test_labels),(balanced_accuracy=balanced,))
end
end
