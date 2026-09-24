# Static high-dimensional experiment. No time decay or embedding projection.
export EmbeddingPartition, EmbeddingCompression, fit_partition, fit_compression,
       embedding_probabilities, embedding_leaf, embedding_logdensity

mutable struct EmbeddingNode
    lo::Vector{Float64}
    hi::Vector{Float64}
    axis::Int
    cut::Float64
    left::Int
    right::Int
    count::Int
    depth::Int
    tau::Float64
end
struct EmbeddingPartition
    nodes::Vector{EmbeddingNode}
    dimensions::Int
    kappa::Float64
end
logvolume(n::EmbeddingNode) = sum(log(n.hi[j]-n.lo[j]) for j in eachindex(n.lo))

"""Mass-prioritized, data-adaptive Mondrian-style partition with Pólya counts.
Cut axes are sampled proportional to observed coordinate ranges; cuts are
uniform in that range. Exponential cut times use the sum of observed ranges.
Cells still partition the entire padded empirical root box. Occupancy stopping
and mass-priority scheduling make this a heuristic, not an exact Mondrian process.
Optional root_bounds=(lower,upper) fixes support without transforming inputs.
Training columns are observations. Temporary row memberships are discarded after fit.
"""
function fit_partition(X; seed=7, max_nodes=2047, min_count=128, max_depth=24, kappa=2., root_bounds=nothing)
    d,n=size(X)
    d>0 && n>0 && all(isfinite,X) || throw(ArgumentError("finite nonempty matrix required"))
    max_nodes>=1 && min_count>=2 && max_depth>=0 && isfinite(kappa) && kappa>0 || throw(ArgumentError("invalid budget"))
    if root_bounds === nothing
        lo=Float64.(vec(minimum(X;dims=2))); hi=Float64.(vec(maximum(X;dims=2)))
        for j in 1:d
            pad=max((hi[j]-lo[j])*1e-6,1e-7); lo[j]-=pad; hi[j]+=pad
        end
    else
        lo=Float64.(root_bounds[1]); hi=Float64.(root_bounds[2])
        length(lo)==d && length(hi)==d && all(isfinite,lo) && all(isfinite,hi) && all(lo .< hi) || throw(ArgumentError("invalid root bounds"))
        all(j -> minimum(view(X,j,:))>=lo[j] && maximum(view(X,j,:))<=hi[j],1:d) || throw(ArgumentError("training data outside root bounds"))
    end
    nodes=[EmbeddingNode(lo,hi,0,0.,0,0,n,0,0.)]
    pending=[(1,collect(1:n))]; rng=MersenneTwister(seed)
    while !isempty(pending) && length(nodes)+2<=max_nodes
        # Most populated cell first; stable node index breaks ties.
        k=argmax([length(p[2]) for p in pending]); i,ids=splice!(pending,k)
        node=nodes[i]
        length(ids)<min_count && continue
        node.depth>=max_depth && continue
        a=fill(Inf,d); b=fill(-Inf,d)
        for row in ids, j in 1:d
            v=Float64(X[j,row]); a[j]=min(a[j],v); b[j]=max(b[j],v)
        end
        widths=b-a; rate=sum(widths); rate>0 || continue
        tau=node.tau+randexp(rng)/rate
        u=rand(rng)*rate; axis=d
        for j in 1:d
            u-=widths[j]
            if u<0; axis=j; break; end
        end
        cut=a[axis]+rand(rng)*widths[axis]
        a[axis]<cut<b[axis] || continue
        left=Int[]; right=Int[]
        for row in ids; push!(X[axis,row]<cut ? left : right,row); end
        (isempty(left) || isempty(right)) && error("invalid cut")
        lh=copy(node.hi); lh[axis]=cut; rl=copy(node.lo); rl[axis]=cut
        node.axis=axis; node.cut=cut; node.left=length(nodes)+1; node.right=length(nodes)+2
        push!(nodes,EmbeddingNode(copy(node.lo),lh,0,0.,0,0,length(left),node.depth+1,tau))
        push!(nodes,EmbeddingNode(rl,copy(node.hi),0,0.,0,0,length(right),node.depth+1,tau))
        push!(pending,(node.left,left),(node.right,right))
    end
    EmbeddingPartition(nodes,d,Float64(kappa))
end

mutable struct EmbeddingMicrocluster
    count::Int
    mean::Vector{Float64}
    m2::Vector{Float64}  # Welford central sum of squares, diagonal only
end
function update_cluster!(c,x)
    c.count+=1
    for j in eachindex(c.mean)
        delta=x[j]-c.mean[j]; c.mean[j]+=delta/c.count
        c.m2[j]+=delta*(x[j]-c.mean[j])
    end
end
struct EmbeddingCompression
    partition::EmbeddingPartition
    kind::Symbol
    active::BitVector
    residual::Vector{Float64}
    clusters::Vector{Vector{EmbeddingMicrocluster}}
    counts::Vector{Float64}
end

function embedding_leaf(t::EmbeddingPartition,x,active=nothing)
    length(x)==t.dimensions && all(isfinite,x) || throw(ArgumentError("invalid query"))
    root=t.nodes[1]
    all(j -> root.lo[j] <= x[j] <= root.hi[j], eachindex(x)) || throw(ArgumentError("query outside root support"))
    i=1
    while t.nodes[i].axis!=0 && (active===nothing || active[i])
        node=t.nodes[i]; i=x[node.axis]<node.cut ? node.left : node.right
    end
    i
end

"""Activate a fixed scaffold prospectively in input order.
Residuals stay at ancestors after splits. Microclusters are leaf-local diagonal
moment summaries; on a split each whole cluster follows its centroid. Original
within-cluster child counts cannot be recovered. No raw post vectors are owned.
"""
function fit_compression(t::EmbeddingPartition,X,kind::Symbol; split_count=128, cluster_cap=4, radius=.2)
    kind in (:residual,:microclusters) || throw(ArgumentError("unknown method"))
    size(X,1)==t.dimensions && split_count>0 && cluster_cap>0 && isfinite(radius) && radius>0 || throw(ArgumentError("invalid configuration"))
    N=length(t.nodes); active=falses(N); residual=zeros(N)
    clusters=[EmbeddingMicrocluster[] for _ in 1:N]; counts=zeros(N)
    for row in axes(X,2)
        x=view(X,:,row); i=embedding_leaf(t,x,active)
        if kind==:residual
            residual[i]+=1
            localcount=residual[i]
        else
            cs=clusters[i]; best=0; dist=Inf
            for (k,c) in enumerate(cs)
                v=sum((x[j]-c.mean[j])^2 for j in eachindex(x))
                if v<dist; best=k; dist=v; end
            end
            if isempty(cs) || (dist>radius^2 && length(cs)<cluster_cap)
                push!(cs,EmbeddingMicrocluster(1,Float64.(x),zeros(t.dimensions)))
            else
                update_cluster!(cs[best],x)
            end
            localcount=sum(c.count for c in cs)
        end
        node=t.nodes[i]
        if node.axis!=0 && localcount>=split_count
            active[i]=true
            if kind==:microclusters
                for c in clusters[i]
                    child=c.mean[node.axis]<node.cut ? node.left : node.right
                    push!(clusters[child],c)
                end
                empty!(clusters[i])
            end
        end
    end
    for i in N:-1:1
        node=t.nodes[i]
        counts[i]=residual[i]+sum((c.count for c in clusters[i]);init=0)
        if active[i]; counts[i]+=counts[node.left]+counts[node.right]; end
    end
    EmbeddingCompression(t,kind,active,residual,clusters,counts)
end

"""Leaf probabilities. Residual model is a uniform-component mixture;
reference and microclusters use Beta posterior means at each active branch.
Only split-axis ratios are used; never exponentiate a 256D box volume.
"""
function embedding_probabilities(t::EmbeddingPartition,m::Union{Nothing,EmbeddingCompression}=nothing)
    p=zeros(length(t.nodes)); p[1]=1.
    counts=m===nothing ? [n.count for n in t.nodes] : m.counts
    for (i,n) in enumerate(t.nodes)
        if m!==nothing && m.kind==:residual
            i==1 && (p[1]=t.kappa/(t.kappa+counts[1]))
            p[i]+=m.residual[i]/(t.kappa+counts[1])
        end
        n.axis==0 && continue
        m!==nothing && !m.active[i] && continue
        q=(n.cut-n.lo[n.axis])/(n.hi[n.axis]-n.lo[n.axis])
        if m===nothing || m.kind!=:residual
            q=(t.kappa*q+counts[n.left])/(t.kappa+counts[n.left]+counts[n.right])
        end
        p[n.left]=p[i]*q; p[n.right]=p[i]*(1-q)
    end
    p
end
function embedding_logdensity(t::EmbeddingPartition,x,m::Union{Nothing,EmbeddingCompression}=nothing; probabilities=embedding_probabilities(t,m))
    i=embedding_leaf(t,x,m===nothing ? nothing : m.active)
    log(probabilities[i])-logvolume(t.nodes[i])
end
