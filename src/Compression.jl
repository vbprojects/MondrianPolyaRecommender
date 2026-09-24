# Experimental temporal compression backends. All use the same admitted stream,
# coordinate system, half-life and bucketed retention contract.
export CompressionModel, ingest!, maintain!, compressed_mass, source_mass,
       represented_count, summary_count, compression_cells, compression_density,
       compression_l1, advance!, micro_moments

mutable struct Microcluster
    mass::Float64
    sx::Float64
    sy::Float64
    sxx::Float64
    syy::Float64
    box::Box
end
Microcluster(x,w) = Microcluster(w,w*x[1],w*x[2],w*x[1]^2,w*x[2]^2,(x[1],x[1],x[2],x[2]))
function add!(c::Microcluster,x,w)
    c.mass+=w; c.sx+=w*x[1]; c.sy+=w*x[2]
    c.sxx+=w*x[1]^2; c.syy+=w*x[2]^2
    b=c.box; c.box=(min(b[1],x[1]),max(b[2],x[1]),min(b[3],x[2]),max(b[4],x[2]))
end

mutable struct TimeBucket
    epoch::Int
    count::Int
    sources::Vector{Float64}
    source_counts::Vector{Int}
    points::Vector{NTuple{3,Float64}} # x, y, weight at bucket origin
    grid::Matrix{Float64}
    local_ppca::Dict{Int,Vector{OnlinePPCAComponent}}
    ppca::Vector{OnlinePPCAComponent}
    clusters::Vector{Microcluster}
end
function TimeBucket(kind,g,nsources)
    TimeBucket(typemin(Int),0,zeros(nsources),zeros(Int,nsources),NTuple{3,Float64}[],zeros(kind==:histogram ? g : 0,kind==:histogram ? g : 0),Dict{Int,Vector{OnlinePPCAComponent}}(),OnlinePPCAComponent[],Microcluster[])
end

mutable struct CompressionModel
    kind::Symbol
    scaffold::Tree
    tree::Tree
    buckets::Vector{TimeBucket}
    bucket_hours::Float64
    window_buckets::Int
    grid_size::Int
    cluster_cap::Int
    radius::Float64
    split_mass::Float64
    merge_mass::Float64
    # Residual model owns each event at exactly one node and time bucket.
    residual::Matrix{Float64}
    split::BitVector
    time::Float64
    fitted_time::Float64
    dirty::Bool
end

"""Day-scale reference or bounded compressor.
Kinds: :replay, :histogram, :microclusters, :residual, :ppca, :mppca.
Retains the current bucket and window_buckets-1 preceding buckets; the oldest
bucket expires in its entirety. This is bucket-aligned retention, not exact
per-event sliding-window expiry. No IDs, text, or durable deduplication are stored.
"""
function CompressionModel(kind::Symbol; half_life=6., bucket_hours=1., window_buckets=24,
        grid_size=16, cluster_cap=64, radius=.045, split_mass=60., merge_mass=20.,
        seed=7, max_nodes=127, lifetime=12., nsources=4)
    kind in (:replay,:histogram,:microclusters,:residual,:ppca,:mppca) || throw(ArgumentError("unknown compression method"))
    (isfinite(bucket_hours) && bucket_hours>0 && window_buckets>0 && grid_size>0 && cluster_cap>0 && nsources>0) || throw(ArgumentError("invalid summary budget"))
    (isfinite(radius) && radius>0 && isfinite(split_mass) && isfinite(merge_mass) && 0<=merge_mass<split_mass) || throw(ArgumentError("invalid maintenance policy"))
    scaffold=Tree(;half_life,seed,max_nodes,lifetime)
    # Weights are stored relative to a nearby bucket origin; reject parameter
    # combinations that would make those weights overflow.
    scaffold.beta*bucket_hours < 600 || throw(ArgumentError("bucket too wide for half-life"))
    n=length(scaffold.nodes)
    model=CompressionModel(kind,scaffold,deepcopy(scaffold),[TimeBucket(kind,grid_size,nsources) for _ in 1:window_buckets],
        bucket_hours,window_buckets,grid_size,cluster_cap,radius,split_mass,merge_mass,
        zeros(kind==:residual ? n : 0,kind==:residual ? window_buckets : 0),falses(n),0.,0.,true)
    maintain!(model;now=0.)
end

epoch(m::CompressionModel,t)=floor(Int,t/m.bucket_hours)
function reset_bucket!(m,i)
    b=m.buckets[i]
    b.epoch=typemin(Int); b.count=0; fill!(b.sources,0.); fill!(b.source_counts,0)
    # Replace vectors so expired reference records release their storage.
    b.points=NTuple{3,Float64}[]; b.clusters=Microcluster[]; b.ppca=OnlinePPCAComponent[]; empty!(b.local_ppca); fill!(b.grid,0.)
    m.kind==:residual && (m.residual[:,i].=0.)
end

function advance!(m::CompressionModel,now)
    (isfinite(now) && now>=m.time) || throw(ArgumentError("clock must be monotone and finite"))
    cutoff=epoch(m,now)-m.window_buckets+1
    for (i,b) in enumerate(m.buckets)
        if b.epoch!=typemin(Int) && b.epoch<cutoff
            reset_bucket!(m,i)
        end
    end
    m.dirty |= now!=m.time
    m.time=now
    m
end

"""Ingest a point into one temporal summary. Call maintain! before density queries.
Late events are accepted only while their event-time bucket remains retained.
The caller is responsible for canonical IDs, deduplication and post storage.
"""
function ingest!(m::CompressionModel,x; source=1,event_time=m.time,now=event_time,weight=1.)
    (isfinite(now) && now>=m.time && isfinite(event_time) && 0<=event_time<=now) || throw(ArgumentError("invalid event/ingestion time"))
    (length(x)==2 && all(isfinite,x) && inside((0.,1.,0.,1.),x)) || throw(ArgumentError("point outside unit square"))
    (isfinite(weight) && weight>=0 && source isa Integer && 1<=source<=length(m.buckets[1].sources)) || throw(ArgumentError("invalid weight/source"))
    e=epoch(m,event_time)
    e>=epoch(m,now)-m.window_buckets+1 || throw(ArgumentError("event's time bucket has expired"))
    w=weight*exp(m.scaffold.beta*(event_time-e*m.bucket_hours))
    isfinite(w) || throw(ArgumentError("scaled weight overflow"))
    advance!(m,now)
    slot=mod(e,m.window_buckets)+1; b=m.buckets[slot]
    if b.epoch!=e
        reset_bucket!(m,slot); b.epoch=e
    end
    b.count+=1; b.sources[source]+=w; b.source_counts[source]+=1
    if m.kind==:replay
        push!(b.points,(x[1],x[2],w))
    elseif m.kind==:histogram
        i=min(m.grid_size,floor(Int,x[1]*m.grid_size)+1)
        j=min(m.grid_size,floor(Int,x[2]*m.grid_size)+1)
        b.grid[i,j]+=w
    elseif m.kind==:microclusters && w>0
        best=0; dmin=Inf
        for (i,c) in enumerate(b.clusters)
            d=(x[1]-c.sx/c.mass)^2+(x[2]-c.sy/c.mass)^2
            if d<dmin; best=i; dmin=d; end
        end
        if isempty(b.clusters) || (dmin>m.radius^2 && length(b.clusters)<m.cluster_cap)
            push!(b.clusters,Microcluster(x,w))
        else
            add!(b.clusters[best],x,w)
        end
    elseif m.kind==:mppca && w>0
        i=1
        while m.split[i]
            n=m.scaffold.nodes[i]; i=x[n.axis]<n.cut ? n.left : n.right
        end
        cs=get!(() -> OnlinePPCAComponent[],b.local_ppca,i)
        ppca_ingest!(cs,x,w,min(8,m.cluster_cap),m.radius)
    elseif m.kind==:ppca && w>0
        ppca_ingest!(b.ppca,x,w,min(8,m.cluster_cap),m.radius)
    elseif m.kind==:residual
        i=1
        while m.split[i]
            n=m.scaffold.nodes[i]; i=x[n.axis]<n.cut ? n.left : n.right
        end
        m.residual[i,slot]+=w
    end
    m.dirty=true
    m
end

bucket_factor(m,b)=b.epoch==typemin(Int) ? 0. : exp(-m.scaffold.beta*(m.time-b.epoch*m.bucket_hours))
function source_mass(m::CompressionModel)
    result=zeros(length(m.buckets[1].sources))
    for b in m.buckets
        result .+= b.sources .* bucket_factor(m,b)
    end
    result
end
compressed_mass(m::CompressionModel)=sum(source_mass(m))
represented_count(m::CompressionModel)=sum(b.count for b in m.buckets)
function summary_count(m::CompressionModel)
    m.kind==:replay && return sum(length(b.points) for b in m.buckets)
    m.kind==:histogram && return sum(count(>(0),b.grid) for b in m.buckets)
    m.kind==:microclusters && return sum(length(b.clusters) for b in m.buckets)
    m.kind==:mppca && return sum((length(cs) for b in m.buckets for cs in values(b.local_ppca));init=0)
    m.kind==:ppca && return sum(length(b.ppca) for b in m.buckets)
    count(>(0),m.residual)
end

# Replaying a uniform rectangle gives exact branch counts for that assumed
# component density. The assumption is approximate for discarded observations.
# Degenerate components route as points in the corresponding coordinate.
function distribute!(t::Tree,box::Box,w,i=1)
    w==0 && return
    n=t.nodes[i]; n.evidence+=w
    n.axis==0 && return
    k=n.axis==1 ? 1 : 3; lo=box[k]; hi=box[k+1]
    if hi<=n.cut && lo<n.cut
        distribute!(t,box,w,n.left)
    elseif lo>=n.cut
        distribute!(t,box,w,n.right)
    else
        q=(n.cut-lo)/(hi-lo)
        bl=ntuple(j->j==k+1 ? n.cut : box[j],4)
        br=ntuple(j->j==k ? n.cut : box[j],4)
        distribute!(t,bl,w*q,n.left); distribute!(t,br,w*(1-q),n.right)
    end
end

function fit_summaries(m)
    t=deepcopy(m.scaffold); t.time=m.time
    for n in t.nodes; n.evidence=0.; n.stamp=m.time; end
    for b in m.buckets
        factor=bucket_factor(m,b); factor==0 && continue
        if m.kind==:replay
            for (x,y,w) in b.points
                distribute!(t,(x,x,y,y),w*factor)
            end
        elseif m.kind==:histogram
            g=m.grid_size
            for j in 1:g,i in 1:g
                w=b.grid[i,j]; w==0 && continue
                distribute!(t,((i-1)/g,i/g,(j-1)/g,j/g),w*factor)
            end
        elseif m.kind==:mppca
            for i in sort!(collect(keys(b.local_ppca))), c in b.local_ppca[i]
                points=ppca_samples(c,m.scaffold.nodes[i].box)
                for (x,y) in points
                    distribute!(t,(x,x,y,y),c.mass*factor/length(points))
                end
            end
        elseif m.kind==:ppca
            for c in b.ppca
                points=ppca_samples(c)
                for (x,y) in points
                    distribute!(t,(x,x,y,y),c.mass*factor/length(points))
                end
            end
        else
            for c in b.clusters
                distribute!(t,c.box,c.mass*factor)
            end
        end
    end
    compact=Node[]
    function visit(i)
        n=deepcopy(t.nodes[i]); push!(compact,n); j=length(compact)
        if n.axis!=0 && n.evidence>=m.split_mass
            n.left=visit(n.left); n.right=visit(n.right)
        else
            n.axis=0; n.left=0; n.right=0; n.cut=0.
        end
        j
    end
    visit(1); t.nodes=compact
    t
end

function residual_masses(m)
    masses=zeros(size(m.residual,1))
    for (i,b) in enumerate(m.buckets)
        masses .+= view(m.residual,:,i).*bucket_factor(m,b)
    end
    masses
end

function adapt_residual!(m)
    localmass=residual_masses(m)
    function subtotal(i)
        n=m.scaffold.nodes[i]
        localmass[i]+(m.split[i] ? subtotal(n.left)+subtotal(n.right) : 0.)
    end
    function pool!(root,i)
        n=m.scaffold.nodes[i]
        if m.split[i]
            pool!(root,n.left); pool!(root,n.right)
        end
        m.residual[root,:].+=m.residual[i,:]; m.residual[i,:].=0.
        m.split[i]=false
    end
    function visit(i)
        n=m.scaffold.nodes[i]
        if m.split[i]
            if subtotal(i)<m.merge_mass
                pool!(i,n.left); pool!(i,n.right); m.split[i]=false
            else
                visit(n.left); visit(n.right)
            end
        elseif n.axis!=0 && localmass[i]>=m.split_mass
            # Keep historical evidence here; both children start empty. Grow at
            # most one level along any path per maintenance call.
            m.split[i]=true
        end
    end
    visit(1)
end

"""Expire temporal buckets and publish a density at `now`.
Replay/grid/microclusters refit and prune the scaffold. Residual maintenance
splits prospectively or pools weak subtrees without assigning old evidence to
new children. It is a mixture model, not Beta branch-count inference.
"""
function maintain!(m::CompressionModel;now=m.time)
    advance!(m,now)
    m.dirty || return m
    if m.kind==:residual
        adapt_residual!(m)
    else
        m.tree=fit_summaries(m)
    end
    if m.kind==:mppca
        # Historical mixtures remain at owners. New children start empty.
        localmass=zeros(length(m.scaffold.nodes))
        for b in m.buckets, (i,cs) in b.local_ppca
            localmass[i]+=bucket_factor(m,b)*sum(c.mass for c in cs)
        end
        function activate(i)
            n=m.scaffold.nodes[i]
            if m.split[i]
                activate(n.left); activate(n.right)
            elseif n.axis!=0 && localmass[i]>=m.split_mass
                m.split[i]=true
            end
        end
        activate(1)
    end
    m.fitted_time=now; m.dirty=false
    m
end

function compression_cells(m::CompressionModel)
    m.dirty && throw(ArgumentError("call maintain! before querying the current density"))
    m.kind!=:residual && return cells(m.tree,m.time)
    masses=residual_masses(m); total=sum(masses); prior=m.scaffold.kappa
    result=NamedTuple{(:box,:probability,:density,:evidence,:depth),Tuple{Box,Float64,Float64,Float64,Int}}[]
    function visit(i,inherited,depth)
        n=m.scaffold.nodes[i]
        v=inherited+masses[i]/area(n.box)
        if m.split[i]
            visit(n.left,v,depth+1); visit(n.right,v,depth+1)
        else
            d=(prior+v)/(prior+total)
            push!(result,(box=n.box,probability=d*area(n.box),density=d,evidence=masses[i],depth=depth))
        end
    end
    visit(1,0.,0)
    result
end
function compression_density(m::CompressionModel,x)
    (length(x)==2 && all(isfinite,x)) || throw(ArgumentError("invalid query point"))
    m.dirty && throw(ArgumentError("call maintain! before querying"))
    inside((0.,1.,0.,1.),x) || return 0.
    m.kind!=:residual && return density(m.tree,x,m.time)
    masses=residual_masses(m); i=1; v=0.
    while true
        n=m.scaffold.nodes[i]; v+=masses[i]/area(n.box)
        m.split[i] || break
        i=x[n.axis]<n.cut ? n.left : n.right
    end
    (m.scaffold.kappa+v)/(m.scaffold.kappa+sum(masses))
end

"""Exact L1 between two piecewise uniform, complete rectangular densities."""
function compression_l1(a,b)
    value=0.
    for x in a,y in b
        u=x.box; v=y.box
        overlap=max(0.,min(u[2],v[2])-max(u[1],v[1]))*max(0.,min(u[4],v[4])-max(u[3],v[3]))
        value+=overlap*abs(x.density-y.density)
    end
    value
end
function micro_moments(m::CompressionModel)
    m.kind==:microclusters || throw(ArgumentError("microcluster model required"))
    result=zeros(5)
    for b in m.buckets,c in b.clusters
        result .+= bucket_factor(m,b).*[c.mass,c.sx,c.sy,c.sxx,c.syy]
    end
    result
end
