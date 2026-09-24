module MondrianPolyaToy

using Random
export Tree, ReferenceModel, observe!, remove!, rebuild!, density, cells, mass, integral, Observation

const Box = NTuple{4,Float64} # xmin, xmax, ymin, ymax
area(b::Box) = (b[2]-b[1])*(b[4]-b[3])
inside(b::Box, x) = b[1] <= x[1] <= b[2] && b[3] <= x[2] <= b[4]

mutable struct Node
    box::Box
    axis::Int
    cut::Float64
    left::Int
    right::Int
    evidence::Float64
    stamp::Float64
end
Node(b::Box, t) = Node(b, 0, 0., 0, 0, 0., t)

mutable struct Tree
    nodes::Vector{Node}
    beta::Float64
    kappa::Float64
    time::Float64
end

"""Complete unit-square Mondrian partition, stopped by lifetime and a hard node cap.
Cut times are geometric refinement times, independent of the stream clock.
"""
function Tree(; seed=7, half_life=12., kappa=2., lifetime=12., max_nodes=127, now=0.)
    (half_life > 0 && !isnan(half_life)) || throw(ArgumentError("half_life must be positive (Inf disables decay)"))
    (isfinite(kappa) && kappa > 0) || throw(ArgumentError("kappa must be positive and finite"))
    (isfinite(lifetime) && lifetime >= 0) || throw(ArgumentError("invalid lifetime"))
    (max_nodes >= 1 && isfinite(now)) || throw(ArgumentError("invalid node budget or clock"))
    rng = MersenneTwister(seed)
    nodes = [Node((0.,1.,0.,1.), Float64(now))]
    # Schedule all next cuts and install in geometric-time order; a cap truncates
    # the process globally, without favoring a depth-first traversal's first branch.
    pending = [(randexp(rng)/2, 1)]
    while !isempty(pending) && length(nodes)+2 <= max_nodes
        sort!(pending; by=first, rev=true)
        tau, i = pop!(pending)
        tau > lifetime && break
        n = nodes[i]; b = n.box
        dx, dy = b[2]-b[1], b[4]-b[3]
        axis = rand(rng)*(dx+dy) < dx ? 1 : 2
        lo, hi = axis == 1 ? (b[1],b[2]) : (b[3],b[4])
        cut = lo + rand(rng)*(hi-lo)
        lo < cut < hi || continue
        bl = axis == 1 ? (b[1],cut,b[3],b[4]) : (b[1],b[2],b[3],cut)
        br = axis == 1 ? (cut,b[2],b[3],b[4]) : (b[1],b[2],cut,b[4])
        n.axis = axis; n.cut = cut
        n.left = length(nodes)+1; n.right = length(nodes)+2
        push!(nodes, Node(bl, Float64(now)), Node(br, Float64(now)))
        for child in (n.left,n.right)
            cb = nodes[child].box
            rate = cb[2]-cb[1]+cb[4]-cb[3]
            push!(pending, (tau+randexp(rng)/rate, child))
        end
    end
    Tree(nodes, log(2)/half_life, Float64(kappa), Float64(now))
end

function checktime(t::Tree, now)
    (isfinite(now) && now >= t.time) || throw(ArgumentError("time must be finite and >= last mutation"))
end
mass(n::Node, t::Tree, now) = n.evidence * exp(-t.beta*(now-n.stamp))
function mass(t::Tree, now=t.time)
    checktime(t, now)
    mass(t.nodes[1], t, now)
end

function path(t::Tree, x)
    result = Int[]; i = 1
    while true
        push!(result, i)
        n = t.nodes[i]
        n.axis == 0 && return result
        i = x[n.axis] < n.cut ? n.left : n.right
    end
end

function validate(t::Tree, x, event_time, now, weight)
    checktime(t, now)
    (length(x) == 2 && all(isfinite, x) && inside(t.nodes[1].box,x)) || throw(ArgumentError("point must lie in [0,1]²"))
    (isfinite(event_time) && event_time <= now) || throw(ArgumentError("future/invalid event timestamp"))
    (isfinite(weight) && weight >= 0) || throw(ArgumentError("invalid observation weight"))
end

function change!(t::Tree, x, event_time, now, weight, sign)
    validate(t, x, event_time, now, weight)
    w = weight*exp(-t.beta*(now-event_time))
    ids = path(t,x)
    # Validate the complete subtraction before mutating. Identity ownership is
    # enforced by ReferenceModel; the raw tree only knows additive statistics.
    if sign < 0
        all(mass(t.nodes[i],t,now) + 1e-10 >= w for i in ids) || throw(ArgumentError("insufficient evidence"))
    end
    for i in ids
        n = t.nodes[i]
        n.evidence = max(0., mass(n,t,now) + sign*w)
        n.stamp = now
    end
    t.time = now
    t
end
observe!(t::Tree, x; event_time=t.time, now=event_time, weight=1.) = change!(t,x,event_time,now,weight,1)

"""Subtract a known contribution; caller must own that contribution. Prefer ReferenceModel for ID-safe removal."""
remove!(t::Tree, x; event_time, now=t.time, weight=1.) = change!(t,x,event_time,now,weight,-1)

function leftprob(t::Tree, n::Node, now)
    q = area(t.nodes[n.left].box)/area(n.box)
    l = mass(t.nodes[n.left], t, now)
    r = mass(t.nodes[n.right], t, now)
    (t.kappa*q+l)/(t.kappa+l+r)
end

"""Piecewise uniform normalized density. Queries do not mutate lazy statistics."""
function density(t::Tree, x, now=t.time)
    checktime(t, now)
    (length(x)==2 && all(isfinite,x)) || throw(ArgumentError("expected finite 2D point"))
    inside(t.nodes[1].box,x) || return 0.
    p = 1.; i = 1
    while t.nodes[i].axis != 0
        n = t.nodes[i]; q = leftprob(t,n,now)
        left = x[n.axis] < n.cut
        p *= left ? q : 1-q
        i = left ? n.left : n.right
    end
    p/area(t.nodes[i].box)
end

function cells(t::Tree, now=t.time)
    checktime(t, now)
    result = NamedTuple{(:box,:probability,:density,:evidence,:depth),Tuple{Box,Float64,Float64,Float64,Int}}[]
    function visit(i,p,depth)
        n = t.nodes[i]
        if n.axis == 0
            push!(result,(box=n.box,probability=p,density=p/area(n.box),evidence=mass(n,t,now),depth=depth))
        else
            q = leftprob(t,n,now)
            visit(n.left,p*q,depth+1); visit(n.right,p*(1-q),depth+1)
        end
    end
    visit(1,1.,0)
    result
end
integral(t::Tree, now=t.time) = sum(c.probability for c in cells(t,now))

struct Observation
    id::Int
    x::NTuple{2,Float64}
    event_time::Float64
    weight::Float64
end

mutable struct ReferenceModel
    tree::Tree
    records::Vector{Observation}
    capacity::Int
    seed::Int
    half_life::Float64
    kappa::Float64
    lifetime::Float64
    max_nodes::Int
    min_mass::Float64
    generation::Int
end

"""FIFO by ingestion order, with exact decayed deletion and replay of retained points.
IDs are unique only within the retained cache, not a durable deduplication ledger.
"""
function ReferenceModel(; capacity=600, seed=7, half_life=12., kappa=2., lifetime=12., max_nodes=127, min_mass=8.)
    capacity >= 1 || throw(ArgumentError("capacity must be positive"))
    (isfinite(min_mass) && min_mass >= 0) || throw(ArgumentError("invalid split mass"))
    t = Tree(;seed,half_life,kappa,lifetime,max_nodes)
    ReferenceModel(t,Observation[],capacity,seed,half_life,kappa,lifetime,max_nodes,min_mass,0)
end

function observe!(m::ReferenceModel, id::Int, x; event_time=m.tree.time, now=event_time, weight=1.)
    validate(m.tree,x,event_time,now,weight)
    any(r.id==id for r in m.records) && throw(ArgumentError("duplicate retained ID"))
    r = Observation(id,(Float64(x[1]),Float64(x[2])),event_time,weight)
    if length(m.records) == m.capacity
        old = first(m.records)
        remove!(m.tree,old.x;event_time=old.event_time,now,weight=old.weight)
        popfirst!(m.records)
    end
    observe!(m.tree,r.x;event_time,now,weight)
    push!(m.records,r)
    m
end

function remove!(m::ReferenceModel, id::Int; now=m.tree.time)
    checktime(m.tree,now)
    i = findfirst(r->r.id==id,m.records)
    isnothing(i) && throw(ArgumentError("ID not retained; exact deletion unsupported"))
    r = m.records[i]
    remove!(m.tree,r.x;event_time=r.event_time,now,weight=r.weight)
    deleteat!(m.records,i)
    m
end

"""Replay the cache into a seeded scaffold, then collapse low-evidence subtrees.
This is an engineering adaptation policy, not published streaming sMPT inference.
Fresh compact storage actually releases collapsed nodes.
"""
function rebuild!(m::ReferenceModel; now=m.tree.time)
    checktime(m.tree,now)
    t = Tree(;seed=m.seed,half_life=m.half_life,kappa=m.kappa,lifetime=m.lifetime,max_nodes=m.max_nodes,now)
    for r in m.records
        observe!(t,r.x;event_time=r.event_time,now,weight=r.weight)
    end
    compact = Node[]
    function copy_node(i)
        n = deepcopy(t.nodes[i]); push!(compact,n); j = length(compact)
        if n.axis != 0 && mass(n,t,now) >= m.min_mass
            n.left = copy_node(n.left); n.right = copy_node(n.right)
        else
            n.axis=0; n.left=0; n.right=0; n.cut=0.
        end
        j
    end
    copy_node(1)
    t.nodes = compact
    m.tree = t; m.generation += 1
    m
end

include("OnlinePPCA.jl")
include("Compression.jl")
include("EmbeddingTree.jl")
include("Localization.jl")

end
