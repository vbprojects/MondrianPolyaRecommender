using MondrianPolyaToy, Random, Printf
const MP=MondrianPolyaToy
include("export_utils.jl")

const METHOD_NAMES=["Full-day replay","Spatial histogram","Microclusters","Ancestor residuals","Leaf-local MPPCA"]
const SOURCE_NAMES=["Mastodon","Nostr","Bluesky","Threads"]
phase(t)=t<8 ? "Morning topics" : t<12 ? "Topics drift" : t<14 ? "Cross-source burst + duplicates" : t<20 ? "New topic emerges" : t<24 ? "Quieter arrivals" : t<30 ? "Old hourly buckets expire" : "New day, returning topics"

function source_pick(rng,t)
    p=12<=t<14 ? (.12,.5,.25,.13) : (.22,.13,.40,.25)
    u=rand(rng); acc=0.
    for i in 1:4; acc+=p[i]; u<acc && return i; end
    4
end
function stream_point(rng,t,source)
    rand(rng)<.06 && return (rand(rng),rand(rng))
    # A synthetic duplicate storm makes the difference between volume and
    # independent evidence visible. No anti-spam correction is applied.
    if 12<=t<14 && source==2 && rand(rng)<.8
        return (.81,.23)
    end
    if t<8
        c=source<=2 ? (.23,.30) : (.70,.70)
    elseif t<12
        a=(t-8)/4; c=source<=2 ? (.23+.35a,.30+.25a) : (.70-.20a,.70-.25a)
    elseif t<20
        c=source==1 ? (.22,.72) : (.78,.25)
    elseif t<30
        c=source<=2 ? (.35,.75) : (.60,.62)
    else
        c=source<=2 ? (.25,.30) : (.70,.70)
    end
    while true
        x=(c[1]+.055randn(rng),c[2]+.055randn(rng))
        all(v->0<=v<=1,x) && return x
    end
end

function ellipses(m)
    m.kind in (:ppca,:mppca) || return []
    b=m.buckets[mod(floor(Int,m.time),m.window_buckets)+1]
    cs=m.kind==:ppca ? b.ppca : [c for i in sort!(collect(keys(b.local_ppca))) for c in b.local_ppca[i]]
    [(c.mx,c.my,2sqrt(MP.ppca_geometry(c).large),2sqrt(MP.ppca_geometry(c).noise),atan(MP.ppca_geometry(c).uy,MP.ppca_geometry(c).ux)) for c in cs]
end
function overlays(m)
    # Only current-hour grid/micro summaries are drawn, to keep the visual legible.
    b=m.buckets[mod(floor(Int,m.time),m.window_buckets)+1]
    if m.kind==:histogram
        g=m.grid_size
        return [(Float64(i-1)/g,i/g,Float64(j-1)/g,j/g) for j in 1:g for i in 1:g if b.grid[i,j]>0]
    elseif m.kind==:microclusters
        return [c.box for c in b.clusters]
    elseif m.kind==:residual
        a=MP.residual_masses(m); total=sum(a)
        return [m.scaffold.nodes[i].box for i in eachindex(a) if a[i]>max(1e-9,.01total)]
    end
    NTuple{4,Float64}[]
end

function main()
    seed=length(ARGS)>=1 ? parse(Int,ARGS[1]) : 42
    per_tick=length(ARGS)>=2 ? parse(Int,ARGS[2]) : 2000
    per_tick>0 || error("events per tick must be positive")
    out=length(ARGS)>=3 ? abspath(ARGS[3]) : joinpath(@__DIR__,"..","output","compression")
    mkpath(out)
    kinds=(:replay,:histogram,:microclusters,:residual,:mppca)
    # Warm up all backend paths before recording illustrative timing diagnostics.
    for kind in kinds
        w=CompressionModel(kind)
        for i in 1:80; ingest!(w,(i/100.,.3);now=.1,event_time=.1); end
        maintain!(w); compression_cells(w)
    end
    models=[CompressionModel(k) for k in kinds]
    rng=MersenneTwister(seed); audit=MersenneTwister(seed+177)
    frames=NamedTuple[]; total=0
    scores=[Float64[] for _ in models]; errors=[Float64[] for _ in models]
    ingest_seconds=zeros(length(models)); maintenance_seconds=zeros(length(models)); peaks=zeros(Int,length(models))
    maxnorm=0.; maxmass=0.
    rows=["hour,method,arrivals,total_ingested,represented_events,effective_mass,source_mastodon,source_nostr,source_bluesky,source_threads,model_bytes,summary_count,leaves,exact_L1_vs_replay,heldout_log_density,normalization_error,ingest_seconds,maintenance_seconds,maintenance_allocated_bytes"]
    for step in 0:144
        t=step/4
        n=step==0 ? 0 : round(Int,per_tick*(12<=t<14 ? 3. : 20<=t<24 ? .2 : 1.))
        batch=[(x=stream_point(rng,t,s),source=s) for s in (source_pick(rng,t) for _ in 1:n)]
        total+=n
        probes=[stream_point(audit,t,source_pick(audit,t)) for _ in 1:80]
        ingest=zeros(length(models)); maintenance=zeros(length(models)); alloc=zeros(Int,length(models))
        for (j,m) in enumerate(models)
            ingest[j]=@elapsed begin
                advance!(m,t)
                for r in batch; ingest!(m,r.x;source=r.source,event_time=t,now=t); end
            end
            timed=@timed maintain!(m;now=t)
            maintenance[j]=timed.time; alloc[j]=timed.bytes
        end
        reference=compression_cells(models[1]); states=NamedTuple[]
        for (j,m) in enumerate(models)
            cs=compression_cells(m); z=sum(c.probability for c in cs)
            l1=compression_l1(reference,cs)
            score=sum(log(compression_density(m,x)) for x in probes)/length(probes)
            bytes=Base.summarysize(m); peaks[j]=max(peaks[j],bytes)
            ingest_seconds[j]+=ingest[j]; maintenance_seconds[j]+=maintenance[j]
            if step>0; push!(scores[j],score); push!(errors[j],l1); end
            maxnorm=max(maxnorm,abs(z-1))
            maxmass=max(maxmass,abs(compressed_mass(m)-compressed_mass(models[1])))
            push!(states,(cells=[(round.(c.box;digits=7)...,round(c.density;digits=7)) for c in cs],
                bytes=bytes,summaries=summary_count(m),mass=compressed_mass(m),l1=l1,score=score,
                local_mixtures=m.kind==:mppca ? sum(length(b.local_ppca) for b in m.buckets) : 0,overlays=overlays(m),ellipses=ellipses(m),ingest_ms=ingest[j]*1000,maintenance_ms=maintenance[j]*1000))
            push!(rows,join((t,METHOD_NAMES[j],n,total,represented_count(m),compressed_mass(m),source_mass(m)...,
                bytes,summary_count(m),length(cs),l1,score,abs(z-1),ingest[j],maintenance[j],alloc[j]),','))
        end
        points=[(r.x...,r.source) for r in batch[1:min(120,length(batch))]]
        counts=reduce(+, (b.source_counts for b in models[1].buckets))
        push!(frames,(hour=t,phase=phase(t),arrivals=n,total=total,represented=represented_count(models[1]),
            sources=source_mass(models[1]),source_counts=counts,points=points,models=states))
        step%24==0 && println("Simulated ",t," h; admitted ",total," events; retained window ",represented_count(models[1]))
    end
    template=read(joinpath(@__DIR__,"compression_viewer.html"),String)
    data=sprint(io->json(io,frames))
    write(joinpath(out,"demo.html"),replace(template,"__FRAMES__"=>data,"__SEED__"=>string(seed),"__RATE__"=>string(per_tick)))
    write(joinpath(out,"metrics.csv"),join(rows,'\n')*"\n")
    lines=["Adaptive-tree compression experiment / seed $seed", "Synthetic 36-hour stream; $total total admitted events; $per_tick base events per 15-minute tick.",
        "Current hour + preceding 23 hourly buckets. Six-hour decay half-life. 127 candidate nodes.",
        "Each model represents ALL events in the same retained buckets. No 600-point cap.",
        "Maximum normalized-density integral error: $maxnorm", "Maximum cross-method effective-mass difference: $maxmass", "",
        "Method | peak reachable model MiB | mean exact L1 vs replay | mean held-out log density | ingest seconds | maintenance seconds"]
    for j in eachindex(models)
        push!(lines,@sprintf("%s | %.4f | %.4f | %.4f | %.4f | %.4f",METHOD_NAMES[j],peaks[j]/2.0^20,sum(errors[j])/length(errors[j]),sum(scores[j])/length(scores[j]),ingest_seconds[j],maintenance_seconds[j]))
    end
    append!(lines,["", "Histogram: 16×16 bins/hour (6144 maximum). Microclusters: at most 64/hour (1536 maximum).",
        "MPPCA: at most 8 rank-one components per owning tree node per hour (127*8*24 hard upper bound); prospective splits retain historical mixtures at ancestors, soft online assignments; covariance moments retained, not original vectors or per-post codes. Gaussian-to-tree allocation uses 64-node bounded quadrature. In 2D rank-one PPCA spans full covariance; this is not a high-dimensional codec benchmark.",
        "Residual: 127×24 evidence slots. Budgets are NOT byte-matched; compare measured memory and error together.",
        "L1 is integrated exactly over intersecting leaf rectangles, before display rounding. Residual is a different density model.",
        "Held-out probes are independent and evaluated after the tick's update, not prequential. One stream and one scaffold seed; no uncertainty interval.",
        "Timing excludes generation, audit queries, metrics and HTML export; includes backend ingestion or maintenance respectively. Not a production benchmark.",
        "Reachable object size includes scaffolds, active tree and temporal summaries; excludes text, IDs, encoder, archive and viewer history.",
        "maintenance_allocated_bytes is cumulative allocation during a call, NOT peak live memory or RSS."])
    write(joinpath(out,"summary.txt"),join(lines,'\n')*"\n")
    println(join(lines,'\n')); println("Open ",joinpath(out,"demo.html"))
end
main()
