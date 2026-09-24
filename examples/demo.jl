using MondrianPolyaToy, Random, Printf

# No plotting or JSON packages required: Julia computes all model states; a tiny
# browser viewer renders the exported cells. All random inputs have explicit seeds.
include("export_utils.jl")

function centers(t)
    if t<=60
        return ((.25,.28),(.72,.72))
    elseif t<=120
        a=(t-60)/60
        return ((.25+.42a,.28+.40a),(.72-.4a,.72-.3a))
    elseif t<=240
        return ((.78,.22),(.60,.28))
    else
        return ((.23,.78),(.40,.65))
    end
end
phase(t) = t<=60 ? "Two stationary clusters" : t<=120 ? "Clusters move" : t<=160 ? "Abrupt shift" : t<=240 ? "Silence: no arrivals" : "Activity returns elsewhere"
function point(rng,t)
    rand(rng)<.08 && return (rand(rng),rand(rng))
    c=centers(t)[rand(rng)<.65 ? 1 : 2]
    # Rejection samples bounded Gaussian components; clamping would invent edge mass.
    while true
        x=(c[1]+.065randn(rng),c[2]+.065randn(rng))
        all(v->0<=v<=1,x) && return x
    end
end

function main()
    seed=length(ARGS)>=1 ? parse(Int,ARGS[1]) : 42
    outdir=length(ARGS)>=2 ? abspath(ARGS[2]) : joinpath(@__DIR__,"..","output")
    mkpath(outdir)
    rng=MersenneTwister(seed); audit_rng=MersenneTwister(seed+10000)
    models=[ReferenceModel(half_life=Inf),ReferenceModel(),ReferenceModel()]
    rebuild!(models[3];now=0.)
    frames=NamedTuple[]; id=0; worst_error=0.
    rows=String["time,phase,model,mass,retained,nodes,integral,audit_log_score,model_bytes,rebuild_seconds,rebuild_l1_jump"]
    names=["Fixed / no decay","Fixed / decay","Adaptive / decay"]
    scores=[Float64[] for _ in models]
    for t in 0:300
        quiet=161<=t<=240
        logs=fill(NaN,3)
        # Independent held-out probes before this tick's updates; no probe enters the model.
        if t>0 && !quiet
            probes=[point(audit_rng,t) for _ in 1:100]
            for (j,m) in enumerate(models)
                logs[j]=sum(log(density(m.tree,x,Float64(t))) for x in probes)/length(probes)
                push!(scores[j],logs[j])
            end
            for _ in 1:12
                id+=1; x=point(rng,t)
                for m in models
                    observe!(m,id,x;event_time=Float64(t))
                end
            end
        end
        elapsed=0.; jump=0.
        if t>0 && t%10==0
            # Midpoint quadrature diagnostic, not an exact TV bound.
            grid=[((i-.5)/40,(j-.5)/40) for i in 1:40 for j in 1:40]
            before=[density(models[3].tree,x,Float64(t)) for x in grid]
            elapsed=@elapsed rebuild!(models[3];now=Float64(t))
            jump=sum(abs(density(models[3].tree,x,Float64(t))-v) for (x,v) in zip(grid,before))/length(grid)
        end
        states=NamedTuple[]
        for (j,m) in enumerate(models)
            cs=cells(m.tree,Float64(t)); z=integral(m.tree,Float64(t))
            worst_error=max(worst_error,abs(z-1))
            push!(states,(cells=[(c.box...,c.density) for c in cs],mass=mass(m.tree,Float64(t)),nodes=length(m.tree.nodes),score=logs[j]))
            push!(rows,join((t,phase(t),names[j],mass(m.tree,Float64(t)),length(m.records),length(m.tree.nodes),z,logs[j],Base.summarysize(m),j==3 ? elapsed : 0.,j==3 ? jump : 0.),','))
        end
        points=[(r.x...,exp(-log(2)/12*(t-r.event_time))) for r in models[3].records]
        push!(frames,(time=t,phase=phase(t),points=points,models=states,rebuild=(t>0 && t%10==0 ? 1 : 0)))
    end
    open(joinpath(outdir,"metrics.csv"),"w") do io
        write(io,join(rows,'\n'),"\n")
    end
    template=read(joinpath(@__DIR__,"viewer.html"),String)
    data=sprint(io->json(io,frames))
    write(joinpath(outdir,"demo.html"),replace(template,"__FRAMES__"=>data,"__SEED__"=>string(seed)))
    report="""
    Seed: $seed; 300 time units; half-life: 12; cache cap: 600; node cap: 127.
    Maximum exact leaf probability normalization error: $worst_error
    Mean independent pre-update audit log density (higher is better; uniform baseline = 0):
    $(join(["  $(names[j]): $(round(sum(scores[j])/length(scores[j]);digits=4))" for j in 1:3],"\n"))
    These are single-stream diagnostics, not statistical evidence of superiority.
    CSV model_bytes measures retained Julia object size, not process RSS or peak rebuild memory.
    Rebuilds temporarily retain old/new trees and compacted nodes; HTML history is offline export storage.
    """
    write(joinpath(outdir,"summary.txt"),report)
    println(report)
    println("Open ",joinpath(outdir,"demo.html"))
end
main()
