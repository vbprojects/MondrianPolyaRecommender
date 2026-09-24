using MondrianPolyaToy, Random, Printf
const MP=MondrianPolyaToy
include("export_utils.jl")
function main()
    trials=length(ARGS)>0 ? parse(Int,ARGS[1]) : 200
    budget=length(ARGS)>1 ? parse(Int,ARGS[2]) : 80
    trials>0 && budget>0 || error("positive trials and budget required")
    out=joinpath(@__DIR__,"..","output","localization");mkpath(out)
    land=static_landscape()
    policies=[:random,:greedy,:information,:tree_information,:tree_midpoint,:tree_path_information]
    names=["Random posts","Highest predicted like","Information gain","Tree representatives + information","Tree path midpoint only","Tree path asymmetric probes"]
    rng=MersenneTwister(333)
    targets=[MP.draw_index(rng,land.prior) for _ in 1:trials]
    for policy in policies; localization_trial(land,policy;target=targets[1],budget=2); end
    results=NamedTuple[]; examples=NamedTuple[]
    rows=["policy,trial,target,stopping_sample,correct_stop,samples_used,likes_until_stop_or_budget,final_error_after_budget,candidates_evaluated,path_edges_recovered,query_selection_seconds"]
    for (j,policy) in enumerate(policies)
        runs=NamedTuple[]
        elapsed=@elapsed for trial in 1:trials
            r=localization_trial(land,policy;seed=10000+trial,target=targets[trial],budget,history=trial==1)
            push!(runs,r)
            trial==1 && push!(examples,(name=names[j],history=r.history,stop=r.stop,correct=Int(r.correct)))
            push!(rows,join((policy,trial,r.target,r.stop,Int(r.correct),r.used,r.likes,r.final_error,r.evaluated,r.path_edges,r.selection_seconds),','))
        end
        successes=count(r->r.stop>0 && r.correct,runs)
        false_stops=count(r->r.stop>0 && !r.correct,runs)
        successful=[r.stop for r in runs if r.stop>0 && r.correct]
        curve=[count(r->r.correct && 0<r.stop<=t,runs)/trials for t in 0:budget]
        p=successes/trials; z=1.96; den=1+z^2/trials
        mid=(p+z^2/(2trials))/den; half=z*sqrt(p*(1-p)/trials+z^2/(4trials^2))/den
        push!(results,(name=names[j],success=successes/trials,ci_low=mid-half,ci_high=mid+half,false_stop=false_stops/trials,
            exhausted=count(r->r.stop==0,runs)/trials,
            mean_used=sum(r.used for r in runs)/trials,
            mean_success=isempty(successful) ? NaN : sum(successful)/length(successful),
            mean_evaluated=sum(r.evaluated for r in runs)/sum(r.used for r in runs),
            mean_path_edges=sum(r.path_edges for r in runs)/sum(r.used for r in runs),
            mean_selection_ms=1000sum(r.selection_seconds for r in runs)/sum(r.used for r in runs),
            curve=curve,seconds=elapsed))
        println(names[j],": ",successes,"/",trials," correct stops; ",false_stops," incorrect stops; ",round(elapsed;digits=2)," s")
    end
    data=(grid=land.grid_size,points=land.points,posts=land.posts,density=land.density,
        cells=[(c.box...,c.density) for c in cells(land.tree)],radius=land.radius,
        scale=land.response_scale,target=targets[1],budget=budget,trials=trials,
        candidates=length(land.candidates),representatives=length(land.representatives),results=results,examples=examples)
    write(joinpath(out,"demo.html"),replace(read(joinpath(@__DIR__,"localization_viewer.html"),String),"__DATA__"=>sprint(io->json(io,data))))
    write(joinpath(out,"metrics.csv"),join(rows,'\n')*"\n")
    lines=["Static user localization / 2D tree Fermat graph", "Trials per policy: $trials; maximum responses: $budget; confidence threshold: 90%.",
        "Fermat radius: $(land.radius); response scale: $(land.response_scale).",
        "$(length(land.points)) grid states; $(length(land.candidates)) occupied query locations; $(length(land.representatives)) tree representatives.",
        "Dense-region prior and likelihood are known to inference and match the simulated user.",
        "Policy | correct stop % (95% Wilson interval) | incorrect stop % | exhausted % | mean interactions consumed | mean interactions among correct stops | candidates scored/response | path edges/response | selection ms/response"]
    for r in results
        push!(lines,@sprintf("%s | %.1f (%.1f–%.1f) | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f | %.3f",r.name,100r.success,100r.ci_low,100r.ci_high,100r.false_stop,100r.exhausted,r.mean_used,r.mean_success,r.mean_evaluated,r.mean_path_edges,r.mean_selection_ms))
    end
    append!(lines,["", "Mean used counts exhausted trials at budget and includes false stops; it is not an uncensored mean time to correct localization.",
        "Correct-stop means exclude failures and are selection-biased. Use success curves and all outcome categories together.",
        "Same targets and per-trial RNG seeds across policies; new path proposals use a separate RNG from feedback. One landscape; Monte Carlo estimates, not guarantees.",
        "No user movement, time decay, text encoder, item-specific effects or real user data. Grid targets; graph-approximated Fermat distance.",
        "Feedback is independent conditional on query and target, including repeated query locations. Posts sampled with replacement.",
        "Query metrics count work only up to first stop or budget; timings include proposal generation but exclude shared precomputation and posterior updates.\nExample histories continue after first stop for diagnosis; metrics use first confidence stop. No truth used in query/stopping."])
    write(joinpath(out,"summary.txt"),join(lines,'\n')*"\n");println(join(lines,'\n'))
end
main()
