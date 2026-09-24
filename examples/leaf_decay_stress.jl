using Random, LinearAlgebra, Statistics
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
include("export_utils.jl")
function synthetic(seed;rate=1000)
    rng=MersenneTwister(seed);ts=[j/rate for j in 1:120rate if !(60<=j/rate<75)]
    X=zeros(Float64,2,length(ts))
    for (j,t) in enumerate(ts)
        if t<40
            c=rand(rng)<.7 ? [-.65+.012t,-.4] : [.5,.45]
        elseif t<90
            c=rand(rng)<.8 ? [.55,.6-.008(t-40)] : [-.6,.55]
        else
            c=[-.55+.006(t-90),-.55]
        end
        theta=t/12;z=randn(rng,2);v=[.085z[1],.025z[2]]
        X[:,j]=clamp.(c+[cos(theta) -sin(theta);sin(theta) cos(theta)]*v,-.98,.98)
    end
    X,ts
end
function main()
    BLAS.set_num_threads(1);out=normpath(joinpath(@__DIR__,"../output/leaf_decay_stress"));mkpath(out)
    records=NamedTuple[];frames=NamedTuple[]
    fit_leaf_tree(.1randn(MersenneTwister(1),2,100);configs=[(:ppca,1)],split_after=32,max_leaves=2)
    for seed in (31,32,33), h in (Inf,5.,1.)
        X,ts=synthetic(seed);name=isinf(h) ? "No decay" : "$(Int(h))-minute half-life"
        function checkpoint(row,f)
            q=leaf_probabilities(f);ids=findall(n->n.axis==0,f.nodes);w=exp2.(-(f.time.-view(ts,1:row))./h)
            exact=zeros(length(f.nodes))
            for j in 1:row;exact[leaf_index(f.nodes,view(X,:,j))]+=w[j];end
            total=sum(w);mass=sum(q.masses[ids]);mean_est=sum(q.masses[i]*f.nodes[i].states[1].mean for i in ids)/mass
            mean_exact=view(X,:,1:row)*w/total
            err=norm(mean_est-mean_exact);rel=abs(mass-total)/total
            @assert rel<1e-10 && err<1e-9
            @assert isapprox(sum(q.probabilities),1.;atol=1e-10)
            tv=.5sum(abs.(q.masses[ids]-exact[ids]))/total
            recent=findfirst(t->t>=f.time-1,view(ts,1:row));recent_mean=vec(mean(view(X,:,recent:row);dims=2))
            rec=(seed=seed,method=name,time=f.time,posts=row,leaves=length(ids),tree_bytes=Base.summarysize(f.nodes),mass=mass,mass_tv=tv,mass_relative_error=rel,mean_error=err,recent_mean_distance=norm(mean_est-recent_mean))
            push!(records,rec)
            if seed==31
                cells=NamedTuple[]
                for i in ids
                    n=f.nodes[i];p=parameters(n.states[1]);C=p.W*p.W'+Diagonal(p.D)
                    push!(cells,(lo=n.lo,hi=n.hi,mean=p.mean,cov=[C[1,1],C[1,2],C[2,2]],mass=q.masses[i]/mass,exact=exact[i]/total,prob=q.probabilities[i]))
                end
                points=[collect(X[:,j]) for j in max(1,row-299):row]
                push!(frames,(method=name,time=f.time,metrics=rec,cells=cells,points=points))
            end
        end
        elapsed=@elapsed f=fit_leaf_tree(X;timestamps=ts,half_life=h,configs=[(:ppca,1)],split_after=64,max_leaves=1024,max_depth=20,reservoir_cap=32,seed=seed+100,on_checkpoint=checkpoint,observe_every=1000)
        # No-arrival gap check uses a freshly fitted prefix so it cannot see future data.
        lastbefore=searchsortedlast(ts,59.999)
        before=fit_leaf_tree(view(X,:,1:lastbefore);timestamps=ts[1:lastbefore],half_life=h,configs=[(:ppca,1)],split_after=64,max_leaves=1024,max_depth=20,reservoir_cap=32,seed=seed+100)
        a=leaf_probabilities(before);b=leaf_probabilities(before;time=before.time+15)
        @assert isapprox(b.masses,a.masses*exp2(-15/h);rtol=1e-10)
        println("Finished seed=",seed," ",name," fit+audits=",elapsed," seconds")
    end
    payload=(records=records,frames=frames,description="105,000 arrivals per run, 120 simulated minutes, 15-minute silence, abrupt topic switches at 40 and 90 minutes, rotating anisotropic clusters; three seeds. PPCA rank 1 in 2D.")
    write(joinpath(out,"results.json"),sprint(io->json(io,payload)))
    open(joinpath(out,"metrics.csv"),"w") do io
        println(io,join(keys(first(records)),","));for r in records;println(io,join(values(r),","));end
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
