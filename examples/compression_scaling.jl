# Memory scaling check, not a throughput benchmark or density-quality experiment.
using MondrianPolyaToy, Random, Printf
function main()
    out=joinpath(@__DIR__,"..","output","compression"); mkpath(out)
    rows=["admitted_events,method,represented_events,reachable_model_bytes,summaries,leaves"]
    for n in (10_000,100_000,1_000_000)
        rng=MersenneTwister(100)
        kinds=(:replay,:histogram,:microclusters,:residual)
        models=[CompressionModel(k) for k in kinds]
        for hour in 0:23
            count=div(n,24)+(hour<rem(n,24) ? 1 : 0)
            for _ in 1:count
                x=(rand(rng),rand(rng)); s=rand(rng,1:4)
                for m in models
                    ingest!(m,x;event_time=hour+.5,source=s)
                end
            end
            for m in models; maintain!(m); end
        end
        for m in models
            @assert represented_count(m)==n
            @assert isapprox(compressed_mass(m),compressed_mass(models[1]))
            @assert isapprox(sum(c.probability for c in compression_cells(m)),1.)
            push!(rows,join((n,m.kind,represented_count(m),Base.summarysize(m),summary_count(m),length(compression_cells(m))),','))
        end
        println("Checked ",n," admitted events: all represented by all four backends.")
    end
    write(joinpath(out,"scaling.csv"),join(rows,'\n')*"\n")
end
main()
