# Export frozen local models for a paired proposal experiment.
include("bluesky_mppca.jl")
include("export_utils.jl")
function proposal_export()
    root=normpath(joinpath(@__DIR__,".."));out=joinpath(root,"output/proposal_experiment");mkpath(out)
    data=deserialize(joinpath(root,"output/bluesky_mppca/models.jls")); t=data.partition
    X=load_bluesky_embeddings(joinpath(root,"data/bluesky_100000"));rng=MersenneTwister(804)
    rows=randperm(rng,size(X,2))[1:64]
    means=Vector{Vector{Float64}}();bases=Matrix{Float64}[];vars=Vector{Float64}[];noises=Float64[];weights=Float64[];owners=Int[]
    for (i,cs) in enumerate(data.model.mixtures),c in cs
        push!(means,c.mean);B=zeros(256,4);B[:,1:size(c.basis,2)].=c.basis;push!(bases,B)
        noise,v=geometry(c);push!(vars,vcat(v,fill(noise,4-length(v))));push!(noises,noise);push!(weights,c.mass);push!(owners,i)
    end
    localids=Vector{Vector{Int}}();bounds=Vector{Vector{Float64}}();regions=Int[]
    for row in rows
        x=X[:,row];i=1;owner=1
        while true
            !isempty(data.model.mixtures[i]) && (owner=i)
            n=t.nodes[i];n.axis==0 && break;i=x[n.axis]<n.cut ? n.left : n.right
        end
        push!(regions,owner);push!(localids,findall(==(owner),owners).-1)
        push!(bounds,vcat(t.nodes[owner].lo,t.nodes[owner].hi))
    end
    payload=(rows=rows.-1,regions=regions,components=localids,bounds=bounds,means=means,
        bases=[[[B[j,k] for j in 1:256] for k in 1:4] for B in bases],variances=vars,noise=noises,mass=weights)
    write(joinpath(out,"models.json"),sprint(io->json(io,payload)))
    python=get(ENV,"EMBEDDING_PYTHON",joinpath(root,".venv-embeddings/bin/python"))
    run(`$python $(joinpath(root,"scripts/proposal_experiment.py"))`)
    run(`$python $(joinpath(root,"scripts/proposal_report.py"))`)
end
proposal_export()
