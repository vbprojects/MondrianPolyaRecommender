using LinearAlgebra, Serialization
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
include("load_bluesky.jl")
include("export_utils.jl")
BLAS.set_num_threads(1)
root=normpath(joinpath(@__DIR__,".."));out=joinpath(root,"output/leaf_decay")
entries=split.(readlines(joinpath(out,"chronology.csv")),',')
ids=[parse(Int,e[1]) for e in entries];ts=[parse(Float64,e[2]) for e in entries]
X=load_bluesky_embeddings(joinpath(root,"data/bluesky_100000"));stream=view(X,:,ids)
fit_leaf_tree(view(stream,:,1:512);configs=[(:ppca,4)],max_leaves=2,split_after=128)
results=NamedTuple[]
for h in (Inf,6*3600.)
    elapsed=@elapsed f=fit_leaf_tree(stream;timestamps=ts,half_life=h,configs=[(:ppca,4)],max_leaves=2048,split_after=128)
    query=leaf_probabilities(f);weights=exp2.(-(ts[end].-ts)./h);exact=zeros(length(f.nodes))
    for j in eachindex(ids);exact[leaf_index(f.nodes,view(stream,:,j))]+=weights[j];end
    leaves=findall(n->n.axis==0,f.nodes);mass=sum(query.masses[leaves])
    @assert isapprox(mass,sum(weights);rtol=1e-10)
    weighted_mean=sum(f.nodes[i].states[1].mass*f.nodes[i].states[1].mean for i in leaves)/mass
    @assert isapprox(weighted_mean,stream*weights/sum(weights);atol=1e-10)
    rec=(half_life_hours=isinf(h) ? "disabled" : "6",posts=length(ids),span_hours=(ts[end]-ts[1])/3600,leaves=length(leaves),cuts=length(leaves)-1,effective_mass=mass,mass_tv=.5sum(abs.(query.masses[leaves]-exact[leaves]))/mass,tree_bytes=Base.summarysize(f.nodes),seconds=elapsed)
    push!(results,rec);println(rec)
    isfinite(h) && serialize(joinpath(out,"model.jls"),f)
end
write(joinpath(out,"results.json"),sprint(io->json(io,results)))
write(joinpath(out,"summary.txt"),join(string.(results),"\n"))
