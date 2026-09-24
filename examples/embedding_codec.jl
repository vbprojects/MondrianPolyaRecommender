# Julia exports the frozen online MPPCA codebook; isolated NumPy bridge packs
# codes and runs retrieval metrics. Original archives are never overwritten.
include("bluesky_mppca.jl")
function codec_main()
    root=normpath(joinpath(@__DIR__,".."));out=joinpath(root,"output/embedding_codec");mkpath(out)
    saved=deserialize(joinpath(root,"output/bluesky_mppca/models.jls")); t=saved.partition
    cs=LocalPPCA[]; owners=Int[]
    for (i,bucket) in enumerate(saved.model.mixtures),c in bucket
        push!(cs,c);push!(owners,i)
    end
    write(joinpath(out,"means.f32"),reduce(hcat,Float32.(c.mean) for c in cs))
    B=zeros(Float32,256,4,length(cs))
    for (i,c) in enumerate(cs); B[:,1:size(c.basis,2),i].=c.basis;end
    write(joinpath(out,"bases.f32"),B)
    write(joinpath(out,"owners.i32"),Int32.(owners))
    X=load_bluesky_embeddings(joinpath(root,"data/bluesky_100000"))
    # Shared assignment: nearest component centroid among owners on the actual
    # original point's tree path. Tree locality is preserved, codes then frozen.
    owner_components=[findall(==(i),owners) for i in eachindex(t.nodes)]
    assignments=zeros(UInt16,size(X,2))
    for row in axes(X,2)
        x=view(X,:,row);i=1;best=0;distance=Inf
        while true
            for k in owner_components[i]
                v=sum(abs2,x-cs[k].mean)
                if v<distance;distance=v;best=k;end
            end
            n=t.nodes[i];n.axis==0 && break;i=x[n.axis]<n.cut ? n.left : n.right
        end
        @assert best>0;assignments[row]=best-1
    end
    write(joinpath(out,"components.u16"),assignments)
    write(joinpath(out,"post_ids.u64"),UInt64.(0:size(X,2)-1))
    python=get(ENV,"EMBEDDING_PYTHON",joinpath(root,".venv-embeddings/bin/python"))
    run(`$python $(joinpath(root,"scripts/embedding_codec.py"))`)
end
codec_main()
