using JSON3, SHA
include("live_modules.jl")
using .LiveIndex, .ArrowPostStore, .LeafTreeKernels
function report(root)
    s=restore(root);a=audit(s);decoded=Dict{UInt64,Vector{Float32}}()
    for block in s.blocks
        X=decode_centroid(s.backend,block.codes)
        for (j,id) in enumerate(block.post_ids);decoded[id]=X[:,j];end
    end
    io=IOBuffer()
    for id in sort!(collect(keys(decoded)))
        write(io,htol(id));write(io,htol.(reinterpret(UInt32,decoded[id])))
    end
    hash=bytes2hex(sha256(take!(io)))
    payload=sum(b->sizeof(b.codes.residuals)+sizeof(b.codes.scales)+sizeof(b.codes.center)+sizeof(b.post_ids)+sizeof(b.regions),s.blocks;init=0)
    path=checkpoint_path(root)
    (counts=a,dimension=s.config.dimension,rank=s.config.rank,clock=s.clock,
     index_reachable_bytes=Base.summarysize(s),codec_array_bytes=payload,
     checkpoint_bytes=sum(filesize(joinpath(path,n)) for n in readdir(path)),
     retained_raw_reservoir_points=s.fitted===nothing ? 0 : sum(size(n.reservoir,2) for n in s.fitted.nodes),
     decoded_sha256=hash,embedding_model=s.config.embedding_model)
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    length(ARGS)==1 || error("usage: inspect_live.jl ARROW_STORE")
    println(JSON3.write(report(only(ARGS))))
end
