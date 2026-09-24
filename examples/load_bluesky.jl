"""Read exported post-major little-endian Float32 embeddings as dimensions × posts.
Column j corresponds to JSONL record j (embedding_row j-1 in the Python metadata).
"""
function load_bluesky_embeddings(directory=joinpath(@__DIR__,"..","data","bluesky_1000"))
    n,d = parse.(Int,split(read(joinpath(directory,"shape.txt"),String)))
    n>0 && d>0 || error("Invalid shape")
    bytes=read(joinpath(directory,"embeddings.f32"))
    length(bytes)==4n*d || error("Embedding payload length mismatch")
    words=ltoh.(reinterpret(UInt32,bytes))
    vectors=reshape(copy(reinterpret(Float32,words)),d,n)
    all(isfinite,vectors) || error("Nonfinite embeddings")
    maximum(abs.(sum(abs2,vectors;dims=1).-1))<1e-5 || error("Expected unit vectors")
    vectors
end
if abspath(PROGRAM_FILE)==@__FILE__
    x=load_bluesky_embeddings(isempty(ARGS) ? joinpath(@__DIR__,"..","data","bluesky_1000") : ARGS[1])
    println("Loaded Julia matrix: ",size(x)," (dimensions × posts)")
end
