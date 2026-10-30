include("GteTiny.jl")
using .GteTiny, JSON3

length(ARGS) >= 2 || error("Usage: julia --threads=2 --project=experiments/gte_tiny_julia experiments/gte_tiny_julia/embed.jl MODEL_DIR TEXT [TEXT ...]")
model = Model(ARGS[1])
vectors = encode(model,ARGS[2:end])
JSON3.write(stdout,Dict("model"=>"TaylorAI/gte-tiny", "revision"=>GteTiny.REVISION,
    "backend"=>"julia-int8", "dimension"=>384,
    "vectors"=>[collect(vectors[:,i]) for i in axes(vectors,2)]))
println()
