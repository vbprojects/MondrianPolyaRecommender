length(ARGS)>=3 || error("Usage: kernel_embed.jl cpu|cpu-polynomial|cuda MODEL_DIR TEXT [TEXT ...]")
backend=ARGS[1]
backend in ("cpu","cpu-polynomial","cuda") || error("Unknown backend")
include("KernelGte.jl")
using JSON3
reference=GteTiny.Model(ARGS[2])
if backend=="cuda"
    include("GPUKernels.jl")
    GPUKernels.CUDA.allowscalar(false)
    model=GPUKernels.GPUModel(reference)
    vectors=GPUKernels.encode(model,ARGS[3:end])
    implementation="cuda-dp4a"
else
    model=KernelGte.KernelModel(reference;polynomial_math=backend=="cpu-polynomial")
    vectors=KernelGte.encode(model,ARGS[3:end])
    implementation=model.vnni ? "cpu-avx512-vnni" : "cpu-portable"
end
JSON3.write(stdout,Dict("model"=>"TaylorAI/gte-tiny","revision"=>GteTiny.REVISION,
    "backend"=>backend,"implementation"=>implementation,"dimension"=>384,
    "vectors"=>[collect(vectors[:,i]) for i in axes(vectors,2)]))
println()
