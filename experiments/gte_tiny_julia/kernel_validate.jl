include("KernelGte.jl")
using .GteTiny, JSON3, Test, Random, SHA
const WITH_GPU = !("--cpu-only" in ARGS)
WITH_GPU && include("GPUKernels.jl")
artifact = normpath(joinpath(@__DIR__,"../../output/gte_tiny_julia"))
readjson(name)=JSON3.read(read(joinpath(artifact,name),String))
columns(rows)=hcat((Float32.(r) for r in rows)...)
cosines(a,b)=[sum(a[:,i].*b[:,i])/sqrt(sum(abs2,a[:,i])*sum(abs2,b[:,i])) for i in axes(a,2)]
model=GteTiny.Model(joinpath(artifact,"model"))
portable=KernelGte.KernelModel(model;vnni=false)
native=KernelGte.KernelModel(model)
poly=KernelGte.KernelModel(model;polynomial_math=true)
gpu=WITH_GPU ? GPUKernels.GPUModel(model) : nothing
WITH_GPU && GPUKernels.CUDA.allowscalar(false)
Random.seed!(42)
results=Any[]

@testset "GTE kernel experiment" begin
    @testset "Quantization and optional polynomial math" begin
        for x in (randn(Float32,384,17),zeros(Float32,384,3),fill(-2f0,384,3),fill(2f0,384,3))
            a=GteTiny.dynamic_quantize(x);b=KernelGte.quantize(x)
            @test a==b
        end
        xs=collect(range(-80f0,0f0;length=10001))
        approx=KernelGte.polynomial_exp.(xs);exact=exp.(xs)
        @test maximum(abs.(approx./exact.-1)) < 5f-6
        @test KernelGte.polynomial_exp(-Inf32)==0f0
        @test KernelGte.polynomial_exp(0f0)==1f0
        xs=collect(range(-8f0,8f0;length=10001))
        @test maximum(abs.(KernelGte.gelu_polynomial.(xs).-GteTiny.gelu.(xs))) < 1f-6
    end
    @testset "Full-range integer dense products and tails" begin
        for d in (model.layers[1].query,model.layers[1].up,model.layers[1].down), tokens in (1,3,4,5,17)
            a=rand(UInt8,size(d.weight,1),tokens)
            quant=(a,0.01f0,Int32(127))
            expected=GteTiny.dense(d,quant)
            @test KernelGte.dense(portable,d,quant)==expected
            @test KernelGte.dense(native,d,quant)==expected
        end
        # Signed-extreme weights exercise overflow/saturation and zero-point corrections.
        d=GteTiny.QDense(rand(Int8,384,384),1f0,zeros(Float32,384),zeros(Int32,384))
        d.sums .= vec(sum(Int32.(d.weight);dims=1))
        native.packed[d]=KernelGte.PackedDense(d)
        a=fill(0xff,384,5)
        @test KernelGte.dense(native,d,(a,1f0,Int32(0)))==GteTiny.dense(d,(a,1f0,Int32(0)))
        if WITH_GPU
            CUDA=GPUKernels.CUDA
            original=model.layers[1].query;dd=gpu.layers[1].query
            quant=(rand(UInt8,384,5),0.01f0,Int32(127))
            got=Array(GPUKernels.dense(dd,(CUDA.CuArray(quant[1]),quant[2],quant[3])))
            @test isapprox(got,GteTiny.dense(original,quant);atol=2f-5,rtol=2f-6)
        end
    end
    @testset "Attention masking, odd lengths and two batches" begin
        for seq in (3,17,32),batch in (1,2)
            q=randn(Float32,384,seq*batch);k=randn(Float32,size(q));v=randn(Float32,size(q))
            mask=fill(true,seq,batch);mask[end,:].=false
            expected=GteTiny.attention(q,k,v,mask)
            @test KernelGte.attention(q,k,v,mask)==expected
            if native.vnni
                @test KernelGte.attention(q,k,v,mask,KernelGte.Workspace(),Val(false),true)==expected
            end
        end
    end
    @testset "Complete embeddings and workspace reuse" begin
        for (i,c) in enumerate(readjson("parity.json"))
            texts=String.(c.texts);bs=length(texts)
            expected=GteTiny.encode(model,texts;batch_size=bs)
            ort=columns(c.quant)
            backends=WITH_GPU ? [("ka_cpu",native),("ka_cpu_polynomial",poly),("ka_cuda",gpu)] :
                               [("ka_cpu",native),("ka_cpu_polynomial",poly)]
            for (name,m) in backends
                actual=name=="ka_cuda" ? GPUKernels.encode(m,texts;batch_size=bs) : KernelGte.encode(m,texts;batch_size=bs)
                cr=minimum(cosines(actual,expected));co=minimum(cosines(actual,ort))
                err=maximum(abs.(actual.-expected))
                push!(results,Dict("case"=>i,"backend"=>name,"reference_cosine_min"=>cr,
                    "onnx_cosine_min"=>co,"reference_max_abs"=>err))
                println(name," case ",i," reference cosine=",cr," ONNX cosine=",co," error=",err)
                @test all(isfinite,actual)
                @test maximum(abs.(sum(abs2,actual;dims=1).-1)) < 2f-6
                @test co >= 0.999
                @test cr >= 0.999
                @test err <= 0.01
                name=="ka_cpu" && @test actual==expected
            end
        end
        a=KernelGte.encode(native,["Short.","Short."])
        @test a[:,1]==a[:,2]
        @test a==KernelGte.encode(native,["Short.","Short."])
        @test size(KernelGte.encode(native,String[]))==(384,0)
        @test_throws ArgumentError KernelGte.encode(native,["hello"];batch_size=0)
    end
end
sources=("GteTiny.jl","KernelGte.jl","CPUKernels.jl","GPUKernels.jl","kernel_validate.jl")
stats=Dict("passed"=>true,"gpu_tested"=>WITH_GPU,"cpu_vnni"=>native.vnni,"julia"=>string(VERSION),
    "threads"=>Threads.nthreads(),"cpu"=>Sys.CPU_NAME,"parity"=>results,
    "sources"=>Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in sources),
    "fixtures"=>Dict(f=>bytes2hex(sha256(read(joinpath(artifact,f)))) for f in
        ("parity.json","retrieval.json","tokens.json","panel.json","model/weights.json")))
open(joinpath(artifact,"kernel-validation.json"),"w") do io
    JSON3.pretty(io,stats)
end
