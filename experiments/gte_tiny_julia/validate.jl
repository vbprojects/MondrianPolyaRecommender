include("GteTiny.jl")
using .GteTiny, JSON3, Test, Statistics, SHA

artifact = isempty(ARGS) ? normpath(joinpath(@__DIR__,"../../output/gte_tiny_julia")) : abspath(ARGS[1])
readjson(name) = JSON3.read(read(joinpath(artifact,name),String))
ascolumns(rows) = hcat((Float32.(r) for r in rows)...)
cosines(a,b) = [sum(a[:,i].*b[:,i])/sqrt(sum(abs2,a[:,i])*sum(abs2,b[:,i])) for i in axes(a,2)]
load_seconds = @elapsed model = Model(joinpath(artifact,"model"))
stats = Dict{String,Any}("julia"=>string(VERSION),"threads"=>Threads.nthreads(),
    "cpu"=>Sys.CPU_NAME,"model_load_seconds"=>load_seconds,
    "source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"GteTiny.jl")))))
stats["fixtures_sha256"] = Dict(name=>bytes2hex(sha256(read(joinpath(artifact,name))))
    for name in ("tokens.json","parity.json","layers.json","panel.json","reference.json","retrieval.json","model/weights.json"))

@testset "Quantized GTE-tiny" begin
    @testset "Tokenizer against Hugging Face" begin
        cases = readjson("tokens.json")
        for c in cases
            @test tokenize(model.tokenizer,c.text) == c.ids
        end
        stats["tokenizer_cases"] = length(cases)
        @test_throws ArgumentError tokenize(model.tokenizer,"hello";max_tokens=1)
    end
    @testset "Integer kernel and zero points" begin
        w = reshape(Int8[-128,127,0,1,-1,64],3,2)
        a = reshape(UInt8[0,255,128,1,2,3],3,2)
        d = GteTiny.QDense(w,0.25f0,Float32[0.5,-0.2],vec(sum(Int32.(w);dims=1)))
        actual = GteTiny.dense(d,(a,0.5f0,Int32(127)))
        expected = Float32.(transpose(Int32.(w))*(Int32.(a).-127)) .* 0.125f0 .+ d.bias
        @test actual == expected
        for x in (zeros(Float32,2,2),fill(1f0,2,2),fill(-1f0,2,2),Float32[-1 0;0.5 1])
            q,s,z = GteTiny.dynamic_quantize(x)
            @test maximum(abs.((Float32.(q).-Float32(z)).*s.-x)) <= s/2+1f-6
        end
        @test GteTiny.gelu(0f0) == 0f0
        @test abs(GteTiny.gelu(1f0)-0.841344746f0) < 2f-7
        @test abs(GteTiny.gelu(-1f0)+0.158655254f0) < 2f-7
    end
    @testset "Intermediate layers" begin
        case = readjson("layers.json")
        tokens = [tokenize(model.tokenizer,t) for t in case.texts]
        seq = maximum(length,tokens)
        ids = zeros(Int,seq,length(tokens)); mask = fill(false,size(ids))
        for (b,t) in enumerate(tokens)
            ids[1:length(t),b] = t; mask[1:length(t),b] .= true
        end
        trace = Matrix{Float32}[]
        GteTiny.hidden_states(model,ids,mask;trace)
        layer_stats = []
        for (i,hidden) in enumerate(case.hidden)
            expected = hcat((ascolumns(batch) for batch in hidden)...)
            actual = trace[i]
            cosine = minimum(cosines(actual,expected))
            err = maximum(abs.(actual.-expected))
            push!(layer_stats,Dict("layer"=>i-1,"min_token_cosine"=>cosine,"max_abs"=>err))
            println("layer $(i-1): min token cosine=$cosine max abs=$err")
            @test cosine >= 0.995
        end
        stats["layers"] = layer_stats
    end
    @testset "Normalized embeddings against ONNX" begin
        cases = readjson("parity.json"); parity = []
        for (i,c) in enumerate(cases)
            actual = encode(model,String.(c.texts);batch_size=length(c.texts))
            quant = ascolumns(c.quant); fp32 = ascolumns(c.fp32)
            cq = minimum(cosines(actual,quant)); cf = minimum(cosines(actual,fp32))
            err = maximum(abs.(actual.-quant))
            push!(parity,Dict("case"=>i,"texts"=>length(c.texts),"int8_cosine_min"=>cq,
                "fp32_cosine_min"=>cf,"int8_max_abs"=>err,
                "onnx_int8_fp32_cosine_min"=>minimum(cosines(quant,fp32))))
            println("case $i: INT8 cosine=$cq FP32 cosine=$cf max abs=$err")
            @test all(isfinite,actual)
            @test maximum(abs.(sum(abs2,actual;dims=1).-1)) < 2f-6
            @test cq >= 0.999
            @test err <= 0.01
        end
        stats["parity"] = parity
        single = encode(model,["Short."])
        repeated = encode(model,["Short.","Short."])
        @test maximum(abs.(single.-repeated[:,1:1])) < 2f-5
        @test repeated[:,1] == repeated[:,2]
        @test encode(model,["Short."]) == single
        @test size(encode(model,String[])) == (384,0)
        @test_throws ArgumentError encode(model,["hello"];batch_size=33)
        @test_throws ArgumentError hidden_states(model,fill(30522,2,1),fill(true,2,1))
        # Quantization scales span the batch: mixed batches need not match singleton encodes.
    end
end
stats["passed"] = true
open(joinpath(artifact,"validation.json"),"w") do io
    JSON3.pretty(io,stats)
end
println("Validation saved to ",joinpath(artifact,"validation.json"))
