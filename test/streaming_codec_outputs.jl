include("streaming_codecs.jl")
using SHA
BLAS.set_num_threads(1)
@testset "Saved streaming codec payloads" begin
    root=normpath(joinpath(@__DIR__,".."));out=joinpath(root,"output/streaming_codecs")
    X=open(joinpath(root,"data/bluesky_100000/embeddings.f32")) do io
        reshape(read!(io,Vector{Float32}(undef,256*100000)),256,100000)
    end
    for (name,mode) in SC.specifications()
        path=joinpath(out,string(mode)*".bin")
        blocks=open(read_blocks,path);allids=UUID[id for b in blocks for id in b.ids]
        @test length(allids)==length(unique(allids))==40000
        @test Set(Int(id.value & UInt128(typemax(UInt64))) for id in allids)==Set(60001:100000)
        @test all(id.value>>64==1 for id in allids)
        @test Set(b.born for b in blocks)==Set((4,5))
        checksum=SHA.SHA2_256_CTX()
        for b in blocks
            Y=SC.decode(b);@test all(isfinite,Y)
            @test size(Y)==(256,length(b.ids))
            SHA.update!(checksum,reinterpret(UInt8,UInt128[id.value for id in b.ids]))
            SHA.update!(checksum,reinterpret(UInt8,vec(Y)))
            if mode==:f32
                rows=[Int(id.value & UInt128(typemax(UInt64))) for id in b.ids]
                @test Y==X[:,rows]
            end
        end
        @test bytes2hex(SHA.digest!(checksum))==strip(read(joinpath(out,string(mode)*".sha256"),String))
    end
end
