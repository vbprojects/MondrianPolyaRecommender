include("../examples/load_mnist.jl")
@testset "MNIST flattening and fixed support" begin
    mktempdir() do dir
        path=joinpath(dir,"images")
        pixels=UInt8.(mod.(0:1567,256))
        open(path,"w") do io
            for v in (2051,2,28,28); write(io,hton(UInt32(v))); end
            write(io,pixels)
        end
        x=read_mnist_images(path)
        @test size(x)==(784,2)
        @test eltype(x)==UInt8
        @test vec(x)==pixels
        @test x[29,1]==28 # original raster second row begins here
        labels=joinpath(dir,"labels")
        open(labels,"w") do io
            write(io,hton(UInt32(2049)),hton(UInt32(2)),UInt8[3,7])
        end
        @test read_mnist_labels(labels)==[3,7]
        open(path,"a") do io; write(io,UInt8(0)); end
        @test_throws ErrorException read_mnist_images(path)
    end
    x=UInt8[0 0 100; 0 10 255]
    t=fit_partition(x;root_bounds=(zeros(2),fill(255.,2)),min_count=2)
    @test t.nodes[1].lo==[0,0]
    @test t.nodes[1].hi==[255,255]
    @test isfinite(embedding_logdensity(t,[255,255]))
    @test_throws ArgumentError fit_partition(x;root_bounds=(zeros(2),ones(2)))
    @test_throws ArgumentError fit_partition(x;root_bounds=(zeros(1),ones(1)))
end
