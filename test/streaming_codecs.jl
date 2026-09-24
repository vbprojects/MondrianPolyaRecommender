using Test, Random, LinearAlgebra, UUIDs
include("../src/StreamingCodecs.jl")
include("../src/LeafLatentTree.jl")
const SC=StreamingCodecs
const LT=LeafLatentTrees

# Independent parser audits the actual byte accounting and stored decoders.
function read_blocks(io)
    @test String(read(io,8))=="MPCODE01"
    d=Int(read(io,UInt32));nb=Int(read(io,UInt32));nd=Int(read(io,UInt32));nc=Int(read(io,UInt32))
    bases=Dict{UInt32,SC.Basis}();decs=Dict{UInt32,SC.Decoder}()
    for _ in 1:nb
        id=read(io,UInt32);parent=read(io,UInt32);rows=Int(read(io,UInt32));cols=Int(read(io,UInt32))
        U=read!(io,zeros(Float32,rows,cols));bases[id]=SC.Basis(get(bases,parent,nothing),U)
    end
    for _ in 1:nd
        id=read(io,UInt32);basis=read(io,UInt32);mu=read!(io,zeros(Float32,d))
        decs[id]=SC.Decoder(mu,get(bases,basis,nothing))
    end
    blocks=SC.Block[]
    for _ in 1:nc
        n=Int(read(io,UInt32));born=Int(read(io,Int32));format=read(io,UInt8);bits=Int(read(io,UInt8))
        decoder=read(io,UInt32);rank=Int(read(io,UInt16))
        ids=UUID.(read!(io,zeros(UInt128,n)));regions=read!(io,zeros(UInt64,n))
        raw=format==1 ? read!(io,zeros(Float32,d,n)) : format==2 ? read!(io,zeros(Float16,d,n)) : nothing
        z=read!(io,zeros(Float32,rank,n))
        q=bits==0 ? nothing : SC.Packed(d,n,bits,read!(io,zeros(Float32,n)),read(io,cld(d*n*bits,8)))
        push!(blocks,SC.Block(ids,regions,born,:decoded,get(decs,decoder,nothing),raw,z,q))
    end
    nh=Int(read(io,UInt32))
    for _ in 1:nh;read(io,UInt64);@test haskey(bases,read(io,UInt32));end
    @test eof(io)
    blocks
end

@testset "Streaming immutable local codecs" begin
    rng=MersenneTwister(9)
    for (d,n) in ((1,1),(3,5),(32,41)),bits in (4,8)
        X=randn(rng,Float32,d,n);X[:,1].=0
        q=SC.pack_coordinates(X,bits);Y=SC.unpack_coordinates(q)
        @test length(q.bytes)==cld(d*n*bits,8)
        @test all(abs.(Y-X).<=reshape(q.scales,1,:)./2 .+ 1e-6)
        @test all(isfinite,Y)
        for j in 1:n,i in 1:d;@test Y[i,j]==SC.packed_coordinate(q,i,j);end
    end
    @test_throws ArgumentError SC.pack_coordinates([NaN;;],8)
    @test_throws ArgumentError SC.pack_coordinates([1.;;],3)
    d=32;U=Matrix(qr(randn(rng,d,4)).Q)[:,1:4]
    X=Float32.(.05U*randn(rng,4,64));state=LT.Sketch(:ppca,4,d)
    for x in eachcol(X);LT.update!(state,x);end
    ids=UUID.(UInt128.(1:64))
    for (name,mode) in SC.specifications()
        s=SC.Store(name,mode,d);SC.ingest!(s,ids,UInt64(1),1,X,state)
        initial=SC.decode(only(s.blocks));st=SC.storage(s)
        @test st.posts==64
        io=IOBuffer();SC.write_payload(io,s);@test position(io)==st.packed_bytes
        seekstart(io);loaded=read_blocks(io)
        @test SC.decode(only(loaded))==initial
        @test only(loaded).ids==ids
        @test SC.axis_values(only(s.blocks),5,1:64)≈Float64.(initial[5,:]) atol=1e-7
        L,R,_=LT.conditional_children(state,1,0.,U,U)
        SC.split!(s,UInt64(1),UInt64(2),UInt64(3),1,0.,L,R)
        @test Set(id for b in s.blocks for id in b.ids)==Set(ids)
        if mode!=:sequential
            @test SC.decode(only(s.blocks))==initial
            @test only(s.blocks).regions==[v<0 ? 2 : 3 for v in initial[1,:]]
        end
        parent=LT.merge_sketches(L,R);SC.merge!(s,UInt64(2),UInt64(3),UInt64(4),parent)
        @test all(r==4 for b in s.blocks for r in b.regions)
        # Inherit and extend using a changed distribution; old bytes must stay fixed.
        SC.ingest!(s,UUID.(UInt128.(65:128)),UInt64(4),2,Float32.(.01randn(rng,d,64)),parent)
        @test Set(id for b in s.blocks for id in b.ids)==Set(UUID.(UInt128.(1:128)))
        io=IOBuffer();SC.write_payload(io,s);@test position(io)==SC.storage(s).packed_bytes
        seekstart(io);loaded=read_blocks(io)
        @test all(SC.decode(a)==SC.decode(b) for (a,b) in zip(loaded,s.blocks))
        SC.expire!(s,2);@test SC.storage(s).posts==64
        SC.expire!(s,3);empty_stats=SC.storage(s)
        @test empty_stats.posts==empty_stats.decoders==empty_stats.basis_nodes==0
        @test isempty(s.hints) && empty_stats.packed_bytes==28
    end
end
