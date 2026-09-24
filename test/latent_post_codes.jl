using Test, Random, LinearAlgebra, UUIDs
include("../src/LatentPostCodes.jl")
using .LatentPostCodes
@testset "Per-post coordinate transport" begin
    rng=MersenneTwister(29);ids=[uuid4(rng) for _ in 1:80]
    for (d,r,s) in ((2,2,2),(16,4,4),(16,4,8),(16,8,4),(5,0,2))
        Q=Matrix(qr(randn(rng,d,d)).Q)
        old=Decoder(1,randn(rng,d),Q[:,1:r]);new=Decoder(2,randn(rng,d),Matrix(qr(randn(rng,d,d)).Q)[:,1:s])
        X=randn(rng,d,80);c=encode_posts(old,ids,X;T=Float64)
        before=decode_posts(c);next=migrate_posts(c,new)
        @test next.ids==ids
        @test next.z≈new.U'*(before.-new.mean) atol=1e-12
        if r>0 && s>0;@test opnorm(coordinate_map(old,new).A)<=1+1e-12;end
        # Snapshot identity survives mutation of source arrays.
        mu=zeros(d);U=copy(Q[:,1:r]);saved=Decoder(3,mu,U);mu.+=10
        @test saved.mean==zeros(d)
        for T in (Float32,Float16)
            narrow=migrate_posts(c,new;T)
            @test eltype(narrow.z)==T && narrow.ids==ids
            @test Float64.(narrow.z)==Float64.(T.(next.z))
        end
        if r==s && r==d
            back=migrate_posts(next,old)
            @test decode_posts(back)≈before atol=1e-12
        end
    end
    old=Decoder(1,zeros(2),Matrix{Float64}(I,2,2));c=encode_posts(old,ids,randn(rng,2,80);T=Float64)
    l=Decoder(2,[-1.,0.],old.U);r=Decoder(3,[1.,0.],old.U)
    a,b=split_posts(c,l,r,1,0.)
    @test sort(vcat(a.ids,b.ids))==sort(ids)
    @test isempty(intersect(a.ids,b.ids))
    @test all(decode_posts(a)[1,:].<0) && all(decode_posts(b)[1,:].>=0)
    merged=merge_posts(a,b,old);lookup=Dict(id=>i for (i,id) in enumerate(merged.ids))
    @test decode_posts(merged)[:,[lookup[id] for id in ids]]≈decode_posts(c)
    @test_throws ArgumentError merge_posts(a,a,old)
    @test_throws ArgumentError Decoder(1,zeros(2),ones(2,1))
    # Gauge changes inside the same subspace must not move reconstructed posts.
    U=Matrix(qr(randn(rng,16,4)).Q)[:,1:4];base=Decoder(1,zeros(16),U)
    c=encode_posts(base,ids,randn(rng,16,80);T=Float64);before=decode_posts(c)
    for step in 1:200
        rot=Matrix(qr(randn(rng,4,4)).Q)
        c=migrate_posts(c,Decoder(step+1,zeros(16),U*rot))
    end
    @test norm(decode_posts(c)-before)/norm(before)<1e-12
    # A rank-one transport can have condition number 1 yet progressively erase data.
    theta=pi/9;a=Decoder(1,zeros(2),reshape([1.,0.],2,1));b=Decoder(2,zeros(2),reshape([cos(theta),sin(theta)],2,1))
    c=PostCodes([ids[1]],a,ones(1,1))
    for _ in 1:100;c=migrate_posts(migrate_posts(c,b),a);end
    @test only(c.z)≈cos(theta)^200 rtol=1e-12
    @test cond(coordinate_map(a,b).A)==1.
    @test only(c.z)<1e-5
    # A sequence of changing subspaces can return to the same plane while
    # carrying a fixed post to its opposite side. This is geometric drift.
    initial=PostCodes([ids[1]],a,ones(1,1));c=initial
    for j in 1:200
        angle=j*pi/200
        target=Decoder(j+2,zeros(2),reshape([cos(angle),sin(angle)],2,1))
        c=migrate_posts(c,target)
    end
    @test decode_posts(c)[1,1]≈-cos(pi/200)^200 rtol=1e-12
    @test decode_posts(migrate_posts(initial,c.decoder))≈decode_posts(initial) atol=1e-12

end
