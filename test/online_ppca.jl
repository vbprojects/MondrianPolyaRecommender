@testset "Online PPCA mixture" begin
    m=CompressionModel(:ppca;half_life=Inf,window_buckets=2)
    rng=MersenneTwister(111); sx=0.; sy=0.; xx=0.; xy=0.; yy=0.
    for i in 1:1000
        x=.2+.6rand(rng); y=clamp(.1+.75x+.015randn(rng),0.,1.)
        w=.5+rand(rng)
        ingest!(m,(x,y);weight=w,now=.5)
        sx+=w*x; sy+=w*y; xx+=w*x*x; xy+=w*x*y; yy+=w*y*y
    end
    cs=m.buckets[1].ppca
    @test length(cs)<=8
    @test sum(c.mass for c in cs)≈compressed_mass(m)
    @test sum(c.mass*c.mx for c in cs)≈sx
    @test sum(c.mass*c.my for c in cs)≈sy
    @test sum(c.xx+c.mass*c.mx^2 for c in cs)≈xx
    @test sum(c.xy+c.mass*c.mx*c.my for c in cs)≈xy
    @test sum(c.yy+c.mass*c.my^2 for c in cs)≈yy
    @test all(M.ppca_geometry(c).large>=M.ppca_geometry(c).noise>0 for c in cs)
    @test all(!isempty(M.ppca_samples(c)) for c in cs)
    maintain!(m)
    @test sum(c.probability for c in compression_cells(m))≈1
    @test mass(m.tree)≈compressed_mass(m)
    @test isfinite(compression_density(m,(.4,.4)))
    maintain!(m;now=2.)
    @test summary_count(m)==0
    @test compressed_mass(m)==0
    @test compression_density(m,(.4,.4))≈1
    d=CompressionModel(:ppca;half_life=1.,window_buckets=3)
    ingest!(d,(.5,.5);event_time=.2,now=.4)
    ingest!(d,(.5,.5);event_time=.1,now=.4,weight=2.)
    maintain!(d;now=1.)
    @test compressed_mass(d)≈2.0^(-.8)+2*2.0^(-.9)
    @test mass(d.tree)≈compressed_mass(d)
    @test all(isfinite(c.density) for c in compression_cells(d))
end

@testset "Leaf-local MPPCA ownership" begin
    m=CompressionModel(:mppca;half_life=Inf,window_buckets=2,split_mass=8.,merge_mass=2.,max_nodes=31)
    rng=MersenneTwister(90); total=zeros(6)
    for batch in 1:8
        for j in 1:100
            x=(rand(rng),rand(rng)); w=.5+rand(rng)
            ingest!(m,x;now=.5,weight=w)
            total .+= w.*[1.,x[1],x[2],x[1]^2,x[1]*x[2],x[2]^2]
        end
        maintain!(m)
    end
    b=m.buckets[1]
    @test length(b.local_ppca)>1
    @test haskey(b.local_ppca,1) && m.split[1]
    @test all(length(cs)<=8 for cs in values(b.local_ppca))
    result=zeros(6)
    for (i,cs) in b.local_ppca,c in cs
        @test M.inside(m.scaffold.nodes[i].box,(c.mx,c.my))
        @test all(x->M.inside(m.scaffold.nodes[i].box,x),M.ppca_samples(c,m.scaffold.nodes[i].box))
        result .+= [c.mass,c.mass*c.mx,c.mass*c.my,c.xx+c.mass*c.mx^2,c.xy+c.mass*c.mx*c.my,c.yy+c.mass*c.my^2]
    end
    @test result≈total
    @test mass(m.tree)≈total[1]
    @test sum(c.probability for c in compression_cells(m))≈1
    @test summary_count(m)==sum(length,values(b.local_ppca))
    maintain!(m;now=2.)
    @test summary_count(m)==0
    @test all(isempty(b.local_ppca) for b in m.buckets)
    @test compressed_mass(m)==0
end
