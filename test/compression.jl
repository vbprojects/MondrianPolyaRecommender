@testset "Compression: shared clocks, sources and bucket expiry" begin
    rng=MersenneTwister(88)
    kinds=(:replay,:histogram,:microclusters,:residual)
    models=[CompressionModel(k;window_buckets=3,half_life=2.,grid_size=8,cluster_cap=8,split_mass=4.,merge_mass=1.) for k in kinds]
    events=[]
    for step in 1:30
        now=step/4
        for j in 1:12
            event_time=max(0.,now-rand(rng)*.5)
            x=(rand(rng),rand(rng)); w=rand(rng)*2; src=rand(rng,1:4)
            push!(events,(x=x,et=event_time,w=w,src=src))
            for m in models; ingest!(m,x;event_time,now,weight=w,source=src); end
        end
        retained=filter(r->floor(Int,r.et)>=floor(Int,now)-2,events)
        expected=[sum((r.w*exp(-log(2)/2*(now-r.et)) for r in retained if r.src==s);init=0.) for s in 1:4]
        for m in models
            @test source_mass(m) ≈ expected atol=1e-11
            @test represented_count(m)==length(retained)
            maintain!(m;now)
            cs=compression_cells(m)
            @test sum(c.probability for c in cs) ≈ 1
            @test sum(M.area(c.box) for c in cs) ≈ 1
            @test all(c.density>=0 && isfinite(c.density) for c in cs)
            if m.kind==:residual
                @test sum(M.residual_masses(m)) ≈ sum(expected)
            else
                @test mass(m.tree) ≈ sum(expected)
            end
            @test length(cs)<=64
        end
        expectedmom=[sum(r.w*exp(-log(2)/2*(now-r.et))*v(r.x) for r in retained) for v in (x->1.,x->x[1],x->x[2],x->x[1]^2,x->x[2]^2)]
        @test micro_moments(models[3]) ≈ expectedmom
        @test summary_count(models[3])<=24
        @test summary_count(models[2])<=3*64
        @test compression_l1(compression_cells(models[1]),compression_cells(models[1])) ≈ 0
    end
    for m in models
        before=compressed_mass(m)
        @test_throws ArgumentError ingest!(m,(.2,.2);event_time=0.,now=8.)
        @test_throws ArgumentError ingest!(m,(2.,.2);event_time=8.,now=8.)
        @test_throws ArgumentError ingest!(m,(.2,.2);event_time=9.,now=8.)
        @test_throws ArgumentError ingest!(m,(.2,.2);source=5,event_time=8.,now=8.)
        @test_throws ArgumentError ingest!(m,(.2,.2);source=1.5,event_time=8.,now=8.)
        @test compressed_mass(m)==before
        maintain!(m;now=11.)
        @test represented_count(m)==0
        @test compressed_mass(m)==0
        @test all(c.density≈1 for c in compression_cells(m))
        @test compression_density(m,(-.1,.3))==0
    end
end

@testset "Compression: exact replay and approximation boundaries" begin
    rng=MersenneTwister(23)
    m=CompressionModel(:replay;split_mass=.001,merge_mass=0.)
    t=Tree(half_life=6.)
    for j in 1:100
        x=(rand(rng),rand(rng)); et=j/100
        ingest!(m,x;event_time=et)
        observe!(t,x;event_time=et)
    end
    maintain!(m)
    for c in cells(t)
        x=((c.box[1]+c.box[2])/2,(c.box[3]+c.box[4])/2)
        @test compression_density(m,x) ≈ density(t,x)
    end
    # A histogram routes uniform cell mass exactly for its declared approximation.
    h=CompressionModel(:histogram;grid_size=1,split_mass=.001,merge_mass=0.)
    for _ in 1:100; ingest!(h,(.1,.1)); end
    maintain!(h)
    @test all(c.density≈1 for c in compression_cells(h))
    # One degenerate microcluster routes repeated boundary points without NaNs.
    mc=CompressionModel(:microclusters;cluster_cap=1,split_mass=.001,merge_mass=0.)
    for _ in 1:30; ingest!(mc,(1.,1.)); end
    maintain!(mc)
    @test compressed_mass(mc)==30
    @test sum(c.probability for c in compression_cells(mc))≈1
    @test compression_density(mc,(1.,1.))>1
    # A future decay-only clock change makes stale published densities explicit.
    advance!(mc,1.)
    @test_throws ArgumentError compression_cells(mc)
    maintain!(mc)
    @test compressed_mass(mc)≈30*exp(-log(2)/6)
end

@testset "Residual: preserving splits, ancestor overlap, pooling and expiry" begin
    m=CompressionModel(:residual;split_mass=4.,merge_mass=2.,half_life=1.,window_buckets=24)
    for _ in 1:10; ingest!(m,(.2,.2)); end
    # First split leaves all history at the root; the density must stay uniform.
    maintain!(m)
    @test m.split[1]
    @test all(c.density≈1 for c in compression_cells(m))
    n=m.scaffold.nodes[1]; b=m.scaffold.nodes[n.left].box
    x=((b[1]+b[2])/2,(b[3]+b[4])/2)
    ingest!(m,x;event_time=.1)
    maintain!(m)
    @test sum(M.residual_masses(m))≈compressed_mass(m)
    expected=(2+10*exp(-log(2)*.1)+1/M.area(b))/(2+10*exp(-log(2)*.1)+1)
    @test compression_density(m,x)≈expected
    # Long quiet time pools descendants, keeps evidence, and returns one leaf.
    before=compressed_mass(m)
    maintain!(m;now=10.1)
    @test compressed_mass(m)≈before/1024
    @test length(compression_cells(m))==1
    @test sum(M.residual_masses(m))≈compressed_mass(m)
    # An hour-0 event is absent at hour 24; hour-1 event remains until 25.
    r=CompressionModel(:replay)
    ingest!(r,(.2,.3);event_time=.9)
    ingest!(r,(.2,.3);event_time=1.1)
    maintain!(r;now=24.)
    @test represented_count(r)==1
    maintain!(r;now=25.)
    @test represented_count(r)==0
end
