using Test, Random, MondrianPolyaToy
const M = MondrianPolyaToy

@testset "Complete geometry and uniform prior" begin
    for seed in 1:15
        t = Tree(;seed,max_nodes=80)
        @test length(t.nodes) <= 80
        @test sum(M.area(c.box) for c in cells(t)) ≈ 1
        @test integral(t) ≈ 1
        @test all(c.density ≈ 1 for c in cells(t))
        @test density(t,(0.,0.)) ≈ 1
        @test density(t,(1.,1.)) ≈ 1
        @test density(t,(-0.1,0.)) == 0
    end
    @test length(Tree(lifetime=0.).nodes) == 1
    @test_throws ArgumentError Tree(half_life=0.)
end

@testset "Lazy evidence versus independent weighted replay" begin
    rng = MersenneTwister(91); t = Tree(half_life=7.)
    records = []
    for i in 1:150
        x = (rand(rng),rand(rng)); event_time = i-rand(rng)*10; weight = rand(rng)*2
        observe!(t,x;event_time,now=Float64(i),weight)
        push!(records,(x,event_time,weight))
    end
    for now in (150.,157.,200.)
        expected = zeros(length(t.nodes))
        for (x,et,w) in records
            for j in M.path(t,x)
                expected[j] += w*exp(-t.beta*(now-et))
            end
        end
        for (j,n) in enumerate(t.nodes)
            @test M.mass(n,t,now) ≈ expected[j] atol=1e-12
            if n.axis != 0
                @test M.mass(n,t,now) ≈ M.mass(t.nodes[n.left],t,now)+M.mass(t.nodes[n.right],t,now) atol=1e-12
            end
        end
        @test integral(t,now) ≈ 1
        # Integrate actual density calls at one interior point per exact cell.
        @test sum(M.area(c.box)*density(t,((c.box[1]+c.box[2])/2,(c.box[3]+c.box[4])/2),now) for c in cells(t,now)) ≈ 1
    end
    @test mass(t,157.) ≈ mass(t,150.)/2
    @test all(c.density ≈ 1 for c in cells(t,10000.))
    @test t.time == 150.
    @test_throws ArgumentError observe!(t,(.5,.5);now=149.,event_time=149.)
    @test_throws ArgumentError observe!(t,(1.5,.5);now=150.,event_time=150.)
    @test_throws ArgumentError observe!(t,(.5,.5);now=150.,event_time=151.)
    @test_throws ArgumentError density(t,(NaN,.5))
end

@testset "Cache, identity, replay, collapse and prospective learning" begin
    rng=MersenneTwister(4)
    m=ReferenceModel(capacity=25,half_life=5.,min_mass=3.)
    for i in 1:80
        observe!(m,i,(rand(rng),rand(rng));event_time=i-2.,now=Float64(i))
        i%7==0 && rebuild!(m)
        @test length(m.records)<=25
        @test mass(m.tree) ≈ sum(exp(-m.tree.beta*(i-r.event_time))*r.weight for r in m.records)
    end
    remove!(m,80;now=83.)
    @test_throws ArgumentError remove!(m,80;now=83.)
    @test_throws ArgumentError observe!(m,79,(.1,.2);now=83.,event_time=83.)
    before=mass(m.tree)
    rebuild!(m;now=83.)
    @test mass(m.tree) ≈ before
    for n in m.tree.nodes
        n.axis==0 && continue
        @test M.mass(n,m.tree,83.) ≈ M.mass(m.tree.nodes[n.left],m.tree,83.)+M.mass(m.tree.nodes[n.right],m.tree,83.) atol=1e-12
    end
    rebuild!(m;now=300.)
    @test length(m.tree.nodes)==1
    @test density(m.tree,(.2,.4))==1
    for i in 81:110
        observe!(m,i,(.2,.4);event_time=300.,now=300.)
    end
    rebuild!(m;now=300.)
    @test length(m.tree.nodes)>1
    @test integral(m.tree) ≈ 1
    @test mass(m.tree) ≈ 25
    for r in copy(m.records)
        remove!(m,r.id;now=301.)
    end
    @test mass(m.tree) ≈ 0 atol=1e-12
    @test all(c.density ≈ 1 for c in cells(m.tree))
end

@testset "No decay and rebuild equivalence with no pruning" begin
    m=ReferenceModel(half_life=Inf,min_mass=0.)
    for i in 1:40
        observe!(m,i,(.3,.7);event_time=Float64(i))
    end
    before=cells(m.tree,1000.)
    rebuild!(m;now=1000.)
    @test mass(m.tree)==40
    @test cells(m.tree)==before
    @test density(m.tree,(.3,.7),1e6)==density(m.tree,(.3,.7))
end

include("compression.jl")
include("localization.jl")

include("embedding_tree.jl")
include("mnist.jl")
include("online_ppca.jl")
