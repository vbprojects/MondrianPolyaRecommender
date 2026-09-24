using Test, Serialization, LinearAlgebra
include("../src/LatentPostCodes.jl")
using .LatentPostCodes
path=joinpath(@__DIR__,"../output/latent_migration/transition_audit.jls")
a=deserialize(path)
@testset "Saved per-post split/merge transitions" begin
    @test length(a.journal)>160
    @test length(unique(a.ids))==length(a.ids)==384
    function verify_codes(actual,expected)
        @test actual.ids==expected.ids
        @test actual.decoder.id==expected.decoder.id
        @test actual.z≈expected.z atol=1e-12
        @test all(isfinite,actual.z)
    end
    live=Dict(a.anchor.decoder.id=>a.anchor)
    for event in a.journal
        if event.kind=="split"
            old=pop!(live,event.old.decoder.id);verify_codes(old,event.old)
            l,r=split_posts(old,event.left.decoder,event.right.decoder,event.axis,event.cut;T=Float64)
            verify_codes(l,event.left);verify_codes(r,event.right)
            live[l.decoder.id]=l;live[r.decoder.id]=r
        else
            l=pop!(live,event.left.decoder.id);r=pop!(live,event.right.decoder.id)
            verify_codes(l,event.left);verify_codes(r,event.right)
            combined=merge_posts(l,r,event.parent.decoder;T=Float64)
            verify_codes(combined,event.parent);live[combined.decoder.id]=combined
        end
        allids=vcat([c.ids for c in values(live)]...)
        @test length(allids)==length(unique(allids))==length(a.ids)
        @test Set(allids)==Set(a.ids)
    end
    @test length(live)==1
    verify_codes(only(values(live)),a.final["Sequential Float64"])
    # Current codes under a retained original decoder remain a distinct policy.
    f=a.final["Frozen initial decoder"];index=Dict(id=>i for (i,id) in enumerate(f.ids))
    @test decode_posts(f)[:,[index[id] for id in a.ids]]≈decode_posts(a.anchor)
end
