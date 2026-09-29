using Test, Random
include("../bin/replay_tree.jl")
@testset "Offline retained-post tree replay" begin
    mktempdir() do root
        src=joinpath(root,"original");dst=joinpath(root,"new");report=joinpath(root,"report.json")
        s=DS.open_store(src;create_config=DS.config(dimension=4,split_after=256,max_leaves=8,reservoir_cap=16))
        ex=KE.HashtagsOnlyExtractor();DS.enable_keywords!(s,ex)
        rng=MersenneTwister(5)
        for batch in 1:3
            ps=[PostRecord(PostKey(:threads,"$batch-$i");text="Test #Space",tags=["Space"],created_at=100.,observed_at=100. + batch,
                aliases=i==1 ? [PostKey(:threads,"alias-$batch")] : PostKey[]) for i in 1:64]
            DS.append_batch!(s,ps,rand(rng,Float32,4,64).*1.8f0.-.9f0;time=100. + batch,keywords=KE.prepare_keywords(ex,KE.keyword_input.(ps)))
        end
        DS.delete_post!(s,PostKey(:threads,"1-2");revision_at=104.,time=104.)
        original=DS.tree_signature(s);gen=s.generation;DS.close_store(s)
        r=replay_tree(src,dst,report)
        @test r.verified_posts==191
        @test r.settings.split_after==32 && r.settings.max_leaves==1024 && r.settings.max_depth==64 && r.settings.merge_below==2.
        @test r.after.leaves>r.before.leaves
        @test r.decoded_vectors_identical && r.global_hashtags_identical
        a=DS.open_store(src);b=DS.open_store(dst)
        try
            @test DS.tree_signature(a)==original && a.generation==gen
            @test DS.lookup(b,PostKey(:threads,"alias-1")).id==DS.lookup(b,PostKey(:threads,"1-1")).id
            @test DS.lookup(b,PostKey(:threads,"1-2"))===nothing
        finally;DS.close_store(a);DS.close_store(b);end
        @test_throws ErrorException replay_tree(src,dst,report)
    end
end
