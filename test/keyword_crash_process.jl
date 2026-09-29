using Test
include("../examples/disk_ablation_common.jl")
using .DiskTreeStore.KeywordExtraction
if !isempty(ARGS)
    root,fault=ARGS
    s=DS.open_store(root)
    ps=[PostRecord(PostKey(:threads,"crash$i");text="Space #galaxy",created_at=0.,observed_at=0.) for i in 1:30]
    kw=prepare_keywords(HashtagsOnlyExtractor(),keyword_input.(ps))
    DS.append_batch!(s,ps,fill(0.1f0,2,30);keywords=kw,fault=Symbol(fault))
    error("fault failed")
end
@testset "Keyword atomic publication under abrupt exit" begin
    for fault in (:exit_before_files,:exit_before_commit,:exit_after_commit)
        mktempdir() do root
            s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=DS.config(dimension=2));DS.enable_keywords!(s,HashtagsOnlyExtractor());DS.close_store(s)
            cmd=`$(Base.julia_cmd()) --project=$(joinpath(ROOT,"environments/storage")) $(@__FILE__) $root $(String(fault))`
            p=run(ignorestatus(cmd));@test p.exitcode==93
            s=DS.open_store(root);n=fault==:exit_after_commit ? 30 : 0
            @test DS.audit!(s)==n
            @test DS.keyword_stats(s).covered==n
            summary=DS.leaf_keywords(s,1;ranking=:common)
            @test summary.covered==n
            @test n==0 ? isempty(summary.terms) : only(summary.terms).count==30
            DS.close_store(s)
        end
    end
end
