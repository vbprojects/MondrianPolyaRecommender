using Test
include("../examples/disk_live_modules.jl")
include("../src/SourceAdapters.jl")
include("../src/DiskIngestionRunner.jl")
using .PostRecords
const DS=DiskTreeStore
const DR=DiskIngestionRunner
if !isempty(ARGS)
    root,phase=ARGS
    s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2)
    post=PostRecord(PostKey(:threads,"crash");text="Space science #NASA",created_at=100.,observed_at=100.)
    batch=(events=[PostEvent(post)],cursor="published",received=1,complete=true)
    DR.commit_batch!(s,"source",batch,ts->fill(.2f0,4,length(ts));now=100.,fault=Symbol(phase))
    error("fault did not terminate process")
end
@testset "Process crash: data, keywords and cursor publish together" begin
    for phase in (:exit_before_files,:exit_before_commit,:exit_after_commit)
        mktempdir() do root
            s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2,create_config=DS.config(dimension=4))
            DS.enable_keywords!(s,KeywordExtraction.JuliaYakeExtractor());generation=s.generation
            DS.close_store(s)
            project=dirname(Base.active_project())
            proc=run(ignorestatus(`$(Base.julia_cmd()) --project=$project $(@__FILE__) $root $(string(phase))`))
            @test proc.exitcode==93
            s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2)
            published=phase==:exit_after_commit
            @test s.generation==generation+Int(published)
            @test DR.source_cursor(s,"source")==(published ? "published" : "")
            @test DR.status(s).active==Int(published)
            @test DS.keyword_stats(s).covered==Int(published)
            @test DS.audit!(s)!==nothing
            DS.collect_garbage!(s)
            DS.close_store(s)
        end
    end
end
