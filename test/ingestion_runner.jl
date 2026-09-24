using Test, Random
include("../examples/live_modules.jl")
include("../src/SourceAdapters.jl")
include("../src/LiveSources.jl")
include("../src/IngestionRunner.jl")
using .PostRecords, .LiveIndex, .ArrowPostStore, .LiveSources, .IngestionRunner
config=IndexConfig(dimension=4,rank=2,split_after=24,reservoir_cap=24,retention_seconds=86400.)
embed=texts->fill(.2f0,4,length(texts))
event=PostEvent(PostRecord(PostKey(:threads,"000123");text="Julia",created_at=100.,observed_at=100.))
@testset "Writer ownership, atomic cursor commit, and daily retention" begin
    mktempdir() do root
        with_store_lock(root) do
            @test_throws ErrorException with_store_lock(()->nothing,root)
            s=Index(config);checkpoint!(s,root)
            batch=SourceBatch([event],"first",1,true)
            @test_throws ErrorException commit_batch!(s,root,"threads",batch,embed;now=100.,checkpoint_fault="publish")
            restored=restore(root)
            @test isempty(restored.posts) && !haskey(restored.cursors,"threads")
            result=commit_batch!(restored,root,"threads",batch,embed;now=100.)
            @test result.counts.inserted==1 && audit(restore(root)).active==1
            again=restore(root)
            result=commit_batch!(again,root,"threads",batch,t->error("duplicate embedding");now=101.)
            @test result.counts.inserted==0 && again.fitted.processed==1
            @test restore(root).cursors["threads"]=="first"
            # A midnight sweep keeps a post younger than one day; the next
            # midnight removes it, while retaining its replay tombstone.
            daily_maintenance!(again;now=86400.);@test length(again.locations)==1
            daily_maintenance!(again;now=172800.);@test isempty(again.locations)
            @test lookup(again,event.key).status==3
            checkpoint!(again,root)
            @test isempty(restore(root).locations)
        end
        @test with_store_lock(()->true,root)
    end
end
