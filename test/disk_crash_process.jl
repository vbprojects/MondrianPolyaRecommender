using Test
include("../examples/disk_ablation_common.jl")
if !isempty(ARGS)
    root,fault=ARGS[1:2]
    bootstrap=length(ARGS)>2 && ARGS[3]=="bootstrap"
    if length(ARGS)>2 && ARGS[3]=="merge"
        s=DS.open_store(root)
        DS.commit!(s;fault=Symbol(fault)) do
            s.clock=1000.
            DS.merge_cold!(s,s.clock)
        end
        error("merge fault did not exit")
    end
    s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=DS.config(dimension=12,max_leaves=2),create_fault=bootstrap ? Symbol(fault) : nothing)
    rs=[PostRecord(PostKey(:threads,"crash$i");text="crash",created_at=0.,observed_at=0.) for i in 1:30]
    DS.append_batch!(s,rs,fill(.1f0,12,30);fault=Symbol(fault),cursor=("crash","30"))
    error("fault did not exit")
end
@testset "Abrupt death during leaf payload migration" begin
    for fault in (:exit_before_files,:exit_before_commit,:exit_after_commit)
        mktempdir() do root
            cfg=DS.config(dimension=12,max_leaves=2,split_after=24,reservoir_cap=24,
                mode=:pca16,layout=:leaf,half_life=10.,merge_below=2.)
            s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=cfg)
            X=randn(MersenneTwister(41),Float32,12,120);X./=sqrt.(sum(abs2,X;dims=1))
            rs=[PostRecord(PostKey(:threads,"migration$i");text="migration",created_at=0.,observed_at=0.) for i in 1:120]
            DS.append_batch!(s,rs,X)
            before=DS.fetch_vectors(s,collect(1:120));generation=s.generation
            @test DS.storage_stats(s).leaves==2
            DS.close_store(s)
            command=`$(Base.julia_cmd()) --project=$(joinpath(ROOT,"environments/storage")) $(@__FILE__) $root $(String(fault)) merge`
            p=run(ignorestatus(command));@test p.exitcode==93
            s=DS.open_store(root)
            @test DS.audit!(s)==120
            @test DS.fetch_vectors(s,collect(1:120))==before
            @test DS.storage_stats(s).leaves==(fault==:exit_after_commit ? 1 : 2)
            @test s.generation==generation+(fault==:exit_after_commit)
            DS.collect_garbage!(s)
            @test DS.audit!(s)==120
            DS.close_store(s)
        end
    end
end
@testset "Atomic store bootstrap" begin
    for fault in (:exit_before_files,:exit_before_commit,:exit_after_commit)
        mktempdir() do root
            command=`$(Base.julia_cmd()) --project=$(joinpath(ROOT,"environments/storage")) $(@__FILE__) $root $(String(fault)) bootstrap`
            p=run(ignorestatus(command));@test p.exitcode==93
            s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=DS.config(dimension=12,max_leaves=2))
            @test DS.audit!(s)==0
            @test s.generation==1 && length(s.nodes)==1
            DS.close_store(s)
        end
    end
end
@testset "Abrupt process death at publication boundaries" begin
    for fault in (:exit_before_files,:exit_before_commit,:exit_after_commit)
        mktempdir() do root
            command=`$(Base.julia_cmd()) --project=$(joinpath(ROOT,"environments/storage")) $(@__FILE__) $root $(String(fault))`
            p=run(ignorestatus(command));@test p.exitcode==93
            s=DS.open_store(root);expected=fault==:exit_after_commit ? 30 : 0
            @test DS.audit!(s)==expected
            @test length(DS.rows(s,"SELECT * FROM cursors"))==(expected>0 ? 1 : 0)
            DS.collect_garbage!(s);@test DS.audit!(s)==expected
            DS.close_store(s)
        end
    end
end
