using Test, Random, LinearAlgebra, SHA, JSON3
include("../src/LeafLatentTree.jl")
include("../src/PostRecords.jl")
include("../src/DiskCodecs.jl")
include("../src/DiskTreeStore.jl")
const DS=DiskTreeStore
const DC=DiskCodecs
const LT=LeafLatentTrees
using .PostRecords
BLAS.set_num_threads(1)
function posts(first,last;t=0.,revision=t)
    [PostRecord(PostKey(:bluesky,"at://did:plc:disk/app.bsky.feed.post/$i");text="Post $i + 🦋",author="author $(i%7)",
        created_at=t,observed_at=revision,revision_at=revision,tags=["test"]) for i in first:last]
end
function vectors(d,n;seed=41)
    X=randn(MersenneTwister(seed),Float32,d,n);X./=sqrt.(sum(abs2,X;dims=1));X
end
@testset "Aliases, unknown deletes and open failures" begin
    mktempdir() do root
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=DS.config(dimension=12))
        key=PostKey(:threads,"primary");alias=PostKey(:activitypub,"https://example.test/posts/primary+suffix")
        a=PostRecord(key;text="original",created_at=0.,observed_at=0.)
        DS.append_batch!(s,[a],vectors(12,1))
        metadata=PostRecord(key;text="original",created_at=0.,observed_at=1.,aliases=[alias])
        @test DS.append_batch!(s,[metadata],vectors(12,1);time=1.).metadata==1
        @test DS.lookup(s,alias).id==DS.lookup(s,key).id
        @test key_from_bytes(DS.lookup(s,1).kind,DS.lookup(s,1).native)==key
        @test DS.lookup(s,999)===nothing
        bare=PostRecord(key;text="original",created_at=0.,observed_at=1.)
        aliased=PostRecord(alias;text="original",created_at=0.,observed_at=1.)
        @test_throws ErrorException DS.append_batch!(s,[bare,aliased],vectors(12,2);time=1.)
        DS.close_store(s);s=DS.open_store(root)
        @test DS.audit!(s)==1
        duplicate=PostRecord(alias;text="original",created_at=0.,observed_at=1.)
        @test_throws ErrorException DS.append_batch!(s,[metadata,duplicate],vectors(12,2);time=1.)
        unseen=PostKey(:threads,"deleted-before-arrival")
        @test DS.delete_post!(s,unseen;revision_at=2.,time=2.)
        old=PostRecord(unseen;text="old event",created_at=1.,observed_at=1.)
        @test DS.append_batch!(s,[old],vectors(12,1);time=2.).ignored==1
        @test DS.audit!(s)==1
        DS.close_store(s)
        @test_throws ErrorException DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=DS.config(dimension=24))
        s=DS.open_store(root);@test DS.audit!(s)==1;DS.close_store(s)
    end
end
function fingerprints(s)
    Dict(r.id=>DC.signature(DS.frame_for(s,r.block),r.row) for r in DS.rows(s,"SELECT id,block,row FROM posts WHERE status=1 ORDER BY id"))
end
@testset "Codec relocation is bit preserving" begin
    X=vectors(24,60)
    for mode in DC.FORMATS
        f=DC.encode(mode,X);g=DC.read_frame(DC.frame_bytes(f))
        @test DC.frame_bytes(f)==DC.frame_bytes(g)
        @test DC.decode(f)==DC.decode(g)
        @test DC.decode(DC.subset(f,[1,17,60]))==DC.decode(f)[:,[1,17,60]]
        @test DC.decode(DC.read_frame(DC.frame_bytes(DC.subset(f,[1]))))==DC.decode(f)[:,1:1]
        @test DC.decode(DC.concatenate(DC.subset(f,1:20),DC.subset(f,21:60)))==DC.decode(f)
        @test all(isfinite,DC.decode(DC.encode(mode,X[:,1:1])))
        @test length(DC.frame_bytes(f))==sum(values(DC.payload_stats(f))[1:3])
    end
end
@testset "Layouts, identities, model parity and mutation" begin
    X=vectors(12,600);rs=posts(1,600;t=1.)
    for mode in DC.FORMATS
        signatures=[];topologies=[]
        for layout in (:append,:leaf)
            mktempdir() do root
                cfg=DS.config(;dimension=12,mode,layout,max_leaves=8,split_after=24,reservoir_cap=24,half_life=10.,merge_below=2.)
                s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=cfg,budget=32*1024^2)
                DS.append_batch!(s,rs[1:200],X[:,1:200];time=1.)
                frozen=fingerprints(s)
                DS.append_batch!(s,rs[201:end],X[:,201:end];time=1.)
                after=fingerprints(s)
                @test all(after[id]==sig for (id,sig) in frozen)
                @test DS.audit!(s)==600
                f=LT.fit_leaf_tree(X;max_leaves=8,split_after=24,reservoir_cap=24,seed=17,
                    configs=[(:ppca,4)],timestamps=ones(600),half_life=10.,merge_below=0.)
                @test [(n.axis,n.cut,n.left,n.right) for n in f.nodes]==[(n.axis,n.cut,n.left,n.right) for n in sort(collect(values(s.nodes));by=n->n.id)]
                for time in (1.,1001.)
                    expected=LT.leaf_probabilities(f;time);actual=DS.polya_weights(s;time)
                    @test [actual.probabilities[i] for i in eachindex(f.nodes)]≈expected.probabilities
                    @test [actual.masses[i] for i in eachindex(f.nodes)]≈expected.masses
                end
                for (id,node) in enumerate(f.nodes)
                    node.axis==0 || continue
                    a=only(node.states);b=DS.get_state(s,id).sketch
                    @test a.mass≈b.mass
                    @test a.mean≈b.mean atol=1e-10
                    @test a.U*Diagonal(a.eigenvalues)*a.U'≈b.U*Diagonal(b.eigenvalues)*b.U' atol=1e-9
                end
                sig=fingerprints(s);push!(signatures,sig);push!(topologies,DS.tree_signature(s))
                @test DS.append_batch!(s,rs,X;time=1.).ignored==600
                @test DS.storage_stats(s).active==600
                sample=DS.sample_ids(s,X[:,1];leaf_budget=8,rng=MersenneTwister(9))
                @test length(sample.ids)==length(unique(sample.ids))==20
                @test length(DS.fetch_records(s,sample.ids))==20
                nblocks=DS.storage_stats(s).blocks
                DS.compact!(s);@test fingerprints(s)==sig
                layout==:append && @test DS.storage_stats(s).blocks==nblocks
                DS.maintain!(s;time=201.);@test DS.storage_stats(s).leaves==1
                @test fingerprints(s)==sig
                generation=s.generation;DS.close_store(s)
                s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2)
                @test isempty(s.block_cache.entries) && isempty(s.state_cache.entries)
                @test s.generation==generation && fingerprints(s)==sig
                edit=PostRecord(rs[1].key;text="Edited",created_at=1.,observed_at=202.,revision_at=202.)
                DS.append_batch!(s,[edit],X[:,2:2];time=202.)
                @test DS.lookup(s,edit.key).id==1
                @test DS.fetch_records(s,[1])[1].text=="Edited"
                DS.delete_post!(s,rs[2].key;revision_at=203.,time=203.)
                @test DS.lookup(s,rs[2].key).status==2
                @test_throws KeyError DS.fetch_vectors(s,[2])
                @test DS.append_batch!(s,rs[2:2],X[:,2:2];time=203.).ignored==1
                @test DS.audit!(s)==599
                # Independent weighted mean/diagonal check, with bounded rebuild.
                DS.rebuild_density!(s)
                for node in values(s.nodes)
                    node.axis==0 || continue
                    ps=DS.rows(s,"SELECT id,density_at FROM posts WHERE status=1 AND leaf=? ORDER BY sequence",(node.id,))
                    isempty(ps) && continue
                    Y=DS.fetch_vectors(s,[p.id for p in ps]);w=[exp2(-(s.clock-p.density_at)/cfg.half_life) for p in ps]
                    mu=Y*w/sum(w);scatter=vec(sum((Float64.(Y).-mu).^2 .* reshape(w,1,:);dims=2))
                    st=DS.get_state(s,node.id).sketch
                    @test st.mass≈sum(w)
                    @test st.mean≈mu atol=1e-10
                    @test st.diagonal≈scatter atol=1e-9
                    reference=LT.Sketch(:ppca,4,12);previous=minimum(p.density_at for p in ps)
                    for (j,p) in enumerate(ps)
                        LT.advance_state!(reference,Y[:,j],p.density_at-previous,cfg.half_life);previous=p.density_at
                    end
                    LT.decay!(reference,s.clock-previous,cfg.half_life)
                    @test st.U*Diagonal(st.eigenvalues)*st.U'≈reference.U*Diagonal(reference.eigenvalues)*reference.U' atol=1e-9
                end
                late=cfg.retention_seconds+1000
                @test isempty(DS.sample_ids(s,X[:,1];time=late,leaf_budget=8).ids)
                @test DS.expire!(s;time=late)==599
                @test DS.collect_garbage!(s)>0
                @test DS.audit!(s)==0
                @test DS.storage_stats(s).vector_files==0 && DS.storage_stats(s).text_files==0
                DS.close_store(s)
            end
        end
        @test signatures[1]==signatures[2]
        @test topologies[1]==topologies[2]
    end
end
@testset "Crash publication and corruption" begin
    for fault in (:before_files,:before_commit,:after_commit)
        mktempdir() do root
            cfg=DS.config(dimension=12,max_leaves=2)
            s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),create_config=cfg)
            @test_throws ErrorException DS.append_batch!(s,posts(1,30),vectors(12,30);cursor=("test","30"),fault)
            DS.close_store(s);s=DS.open_store(root)
            @test DS.storage_stats(s).active==(fault==:after_commit ? 30 : 0)
            @test length(DS.rows(s,"SELECT * FROM cursors"))==(fault==:after_commit ? 1 : 0)
            DS.collect_garbage!(s)
            @test DS.audit!(s)==(fault==:after_commit ? 30 : 0)
            if fault==:after_commit
                b=first(DS.rows(s,"SELECT * FROM blocks LIMIT 1"));DS.clear_caches!(s)
                open(joinpath(root,b.file),"r+") do io;seek(io,b.offset+15);x=read(io,UInt8);seek(io,b.offset+15);write(io,x⊻0xff);end
                @test_throws ErrorException DS.frame_for(s,b.id)
            end
            DS.close_store(s)
        end
    end
end
@testset "Forced eviction and restart" begin
    mktempdir() do root
        cfg=DS.config(dimension=384,mode=:f32,max_leaves=128,split_after=32,reservoir_cap=64)
        s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2,create_config=cfg)
        X=vectors(384,12000)
        for first in 1:1000:12000
            DS.append_batch!(s,posts(first,first+999),X[:,first:first+999])
        end
        @test s.block_cache.evictions>0
        @test s.state_cache.evictions>0
        @test s.block_cache.peak<=s.block_cache.capacity && s.state_cache.peak<=s.state_cache.capacity
        @test s.workspace+s.sqlite_budget+s.block_cache.capacity+s.state_cache.capacity<=s.budget
        expected=DS.fetch_vectors(s,[1,5001,12000]);sig=DS.tree_signature(s)
        DS.close_store(s);s=DS.open_store(root;backend=Symbol(get(ENV,"MONDRIAN_TEST_BACKEND","auto")),budget=32*1024^2)
        @test isempty(s.block_cache.entries) && isempty(s.state_cache.entries)
        @test DS.fetch_vectors(s,[1,5001,12000])==expected
        @test DS.tree_signature(s)==sig
        @test DS.audit!(s)==12000
        DS.close_store(s)
    end
end
