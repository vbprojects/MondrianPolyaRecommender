using Test, Random
include("../src/MondrianServer.jl")
const DS=DiskTreeStore;const RM=DS.RocksMetadata
using .PostRecords

@testset "Compact metadata wire format and legacy decoding" begin
    for table in RM.ROW_TABLES
        fields=RM.SCHEMA[table]
        vals=map(fields) do c
            T=RM.fieldtype(table,c)
            T==Int64 ? Int64(-7) : T==Float64 ? -0.0 :
                T==String ? "λ/東京\0 café" : UInt8[0,1,127,255]
        end
        r=NamedTuple{fields}(vals)
        for row in (r,NamedTuple{fields}(map(_->missing,fields)))
            @test isequal(RM.unpack(RM.pack(row)),row)
            @test isequal(RM.unpack(RM.legacy_pack(row)),row)
        end
        @test length(RM.pack(r))<length(RM.legacy_pack(r))
        @test_throws Exception RM.unpack(RM.pack(r)[1:end-1])
        @test_throws Exception RM.unpack(vcat(RM.pack(r),0x00))
    end
    for n in (typemin(Int64),-1,0,1,typemax(Int64))
        @test RM.unpack(RM.pack(n))==n
        @test RM.unpack(RM.legacy_pack(n))==n
    end
    for f in (NaN,Inf,-Inf,-0.,1e200)
        r=(leaf=1,term=2,n=3,mass=f,at=7.)
        @test isequal(RM.unpack(RM.pack(r)),r)
    end
    @test_throws Exception RM.unpack(UInt8[0x4d,0x52,3,1])
    @test_throws Exception RM.unpack(UInt8[0x4d,0x52,2,255])
    # Byte ordering and exact historical key representation must be preserved.
    rng=MersenneTwister(900)
    for _ in 1:200
        args=(rand(rng,Int64),randn(rng),"東京/"*string(rand(rng,UInt32)),rand(rng,UInt8,7))
        @test RM.key("X/test/",args...)=="X/test/"*join(RM.part.(args),"/")
    end
    @test issorted([RM.key("",x) for x in (typemin(Int64),-1,0,1,typemax(Int64))])
    @test issorted([RM.key("",x) for x in (-Inf,-1.,-0.,0.,1.,Inf)])
end

@testset "Mixed legacy/compact records and atomic format upgrade" begin
    mktempdir() do root
        s=DS.open_store(root;backend=:rocksdb,create_config=DS.config(dimension=4),budget=32*1024^2)
        rs=[PostRecord(PostKey(:threads,"legacy-$i");text="legacy text $i",created_at=10.,observed_at=10.) for i in 1:2]
        X=Float32[1 0;0 1;0 0;0 0];DS.append_batch!(s,rs,X;time=10.)
        generation=s.generation
        # Construct the previous on-disk encoding without changing logical data.
        for prefix in ("R/","C/")
            RM.rawscan(s.db,prefix) do k,v
                RM.direct!(s.db,[k=>RM.legacy_pack(RM.unpack(v))];sync=true)
                nothing
            end
        end
        RM.direct!(s.db,["!format"=>RM.bstr(RM.LEGACY_VERSION_TAG)];sync=true)
        DS.close_store(s)
        s=DS.open_store(root;budget=32*1024^2)
        @test s.generation==generation
        @test DS.fetch_vectors(s,[1,2])==X
        @test String(RM.rawget(s.db,"!format"))==RM.LEGACY_VERSION_TAG
        s.db.buffer_limit=32 # Force the format marker and staged rows onto disk.
        @test_throws ErrorException DS.commit!(s;fault=:before_commit) do
            DS.exec(s,"INSERT OR REPLACE INTO cursors VALUES (?,?)",("source","not committed"))
        end
        DS.close_store(s);s=DS.open_store(root;budget=32*1024^2)
        @test String(RM.rawget(s.db,"!format"))==RM.LEGACY_VERSION_TAG
        @test isempty(DS.rows(s,"SELECT * FROM cursors"))
        DS.commit!(s) do
            DS.exec(s,"INSERT OR REPLACE INTO cursors VALUES (?,?)",("source","committed"))
            # Indexed fields do not change; secondary index keys need no writes.
            empty_before=Set(keys(s.db.pending))
            DS.exec(s,"UPDATE posts SET observed=? WHERE id=?",(11.,1))
            @test !any(k->startswith(k,"X/"),setdiff(Set(keys(s.db.pending)),empty_before))
        end
        @test String(RM.rawget(s.db,"!format"))==RM.VERSION_TAG
        @test RM.rawget(s.db,RM.rkey("posts",1))[1:3]==RM.ROW_MAGIC
        @test RM.rawget(s.db,RM.rkey("posts",2))[1:3]!=RM.ROW_MAGIC
        DS.close_store(s);s=DS.open_store(root;budget=32*1024^2)
        @test DS.audit!(s)==2
        @test DS.fetch_vectors(s,[1,2])==X
        @test only(DS.rows(s,"SELECT * FROM cursors")).cursor=="committed"
        # Prepared operations enforce arity and uniqueness just like the fallback.
        @test_throws ErrorException DS.exec(s,"SELECT count FROM leaves WHERE id=?",())
        @test_throws ErrorException DS.commit!(s) do
            DS.exec(s,"INSERT INTO leaves VALUES (?,?)",(1,9))
        end
        DS.close_store(s)
    end
end
