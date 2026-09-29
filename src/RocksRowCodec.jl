# Included in RocksMetadata. Fixed schema IDs are part of the on-disk contract.
# Opaque PPCA/runtime/keyword contribution blobs retain their existing format.
const ROW_TABLES=("meta","nodes","states","leaves","identities","posts","blocks",
    "files","cursors","kw_terms","kw_posts","kw_counts")
const ROW_MAGIC=UInt8[0x4d,0x52,0x02]
legacy_pack(x)=take!(let io=IOBuffer();serialize(io,x);io;end)
legacy_unpack(x)=deserialize(IOBuffer(x))
function fieldtype(table,c)
    c in REAL_FIELDS && return Float64
    (c in INTEGER_FIELDS || (c==:kind && table in ("posts","identities")) ||
        (c==:term && table=="kw_counts")) && return Int64
    c in (:value,:payload,:native,:terms) && return Vector{UInt8}
    String
end
wirewrite(io,x::Int64)=write(io,htol(reinterpret(UInt64,x)))
wirewrite(io,x::Float64)=write(io,htol(reinterpret(UInt64,x)))
function wirewrite(io,x::Union{AbstractString,AbstractVector{UInt8}})
    b=x isa AbstractString ? codeunits(x) : x
    length(b)<=typemax(UInt32) || error("metadata field too large")
    write(io,htol(UInt32(length(b))));write(io,b)
end
wireread(io,::Type{Int64})=reinterpret(Int64,ltoh(read(io,UInt64)))
wireread(io,::Type{Float64})=reinterpret(Float64,ltoh(read(io,UInt64)))
function wireread(io,::Type{Vector{UInt8}})
    n=Int(ltoh(read(io,UInt32)));n<=bytesavailable(io) || error("truncated metadata field")
    read(io,n)
end
wireread(io,::Type{String})=String(wireread(io,Vector{UInt8}))
pack(x)=legacy_pack(x)
function pack(x::Integer)
    io=IOBuffer();write(io,ROW_MAGIC);write(io,UInt8(0));wirewrite(io,Int64(x));take!(io)
end

# Emit specialized field accesses, conversions and codecs once, rather than
# rediscovering field types or serializing Julia datatype objects on every row.
for (id,table) in enumerate(ROW_TABLES)
    fields=SCHEMA[table];tag=Val(Symbol(table))
    normalized=[:(let v=getproperty(r,$(QuoteNode(c)))
        v===nothing || ismissing(v) ? missing : convert($(fieldtype(table,c)),v)
    end) for c in fields]
    readvalues=[:((mask & (UInt32(1)<<$(i-1)))!=0 ? missing : wireread(io,$(fieldtype(table,c)))) for (i,c) in enumerate(fields)]
    writes=[:(ismissing(r.$c) || wirewrite(io,r.$c)) for c in fields]
    maskparts=[:(ismissing(r.$c) ? (UInt32(1)<<$(i-1)) : UInt32(0)) for (i,c) in enumerate(fields)]
    maskexpr=foldl((a,b)->:($a | $b),maskparts)
    pk=[:(r.$c) for c in PRIMARY[table]]
    @eval begin
        normalize_row(::$(typeof(tag)),r)=NamedTuple{$fields}(($(normalized...),))
        rowkey(r::NamedTuple{$fields})=key($("R/"*table*"/"),$(pk...))
        function pack(r::NamedTuple{$fields})
            io=IOBuffer();write(io,ROW_MAGIC);write(io,UInt8($id))
            mask=$maskexpr;write(io,htol(mask));$(writes...)
            take!(io)
        end
        function readrow(io,::Val{$id})
            mask=ltoh(read(io,UInt32))
            (mask >> $(length(fields)))==0 || error("invalid metadata null mask")
            NamedTuple{$fields}(($(readvalues...),))
        end
    end
end
normalize_row(table,r)=normalize_row(Val(Symbol(table)),r)
function unpack(x)
    if length(x)>=2 && x[1]==0x4d && x[2]==0x52
        length(x)>=4 && x[3]==0x02 || error("unsupported compact metadata record")
        io=IOBuffer(x);seek(io,4);id=Int(x[4])
        value=if id==0
            wireread(io,Int64)
        else
            1<=id<=length(ROW_TABLES) || error("unknown metadata schema")
            readrow(io,Val(id))
        end
        eof(io) || error("trailing metadata bytes")
        return value
    end
    legacy_unpack(x)
end
