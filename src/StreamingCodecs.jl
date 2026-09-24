"""Experimental immutable post codecs, separate from mutable density-tree state.
Only sequential PPCA rewrites historical codes. No codec reads audit originals.
"""
module StreamingCodecs
using LinearAlgebra, Statistics, UUIDs
export Packed, pack_coordinates, unpack_coordinates, packed_coordinate, Basis, Decoder,
       Block, Store, specifications, ingest!, split!, merge!, expire!, decode, axis_values,
       storage, write_payload, basis_matrix, rankof

struct Packed
    d::Int
    n::Int
    bits::Int
    scales::Vector{Float32}
    bytes::Vector{UInt8}
end
function pack_coordinates(X,bits)
    bits in (4,8) || throw(ArgumentError("expected 4 or 8 bits"))
    all(isfinite,X) || throw(ArgumentError("nonfinite coordinates"))
    d,n=size(X);m=(1<<(bits-1))-1
    scales=Float32[max(maximum(abs,view(X,:,j);init=0.)/m,floatmin(Float32)) for j in 1:n]
    bytes=zeros(UInt8,cld(d*n*bits,8))
    for j in 1:n,i in 1:d
        q=UInt8(clamp(round(Int,X[i,j]/scales[j]),-m,m)+m);k=(j-1)*d+i
        if bits==8;bytes[k]=q
        else;bytes[cld(k,2)] |= q << (4*((k-1)%2));end
    end
    Packed(d,n,bits,scales,bytes)
end
function packed_coordinate(q::Packed,i,j)
    k=(j-1)*q.d+i
    code=q.bits==8 ? q.bytes[k] : (q.bytes[cld(k,2)]>>(4*((k-1)%2)))&0x0f
    Float32(Int(code)-((1<<(q.bits-1))-1))*q.scales[j]
end
function unpack_coordinates(q::Packed)
    [packed_coordinate(q,i,j) for i in 1:q.d,j in 1:q.n]
end
struct Basis
    parent::Union{Nothing,Basis}
    extra::Matrix{Float32}
end
rankof(b::Nothing)=0
rankof(b::Basis)=size(b.extra,2)+rankof(b.parent)
basis_matrix(b::Basis)=b.parent===nothing ? b.extra : hcat(basis_matrix(b.parent),b.extra)
struct Decoder
    mean::Vector{Float32}
    basis::Union{Nothing,Basis}
end
struct Block
    ids::Vector{UUID}
    regions::Vector{UInt64}
    born::Int
    kind::Symbol
    decoder::Union{Nothing,Decoder}
    raw::Union{Nothing,Matrix{Float32},Matrix{Float16}}
    z::Matrix{Float32}
    residual::Union{Nothing,Packed}
end
mutable struct Store
    name::String
    mode::Symbol
    d::Int
    blocks::Vector{Block}
    hints::Dict{UInt64,Basis}
    encode_seconds::Float64
    maintenance_seconds::Float64
    expiry_seconds::Float64
    routed::Int
    rewrites::Int
end
Store(name,mode,d)=Store(name,mode,d,Block[],Dict{UInt64,Basis}(),0.,0.,0.,0,0)
specifications()=[("Float32 reference",:f32),("Float16 coordinates",:f16),
    ("8-bit coordinates",:q8),("4-bit coordinates",:q4),
    ("Centroid + 8-bit residual",:c8),("Centroid + 4-bit residual",:c4),
    ("Frozen local PPCA r4",:frozen),("Frozen batch PCA r16",:rank16),("PPCA r4 + 4-bit residual",:tail),
    ("Adaptive rank / 8-bit fallback",:adaptive),("Inherited basis + innovations",:inherited),
    ("Sequential PPCA r4",:sequential)]
function decode(b::Block)
    b.raw!==nothing && return Float32.(b.raw)
    d=b.decoder===nothing ? b.residual.d : length(b.decoder.mean)
    Y=b.decoder===nothing ? zeros(Float32,d,length(b.ids)) : repeat(b.decoder.mean,1,length(b.ids))
    !isempty(b.z) && (Y .+= basis_matrix(b.decoder.basis)*b.z)
    b.residual!==nothing && (Y .+= unpack_coordinates(b.residual))
    Y
end
function axis_values(b::Block,axis,indices)
    b.raw!==nothing && return Float64.(b.raw[axis,indices])
    y=fill(b.decoder===nothing ? 0. : Float64(b.decoder.mean[axis]),length(indices))
    !isempty(b.z) && (y .+= vec(Float64.(basis_matrix(b.decoder.basis)[axis:axis,:])*Float64.(b.z[:,indices])))
    if b.residual!==nothing
        for (j,k) in enumerate(indices);y[j]+=packed_coordinate(b.residual,axis,k);end
    end
    y
end
function block(ids,region,born,X,kind;decoder=nothing,bits=0)
    n=length(ids);regions=fill(UInt64(region),n);d=size(X,1)
    raw=kind==:f32 ? Float32.(X) : kind==:f16 ? Float16.(X) : nothing
    z=zeros(Float32,0,n);residual=nothing
    if raw===nothing
        R=decoder===nothing ? Float32.(X) : Float32.(X).-decoder.mean
        if decoder!==nothing && decoder.basis!==nothing
            U=basis_matrix(decoder.basis);z=U'*R;R-=U*z
        end
        bits>0 && (residual=pack_coordinates(R,bits))
    end
    Block(copy(ids),regions,born,kind,decoder,raw,z,residual)
end
function fitted_decoder(state,d)
    U=Float32.(state.U[:,1:min(4,size(state.U,2))])
    Decoder(Float32.(state.mean),Basis(nothing,U))
end
function ingest!(s::Store,ids,region,born,X,state)
    isempty(ids) && return
    n=length(ids);d=s.d;mode=s.mode
    b=if mode in (:f32,:f16)
        block(ids,region,born,X,mode)
    elseif mode in (:q4,:q8)
        block(ids,region,born,X,:quant;bits=mode==:q4 ? 4 : 8)
    elseif mode in (:c4,:c8)
        dec=Decoder(vec(Float32.(mean(X;dims=2))),nothing)
        block(ids,region,born,X,:quant;decoder=dec,bits=mode==:c4 ? 4 : 8)
    elseif mode in (:frozen,:sequential,:tail)
        dec=fitted_decoder(state,d)
        block(ids,region,born,X,:latent;decoder=dec,bits=mode==:tail ? 4 : 0)
    elseif mode==:rank16
        # Rank-matched independent-basis control for the inherited method.
        mu=vec(Float32.(mean(X;dims=2)));R=Float64.(X).-mu
        e=svd(R;full=false);r=min(16,d,max(n-1,0))
        dec=Decoder(mu,Basis(nothing,Float32.(e.U[:,1:r])))
        block(ids,region,born,X,:latent;decoder=dec)
    elseif mode==:adaptive
        # Batch-local, minimum numeric payload satisfying <=10% centered RMS
        # error. Rank candidates 0/2/4/8/16; quantized fallback needs no decoder.
        mu=vec(Float32.(mean(X;dims=2)));R=Float64.(X).-mu
        den=sum(abs2,R);e=svd(R;full=false)
        best=block(ids,region,born,X,:quant;bits=8)
        bestbytes=4n+d*n
        for r in unique(min.([0,2,4,8,16],min(d,max(n-1,0))))
            U=Float32.(e.U[:,1:r]);Z=U'*Float32.(R)
            err=sum(abs2,Float32.(R)-U*Z)
            cost=4d*(r+1)+4r*n+24 # new decoder and basis headers
            if err<=.01max(den,1e-30) && cost<bestbytes
                dec=Decoder(mu,Basis(nothing,U));best=block(ids,region,born,X,:latent;decoder=dec)
                bestbytes=cost
            end
        end
        best
    elseif mode==:inherited
        mu=vec(Float32.(mean(X;dims=2)));R=Float64.(X).-mu
        basis=get(s.hints,UInt64(region),nothing)
        if basis===nothing
            basis=Basis(nothing,Float32.(state.U[:,1:min(4,size(state.U,2))]))
        end
        U=Float64.(basis_matrix(basis));res=R-U*(U'*R)
        cap=min(16,d);available=cap-rankof(basis)
        if available>0 && sum(abs2,res)>.01max(sum(abs2,R),1e-30) && n>4
            e=svd(res;full=false);k=min(4,available,count(>(1e-7),e.S))
            if k>0
                # Twice remove the inherited span before QR; old basis is immutable.
                V=e.U[:,1:k];V-=U*(U'*V);V-=U*(U'*V)
                V=Matrix(qr(V).Q)[:,1:k]
                basis=Basis(basis,Float32.(V))
            end
        end
        s.hints[UInt64(region)]=basis
        block(ids,region,born,X,:latent;decoder=Decoder(mu,basis))
    else
        error("unknown codec")
    end
    push!(s.blocks,b)
end
function recode(b,indices,region,target)
    # Only the sequential control uses this. Reads previous codes, never originals.
    old=b.decoder;A=Float64.(basis_matrix(target.basis))'*Float64.(basis_matrix(old.basis))
    offset=Float64.(basis_matrix(target.basis))'*(Float64.(old.mean)-target.mean)
    z=Float32.(A*Float64.(b.z[:,indices]).+offset)
    Block(b.ids[indices],fill(UInt64(region),length(indices)),b.born,:latent,target,nothing,z,nothing)
end
function split!(s::Store,old,left,right,axis,cut,leftstate,rightstate)
    if s.mode==:sequential
        ld=fitted_decoder(leftstate,s.d);rd=fitted_decoder(rightstate,s.d);next=Block[]
        for b in s.blocks
            if first(b.regions)!=old;push!(next,b);continue;end
            vals=axis_values(b,axis,eachindex(b.ids));L=findall(<(cut),vals);R=findall(>=(cut),vals)
            !isempty(L) && push!(next,recode(b,L,left,ld));!isempty(R) && push!(next,recode(b,R,right,rd))
            s.routed+=length(vals);s.rewrites+=length(vals)
        end
        s.blocks=next
    else
        for b in s.blocks
            ids=findall(==(old),b.regions);isempty(ids) && continue
            vals=axis_values(b,axis,ids)
            for (j,k) in enumerate(ids);b.regions[k]=vals[j]<cut ? left : right;end
            s.routed+=length(ids)
        end
    end
    if haskey(s.hints,old)
        basis=pop!(s.hints,old);s.hints[left]=basis;s.hints[right]=basis
    end
end
function merge!(s::Store,left,right,parent,state)
    if s.mode==:sequential
        dec=fitted_decoder(state,s.d)
        for (i,b) in enumerate(s.blocks)
            first(b.regions) in (left,right) || continue
            s.blocks[i]=recode(b,eachindex(b.ids),parent,dec);s.rewrites+=length(b.ids)
        end
    else
        for b in s.blocks,j in eachindex(b.regions)
            b.regions[j] in (left,right) && (b.regions[j]=parent)
        end
    end
    a=pop!(s.hints,left,nothing);b=pop!(s.hints,right,nothing)
    chosen=rankof(a)>=rankof(b) ? a : b
    chosen!==nothing && (s.hints[parent]=chosen)
end
function expire!(s::Store,minimum_born)
    filter!(b->b.born>=minimum_born,s.blocks)
    live=Set{UInt64}(r for b in s.blocks for r in b.regions)
    filter!(kv->first(kv) in live,s.hints)
    s
end
function registry(s)
    bases=Basis[];decs=Decoder[];bi=IdDict{Basis,UInt32}();di=IdDict{Decoder,UInt32}()
    function visit(b)
        b===nothing && return
        haskey(bi,b) && return
        visit(b.parent);push!(bases,b);bi[b]=UInt32(length(bases))
    end
    for b in s.blocks
        dec=b.decoder;dec===nothing && continue
        visit(dec.basis)
        if !haskey(di,dec);push!(decs,dec);di[dec]=UInt32(length(decs));end
    end
    for b in values(s.hints);visit(b);end
    bases,decs,bi,di
end
function storage(s)
    bases,decs,_,_=registry(s);n=sum(length(b.ids) for b in s.blocks;init=0)
    decoderbytes=sum(16+sizeof(b.extra) for b in bases;init=0)+sum(8+sizeof(c.mean) for c in decs;init=0)
    codebytes=sum((b.raw===nothing ? 0 : sizeof(b.raw))+sizeof(b.z)+
        (b.residual===nothing ? 0 : sizeof(b.residual.scales)+sizeof(b.residual.bytes)) for b in s.blocks;init=0)
    metabytes=28+16length(s.blocks)+24n+12length(s.hints)
    weightedrank=sum(size(b.z,1)*length(b.ids) for b in s.blocks;init=0)/max(n,1)
    (posts=n,packed_bytes=decoderbytes+codebytes+metabytes,decoder_bytes=decoderbytes,code_bytes=codebytes,
        metadata_bytes=metabytes,runtime_bytes=Base.summarysize(s),blocks=length(s.blocks),decoders=length(decs),
        basis_nodes=length(bases),mean_rank=weightedrank)
end
"""Write measured numeric payload, including UUIDs, membership, sharing and headers.
Little-endian native machine format for experiment auditing, not a production API.
"""
function write_payload(io,s)
    bases,decs,bi,di=registry(s)
    write(io,codeunits("MPCODE01"),UInt32(s.d),UInt32(length(bases)),UInt32(length(decs)),UInt32(length(s.blocks)))
    for b in bases
        write(io,bi[b],b.parent===nothing ? UInt32(0) : bi[b.parent],UInt32(size(b.extra,1)),UInt32(size(b.extra,2)),b.extra)
    end
    for c in decs;write(io,di[c],c.basis===nothing ? UInt32(0) : bi[c.basis],c.mean);end
    for b in s.blocks
        format=b.raw isa Matrix{Float32} ? 1 : b.raw isa Matrix{Float16} ? 2 : 3
        write(io,UInt32(length(b.ids)),Int32(b.born),UInt8(format),UInt8(b.residual===nothing ? 0 : b.residual.bits),
            b.decoder===nothing ? UInt32(0) : di[b.decoder],UInt16(size(b.z,1)))
        write(io,UInt128[id.value for id in b.ids],b.regions)
        b.raw!==nothing && write(io,b.raw)
        write(io,b.z)
        b.residual!==nothing && write(io,b.residual.scales,b.residual.bytes)
    end
    write(io,UInt32(length(s.hints)))
    for (region,b) in sort!(collect(s.hints);by=first);write(io,region,bi[b]);end
end
end
