"""Versioned per-post orthogonal codes and forward coordinate migration.
This is an experimental storage layer, independent of the tree's mutable indices.
Decoder arrays are private snapshots by convention; never update them in place.
"""
module LatentPostCodes
using LinearAlgebra, UUIDs
export Decoder, PostCodes, encode_posts, decode_posts, coordinate_map, migrate_posts,
       split_posts, merge_posts, subset
struct Decoder
    id::UInt64
    mean::Vector{Float64}
    U::Matrix{Float64}
    function Decoder(id,mean,U)
        size(U,1)==length(mean) && all(isfinite,mean) && all(isfinite,U) || throw(ArgumentError("invalid decoder"))
        isapprox(U'*U,Matrix{Float64}(I,size(U,2),size(U,2));atol=1e-8,rtol=1e-8) || throw(ArgumentError("decoder basis must be orthonormal"))
        new(UInt64(id),copy(Float64.(mean)),copy(Float64.(U)))
    end
end
struct PostCodes{T<:AbstractFloat}
    ids::Vector{UUID}
    decoder::Decoder
    z::Matrix{T}
    function PostCodes(ids,decoder,z::AbstractMatrix{T}) where T<:AbstractFloat
        size(z)==(size(decoder.U,2),length(ids)) && all(isfinite,z) || throw(ArgumentError("invalid code dimensions/values"))
        length(unique(ids))==length(ids) || throw(ArgumentError("duplicate post IDs"))
        new{T}(copy(ids),decoder,Matrix(z))
    end
end
function encode_posts(decoder,ids,X;T=Float32)
    size(X)==(length(decoder.mean),length(ids)) || throw(ArgumentError("invalid input dimensions"))
    PostCodes(ids,decoder,T.(decoder.U'*(X.-decoder.mean)))
end
decode_posts(c::PostCodes)=c.decoder.mean.+c.decoder.U*Float64.(c.z)
function coordinate_map(old::Decoder,new::Decoder)
    length(old.mean)==length(new.mean) || throw(ArgumentError("ambient dimension mismatch"))
    (A=new.U'*old.U,b=new.U'*(old.mean-new.mean))
end
"""Forward projection of the old reconstruction into the new orthogonal basis.
No inverse, whitening or division by scatter eigenvalues. Lost residuals stay lost.
Computation is Float64; T controls code storage/rounding only.
"""
function migrate_posts(c,new;T=eltype(c.z))
    map=coordinate_map(c.decoder,new)
    PostCodes(c.ids,new,T.(map.A*Float64.(c.z).+map.b))
end
subset(c,indices)=PostCodes(c.ids[indices],c.decoder,c.z[:,indices])
function split_posts(c,left,right,axis,cut;T=eltype(c.z))
    1<=axis<=length(c.decoder.mean) && isfinite(cut) || throw(ArgumentError("invalid cut"))
    positions=c.decoder.mean[axis].+vec(c.decoder.U[axis:axis,:]*Float64.(c.z))
    mask=positions.<cut
    migrate_posts(subset(c,findall(mask)),left;T),migrate_posts(subset(c,findall(.!mask)),right;T)
end
function merge_posts(a,b,parent;T=promote_type(eltype(a.z),eltype(b.z)))
    l=migrate_posts(a,parent;T);r=migrate_posts(b,parent;T)
    PostCodes(vcat(l.ids,r.ids),parent,hcat(l.z,r.z))
end
end
