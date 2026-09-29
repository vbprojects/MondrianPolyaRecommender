"""Immutable, self-contained codec frames. Relocation slices bytes and copies decoders.
The on-disk v1 numeric format is little-endian; big-endian hosts are rejected.
"""
module DiskCodecs
using LinearAlgebra, Statistics, SHA
export Frame, encode, decode, subset, concatenate, frame_bytes, read_frame, signature,
       payload_stats, decoder_key, FORMATS, FORMAT_NAMES
const FORMATS=[:f32,:f16,:q8,:centroid_shared,:centroid_leaf,:pca16]
const FORMAT_NAMES=Dict(:f32=>"Float32",:f16=>"Float16",:q8=>"Per-vector Int8",
    :centroid_shared=>"Current centroid Int8",:centroid_leaf=>"Leaf-local centroid Int8",:pca16=>"Frozen leaf PCA r16")
const MAGIC=UInt32(0x4d504346)
struct Frame
    mode::Symbol
    d::Int
    n::Int
    rank::Int
    center::Vector{Float32}
    basis::Matrix{Float32}
    scales::Vector{Float32}
    payload::Vector{UInt8}
end
rawbytes(A)=copy(reinterpret(UInt8,vec(A)))
function encode(mode::Symbol,X)
    mode in FORMATS || throw(ArgumentError("unknown codec"))
    d,n=size(X);d>0 && 0<n<=256 && all(isfinite,X) || throw(ArgumentError("invalid codec group"))
    x=Float32.(X);center=Float32[];basis=zeros(Float32,d,0);scales=Float32[];r=0
    if mode==:f32
        payload=rawbytes(x)
    elseif mode==:f16
        payload=rawbytes(Float16.(x))
    elseif mode==:pca16
        center=vec(mean(x;dims=2));Y=Float64.(x).-center
        r=min(16,d,max(0,n-1));basis=r==0 ? zeros(Float32,d,0) : Float32.(svd(Y;full=false).U[:,1:r])
        payload=rawbytes(basis'*(x.-center))
    else
        mode!=:q8 && (center=vec(mean(x;dims=2));x=x.-center)
        scales=Float32[max(maximum(abs,view(x,:,j))/127,floatmin(Float32)) for j in 1:n]
        values=UInt8[clamp(round(Int,Float64(x[a,j])/scales[j]),-127,127)+127 for a in 1:d,j in 1:n]
        payload=rawbytes(values)
    end
    Frame(mode,d,n,r,center,basis,scales,payload)
end
rowwidth(f::Frame)=f.mode==:f32 ? 4f.d : f.mode==:f16 ? 2f.d : f.mode==:pca16 ? 4f.rank : f.d
function decode(f::Frame;rows=1:f.n)
    ids=collect(rows);all(i->1<=i<=f.n,ids) || throw(BoundsError())
    X=zeros(Float32,f.d,length(ids));w=rowwidth(f)
    for (j,i) in enumerate(ids)
        if f.mode in (:f32,:f16)
            T=f.mode==:f32 ? Float32 : Float16
            X[:,j]=reinterpret(T,f.payload[(i-1)*w+1:i*w])
        elseif f.mode==:pca16
            z=reinterpret(Float32,f.payload[(i-1)*w+1:i*w])
            # Fixed accumulation order: block slicing must not change GEMM rounding.
            for a in 1:f.d
                v=Float64(f.center[a])
                for k in 1:f.rank;v+=Float64(f.basis[a,k])*Float64(z[k]);end
                X[a,j]=Float32(v)
            end
        else
            for a in 1:f.d
                base=isempty(f.center) ? 0. : Float64(f.center[a])
                X[a,j]=Float32(base+(Int(f.payload[(i-1)*w+a])-127)*Float64(f.scales[i]))
            end
        end
    end
    X
end
function subset(f::Frame,rows)
    ids=collect(rows);all(i->1<=i<=f.n,ids) || throw(BoundsError())
    w=rowwidth(f);payload=UInt8[];sizehint!(payload,length(ids)*w)
    for i in ids;append!(payload,view(f.payload,(i-1)*w+1:i*w));end
    Frame(f.mode,f.d,length(ids),f.rank,copy(f.center),copy(f.basis),isempty(f.scales) ? Float32[] : f.scales[ids],payload)
end
decoder_key(f::Frame)=bytes2hex(sha256(vcat(UInt8[findfirst(==(f.mode),FORMATS)],rawbytes(f.center),rawbytes(f.basis))))
function concatenate(a::Frame,b::Frame)
    a.mode==b.mode && a.d==b.d && a.rank==b.rank && a.center==b.center && a.basis==b.basis || error("incompatible frozen decoders")
    a.n+b.n<=256 || error("frame too large")
    Frame(a.mode,a.d,a.n+b.n,a.rank,a.center,a.basis,vcat(a.scales,b.scales),vcat(a.payload,b.payload))
end
function frame_bytes(f::Frame)
    ENDIAN_BOM==0x04030201 || error("v1 requires little-endian host")
    io=IOBuffer();write(io,MAGIC,UInt8(1),UInt8(findfirst(==(f.mode),FORMATS)),UInt16(f.d),UInt16(f.n),UInt16(f.rank))
    write(io,f.center,vec(f.basis),f.scales,f.payload);take!(io)
end
function read_frame(bytes)
    io=IOBuffer(bytes);read(io,UInt32)==MAGIC && read(io,UInt8)==1 || error("invalid frame header")
    k=Int(read(io,UInt8));1<=k<=length(FORMATS) || error("invalid codec")
    mode=FORMATS[k];d=Int(read(io,UInt16));n=Int(read(io,UInt16));r=Int(read(io,UInt16))
    # A relocated fragment can contain fewer rows than its frozen PCA rank.
    d>0 && 0<n<=256 && 0<=r<=min(16,d) && (mode==:pca16 || r==0) || error("invalid frame dimensions")
    center=mode in (:centroid_shared,:centroid_leaf,:pca16) ? read!(io,zeros(Float32,d)) : Float32[]
    basis=read!(io,zeros(Float32,d,r));scales=mode in (:q8,:centroid_shared,:centroid_leaf) ? read!(io,zeros(Float32,n)) : Float32[]
    f=Frame(mode,d,n,r,center,basis,scales,read(io))
    length(f.payload)==n*rowwidth(f) || error("invalid payload length")
    all(isfinite,center) && all(isfinite,basis) && all(s->isfinite(s)&&s>0,scales) || error("invalid decoder")
    f
end
signature(f::Frame,row)=bytes2hex(sha256(frame_bytes(subset(f,[row]))))
payload_stats(f::Frame)=(vector_bytes=length(f.payload)+sizeof(f.scales),decoder_bytes=sizeof(f.center)+sizeof(f.basis),header_bytes=12,
    posts=f.n,rank=f.rank)
end
