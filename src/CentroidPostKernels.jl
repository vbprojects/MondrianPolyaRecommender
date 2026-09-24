# Included inside LeafTreeKernels. Post storage has no dependency on PPCA bases.
export CentroidCodes, encode_centroid, decode_centroid, route_centroid,
       centroid_distances, kernel_weighted_update!

"""Frozen batch decoder and signed 8-bit residuals (stored as bytes offset by 127).
One Float32 scale per post. Never change the decoder when density/tree state changes.
"""
struct CentroidCodes
    center::Vector{Float32}
    scales::Vector{Float32}
    residuals::Matrix{UInt8}
end

@kernel function centroid_mean_kernel!(center,X)
    a=@index(Global,Linear)
    total=0.0
    for j in 1:size(X,2);total+=Float64(X[a,j]);end
    center[a]=Float32(total/size(X,2))
end
@kernel function centroid_scale_kernel!(scales,X,center)
    j=@index(Global,Linear);largest=0.0
    for a in 1:size(X,1);largest=max(largest,abs(Float64(X[a,j])-Float64(center[a])));end
    scales[j]=max(Float32(largest/127),floatmin(Float32))
end
@kernel function centroid_encode_kernel!(codes,X,center,scales)
    a,j=@index(Global,NTuple)
    delta=Float64(X[a,j])-Float64(center[a])
    codes[a,j]=UInt8(clamp(round(Int32,delta/Float64(scales[j])),-127,127)+127)
end
@inline function centroid_value(center,codes,scales,a,j)
    # Float64 intermediates make CPU/GPU reconstruction independent of Float32
    # multiply/add contraction. Persistent payload remains Float32 + UInt8.
    Float32(Float64(center[a])+Float64(Int32(codes[a,j])-127)*Float64(scales[j]))
end
@kernel function centroid_decode_kernel!(out,center,codes,scales,ids)
    a,j=@index(Global,NTuple);out[a,j]=centroid_value(center,codes,scales,a,ids[j])
end
@kernel function centroid_route_kernel!(out,center,codes,scales,ids,axes,cuts,left,right)
    j=@index(Global,Linear);i=1;row=ids[j]
    while axes[i]!=0
        i=centroid_value(center,codes,scales,axes[i],row)<cuts[i] ? left[i] : right[i]
    end
    out[j]=i
end
@kernel function centroid_distance_kernel!(out,center,codes,scales,ids,query)
    j=@index(Global,Linear);total=0.0
    for a in 1:length(center)
        delta=Float64(centroid_value(center,codes,scales,a,ids[j]))-Float64(query[a])
        total+=delta*delta
    end
    out[j]=total
end
function validate_codes(c::CentroidCodes)
    size(c.residuals)==(length(c.center),length(c.scales)) &&
        all(isfinite,c.center) && all(s->isfinite(s)&&s>0,c.scales) &&
        all(<=(0xfe),c.residuals) || throw(ArgumentError("invalid centroid codes"))
    c
end
function encode_centroid(k,X::AbstractMatrix)
    d,n=size(X)
    d>0 && n>0 && all(isfinite,X) || throw(ArgumentError("expected a nonempty finite embedding batch"))
    host=Float32.(X);all(isfinite,host) || throw(ArgumentError("embedding exceeds Float32 range"))
    dx=k.array(host);center=k.array(zeros(Float32,d));scales=k.array(zeros(Float32,n));codes=k.array(zeros(UInt8,d,n))
    launch(centroid_mean_kernel!,center,dx;ndrange=d)
    launch(centroid_scale_kernel!,scales,dx,center;ndrange=n)
    launch(centroid_encode_kernel!,codes,dx,center,scales;ndrange=(d,n))
    validate_codes(CentroidCodes(Array(center),Array(scales),Array(codes)))
end
function code_indices(c,indices)
    ids=Int.(collect(indices))
    all(i->1<=i<=length(c.scales),ids) || throw(BoundsError(c.scales,ids))
    ids
end
function decode_centroid(k,c::CentroidCodes;indices=eachindex(c.scales))
    ids=code_indices(c,indices);d=length(c.center)
    isempty(ids) && return zeros(Float32,d,0)
    out=k.array(zeros(Float32,d,length(ids)))
    launch(centroid_decode_kernel!,out,k.array(c.center),k.array(c.residuals),k.array(c.scales),k.array(ids);ndrange=size(out))
    Array(out)
end
function route_centroid(k,nodes,c::CentroidCodes;indices=eachindex(c.scales))
    length(c.center)==length(nodes[1].lo) || throw(DimensionMismatch("code/tree dimension mismatch"))
    ids=code_indices(c,indices);isempty(ids) && return Int[]
    out=k.array(zeros(Int,length(ids)))
    launch(centroid_route_kernel!,out,k.array(c.center),k.array(c.residuals),k.array(c.scales),k.array(ids),
        k.array([n.axis for n in nodes]),k.array([n.cut for n in nodes]),
        k.array([n.left for n in nodes]),k.array([n.right for n in nodes]);ndrange=length(ids))
    Array(out)
end
"""Squared Euclidean distances to selected compressed posts; no dense decode matrix."""
function centroid_distances(k,c::CentroidCodes,query;indices=eachindex(c.scales))
    length(query)==length(c.center) && all(isfinite,query) || throw(ArgumentError("invalid query"))
    ids=code_indices(c,indices);isempty(ids) && return Float64[]
    out=k.array(zeros(Float64,length(ids)))
    launch(centroid_distance_kernel!,out,k.array(c.center),k.array(c.residuals),k.array(c.scales),k.array(ids),k.array(Float64.(query));ndrange=length(ids))
    Array(out)
end

@kernel function weighted_segment_kernel!(mass,mu,diag,U,lam,nrank,X,weights,offsets,rows,V,Q,C,E,z,work)
    l=@index(Global,Linear)
    for at in offsets[l]:offsets[l+1]-1
        row=rows[at]
        if weights[row]>0
            update_one!(mass,mu,diag,U,lam,nrank,X,row,0.,Inf,V,Q,C,E,z,work,true,l,1.,weights[row])
        end
    end
end
"""Accumulate a batch of weighted observations at a common clock into region sketches.
Used to rebuild affected density states from retained codes after edits/deletions.
Preserves input order within each region; different regions run independently.
"""
function kernel_weighted_update!(k,states,X,groups,weights)
    d,n=size(X);K=length(states);n==0 && return states
    K>0 || throw(ArgumentError("no destination states"));R=states[1].rank
    length(groups)==n && length(weights)==n && all(i->1<=i<=K,groups) &&
        all(w->isfinite(w)&&w>=0,weights) && all(isfinite,X) || throw(ArgumentError("invalid weighted batch"))
    all(s->length(s.mean)==d&&s.rank==R,states) || throw(DimensionMismatch("incompatible sketches"))
    counts=zeros(Int,K);for i in groups;counts[i]+=1;end
    offsets=vcat(1,1 .+cumsum(counts));rows=zeros(Int,n);cursor=copy(offsets[1:end-1])
    for j in 1:n;i=groups[j];rows[cursor[i]]=j;cursor[i]+=1;end
    packed=zeros(2+2d+d*R+R,K);mass,mu,diag,U,lam,nr=packed_views(packed,d,R,K)
    for (i,s) in enumerate(states)
        mass[i]=s.mass;mu[:,i]=s.mean;diag[:,i]=s.diagonal;r=length(s.eigenvalues)
        U[1:d*r,i]=vec(s.U);lam[1:r,i]=s.eigenvalues;nr[i]=r
    end
    dev=k.array(packed);m,v,dd,uu,ll,rr=packed_views(dev,d,R,K)
    launch(weighted_segment_kernel!,m,v,dd,uu,ll,rr,k.array(X),k.array(Float64.(weights)),k.array(offsets),k.array(rows),
        k.array(zeros(d,K)),k.array(zeros(d,R+1,K)),k.array(zeros(R+1,R+1,K)),k.array(zeros(R+1,R+1,K)),
        k.array(zeros(R+1,K)),k.array(zeros(d,K));ndrange=K)
    copyto!(packed,Array(dev))
    for (i,s) in enumerate(states)
        r=Int(nr[i]);s.mass=mass[i];s.mean=copy(mu[:,i]);s.diagonal=copy(diag[:,i])
        s.U=reshape(copy(U[1:d*r,i]),d,r);s.eigenvalues=copy(lam[1:r,i])
    end
    states
end
