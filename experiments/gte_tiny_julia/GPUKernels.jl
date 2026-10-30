isdefined(@__MODULE__,:KernelGte) || include("KernelGte.jl")
module GPUKernels
using ..GteTiny, ..KernelGte
using CUDA, KernelAbstractions
export GPUModel, encode

struct DeviceDense
    weight::CuArray{Int32,2} # output × packed K/4 signed bytes
    scale::Float32
    bias::CuVector{Float32}
    sums::CuVector{Int32}
end
struct DeviceNorm
    weight::CuVector{Float32}
    bias::CuVector{Float32}
end
struct DeviceLayer
    query::DeviceDense
    key::DeviceDense
    value::DeviceDense
    attention::DeviceDense
    up::DeviceDense
    down::DeviceDense
    attnorm::DeviceNorm
    outnorm::DeviceNorm
end
struct GPUModel
    reference::GteTiny.Model
    word::CuMatrix{UInt8}
    position::CuMatrix{UInt8}
    token_type::CuMatrix{UInt8}
    norm::DeviceNorm
    layers::Vector{DeviceLayer}
end
function GPUModel(m::GteTiny.Model)
    CUDA.functional() || error("CUDA is unavailable")
    function device(d::GteTiny.QDense)
        packed=KernelGte.PackedDense(d).packed
        k4=size(packed,2);n=length(d.bias)
        words=permutedims(reshape(permutedims(packed,(2,1,3)),k4,n))
        DeviceDense(CuArray(reinterpret(Int32,words)),d.scale,CuArray(d.bias),CuArray(d.sums))
    end
    norm(n)=DeviceNorm(CuArray(n.weight),CuArray(n.bias))
    layers=[DeviceLayer((device(getfield(l,name)) for name in (:query,:key,:value,:attention,:up,:down))...,
                        norm(l.attnorm),norm(l.outnorm)) for l in m.layers]
    GPUModel(m,CuArray(m.word.weight),CuArray(m.position.weight),CuArray(m.token_type.weight),norm(m.norm),layers)
end

function launch(kernel,out,args...;ndrange=size(out),group=256)
    kernel(get_backend(out),group)(out,args...;ndrange)
    out
end

@kernel function embeddings!(x,word,position,typ,ids,seq,ws,wz,ps,pz,ts,tz)
    d,t=@index(Global,NTuple)
    j=mod1(t,seq)
    @inbounds begin
        w=Float32(Int32(word[d,ids[t]+1])-wz)*ws
        p=Float32(Int32(position[d,j])-pz)*ps
        tt=Float32(Int32(typ[d,1])-tz)*ts
        x[d,t]=(w+tt)+p
    end
end

@kernel function normalize!(x,weight,bias)
    t=@index(Global,Linear)
    mu=0f0
    @inbounds for d in 1:384
        mu+=x[d,t]
    end
    mu/=384f0; variance=0f0
    @inbounds for d in 1:384
        variance+=(x[d,t]-mu)^2
    end
    denom=sqrt(variance/384f0+1f-12)
    @inbounds for d in 1:384
        x[d,t]=((x[d,t]-mu)/denom)*weight[d]+bias[d]
    end
end
norm!(x,n)=launch(normalize!,x,n.weight,n.bias;ndrange=size(x,2),group=128)

function quantize(x)
    lo,hi=extrema(x) # device reduction; scalar transfer is recorded in end-to-end timing
    lo=min(0f0,lo);hi=max(0f0,hi)
    isfinite(lo)&&isfinite(hi) || error("nonfinite activation")
    scale=lo==hi ? 1f0 : (hi-lo)/255f0
    zero=Int32(clamp(round(-lo/scale),0f0,255f0))
    q=similar(x,UInt8)
    launch(KernelGte.quantize_kernel!,q,x,scale,zero;ndrange=length(q))
    q,scale,zero
end

@kernel function pack4!(words,a)
    k,t=@index(Global,NTuple)
    @inbounds words[k,t]=UInt32(a[4k-3,t]) | (UInt32(a[4k-2,t])<<8) |
        (UInt32(a[4k-1,t])<<16) | (UInt32(a[4k,t])<<24)
end
@kernel function dense_dp4a!(y,w,a,bias,sums,factor,zero)
    o,t=@index(Global,NTuple)
    acc=Int32(0)
    @inbounds for k in 1:size(a,1)
        acc=CUDA.dp4a(w[o,k],a[k,t],acc)
    end
    @inbounds y[o,t]=Float32(acc-zero*sums[o])*factor+bias[o]
end
function dense(d::DeviceDense,quant::Tuple)
    a,scale,zero=quant
    words=CuArray{UInt32}(undef,size(a,1)÷4,size(a,2))
    launch(pack4!,words,a)
    y=CuArray{Float32}(undef,length(d.bias),size(a,2))
    launch(dense_dp4a!,y,d.weight,words,d.bias,d.sums,scale*d.scale,zero;group=(32,4))
end
dense(d::DeviceDense,x)=dense(d,quantize(x))

@kernel function scores!(p,q,k,mask,factor,qz,kz,seq)
    key,query,hb=@index(Global,NTuple)
    head=mod1(hb,12); b=(hb-1)÷12+1
    offset=(head-1)*32;acc=Int32(0)
    @inbounds for d in 1:32
        acc+=(Int32(q[offset+d,(b-1)*seq+query])-qz)*(Int32(k[offset+d,(b-1)*seq+key])-kz)
    end
    @inbounds p[key,query,head,b]=Float32(acc)*factor/sqrt(32f0)+(mask[key,b] ? 0f0 : -floatmax(Float32))
end
@kernel function softmax!(p,seq)
    query,head,b=@index(Global,NTuple)
    best=-Inf32
    @inbounds for key in 1:seq
        best=max(best,p[key,query,head,b])
    end
    total=0f0
    @inbounds for key in 1:seq
        v=exp(p[key,query,head,b]-best)
        p[key,query,head,b]=v;total+=v
    end
    @inbounds for key in 1:seq
        p[key,query,head,b]/=total
    end
end
@kernel function attention_context_gpu!(x,p,v,seq,factor,pz,vz)
    d,query,b=@index(Global,NTuple)
    head=(d-1)÷32+1;acc=Int32(0)
    @inbounds for key in 1:seq
        acc+=(Int32(p[key,query,head,b])-pz)*(Int32(v[d,(b-1)*seq+key])-vz)
    end
    @inbounds x[d,(b-1)*seq+query]=Float32(acc)*factor
end
function attention(q,k,v,mask)
    seq,batch=size(mask)
    q8,qs,qz=quantize(q);k8,ks,kz=quantize(k);v8,vs,vz=quantize(v)
    p=CuArray{Float32}(undef,seq,seq,12,batch)
    launch(scores!,p,q8,k8,mask,qs*ks,qz,kz,seq;ndrange=(seq,seq,12batch),group=(16,8,1))
    launch(softmax!,p,seq;ndrange=(seq,12,batch),group=(32,1,1))
    p8,ps,pz=quantize(p)
    x=CuArray{Float32}(undef,384,seq*batch)
    launch(attention_context_gpu!,x,p8,v8,seq,ps*vs,pz,vz;ndrange=(384,seq,batch),group=(32,4,1))
end

@kernel function pooled!(out,x,mask,seq)
    d,b=@index(Global,NTuple)
    total=0f0;count=0
    @inbounds for j in 1:seq
        if mask[j,b]
            total+=x[d,(b-1)*seq+j];count+=1
        end
    end
    @inbounds out[d,b]=total/Float32(count)
end

function encode_batch(m::GPUModel,texts;max_tokens=256)
    ref=m.reference
    tokens=[GteTiny.tokenize(ref.tokenizer,t;max_tokens) for t in texts]
    seq=maximum(length,tokens);batch=length(tokens)
    ids=zeros(Int32,seq,batch);mask=fill(false,seq,batch)
    for (b,t) in enumerate(tokens)
        ids[1:length(t),b]=t;mask[1:length(t),b].=true
    end
    di=CuArray(ids);dm=CuArray(mask)
    x=CuArray{Float32}(undef,384,seq*batch)
    launch(embeddings!,x,m.word,m.position,m.token_type,di,seq,ref.word.scale,ref.word.zero,
        ref.position.scale,ref.position.zero,ref.token_type.scale,ref.token_type.zero)
    norm!(x,m.norm)
    for l in m.layers
        quant=quantize(x)
        q=dense(l.query,quant);k=dense(l.key,quant);v=dense(l.value,quant)
        a=dense(l.attention,attention(q,k,v,dm));a .+=x;norm!(a,l.attnorm)
        z=dense(l.up,a);launch(KernelGte.gelu_kernel!,z,Val(false);ndrange=length(z))
        x=dense(l.down,z);x .+=a;norm!(x,l.outnorm)
    end
    out=CuArray{Float32}(undef,384,batch)
    launch(pooled!,out,x,dm,seq)
    result=Array(out) # synchronizes all prior work before reporting elapsed time
    for b in 1:batch
        result[:,b]./=sqrt(sum(abs2,view(result,:,b)))
    end
    result
end
function encode(m::GPUModel,texts::AbstractVector{<:AbstractString};batch_size=8,max_tokens=256)
    1<=batch_size<=32 || throw(ArgumentError("batch_size must be in 1:32"))
    2<=max_tokens<=256 || throw(ArgumentError("max_tokens must be in 2:256"))
    out=Matrix{Float32}(undef,384,length(texts))
    for first in 1:batch_size:length(texts)
        last=min(first+batch_size-1,length(texts))
        out[:,first:last]=encode_batch(m,view(texts,first:last);max_tokens)
    end
    out
end
end
