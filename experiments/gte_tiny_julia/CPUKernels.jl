# Included in KernelGte. Quantization and attention preserve the reference's
# integer arithmetic; no floating-point matrix multiply replaces INT8 products.
mutable struct Workspace
    arrays::Dict{Any,Any}
end
Workspace()=Workspace(Dict{Any,Any}())
function buffer(w::Workspace,::Type{T},key::Symbol,dims::NTuple{N,Int}) where {T,N}
    get!(w.arrays,(T,key,dims)) do
        Array{T}(undef,dims)
    end::Array{T,N}
end

# A separately tested optional approximation for x <= 0. Underflows below -87
# are flushed to zero; attention probabilities at that scale are negligible.
@inline function polynomial_exp(x::Float32)
    bounded=max(x,-87f0)
    n=unsafe_trunc(Int32,round(bounded*1.4426950408889634f0))
    r=(bounded-Float32(n)*0.693145751953125f0)-Float32(n)*1.428606765330187f-6
    p=muladd(r,1f0/5040f0,1f0/720f0)
    p=muladd(r,p,1f0/120f0);p=muladd(r,p,1f0/24f0)
    p=muladd(r,p,1f0/6f0);p=muladd(r,p,0.5f0)
    p=muladd(r,p,1f0);p=muladd(r,p,1f0)
    pow2=reinterpret(Float32,reinterpret(UInt32,n+Int32(127))<<23)
    ifelse(x < -87f0,0f0,p*pow2)
end
@inline exp_mode(x,::Val{false})=exp(x)
@inline exp_mode(x,::Val{true})=polynomial_exp(x)

@kernel function quantize_kernel!(q,x,scale,zero)
    i=@index(Global,Linear)
    @inbounds q[i]=unsafe_trunc(UInt8,clamp(round(x[i]/scale)+Float32(zero),0f0,255f0))
end

function quantize(x::AbstractArray{Float32},ws::Workspace=Workspace(),key=:dense_quant)
    lo=0f0;hi=0f0
    @inbounds @simd for i in eachindex(x)
        lo=min(lo,x[i]);hi=max(hi,x[i])
    end
    isfinite(lo)&&isfinite(hi) || error("nonfinite activation")
    scale=hi==lo ? 1f0 : (hi-lo)/255f0
    zero=Int32(clamp(round(-lo/scale),0f0,255f0))
    q=buffer(ws,UInt8,key,size(x))
    quantize_kernel!(CPU(),256)(q,x,scale,zero;ndrange=length(q))
    KernelAbstractions.synchronize(CPU())
    q,scale,zero
end

@kernel function attention_scores!(p,q,k,mask,factor,seq,math)
    query,head,b=@index(Global,NTuple)
    offset=(head-1)*32;colq=(b-1)*seq+query
    best=-Inf32
    @inbounds begin
        for key in 1:seq
            acc=Int32(0);colk=(b-1)*seq+key
            @simd for d in 1:32
                acc+=Int32(q[offset+d,colq])*Int32(k[offset+d,colk])
            end
            score=Float32(acc)*factor/sqrt(32f0)
            mask[key,b] || (score+=-floatmax(Float32))
            p[key,query,head,b]=score;best=max(best,score)
        end
        total=0f0
        for key in 1:seq
            value=exp_mode(p[key,query,head,b]-best,math)
            p[key,query,head,b]=value;total+=value
        end
        for key in 1:seq
            p[key,query,head,b]/=total
        end
    end
end

@kernel function attention_context!(context,p,v,seq,factor)
    query,head,b=@index(Global,NTuple)
    offset=(head-1)*32
    @inbounds for d in 1:32
        acc=Int32(0)
        @simd for key in 1:seq
            acc+=Int32(p[key,query,head,b])*Int32(v[key,offset+d,b])
        end
        context[offset+d,(b-1)*seq+query]=Float32(acc)*factor
    end
end

@kernel function pack_qk!(qw,kw,qsums,ksums,q,k,seq)
    t,hb=@index(Global,NTuple)
    head=mod1(hb,12);b=(hb-1)÷12+1;offset=(head-1)*32
    sq=Int32(0);sk=Int32(0)
    @inbounds for chunk in 1:8
        qword=UInt32(0);kword=UInt32(0)
        for byte in 0:3
            d=offset+4(chunk-1)+byte+1
            qa=q[d,(b-1)*seq+t];ka=k[d,(b-1)*seq+t]
            sq+=Int32(qa);sk+=Int32(ka)-128
            qword|=UInt32(qa)<<(8byte)
            kword|=UInt32(ka ⊻ 0x80)<<(8byte)
        end
        qw[chunk,t,hb]=qword
        kw[mod1(t,16),chunk,(t-1)÷16+1,hb]=kword
    end
    @inbounds qsums[t,hb]=sq
    @inbounds ksums[t,hb]=sk
end

@inline function save_qk!(p,c,ksums,qsums,mask,factor,qz,kz,seq,kb,t,hb)
    head=mod1(hb,12);b=(hb-1)÷12+1
    @inbounds for lane in 1:16
        key=16(kb-1)+lane
        if key<=seq
            acc=c[lane].value-qz*ksums[key,hb]+(128-kz)*(qsums[t,hb]-32qz)
            score=Float32(acc)*factor/sqrt(32f0)
            p[key,t,head,b]=mask[key,b] ? score : score-floatmax(Float32)
        end
    end
end

@kernel function vnni_qk!(p,kw,qw,ksums,qsums,mask,factor,qz,kz,seq)
    kb,tb,hb=@index(Global,NTuple)
    first=4(tb-1)+1;offset=16*8*(kb-1+size(kw,3)*(hb-1))
    if first+3<=seq
        c1=zero16();c2=zero16();c3=zero16();c4=zero16()
        @inbounds for k in 1:8
            w=load16(kw,offset+16(k-1))
            c1=dpbusd(c1,splat16(qw[k,first,hb]),w)
            c2=dpbusd(c2,splat16(qw[k,first+1,hb]),w)
            c3=dpbusd(c3,splat16(qw[k,first+2,hb]),w)
            c4=dpbusd(c4,splat16(qw[k,first+3,hb]),w)
        end
        save_qk!(p,c1,ksums,qsums,mask,factor,qz,kz,seq,kb,first,hb)
        save_qk!(p,c2,ksums,qsums,mask,factor,qz,kz,seq,kb,first+1,hb)
        save_qk!(p,c3,ksums,qsums,mask,factor,qz,kz,seq,kb,first+2,hb)
        save_qk!(p,c4,ksums,qsums,mask,factor,qz,kz,seq,kb,first+3,hb)
    else
        for t in first:seq
            c=zero16()
            @inbounds for k in 1:8
                c=dpbusd(c,splat16(qw[k,t,hb]),load16(kw,offset+16(k-1)))
            end
            save_qk!(p,c,ksums,qsums,mask,factor,qz,kz,seq,kb,t,hb)
        end
    end
end

@kernel function softmax_cpu!(p,seq,math)
    t,h,b=@index(Global,NTuple)
    best=-Inf32
    @inbounds @simd for k in 1:seq
        best=max(best,p[k,t,h,b])
    end
    @inbounds @simd for k in 1:seq
        p[k,t,h,b]=exp_mode(p[k,t,h,b]-best,math)
    end
    total=0f0
    # Separate the elementwise pass from scalar-order summation so polynomial
    # evaluation can vectorize without reassociating the probability sum.
    @inbounds for k in 1:seq
        total+=p[k,t,h,b]
    end
    @inbounds @simd for k in 1:seq
        p[k,t,h,b]/=total
    end
end

function attention(q,k,v,mask,ws::Workspace=Workspace(),math=Val(false),vnni=false)
    seq,batch=size(mask)
    q8,qs,qz=quantize(q,ws,:qquant);k8,ks,kz=quantize(k,ws,:kquant);v8,vs,vz=quantize(v,ws,:vquant)
    # Attention's reduction is over tokens. Make that axis contiguous for SIMD.
    vb=buffer(ws,Int16,:vb,size(v));vb .= Int16.(v8).-Int16(vz)
    vc=buffer(ws,Int16,:vc,(seq,384,batch))
    permutedims!(vc,reshape(vb,384,seq,batch),(2,1,3))
    p=buffer(ws,Float32,:probs,(seq,seq,12,batch))
    if vnni
        qw=buffer(ws,UInt32,:qwords,(8,seq,12batch))
        kw=buffer(ws,UInt32,:kwords,(16,8,cld(seq,16),12batch));fill!(kw,0)
        qsums=buffer(ws,Int32,:qsums,(seq,12batch));ksums=buffer(ws,Int32,:ksums,(seq,12batch))
        pack_qk!(CPU(),(16,1))(qw,kw,qsums,ksums,q8,k8,seq;ndrange=(seq,12batch))
        KernelAbstractions.synchronize(CPU())
        vnni_qk!(CPU(),(2,4,1))(p,kw,qw,ksums,qsums,mask,qs*ks,qz,kz,seq;
            ndrange=(cld(seq,16),cld(seq,4),12batch))
        KernelAbstractions.synchronize(CPU())
        softmax_cpu!(CPU(),(8,1,1))(p,seq,math;ndrange=(seq,12,batch))
        KernelAbstractions.synchronize(CPU())
    else
        qc=buffer(ws,Int16,:qc,size(q));kc=buffer(ws,Int16,:kc,size(k))
        qc .= Int16.(q8).-Int16(qz);kc .= Int16.(k8).-Int16(kz)
        attention_scores!(CPU(),(8,1,1))(p,qc,kc,mask,qs*ks,seq,math;ndrange=(seq,12,batch))
        KernelAbstractions.synchronize(CPU())
    end
    p8,ps,pz=quantize(p,ws,:pquant)
    pz==0 || error("nonnegative probabilities must have zero zero-point")
    context=buffer(ws,Float32,:context,(384,seq*batch))
    attention_context!(CPU(),(8,1,1))(context,p8,vc,seq,ps*vs;ndrange=(seq,12,batch))
    KernelAbstractions.synchronize(CPU())
    context
end

@inline function gelu_polynomial(x)
    z=x/sqrt(2f0);t=1f0/(1f0+0.3275911f0*abs(z))
    poly=((((1.061405429f0*t-1.453152027f0)*t+1.421413741f0)*t-0.284496736f0)*t+0.254829592f0)*t
    e=copysign(1f0-poly*polynomial_exp(-z*z),z)
    (x*(1f0+e))*0.5f0
end
@inline gelu_mode(x,::Val{false})=GteTiny.gelu(x)
@inline gelu_mode(x,::Val{true})=gelu_polynomial(x)
@kernel function gelu_kernel!(x,math=Val(false))
    i=@index(Global,Linear)
    @inbounds x[i]=gelu_mode(x[i],math)
end
function gelu!(x,math=Val(false))
    gelu_kernel!(CPU(),256)(x,math;ndrange=length(x))
    KernelAbstractions.synchronize(CPU())
    x
end
