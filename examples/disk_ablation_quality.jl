# Reference evaluation deliberately runs outside the bounded storage worker.
include("disk_ablation_common.jl")
include("../src/MondrianPolyaToy.jl")
BLAS.set_num_threads(4)
function evaluate(mode,seed)
    name="$(seed)-$(mode)-append";s=DS.open_store(joinpath(STORE,name);budget=256*1024^2)
    N=DS.storage_stats(s).active;X=open(io->read!(io,Matrix{Float32}(undef,384,N)),joinpath(INPUT,"embeddings.f32"))
    Y=similar(X);members=zeros(Int64,N);true_members=zeros(Int64,N)
    for first in 1:256:N
        last=min(N,first+255);Y[:,first:last]=DS.fetch_vectors(s,collect(first:last))
        for p in DS.rows(s,"SELECT id,leaf FROM posts WHERE status=1 AND id BETWEEN ? AND ?",(first,last));members[p.id]=p.leaf;end
    end
    for i in 1:N;true_members[i]=DS.route(s,view(X,:,i));end
    disagreement=mean(members.!=true_members)
    rmse=sqrt(sum(abs2,Float64.(Y).-X)/length(X))
    norms=sqrt.(sum(abs2,Y;dims=1));Y./=max.(norms,eps(Float32))
    X./=sqrt.(sum(abs2,X;dims=1))
    distortion=zeros(Float64,N)
    for i in 1:N
        xy=0.;xx=0.;yy=0.
        for a in 1:384
            x=Float64(X[a,i]);y=Float64(Y[a,i]);xy+=x*y;xx+=x*x;yy+=y*y
        end
        distortion[i]=max(0.,1-xy/sqrt(xx*yy))
    end
    Q,ids=fixture_vectors(seed,64);Q./=sqrt.(sum(abs2,Q;dims=1))
    A=X'*Q;B=Y'*Q;recalls=Float64[];tie_recalls=Float64[];regions=Any[];empty_reference_regions=0
    for j in eachindex(ids)
        A[ids[j],j]=-Inf;B[ids[j],j]=-Inf
        ref=partialsortperm(view(A,:,j),1:10;rev=true);test=partialsortperm(view(B,:,j),1:10;rev=true)
        push!(recalls,length(intersect(ref,test))/10)
        threshold=minimum(A[ref,j]);push!(tie_recalls,mean(A[test,j].>=threshold-1e-6))
        for budget in (1,4,16)
            leaves=DS.candidate_leaves(s,view(Q,:,j);leaf_budget=budget)
            rids=filter(!=(ids[j]),findall(in(leaves),true_members))
            tids=filter(!=(ids[j]),findall(in(leaves),members))
            if isempty(rids);empty_reference_regions+=1;continue;end
            k=min(10,length(rids));kt=min(10,length(tids))
            rr=rids[partialsortperm(A[rids,j],1:k;rev=true)]
            tt=kt==0 ? Int[] : tids[partialsortperm(B[tids,j],1:kt;rev=true)]
            push!(regions,(query=j,leaf_budget=budget,reference_candidates=length(rids),decoded_candidates=length(tids),
                preservation=length(intersect(rr,tt))/k,global_recall=length(intersect(ref,tt))/10,
                float32_global_recall=length(intersect(ref,rr))/10))
        end
    end
    # Reuse the original entropy/response functions. Geometry is cosine-derived,
    # not a validated behavioral model or a high-dimensional Fermat solver.
    rng=MersenneTwister(seed+430);ig=Any[]
    for trial in 1:64
        h=randperm(rng,N)[1:32];c=randperm(rng,N)[1:96]
        belief=exp.(-(2 .-2 .* (X[:,h]'*X[:,h[1]]))/.2);belief./=sum(belief)
        distances=max.(0.,2 .-2 .* (X[:,h]'*X[:,c]));scale=max(median(distances),1e-8)
        original=Float64[];decoded=Float64[]
        for id in c
            d=max.(0.,2 .-2 .* (X[:,h]'*X[:,id]))
            e=max.(0.,2 .-2 .* (X[:,h]'*Y[:,id]))
            p=MondrianPolyaToy.response_probability.(d,scale)
            q=MondrianPolyaToy.response_probability.(e,scale)
            push!(original,MondrianPolyaToy.query_information(belief,p)/log(2))
            push!(decoded,MondrianPolyaToy.query_information(belief,q)/log(2))
        end
        a,b=argmax(original),argmax(decoded)
        push!(ig,(agreement=a==b,regret_bits=max(0.,original[a]-original[b]),best_bits=original[a]))
    end
    savejson(joinpath(OUT,"quality","$(seed)-$(mode).json"),
        (protocol_version=3,mode=mode,seed=seed,count=N,queries=64,empty_reference_regions=empty_reference_regions,cut_disagreement=disagreement,rmse=rmse,
         cosine_error_mean=mean(distortion),cosine_error_p99=quantile(distortion,.99),
         recall10=mean(recalls),tie_aware_recall10=mean(tie_recalls),
         score_error_mean=mean(abs.(A[isfinite.(A)].-B[isfinite.(A)])),
         regional=regions,ig=ig,ig_agreement=mean(x.agreement for x in ig),ig_regret_bits=mean(x.regret_bits for x in ig)))
    DS.close_store(s);println("QUALITY DONE ",seed," ",mode)
end
evaluate(Symbol(ARGS[1]),parse(Int,ARGS[2]))
