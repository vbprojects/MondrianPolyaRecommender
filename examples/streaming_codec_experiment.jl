using LinearAlgebra, Random, Statistics, UUIDs, Printf, SHA
include("../src/LeafLatentTree.jl")
include("../src/StreamingCodecs.jl")
include("export_utils.jl")
const LT=LeafLatentTrees
const SC=StreamingCodecs
rowid(id::UUID)=Int(id.value & UInt128(typemax(UInt64)))
uuid(seed,row)=UUID((UInt128(seed)<<64)|UInt128(row))
distance_matrix(X,Y)=max.(0.,sum(abs2,X;dims=1)'.+sum(abs2,Y;dims=1).-2Float64.(X)'*Float64.(Y))
h(p)=-(p*log2(max(p,1e-300))+(1-p)*log2(max(1-p,1e-300)))
function information_gain(D,prior,scale)
    P=0.05 .+ 0.9 ./ (1 .+ D/scale)
    marginal=vec(prior'*P)
    h.(marginal).-vec(prior'*h.(P))
end
function fixture(name,seed;rounds=40,batch=512)
    rng=MersenneTwister(seed+1700)
    if name=="Bluesky 100,000"
        path=joinpath(@__DIR__,"../data/bluesky_100000/embeddings.f32")
        X=open(path) do io
            reshape(read!(io,Vector{Float32}(undef,256*100000)),256,100000)
        end
        # Original archive row order is held constant; seeds vary tree and audits.
        return X,zeros(Float32,256,0),8,2,2500
    end
    d=name=="2D moving clusters" ? 2 : 32
    Q=Matrix(qr(randn(rng,d,d)).Q);X=zeros(Float32,d,rounds*batch)
    for step in 1:rounds
        if d==2
            angle=step*pi/16;mu=.3 .* [cos(angle),sin(angle)]
            Z=.06randn(rng,d,batch)
            for j in 1:batch;Z[:,j].+=rand(rng,Bool) ? mu : -mu;end
        elseif startswith(name,"Rotating")
            angle=step*pi/16;U=cos(angle)*Q[:,1:4]+sin(angle)*Q[:,5:8]
            Z=U*([.15,.12,.08,.04].*randn(rng,4,batch))+.001randn(rng,d,batch)
        elseif name=="Sixteen active directions"
            Z=.07Q[:,1:16]*randn(rng,16,batch)+.001randn(rng,d,batch)
        else
            Z=Q[:,1:4]*([.15,.12,.08,.04].*randn(rng,4,batch))+.0001randn(rng,d,batch)
        end
        @assert maximum(abs,Z)<1
        X[:,(step-1)*batch+1:step*batch]=Z
    end
    X,Float32.(Q[:,1:2]),4,name=="Rotating, long retention" ? 8 : 2,batch
end
function audit_setup(X,active,truthregions,seed,step)
    rng=MersenneTwister(seed*100000+step);n=length(active);nq=min(24,n-1)
    indices=randperm(rng,n)[1:nq];queries=X[:,indices]
    D=distance_matrix(queries,X)
    for (j,i) in enumerate(indices);D[j,i]=Inf;end
    exact=[partialsortperm(vec(D[j,:]),1:min(10,n-1)) for j in 1:nq]
    Q=zeros(Float32,size(X,1),nq);qregions=UInt64[];localneighbors=Vector{Int}[];globalneighbors=Vector{Int}[]
    for j in 1:nq
        a=indices[j];region=truthregions[a];candidates=findall(==(region),truthregions)
        b=rand(rng,candidates);Q[:,j]=.5X[:,a]+.5X[:,b];push!(qregions,region)
    end
    DP=distance_matrix(Q,X)
    for j in 1:nq
        candidates=findall(==(qregions[j]),truthregions)
        push!(localneighbors,candidates[partialsortperm(DP[j,candidates],1:min(10,length(candidates)))])
        push!(globalneighbors,partialsortperm(vec(DP[j,:]),1:min(10,n)))
    end
    ig=NamedTuple[]
    for _ in 1:4
        hypotheses=randperm(rng,n)[1:min(32,n)];candidates=randperm(rng,n)[1:min(96,n)]
        H=X[:,hypotheses];dist=distance_matrix(H,X[:,candidates]);scale=max(median(vec(dist)),1e-8)
        target=rand(rng,1:size(H,2));prior=exp.(-vec(distance_matrix(H[:,target:target],H))/(.25scale));prior./=sum(prior)
        truth=information_gain(dist,prior,scale)
        push!(ig,(H=H,candidates=candidates,scale=scale,prior=prior,truth=truth))
    end
    (queries=queries,indices=indices,exact=exact,proposals=Q,qregions=qregions,
        localneighbors=localneighbors,globalneighbors=globalneighbors,ig=ig)
end
function evaluate(store,X,active,truthregions,audit,f,regions)
    n=length(active);d=size(X,1);Y=zeros(Float32,d,n);members=zeros(UInt64,n);seen=falses(n)
    # IDs within the retained daily window are contiguous; audit-only lookup.
    firstrow=first(active)
    for b in store.blocks
        indices=[rowid(id)-firstrow+1 for id in b.ids]
        @assert all(1 .<= indices .<= n) && !any(seen[indices])
        seen[indices].=true;Y[:,indices]=SC.decode(b);members[indices]=b.regions
    end
    @assert all(seen) && all(isfinite,Y)
    D=distance_matrix(audit.queries,Y);recalls=Float64[]
    for (j,i) in enumerate(audit.indices)
        D[j,i]=Inf;nearest=partialsortperm(vec(D[j,:]),1:length(audit.exact[j]))
        push!(recalls,length(intersect(nearest,audit.exact[j]))/length(nearest))
    end
    DP=distance_matrix(audit.proposals,Y);localrec=Float64[];globalrec=Float64[]
    for j in axes(DP,1)
        candidates=findall(==(audit.qregions[j]),members);truth=audit.localneighbors[j]
        nearest=isempty(candidates) ? Int[] : candidates[partialsortperm(DP[j,candidates],1:min(length(truth),length(candidates)))]
        push!(localrec,length(intersect(nearest,truth))/length(truth))
        nearest=partialsortperm(vec(DP[j,:]),1:length(audit.globalneighbors[j]))
        push!(globalrec,length(intersect(nearest,audit.globalneighbors[j]))/length(nearest))
    end
    regrets=Float64[];agreement=Float64[]
    for trial in audit.ig
        scores=information_gain(distance_matrix(trial.H,Y[:,trial.candidates]),trial.prior,trial.scale)
        choice=argmax(scores);push!(regrets,maximum(trial.truth)-trial.truth[choice])
        push!(agreement,choice==argmax(trial.truth))
    end
    centered=Float64.(X).-mean(X;dims=2);den=sum(abs2,centered)
    outside=0
    node_by_region=Dict(regions[node.states[1]]=>node for node in f.nodes if node.axis==0)
    for j in 1:n
        node=node_by_region[members[j]];x=view(Y,:,j)
        outside+=any(x .< node.lo .- 1e-6) || any(x .>= node.hi .+ 1e-6)
    end
    metrics=(nrmse=sqrt(sum(abs2,Float64.(Y)-X)/max(den,1e-30)),recall10=mean(recalls),
        proposal_recall10=mean(globalrec),local_proposal_recall10=mean(localrec),
        ig_regret_bits=mean(regrets),ig_choice_agreement=mean(agreement),
        membership_error=mean(members.!=truthregions),outside_fraction=outside/n,
        variance_ratio=sum(abs2,Float64.(Y).-mean(Y;dims=2))/max(den,1e-30),ids_preserved=true)
    metrics,Y
end
function run_experiment(name,seed;rounds=40,batch=512,output=nothing,smoke=false)
    X,Q,perday,keepdays,batch=fixture(name,seed;rounds,batch)
    if name=="Bluesky 100,000";rounds=40;end
    smoke && (rounds=min(rounds,8);X=X[:,1:rounds*batch])
    d=size(X,1);ps=[SC.Store(n,m,d) for (n,m) in SC.specifications()]
    regions=IdDict{LT.Sketch,UInt64}();nextid=Ref(UInt64(0));counts=(splits=Ref(0),merges=Ref(0))
    region!(state)=get!(regions,state) do;nextid[]+=1;nextid[];end
    function children(s,axis,cut,L,R)
        left,right,err=LT.conditional_children(s,axis,cut,L,R)
        old=pop!(regions,s);li=region!(left);ri=region!(right);counts.splits[]+=1
        # Rotate timed codec order; identical inputs and tree for every method.
        for p in circshift(ps,counts.splits[]%length(ps))
            p.maintenance_seconds+=@elapsed SC.split!(p,old,li,ri,axis,cut,left,right)
        end
        left,right,err
    end
    function combine(a,b)
        parent=LT.merge_sketches(a,b);left=pop!(regions,a);right=pop!(regions,b);id=region!(parent);counts.merges[]+=1
        for p in circshift(ps,counts.merges[]%length(ps))
            p.maintenance_seconds+=@elapsed SC.merge!(p,left,right,id,parent)
        end
        parent
    end
    function reservoir(node,x,h,rng,cap)
        region!(node.states[1]);R=node.reservoir
        if size(R,2)<cap;return hcat(R,Float32.(x));end
        R[:,1:end-1].=R[:,2:end];R[:,end].=x;R
    end
    ops=(route=LT.leaf_index,advance=LT.advance_state!,decay=LT.decay!,reservoir=reservoir,
        proposal=LT.cut_proposal,children=children,merge=combine)
    f=nothing;records=NamedTuple[];frames=NamedTuple[];tree_seconds=0.;minimum_born=1
    maxleaves=name=="Bluesky 100,000" ? 32 : 16;splitafter=name=="Bluesky 100,000" ? 96 : 32
    peak_runtime=Dict(p.name=>0 for p in ps);peak_packed=copy(peak_runtime)
    function snapshot(step,phase)
        firstrow=(minimum_born-1)*perday*batch+1;active=collect(firstrow:step*batch);original=X[:,active]
        truthregions=[regions[f.nodes[LT.leaf_index(f.nodes,x)].states[1]] for x in eachcol(original)]
        audit=audit_setup(original,active,truthregions,seed,step)
        treememory=Base.summarysize((f.nodes,f.last_times,f.effective_arrivals,f.reservoir_rows))
        for p in ps
            metrics,Y=evaluate(p,original,active,truthregions,audit,f,regions);st=SC.storage(p)
            @assert st.posts==length(active)
            peak_runtime[p.name]=max(peak_runtime[p.name],st.runtime_bytes);peak_packed[p.name]=max(peak_packed[p.name],st.packed_bytes)
            rec=merge((case=name,seed=seed,step=step,day=cld(step,perday),phase=phase,method=p.name,
                d=d,arrivals=step*batch,retained=length(active),leaves=length(regions),splits=counts.splits[],merges=counts.merges[],
                tree_bytes=treememory,tree_seconds=tree_seconds,raw_input_batch_bytes=sizeof(X[:,1:batch]),
                encode_seconds=p.encode_seconds,maintenance_seconds=p.maintenance_seconds,expiry_seconds=p.expiry_seconds,
                routed=p.routed,rewrites=p.rewrites),st,metrics)
            push!(records,rec)
            if seed==1 && name=="Rotating rank-four manifold" && phase=="grown"
                ids=round.(Int,range(1,length(active);length=min(128,length(active))))
                push!(frames,(step=step,method=p.name,truth=[collect(Q'*original[:,j]) for j in ids],
                    points=[collect(Q'*Y[:,j]) for j in ids],metrics=rec))
            end
        end
    end
    for step in 1:rounds
        day=cld(step,perday);minimum_born=max(1,day-keepdays+1)
        if (step-1)%perday==0
            for p in ps;p.expiry_seconds+=@elapsed SC.expire!(p,minimum_born);end
        end
        rows=(step-1)*batch+1:step*batch;fresh=X[:,rows];start=f===nothing ? 0. : f.time
        times=start.+collect(range(.001,.25;length=batch))
        maintenance_before=sum(p.maintenance_seconds for p in ps)
        elapsed=@elapsed f=LT.fit_leaf_tree(fresh;timestamps=times,half_life=1.,configs=[(:ppca,min(4,d))],
            max_leaves=maxleaves,split_after=splitafter,reservoir_cap=64,seed=seed+91,operations=ops,_initial=f)
        tree_seconds+=max(0.,elapsed-(sum(p.maintenance_seconds for p in ps)-maintenance_before))
        # Compress each incoming microbatch in its final tree region. The original
        # microbatch is transient input; old originals are not available to codecs.
        assignment=[LT.leaf_index(f.nodes,x) for x in eachcol(fresh)]
        for nodeid in sort!(unique(assignment))
            indices=findall(==(nodeid),assignment);node=f.nodes[nodeid];region=regions[node.states[1]]
            ids=[uuid(seed,rows[j]) for j in indices];localX=fresh[:,indices]
            for p in circshift(ps,(step+seed)%length(ps))
                p.encode_seconds+=@elapsed SC.ingest!(p,ids,region,day,localX,node.states[1])
            end
        end
        for p in ps
            st=SC.storage(p);peak_runtime[p.name]=max(peak_runtime[p.name],st.runtime_bytes);peak_packed[p.name]=max(peak_packed[p.name],st.packed_bytes)
        end
        if step%perday==0 || step==rounds
            snapshot(step,"grown")
            maintenance_before=sum(p.maintenance_seconds for p in ps)
            elapsed=@elapsed f=LT.merge_tree!(f;time=f.time+16.,merge_below=3.,operations=ops)
            tree_seconds+=max(0.,elapsed-(sum(p.maintenance_seconds for p in ps)-maintenance_before))
            @assert length(f.nodes)==1 && length(regions)==1
            snapshot(step,"merged")
            println(name," seed ",seed," day ",day,": ",step*batch," arrivals; ",length(ps[1].blocks),
                " retained input groups; ",counts.splits[]," cuts / ",counts.merges[]," merges");flush(stdout)
        end
    end
    finalstats=NamedTuple[]
    for p in ps
        st=SC.storage(p)
        digest=""
        if output!==nothing && seed==1 && name=="Bluesky 100,000"
            file=joinpath(output,string(p.mode)*".bin")
            open(io->SC.write_payload(io,p),file,"w")
            @assert filesize(file)==st.packed_bytes
            # Independent binary reader verifies this reconstruction later.
            checksum=SHA.SHA2_256_CTX()
            for b in p.blocks
                SHA.update!(checksum,reinterpret(UInt8,UInt128[id.value for id in b.ids]))
                SHA.update!(checksum,reinterpret(UInt8,vec(SC.decode(b))))
            end
            digest=bytes2hex(SHA.digest!(checksum))
            write(joinpath(output,string(p.mode)*".sha256"),digest*"\n")
        end
        # End-of-retention drain must remove all post decoders and inherited chains.
        SC.expire!(p,typemax(Int));empty=SC.storage(p)
        @assert empty.posts==empty.decoders==empty.basis_nodes==0 && empty.packed_bytes==28
        push!(finalstats,(case=name,seed=seed,method=p.name,arrivals=rounds*batch,
            encode_seconds=p.encode_seconds,maintenance_seconds=p.maintenance_seconds,expiry_seconds=p.expiry_seconds,
            peak_runtime_bytes=peak_runtime[p.name],peak_packed_bytes=peak_packed[p.name],final_retained=st.posts,
            final_decoders=st.decoders,expired_all_decoders=true,tree_seconds=tree_seconds,
            splits=counts.splits[],merges=counts.merges[],keep_days=keepdays,rounds_per_day=perday,batch=batch,
            reconstruction_sha256=digest))
    end
    records,frames,finalstats
end
function main()
    BLAS.set_num_threads(1);out=normpath(joinpath(@__DIR__,"../output/streaming_codecs"));mkpath(out)
    smoke="--smoke" in ARGS
    println("Warm Julia code paths on an independent disposable stream.")
    run_experiment("Stable rank-four manifold",99;rounds=4,batch=64,smoke=true)
    records=NamedTuple[];frames=NamedTuple[];runs=NamedTuple[]
    cases=smoke ? ["Rotating rank-four manifold"] : ["2D moving clusters","Stable rank-four manifold",
        "Rotating rank-four manifold","Rotating, long retention","Sixteen active directions","Bluesky 100,000"]
    elapsed=@elapsed for name in cases,seed in 1:(smoke ? 1 : 3)
        a,b,c=run_experiment(name,seed;rounds=smoke ? 8 : 40,output=out,smoke)
        append!(records,a);append!(frames,b);append!(runs,c)
        write(joinpath(out,smoke ? "smoke.json" : "results.json"),sprint(io->json(io,(records=records,frames=frames,runs=runs))))
    end
    config=(seeds=smoke ? 1 : 3,methods=length(SC.specifications()),
        isolation="Every codec sees the same original-data tree. Historical originals are audit-only. Density is identical; codecs affect post retrieval and membership.",
        clock="Synthetic .25-half-life ingestion bursts and 16-half-life idle day endings; Bluesky uses archive order, not historical timestamps.",
        retention="Expire complete oldest ingestion days before new-day arrivals; keep 2 days, or 8 in the long-retention case. Density statistics retain an exponentially decayed tail.",
        batching="Each incoming batch is trained into the tree, then encoded by its final leaf. Fresh input may be used for centroid/rank fitting; no future batches or historical originals are read.",
        storage="Measured binary payload includes UUID128, UInt64 membership, decoder/basis sharing, scales and block headers. Julia reachable store memory is separate; shared tree and raw input batch reported separately. Text/catalog, audit truth and viewer history excluded.",
        metrics="Euclidean distances in original coordinates, 24 real-post queries and 24 within-leaf arithmetic midpoints per checkpoint. IG uses four synthetic beliefs (32 hypotheses, 96 candidates); not human feedback or Fermat shortest paths.",
        precision="Float32 decoders and latent coordinates; Float64 transport and metric arithmetic. No noise added during migration.",
        performance="Single Julia process, BLAS one thread, warmup, rotated codec order. Times include allocations/GC and exclude audits; unoptimized reference implementation, not production throughput.")
    payload=(records=records,frames=frames,runs=runs,configuration=config,seconds=elapsed)
    write(joinpath(out,smoke ? "smoke.json" : "results.json"),sprint(io->json(io,payload)))
    println("Finished ",length(runs)," method-runs in ",elapsed," seconds.")
end
if abspath(PROGRAM_FILE)==@__FILE__;main();end
