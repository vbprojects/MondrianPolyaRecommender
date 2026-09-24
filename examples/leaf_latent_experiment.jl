using Random, LinearAlgebra, Serialization, Statistics
include("../src/LeafLatentTree.jl")
using .LeafLatentTrees
include("load_bluesky.jl")
include("export_utils.jl")
function main()
    BLAS.set_num_threads(1)
    root=normpath(joinpath(@__DIR__,".."));out=joinpath(root,"output/leaf_latent");mkpath(out)
    X=load_bluesky_embeddings(joinpath(root,"data/bluesky_100000"));order=randperm(MersenneTwister(192),size(X,2));train=order[1:80000];test=order[80001:end]
    fit_leaf_tree(X[:,train[1:1024]];max_leaves=2) # warm compilation including splits
    elapsed=@elapsed fitted=fit_leaf_tree(X[:,train]);nodes=fitted.nodes;leaves=findall(n->n.axis==0,nodes)
    @assert all(n.axis==0 || (isempty(n.states)&&isempty(n.reservoir)) for n in nodes)
    @assert all(length(nodes[i].states)==5 for i in leaves)
    geometry=[(lo=n.lo,hi=n.hi,axis=n.axis,cut=n.cut,left=n.left,right=n.right) for n in nodes]
    shared_bytes=Base.summarysize(geometry);reservoir_bytes=Base.summarysize([nodes[i].reservoir for i in leaves])
    traincounts=zeros(length(nodes));testcounts=zeros(length(nodes))
    for row in train;traincounts[leaf_index(nodes,view(X,:,row))]+=1;end
    assignments=[leaf_index(nodes,view(X,:,row)) for row in test]
    for i in assignments;testcounts[i]+=1;end
    targets=test[1:64];proposals=NamedTuple[];metrics=NamedTuple[]
    for (k,(kind,rank)) in enumerate(fitted.configs)
        name=string(kind,"_r",rank);states=[nodes[i].states[k] for i in leaves]
        total=sum(s.mass for s in states);@assert isapprox(total,80000.;atol=1e-7)
        @assert all(all(isfinite,s.mean)&&all(isfinite,s.diagonal) for s in states)
        # Pólya probabilities from approximate transferred leaf masses.
        masses=zeros(length(nodes));for i in leaves;masses[i]=nodes[i].states[k].mass;end
        for i in length(nodes):-1:1
            n=nodes[i];n.axis!=0 && (masses[i]=masses[n.left]+masses[n.right])
        end
        probability=zeros(length(nodes));probability[1]=1.
        for (i,n) in enumerate(nodes)
            n.axis==0 && continue
            q=(n.cut-n.lo[n.axis])/(n.hi[n.axis]-n.lo[n.axis]);q=(2q+masses[n.left])/(2+masses[i])
            probability[n.left]=probability[i]*q;probability[n.right]=probability[i]*(1-q)
        end
        @assert isapprox(sum(probability[leaves]),1.;atol=1e-10)
        mse=0.;cosine=0.
        for (j,row) in enumerate(test)
            x=view(X,:,row);s=nodes[assignments[j]].states[k];y=reconstruction(s,x)
            mse+=sum(abs2,y-x)/length(x);cosine+=dot(y,x)/max(norm(y)*norm(x),1e-12)
        end
        allsamples=Vector{Vector{Vector{Float64}}}();draws=Int[];sample_seconds=@elapsed begin
            for (j,row) in enumerate(targets)
                i=leaf_index(nodes,view(X,:,row));samples,attempts=sample_leaf(MersenneTwister(1500+j),nodes[i],nodes[i].states[k],8)
                push!(allsamples,samples);push!(draws,attempts)
            end
        end
        received=sum(length,allsamples)
        params=[parameters(s) for s in states]
        decoder_bytes=sum(4*(length(p.mean)+length(p.D)+length(p.W))+8 for p in params)
        # Isotropic PPCA decoder needs one noise scalar, not 256 repeated values.
        kind==:ppca && (decoder_bytes-=length(leaves)*4*(size(X,1)-1))
        errors=fitted.errors[k]
        m=(name=name,leaves=length(leaves),mass=total,model_bytes=Base.summarysize(states)+shared_bytes+reservoir_bytes,
            shared_tree_bytes=shared_bytes,reservoir_bytes=reservoir_bytes,decoder_float32_bytes=decoder_bytes,
            update_seconds=fitted.update_seconds[k],split_seconds=fitted.split_seconds[k],
            test_rmse=sqrt(mse/length(test)),test_cosine=cosine/length(test),
            train_mass_tv=.5sum(abs.(masses[leaves]/total-traincounts[leaves]/sum(traincounts))),
            test_region_log_probability=sum(log(max(probability[i],1e-300)) for i in assignments)/length(assignments),
            received_samples=received,requested_samples=512,draws=sum(draws),acceptance=received/sum(draws),
            failed_regions=count(isempty,allsamples),sample_seconds=sample_seconds,
            max_split_mass_error=maximum(e.mass_error for e in errors),
            max_split_mean_error=maximum(e.mean_error for e in errors),
            max_conditional_diagonal_error=maximum(e.conditional_diagonal_error for e in errors),
            max_compressed_diagonal_error=maximum(e.compressed_diagonal_error for e in errors))
        push!(metrics,m);push!(proposals,(name=name,points=allsamples));println(m)
    end
    serialize(joinpath(out,"models.jls"),(fitted=fitted,train_rows=train,test_rows=test))
    payload=(target_bounds=[(lo=nodes[leaf_index(nodes,view(X,:,row))].lo,hi=nodes[leaf_index(nodes,view(X,:,row))].hi) for row in targets],train_rows=train.-1,test_rows=test.-1,target_rows=targets.-1,proposals=proposals,metrics=metrics,
        fit_wall_seconds=elapsed,configs=(max_leaves=64,split_after=512,reservoir_cap=64,seed=918,split_seed=192))
    write(joinpath(out,"experiment.json"),sprint(io->json(io,payload)))
    python=get(ENV,"EMBEDDING_PYTHON",joinpath(root,".venv-embeddings/bin/python"))
    run(`$python $(joinpath(root,"scripts/leaf_latent_report.py"))`)
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
