using MondrianPolyaToy, Serialization, Printf
include("load_mnist.jl")
include("export_utils.jl")
function mnist_leaves(t,m)
    ids=Int[]
    function visit(i)
        n=t.nodes[i]
        if n.axis==0 || (m!==nothing && !m.active[i]); push!(ids,i)
        else; visit(n.left); visit(n.right); end
    end
    visit(1); ids
end
function main()
    root=normpath(joinpath(@__DIR__,"..")); directory=joinpath(root,"data/mnist"); out=joinpath(root,"output/mnist"); mkpath(out)
    data=load_mnist(directory); X=data.train; T=data.test
    bounds=(zeros(784),fill(255.,784))
    warm=fit_partition(X[:,1:256];root_bounds=bounds,max_nodes=15)
    fit_compression(warm,X[:,1:256],:residual)
    elapsed=@elapsed t=fit_partition(X;root_bounds=bounds,seed=7,max_nodes=2047,min_count=128,max_depth=24)
    residual_time=@elapsed residual=fit_compression(t,X,:residual;split_count=128)
    serialize(joinpath(out,"models.jls"),(partition=t,residual=residual,config=(seed=7,max_nodes=2047,min_count=128,max_depth=24,root_bounds=(0,255)),preprocessing="flatten original UInt8 28x28 raster only"))
    summaries=String["MNIST: 60000 training / 10000 test images. 784 original UInt8 pixels (0..255). Flattening only.",
        "Training-only adaptive Mondrian-style heuristic; Pólya prior kappa=2. Same prototype as Bluesky, not exact Mondrian process.",
        "Support fixed a priori at [0,255]^784. No time decay. Ancestor residuals processed in original training file order.",
        "Labels used only AFTER fitting for histograms and training-leaf majority diagnostics; test labels never affect fit or predictions.",
        "Classification is a post-hoc majority readout, not a supervised tree or a calibrated class model.",
        "method,nodes,active_leaves,model_bytes,fit_seconds,train_purity,test_accuracy,integral_error,mean_test_log_density"]
    views=NamedTuple[]
    for (name,m,seconds) in (("reference",nothing,elapsed),("ancestor residual",residual,residual_time))
        leaves=mnist_leaves(t,m); p=embedding_probabilities(t,m)
        counts=zeros(Int,10,length(t.nodes)); examples=[Int[] for _ in t.nodes]; assignments=Int[]
        for row in axes(X,2)
            i=embedding_leaf(t,view(X,:,row),m===nothing ? nothing : m.active)
            push!(assignments,i); counts[Int(data.train_labels[row])+1,i]+=1
            length(examples[i])<6 && push!(examples[i],row)
        end
        predicted=[argmax(view(counts,:,i))-1 for i in eachindex(t.nodes)]
        # An empty active cell falls back to the training global majority label.
        globalmajor=argmax([count(==(UInt8(d)),data.train_labels) for d in 0:9])-1
        for i in leaves; sum(counts[:,i])==0 && (predicted[i]=globalmajor); end
        confusion=zeros(Int,10,10); logs=0.; testrows=String["test_row,leaf,label,prediction,log_density"]
        for row in axes(T,2)
            x=view(T,:,row); i=embedding_leaf(t,x,m===nothing ? nothing : m.active)
            label=Int(data.test_labels[row]); pred=predicted[i]
            confusion[label+1,pred+1]+=1
            ld=embedding_logdensity(t,x,m;probabilities=p); @assert isfinite(ld); logs+=ld
            push!(testrows,join((row-1,i,label,pred,ld),','))
        end
        purity=sum(maximum(view(counts,:,i)) for i in leaves)/size(X,2)
        accuracy=sum(confusion[i,i] for i in 1:10)/size(T,2)
        z=sum(p[leaves]); @assert isapprox(z,1.;atol=1e-10)
        @assert (m===nothing ? t.nodes[1].count : m.counts[1])==60000
        slug=m===nothing ? "reference" : "residual"
        write(joinpath(out,slug*"_test.csv"),join(testrows,'\n')*"\n")
        open(joinpath(out,slug*"_train.csv"),"w") do io
            println(io,"train_row,leaf,label")
            for row in axes(X,2); println(io,join((row-1,assignments[row],data.train_labels[row]),',')); end
        end
        records=[(id=i,depth=t.nodes[i].depth,probability=p[i],counts=counts[:,i],
            residual=m===nothing ? 0. : m.residual[i],
            images=[(row=row-1,label=Int(data.train_labels[row]),pixels=Int.(X[:,row])) for row in examples[i]]) for i in leaves]
        push!(views,(name=name,purity=purity,accuracy=accuracy,seconds=seconds,bytes=Base.summarysize(m===nothing ? t : m),leaves=records,confusion=[confusion[i,:] for i in 1:10]))
        push!(summaries,join((name,length(t.nodes),length(leaves),Base.summarysize(m===nothing ? t : m),seconds,purity,accuracy,abs(z-1),logs/size(T,2)),','))
    end
    write(joinpath(out,"summary.txt"),join(summaries,'\n')*"\n")
    open(joinpath(out,"manifest.json"),"w") do io
        json(io,(source=MNIST_BASE,preprocessing="flatten only; original UInt8 0..255",train_count=60000,test_count=10000,
            dimensions=784,seed=7,files=[(file=f*".gz",sha256=bytes2hex(sha256(read(joinpath(directory,f*".gz"))))) for f in MNIST_FILES]))
    end
    payload=sprint(io->json(io,views))
    write(joinpath(out,"demo.html"),replace(read(joinpath(@__DIR__,"mnist_viewer.html"),String),"__DATA__"=>payload))
    println(join(summaries,'\n')); println("Open ",joinpath(out,"demo.html"))
end
main()
