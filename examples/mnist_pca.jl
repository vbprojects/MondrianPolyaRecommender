using MondrianPolyaToy, Serialization, LinearAlgebra, Statistics
include("load_mnist.jl")
include("export_utils.jl")
# Clip a convex polygon by a*u+b*v <= c.
function clip_poly(poly,a,b,c)
    isempty(poly) && return poly
    result=Tuple{Float64,Float64}[]
    previous=last(poly); fp=a*previous[1]+b*previous[2]-c
    for current in poly
        fc=a*current[1]+b*current[2]-c
        if (fp<=0)!=(fc<=0)
            q=fp/(fp-fc)
            push!(result,(previous[1]+q*(current[1]-previous[1]),previous[2]+q*(current[2]-previous[2])))
        end
        fc<=0 && push!(result,current)
        previous=current; fp=fc
    end
    result
end
function main()
    root=normpath(joinpath(@__DIR__,"..")); out=joinpath(root,"output/mnist_pca"); mkpath(out)
    data=load_mnist(); saved=deserialize(joinpath(root,"output/mnist/models.jls")); t=saved.partition
    BLAS.set_num_threads(4)
    A=Float64.(data.train); mu=vec(mean(A;dims=2)); A .-= mu
    println("Fitting PCA on training pixels only")
    covariance=Symmetric(A*A'/size(A,2)); eig=eigen(covariance)
    U=eig.vectors[:,end:-1:end-1]
    # Constant pixels have mathematically zero loadings in nonzero PCs.
    # Remove eigensolver roundoff so support checks do not reject tiny negatives.
    for j in axes(A,1)
        covariance[j,j]==0 && (U[j,:].=0.)
    end
    train=U'*A
    test=U'*(Float64.(data.test).-mu)
    explained=eig.values[end:-1:end-1]/sum(eig.values)
    # Bounds include all projected train/test points. Geometry is visualization only.
    limits=[minimum(vcat(vec(train[j,:]),vec(test[j,:]))) for j in 1:2]
    highs=[maximum(vcat(vec(train[j,:]),vec(test[j,:]))) for j in 1:2]
    for j in 1:2; pad=.04*(highs[j]-limits[j]); limits[j]-=pad; highs[j]+=pad; end
    box=[(limits[1],limits[2]),(highs[1],limits[2]),(highs[1],highs[2]),(limits[1],highs[2])]
    # The exact affine slice must also respect every original root bound.
    rootpoly=box
    for j in 1:784
        rootpoly=clip_poly(rootpoly,U[j,1],U[j,2],t.nodes[1].hi[j]-mu[j])
        rootpoly=clip_poly(rootpoly,-U[j,1],-U[j,2],mu[j]-t.nodes[1].lo[j])
    end
    N=length(t.nodes); labelcounts=zeros(Int,10,N); confusion=zeros(Int,4,N)
    # Evaluate the actual cut restricted to the PCA plane as a predictor of the
    # original cut. Balanced accuracy avoids trivial wins from lopsided splits.
    for row in axes(A,2)
        i=1
        while true
            n=t.nodes[i]; labelcounts[Int(data.train_labels[row])+1,i]+=1
            n.axis==0 && break
            actual=data.train[n.axis,row]<n.cut
            projected=mu[n.axis]+U[n.axis,1]*train[1,row]+U[n.axis,2]*train[2,row]<n.cut
            confusion[actual ? (projected ? 1 : 2) : (projected ? 3 : 4),i]+=1
            i=actual ? n.left : n.right
        end
    end
    agreement=ones(N)
    for i in 1:N
        a,b,c,d=confusion[:,i]
        t.nodes[i].axis!=0 && (agreement[i]=.5*(a/max(1,a+b)+d/max(1,c+d)))
    end
    majority=[argmax(view(labelcounts,:,i))-1 for i in 1:N]
    thresholds=[0.,.55,.60,.65,.70,.75,.80,.90]
    function assignment(x,active,threshold,scores=agreement)
        i=1
        while t.nodes[i].axis!=0 && active[i] && scores[i]>=threshold
            n=t.nodes[i]; i=x[n.axis]<n.cut ? n.left : n.right
        end
        i
    end
    models=NamedTuple[]; lines=["MNIST PCA visualization; PCA fit on 60000 training images only; tree unchanged.","PC1 variance fraction: $(explained[1]); PC2: $(explained[2])","mode,threshold,retained_groups,test_majority_accuracy"]
    for (name,active) in (("Reference",BitVector(n.axis!=0 for n in t.nodes)),("Ancestor residual",saved.residual.active))
        subtree_score=zeros(N)
        for i in N:-1:1
            n=t.nodes[i]
            if n.axis!=0 && active[i]
                subtree_score[i]=max(agreement[i],subtree_score[n.left],subtree_score[n.right])
            end
        end
        polygons=NamedTuple[]
        function slice(i,poly)
            length(poly)<3 && return
            n=t.nodes[i]
            if n.axis==0 || !active[i]
                center=sum(first,poly)/length(poly),sum(last,poly)/length(poly)
                reconstructed=mu+U*collect(center)
                @assert all(j->t.nodes[1].lo[j]-1e-8<=reconstructed[j]<=t.nodes[1].hi[j]+1e-8,1:784)
                @assert assignment(clamp.(reconstructed,0.,255.),active,0.)==i
                push!(polygons,(node=i,digit=majority[i],vertices=poly,count=n.count)); return
            end
            a,b=U[n.axis,1],U[n.axis,2]; c=n.cut-mu[n.axis]
            if abs(a)+abs(b)<1e-14
                slice(mu[n.axis]<n.cut ? n.left : n.right,poly)
            else
                slice(n.left,clip_poly(poly,a,b,c));slice(n.right,clip_poly(poly,-a,-b,-c))
            end
        end
        slice(1,rootpoly)
        levels=NamedTuple[]
        for threshold in thresholds
            groups=Int[]
            function visit(i)
                n=t.nodes[i]
                if n.axis==0 || !active[i] || subtree_score[i]<threshold; push!(groups,i)
                else; visit(n.left);visit(n.right);end
            end
            visit(1)
            ids=[assignment(view(data.test,:,row),active,threshold,subtree_score) for row in axes(data.test,2)]
            acc=mean(majority[ids].==data.test_labels)
            @assert all(in(Set(groups)),ids)
            push!(levels,(threshold=threshold,groups=length(groups),assignments=ids,accuracy=acc))
            push!(lines,join((name,threshold,length(groups),acc),','))
        end
        # Routing reconstructed plane points differs from routing originals.
        mismatch=0; inside=0
        for row in axes(test,2)
            x=mu+U*view(test,:,row)
            if all(v->0<=v<=255,x)
                inside+=1
                mismatch+=assignment(x,active,0.)!=assignment(view(data.test,:,row),active,0.)
            end
        end
        push!(models,(name=name,root_agreement=agreement[1],levels=levels,polygons=polygons,slice_inside=inside,slice_mismatch=mismatch))
    end
    payload=(explained=explained,bounds=(limits[1],highs[1],limits[2],highs[2]),
        points=[(test[1,i],test[2,i],Int(data.test_labels[i])) for i in axes(test,2)],
        majority=majority,models=models,rootpoly=rootpoly)
    write(joinpath(out,"demo.html"),replace(read(joinpath(@__DIR__,"mnist_pca_viewer.html"),String),"__DATA__"=>sprint(io->json(io,payload))))
    serialize(joinpath(out,"pca.jls"),(mean=mu,components=U,explained=explained,train_projection=train,test_projection=test,split_balanced_accuracy=agreement))
    write(joinpath(out,"summary.txt"),join(lines,'\n')*"\n")
    println(join(lines,'\n')); println("Exact slice polygons: ",[length(m.polygons) for m in models]);println("Open ",joinpath(out,"demo.html"))
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
