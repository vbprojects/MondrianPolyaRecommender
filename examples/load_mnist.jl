using Downloads, SHA
const MNIST_FILES=("train-images-idx3-ubyte","train-labels-idx1-ubyte","t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")
const MNIST_BASE="https://storage.googleapis.com/cvdf-datasets/mnist/"
function mnist_download(directory)
    mkpath(directory)
    for file in MNIST_FILES
        gz=joinpath(directory,file*".gz")
        if !isfile(gz)
            println("Downloading ",file)
            Downloads.download(MNIST_BASE*file*".gz",gz*".tmp")
            mv(gz*".tmp",gz;force=true)
        end
        dest=joinpath(directory,file)
        if !isfile(dest)
            open(dest*".tmp","w") do io
                run(pipeline(`gzip -dc $gz`,stdout=io))
            end
            mv(dest*".tmp",dest;force=true)
        end
    end
end
"""Read original IDX bytes as 784×N; column order is image row-major raster.
No normalization, centering, binarization, projection, or feature extraction.
"""
function read_mnist_images(path)
    open(path) do io
        ntoh(read(io,UInt32))==2051 || error("invalid image IDX magic")
        n=Int(ntoh(read(io,UInt32))); h=Int(ntoh(read(io,UInt32))); w=Int(ntoh(read(io,UInt32)))
        h==28 && w==28 && n>0 || error("invalid MNIST shape")
        pixels=read(io); length(pixels)==784n || error("invalid image payload length")
        reshape(pixels,784,n)
    end
end
function read_mnist_labels(path)
    open(path) do io
        ntoh(read(io,UInt32))==2049 || error("invalid label IDX magic")
        n=Int(ntoh(read(io,UInt32))); y=read(io)
        length(y)==n && n>0 && all(v->v<=9,y) || error("invalid MNIST labels")
        y
    end
end
function load_mnist(directory=joinpath(@__DIR__,"../data/mnist");download=true)
    download && mnist_download(directory)
    X=read_mnist_images(joinpath(directory,MNIST_FILES[1])); y=read_mnist_labels(joinpath(directory,MNIST_FILES[2]))
    T=read_mnist_images(joinpath(directory,MNIST_FILES[3])); z=read_mnist_labels(joinpath(directory,MNIST_FILES[4]))
    size(X,2)==length(y)==60000 || error("expected 60000 training images")
    size(T,2)==length(z)==10000 || error("expected 10000 test images")
    (train=X,train_labels=y,test=T,test_labels=z)
end
