using JSON3
function inspect_dispatch(root)
    if isfile(joinpath(root,"index.sqlite")) || isfile(joinpath(root,"metadata.rocks","CURRENT"))
        include("disk_live_modules.jl")
        return Base.invokelatest() do
            s=DiskTreeStore.open_store(root)
            try
                (storage=DiskTreeStore.storage_stats(s),clock=s.clock,model=s.config.model,
                 keywords=DiskTreeStore.keyword_stats(s),
                 leaf_labels=[DiskTreeStore.leaf_keywords(s,n.id;limit=5,ranking=:common)
                     for n in sort(collect(values(s.nodes));by=n->n.id) if n.axis==0 && DiskTreeStore.keyword_enabled(s)])
            finally
                DiskTreeStore.close_store(s)
            end
        end
    end
    include("inspect_arrow_live.jl")
    Base.invokelatest(() -> report(root))
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    length(ARGS)==1 || error("usage: inspect_live.jl STORE")
    println(JSON3.write(inspect_dispatch(only(ARGS))))
end
