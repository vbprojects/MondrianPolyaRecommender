# Repeatable later-time structural probe for an already measured experiment store.
include("disk_ablation_common.jl")
include("keyword_snapshot.jl")
method=only(ARGS)
method in ("disabled","hashtags","yake") || error("invalid method")
path=joinpath(ROOT,"output/keywords/$method.json")
r=JSON3.read(read(path,String),Dict{String,Any})
warmup()
s=DS.open_store(joinpath(ROOT,"data/keyword_experiment",method);budget=32*1024^2)
target=1000. + 72*3600
s.clock<target || error("maintenance already applied")
r["merge_seconds"]=@elapsed DS.maintain!(s;time=target)
r["maintenance_condition"]="separate process, reopened store, warmed numerical kernels"
r["maintenance_metrics"]=copy(s.metrics)
r["metrics"]=mergewith(+,r["metrics"],s.metrics)
if method!="disabled"
    r["snapshots"][end]=snapshot(s,"After 72 hours of decay / merges")
end
r["keyword_stats"]=DS.keyword_stats(s)
savejson(path,r);DS.close_store(s)
