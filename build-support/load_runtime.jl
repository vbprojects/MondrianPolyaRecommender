# Called in the trace process and the final --output-o process, never on launch.
const staged_runtime=ENV["MONDRIAN_BUILD_RUNTIME"]
isdefined(Main,:MondrianPolyaToy) || Base.include(Main,joinpath(staged_runtime,"src","MondrianPolyaToy.jl"))
isdefined(Main,:MondrianServer) || Base.include(Main,joinpath(staged_runtime,"server","src","MondrianServer.jl"))
