# Reuses the frozen model/proposals and trial design exported by proposal_experiment.jl.
root=normpath(joinpath(@__DIR__,".."))
python=get(ENV,"EMBEDDING_PYTHON",joinpath(root,".venv-embeddings/bin/python"))
run(`$python $(joinpath(root,"scripts/proposal_knn.py"))`)
run(`$python $(joinpath(root,"scripts/proposal_knn_report.py"))`)
