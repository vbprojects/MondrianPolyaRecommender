# Julia entry point; ONNX/tokenizer operations run in an isolated Python bridge.
const root = normpath(joinpath(@__DIR__, ".."))
python = get(ENV, "EMBEDDING_PYTHON", joinpath(root, ".venv-embeddings", "bin", "python"))
isfile(python) || error("Create .venv-embeddings and install scripts/requirements-embeddings.txt; see docs/Bluesky_Embeddings.md")
run(`$python $(joinpath(root,"scripts","embed_bluesky.py")) $ARGS`)
