# Bluesky text and embeddings

## 100,000-post GPU corpus

The expanded corpus is ready in `data/bluesky_100000/`: **100,000 distinct-URI posts with original text and 256-dimensional, unit-normalized Float32 embeddings**. It uses the same pinned dataset and Model2Vec model as the initial experiment below.

- `posts.jsonl` contains original text and metadata, source shard/record, token count, and `embedding_row` (zero-based).
- `embeddings.f32` contains aligned post-major vectors (102,400,000 bytes). Julia loads it as a **256 × 100,000** matrix; column `j` corresponds to JSONL line `j`, whose `embedding_row` is `j-1`.
- `embeddings.npy` stores the same vectors in NumPy format (100,000 × 256).
- `shape.txt` and `manifest.json` record shape, pinned revisions, sampling provenance, artifact hashes, validation results, and timings.

GPU inference took **24.10 seconds** for all 100,000 posts on the RTX 4070 SUPER, approximately **0.241 seconds per 1,000**. Tokenization took **1.29 seconds**, recorded separately. Inference excludes warmup, downloads, model initialization, validation and export.

The sampler reads the first 10,000 records from each of 20 shards (200,000 scanned; 192,897 eligible) and chooses 100,000 with seed 42. This is a reproducible sample of shard prefixes, not a uniform whole-corpus sample or a single-day dataset. There are 3,547 repeated-text records under distinct URIs; they are preserved because repetition may affect density. Original timestamps are preserved, including anomalous years. See the manifest's year counts before interpreting this as a temporal dataset.

Reproduce and load from Julia:

```sh
julia --project=. examples/embed_bluesky.jl --count 100000 --device cuda --output data/bluesky_100000
julia --project=. examples/load_bluesky.jl data/bluesky_100000
```

```julia
include("examples/load_bluesky.jl")
X = load_bluesky_embeddings("data/bluesky_100000")
# X[:, j] is the embedding of JSONL record j.
```

The sampler automatically uses `max(1000, ceil(count/10))` records per shard; `--prefix-records` overrides that value. Larger prefixes have separate caches. It reports an error if the eligible pool is too small. For exports above 5,000 posts, the all-pairs neighbor table and HTML inspector are omitted to avoid quadratic memory and computation. The earlier small inspectors remain available.

All 100,000 saved texts and URIs were checked against their source records. Validation also checked unique URIs, row indices, finite unit vectors, binary/NumPy equality, artifact hashes, batch/singleton agreement, independent pooling and Julia loading. The [256D tree experiment](Bluesky_Tree.md) now uses this corpus with ancestor-residual and microcluster compression; the original 2D demos remain separate.

## Original 1,000-post experiment


The requested model has processed **1,000 distinct-URI Bluesky posts**. The result is a **1000 × 256 Float32** array with unit-length rows. These are static Model2Vec embeddings distilled from Qwen3, not full Qwen transformer inference. No new dimensionality reduction or tree construction is performed.

Open [the local inspector](../data/bluesky_1000/inspect.html) to browse sampled posts and their five nearest neighbors by cosine similarity. Text is displayed literally, including HTML-like content, and never executed. The filter is substring search, not semantic search.

## Reproduce from Julia

The project remains Julia. A small isolated Python process supplies the Hugging Face tokenizer and ONNX Runtime (CPU or CUDA). No Julia package dependencies were added.

```sh
python -m venv .venv-embeddings
.venv-embeddings/bin/python -m pip install -r scripts/requirements-embeddings.txt
julia --project=. examples/embed_bluesky.jl
julia --project=. examples/load_bluesky.jl
```

Arguments pass through the Julia entry point:

```sh
julia --project=. examples/embed_bluesky.jl --count 1000 --seed 42 --output data/bluesky_1000
```

`EMBEDDING_PYTHON` can point to another compatible Python executable. The requirements file records the environment actually used. Model downloads are cached in Hugging Face's normal cache; dataset prefixes are cached in `.cache/bluesky/`. Subsequent runs reuse them. The dataset, caches and Python environment are locally retained but excluded from version control by `.gitignore`.

## Input provenance

Dataset: [withalim/bluesky-posts](https://huggingface.co/datasets/withalim/bluesky-posts)

Pinned revision: `45b2e60e2bf6ec6fb21db1fe4a5346b64bcf96f1`

Model: [futur/Qwen3-Embedding-0.6B-model2vec-onnx](https://huggingface.co/futur/Qwen3-Embedding-0.6B-model2vec-onnx)

Pinned revision: `a3f60a85a2570e7976ea7f2d6505ad3e5152a96f`

The sampler reads the first 1,000 nonblank JSON records from each of the 20 JSONL shards. Of these 20,000 records, 19,275 have nonblank text and distinct, present URIs. With seed 42, it samples 1,000 without replacement from that eligible pool. Each saved record includes its source file, 1-based source record and 0-based embedding row.

This is a bounded, reproducible preview spread across shard prefixes. It is **not a uniform random sample of all 7.88 million posts**, a representative platform sample, or a contiguous day. Duplicate text under different URIs is preserved; this sample has three repeated-text records. Language, replies and image-bearing posts are not filtered. Only text is embedded; images are not processed.

Original author, URI, created_at, image flag and reply fields are preserved. The sample contains timestamps with years 2002 (1), 2007 (7), 2008 (38), 2023 (48), and 2024 (906). They are not validated or repaired. Do not directly use this preview as a clean temporal replay dataset. The source lacks user exposure logs or per-user like/dislike labels, so it supports content experiments, not historical evaluation of preference learning by itself.

## Encoding contract

The tokenizer receives original text without an instruction prefix, special-token insertion, padding, or truncation. The longest sampled sequence has 472 tokens. This is a Model2Vec lookup/mean model, so no generative Qwen prompt template is applied.

The ONNX graph accepts a flat Int64 `input_ids` vector and Int64 `offsets` specifying each post's first token. It returns normalized Float16 embeddings with width 256. The bridge converts these to Float32 and renormalizes, correcting Float16 norm rounding. The model's PCA and weighting are already incorporated in its exported embedding table.

The original run used CPUExecutionProvider, batch size 64, and two intra-op threads. The initial recorded batch-inference time was approximately 5.8 seconds for the 1,000 posts, excluding downloads, tokenization, session initialization, reference checks, and neighbor export. See the manifest for the current run's timing; this is not an ingestion throughput benchmark.

## Artifacts and Julia loading

- `posts.jsonl`: metadata and original text, one record per embedding row.
- `embeddings.npy`: NumPy array, shape `(1000, 256)`, Float32.
- `embeddings.f32`: little-endian Float32 values, post-major, 1,024,000 bytes.
- `shape.txt`: post count followed by embedding dimension.
- `manifest.json`: revisions, sampling policy, prefix/model/artifact SHA-256 hashes, dependency versions, checks and timing.
- `neighbors.json`: five cosine neighbors per post, excluding itself.
- `inspect.html`: self-contained local browser viewer; no external scripts or network requests.

```julia
include("examples/load_bluesky.jl")
X = load_bluesky_embeddings()  # 256 × 1000; one post per column
first_post = X[:, 1]          # first JSONL record, embedding_row = 0
```

The Julia loader checks byte length, endianness, finite values and unit norms. Existing tree implementations still assume 2D; this step does not insert these 256D vectors into them or silently truncate them.

## Verification

All 1,000 embeddings are finite, nonzero and normalized. Eight posts were recomputed individually and matched batched ONNX output exactly. An independent mean-pooling and normalization calculation from the exported ONNX weight table agreed within maximum absolute error 0.000166 after normalization. This is a numerical implementation check, not semantic-quality validation.

```sh
.venv-embeddings/bin/python -m unittest discover -s test -p test_embeddings.py
julia --project=. examples/load_bluesky.jl
```

The bridge tests cover ragged batch offsets and JSONL handling of Unicode line/paragraph separators inside multilingual post text. Those separators must not be mistaken for new JSON records. Local inspection of neighbors is the next way to assess whether this particular static embedding model is useful for these posts.

## GPU execution

CUDA inference is installed and verified on the local NVIDIA GeForce RTX 4070 SUPER (12 GB). The default `--device auto` selects CUDA when the installed ONNX Runtime exposes it; `--device cuda` requires CUDA and refuses silent CPU-only fallback. A few shape operations still run on CPU. Profiling confirms GPU execution of embedding gathers, mean pooling, and normalization.

To install the tested GPU environment, replace the CPU runtime (these two distributions share the same Python module):

```sh
.venv-embeddings/bin/python -m pip uninstall -y onnxruntime
.venv-embeddings/bin/python -m pip install -r scripts/requirements-embeddings-gpu.txt
julia --project=. examples/embed_bluesky.jl --device cuda --output data/bluesky_1000_gpu
julia --project=. examples/load_bluesky.jl data/bluesky_1000_gpu
```

GPU outputs and their inspector are in `data/bluesky_1000_gpu/`. The original CPU artifacts remain in `data/bluesky_1000/`. Use `--device cpu` for explicit CPU execution and `--batch-size` to change the default batch size of 64. Each embedding run excludes one warmup batch from its inference timer.

The reproducible comparison uses the same 1,000 posts, batch size 64, one warmup batch per device, and three timed repetitions:

```sh
.venv-embeddings/bin/python scripts/benchmark_embeddings.py
```

Results are saved in `data/bluesky_1000_gpu/benchmark.json`. Timing includes input packing, GPU transfers, inference and output concatenation, but excludes downloads, tokenization, session initialization, normalization and neighbor generation. Kernel profiling is performed separately from timing. CPU and GPU outputs agree within maximum absolute difference 0.000977 before Float32 renormalization; Float16 arithmetic is not bitwise identical across devices. GPU batch/singleton validation and the independent pooling check also pass, as does loading the exported vectors in Julia.

Measured median inference time on this machine: **0.283 seconds per 1,000 posts on GPU**, versus **6.095 seconds on CPU** (21.5× faster). These are local warmed measurements, not full ingestion throughput.
