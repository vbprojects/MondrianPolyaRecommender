# Mondrian–Pólya time-decay toy (Julia)

**Deploy the API and ingester:** [Docker Compose on Lightsail or another Linux VM](deploy/README.md), with persistent disk storage, CPU inference, optional HTTPS and SQLite/RocksDB support.

**Compiled executables:** [PackageCompiler build and deployment](build-support/README.md) for the server and CLI, bundling Julia, SQLite, and the Rust GTE-tiny INT8 encoder.

**Notebook-first encoder experiments:** [NeoBERT block distillation and pretrained Ettin embedding training](encoder_experiments_colab/README.md), with editable end-to-end Colab notebooks and resumable Google Drive checkpoints.

**Qwen → KDA Colab training:** [First-pass runner guide](elastic_embed_colab/README.md) · [Start notebook](elastic_embed_colab/notebooks/01_teacher_cache.ipynb). Includes teacher caches, partial weight initialization, MOHAWK mixer/block alignment, recovery, 2D Matryoshka training, and experimental ONNX export.

**Embedding comparison:** [Arctic Embed XS versus current Model2Vec](output/embedding_comparison/demo.html) on 20,000 historical posts and a hidden-hashtag topic probe, using cosine clustering and the adaptive PPCA tree. [Protocol and results](docs/Embedding_Comparison.md).

**Live ingestion:** [Native cross-platform post IDs, centroid + 8-bit codes, PPCA density, and resumable Arrow checkpoints](docs/Live_Ingestion.md). Start with `julia --project=environments/ingestion examples/ingest_live.jl --config examples/live_sources.toml`. The default run is bounded and uses CPU; CUDA is optional.

**Disk-backed tree experiment:** [Leaf-local compression and storage comparison](output/disk_ablation/demo.html) on 100,000 GTE-tiny posts: six formats, append/leaf layouts, four managed-memory budgets and three seeds. [Implementation, protocol and reproduction](docs/Disk_Storage_Ablation.md). The live index is unchanged.

**Leaf-only latent tree:** [Gaussian representation experiment](output/leaf_latent/demo.html) · [Protocol](docs/Leaf_Latent_Experiment.md).

**MPPCA kNN:** [Latent-space retrieval follow-up](output/proposal_knn/demo.html) · [Protocol](docs/Proposal_KNN.md).

**Active sampling:** [Local MPPCA proposal experiment](output/proposal_experiment/demo.html) · [Protocol and results](docs/Proposal_Experiment.md).

**Embedding storage:** [Packed centroid-residual vs PPCA codec experiment](docs/Embedding_Codecs.md) on 100,000 posts.

**Bluesky MPPCA:** [100,000-post experiment](docs/Bluesky_MPPCA.md) · [Results](output/bluesky_mppca/summary.txt).

**MNIST PCA:** [Projection and subtree-collapse comparison](output/mnist_pca/demo.html) · [Explanation](docs/MNIST_PCA.md).

**MNIST:** [Raw-pixel tree explorer](output/mnist/demo.html) — 784 dimensions, flattening only. [Method and results](docs/MNIST_Tree.md).

**Try it:** [Interactive Like / Dislike demo](output/interactive/demo.html), powered by the ancestor-residual tree. [How it works](docs/Interactive_Recommender.md).

**Real data: 100,000 embedded Bluesky posts** in `data/bluesky_100000/`, with aligned original text and 256-dimensional vectors. GPU inference took 24.10 seconds. See [data preparation and Julia loading](docs/Bluesky_Embeddings.md); the [1,000-post inspector](data/bluesky_1000/inspect.html) remains available. The [256D tree and compression experiment](docs/Bluesky_Tree.md) is built and saved in `output/bluesky_tree/`.

**New: [static user localization](output/localization/demo.html)** — a stationary hidden user, fixed tree/Fermat geometry, binary like/dislike feedback, and measured interactions to localization. Compare six query policies across 200 trials each, including tree-guided Fermat-path midpoint and asymmetric probes. See [mechanism and results](docs/Static_User_Localization.md).

```sh
julia --project=. examples/localization_demo.jl
```

**New: [adaptive-tree compression comparison](output/compression/demo.html)** — full-day replay, histograms, microclusters, and ancestor residuals, all accounting for the same admitted day window. The default simulation admits 294,400 events; a separate scaling check reaches 1,000,000. See [implementation and tradeoffs](docs/Compression_Comparison.md).

```sh
julia --project=. examples/compression_demo.jl
julia --project=. examples/compression_scaling.jl
```

The rest of this README describes the original tree/decay experiment. Its 600-point limit does **not** apply to the new compression models.

A runnable 2D experiment based on the design notes in `Mondrian_Polya_Time_Decay_and_Compression.md`. The model, random stream, evaluation, and snapshot export are Julia. An offline HTML canvas viewer supplies animation, scrubbing, cut/point toggles, and an evidence timeline. No Julia dependencies outside the standard library, no server, and no CDN are needed.

```sh
julia --project=. test/runtests.jl
julia --project=. examples/demo.jl
```

Open **[output/demo.html](output/demo.html)** in a browser. The generated example is already included. Optional arguments set the data seed and output directory:

```sh
julia --project=. examples/demo.jl 123 output/seed123
```

Tested with Julia 1.13.0; the project declares Julia 1.10+ compatibility but older versions have not been tested. `output/metrics.csv` contains per-frame diagnostics; `output/summary.txt` records the run summary. Exported animation history grows with duration; it is separate from the bounded live model.

## What to watch

| Time | Stream |
|---|---|
| 1–60 | Two stationary Gaussian clusters with 8% uniform background |
| 61–120 | Cluster centers move |
| 121–160 | Distribution jumps to the lower right |
| 161–240 | No arrivals; evidence decays through 6⅔ half-lives |
| 241–300 | Arrivals resume in the upper left |

Every active tick generates 12 observations. Gaussian draws outside the unit square are rejected, not clamped. Three models see exactly the same data and retain the same latest 600 arrivals: fixed geometry without decay, fixed geometry with decay, and adaptive geometry with decay. During silence the no-decay model stays unchanged, the fixed decayed model approaches its prior, and the adaptive model eventually collapses to one uniform cell.

## Model and implementation choices

- **Geometry:** a complete axis-aligned partition of `[0,1]²`. Each cell's next cut waits an exponential geometric time with rate `width + height`; choose the cut axis proportional to its span and the location uniformly along that span. Process candidate cuts in increasing geometric time, stopping at lifetime 12 or 127 nodes. The node cap is an additional truncation of the Mondrian process. Geometric lifetime is unrelated to stream time.
- **Evidence:** each node stores a subtree count and a last-update timestamp. At time `t`, its effective evidence is `sum(weight_i * exp(-β*(t-event_time_i)))` for retained observations routed through it. Updates touch only one path; queries compute decay without mutating the tree. `β = log(2)/12` by default.
- **Pólya allocation:** at each split, `q = area(left)/area(parent)`, with fixed prior `Beta(κ*q, κ*(1-q))`, `κ=2`. The predictive left probability is `(κ*q + left_evidence)/(κ + left_evidence + right_evidence)`. Leaf probability is the product of branch probabilities; density is probability divided by cell area. An evidence-free tree has uniform density regardless of cuts. Prior strength is constant per split; this is an explicit finite-tree modeling choice.
- **Retention:** FIFO by ingestion order, not event timestamp. Eviction subtracts the observation's current decayed contribution. Consequently the model targets the retained stream, with a truncated tail, rather than the full exponential history. During silence no records expire automatically. All three models use this same retention policy; even the no-decay control forgets through eviction.
- **Adaptation:** every 10 ticks, regenerate the same seeded scaffold, replay retained observations with their current weights, and collapse subtrees with effective mass below 8. Compact node storage releases collapsed nodes. This lets resolution return along the scaffold when activity returns. Cuts are fixed between rebuilds. Rebuilds can cause density jumps; the CSV estimates their L1 size on a 40×40 midpoint grid. This diagnostic is not an exact bound.
- **Clocks and boundaries:** ingestion/mutation time must be monotone and finite. Late events are accepted at their current age-adjusted weight; future events are rejected. Queries may look forward without changing the mutation clock. Inputs outside the unit square are rejected on insertion and have density zero on query. A point exactly on a cut goes right; outer boundaries are included.
- **Deletion:** the retained cache supports removal by ID. Duplicate retained IDs are rejected. Once an ID is evicted, exact named removal and durable deduplication are unsupported. Low-level tree subtraction assumes the caller owns the contribution; it cannot establish identity from counts.

This is a **complete-partition Mondrian–Pólya prototype**, not a reproduction of the published streaming Mondrian–Pólya Forest. In particular, it does not implement observed-box/complement allocations or projective insertion/deletion. Scaffold pruning is a heuristic, not posterior inference over topology. The design notes' residual-history mixture, sketches, moving cuts, general dimensionality, recommendation pipeline, and durable persistence remain future experiments.

Primary foundations: [Dickens et al., Mondrian Pólya Forests (2020)](https://arxiv.org/abs/2008.01505) and [Lakshminarayanan et al., Mondrian Forests (2014)](https://arxiv.org/abs/1406.2673). The time-decay, cache, and pruning choices above are explicit extensions.

## Initial observations

Seed 42 yields the following mean log predictive densities on 100 independent audit points per active tick, evaluated **before** that tick's arrivals. Uniform density on the unit square scores zero; larger is better.

| Model | Mean audit log density |
|---|---:|
| Fixed, no decay | 0.6835 |
| Fixed, decay | 0.9666 |
| Adaptive, decay | 0.8325 |

This seed supports keeping decay and questioning the current pruning rule. It does not establish general superiority: the tree seed is fixed at 7, no confidence interval is supplied, and temporal samples are dependent. At t=160 the adaptive model has only 11 nodes, versus 127 fixed nodes; pruning can sacrifice useful resolution. It falls to one node by t=240 and recovers to 53 by t=300.

Maximum exact leaf-probability normalization error was `6.66e-16`. Tests also integrate density at exact leaf interiors, compare every node's lazy evidence with direct weighted replay, verify parent/child accounting, check deletion and late arrivals, and test collapse followed by renewed learning. The original experiment contributes 864 assertions; with compression and localization checks, all 2,383 assertions pass.

`model_bytes` measures reachable Julia model objects with `Base.summarysize`, not process RSS or peak allocations. Replay temporarily holds the old tree, the candidate tree, and compacted nodes; the retained cache is bounded but this implementation is not an optimized deployment. FIFO eviction uses a vector shift, and replay loops over retained observations. The first rebuild timing can include Julia compilation. No throughput claim is made.

A useful next iteration is to sweep pruning thresholds and tree seeds on identical streams, comparing score and node count. Then compare a residual-history model separately, so differences in target history and compression remain explicit.

## Small REPL example

```julia
using MondrianPolyaToy
m = ReferenceModel(capacity=100, half_life=10.0)
observe!(m, 1, (0.2, 0.7); event_time=0.0, now=0.0)
observe!(m, 2, (0.3, 0.8); event_time=2.0, now=5.0) # late arrival
mass(m.tree, 15.0)
density(m.tree, (0.2, 0.7), 15.0)
rebuild!(m; now=15.0)
remove!(m, 1; now=16.0)
integral(m.tree) # ≈ 1
```

Million-post scaling: [rank-4 leaf-local PPCA experiment](docs/Bluesky_Million.md), with [memory and cut report](output/bluesky_million/demo.html).

[Exponential decay for leaf-local PPCA](docs/Leaf_Decay.md) includes chronological Bluesky validation and common-time Pólya queries.

[2D short-half-life stress test](docs/Leaf_Decay_Stress.md): [animated comparison](output/leaf_decay_stress/demo.html) on moving clusters, abrupt changes, and a silent interval.

[KernelAbstractions tree backend](docs/Kernel_Tree.md): CPU/NVIDIA parity tests, interchangeable LAPACK/cuSOLVER/portable eigensolvers, and [comparison report](output/kernel_tree/demo.html).

[Live speed benchmark](docs/Kernel_Speed.md): [full tree and batched-routing measurements](output/kernel_speed/demo.html) on synthetic and real Bluesky data.

Optimized fitting: [scheduled CPU/GPU backend and measured speedups](output/kernel_optimized/demo.html), with [usage and execution limits](docs/Kernel_Tree.md).

Sibling merging with decay: [implementation and usage](docs/Sibling_Merging.md) · [live tree comparison and CPU/GPU measurements](output/sibling_merging/demo.html).

Per-post latent migration: [split/merge transport and conditioning experiment](docs/Latent_Migration.md) · [interactive results](output/latent_migration/demo.html).

Streaming post storage: [local codec comparison with arrivals and expiry](docs/Streaming_Codecs.md) · [interactive tradeoffs on synthetic and 100,000-post Bluesky data](output/streaming_codecs/demo.html).

Density ablation: [uniform, diagonal and PPCA models with a shared centroid codec](docs/Density_Ablation.md) · [simulated user-localization outcomes](output/density_ablation/demo.html).

[Pluggable keyword extraction and per-leaf summaries](docs/Leaf_Keywords.md) adds opt-in YAKE preparation, exact decayed keyword counts in the disk tree, and an [interactive 100,000-post comparison](output/keywords/demo.html).

[Native Julia YAKE](julia-yake/README.md) provides a standalone keyword extractor and
an optional `JuliaYakeExtractor` adapter for the disk-backed leaf summaries.

[Disk-backed live ingestion](docs/Disk_Live_Ingestion.md) now connects the source
readers to Float16 storage, rank-4 PPCA and native Julia YAKE summaries with atomic
source cursors. The default overnight store is `data/live_disk_gte`, using pinned 384D GTE-tiny
float ONNX inference and Float16 vector storage. The older Qwen store remains
separate at `data/live_disk`.
