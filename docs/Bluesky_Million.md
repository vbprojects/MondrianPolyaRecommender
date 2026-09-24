# Million-post rank-4 PPCA scaling experiment

The experiment fits one online rank-4 PPCA Gaussian per active leaf to 1,000,000 normalized 256-dimensional Bluesky embeddings. It uses the leaf-only conditional moment transfer implementation: splitting creates two models and releases the parent's model and reservoir. There is no time decay, per-post latent code storage, or recommendation evaluation in this run.

Open [the report](../output/bluesky_million/demo.html) or [the measured summary](../output/bluesky_million/summary.txt). [Memory checkpoints](../output/bluesky_million/memory.csv) record every 10,000 arrivals; [cuts](../output/bluesky_million/cuts.csv) records the arrival number, node, axis, threshold, depth, child IDs, and inherited masses for every split. Axis indices and node IDs are Julia 1-based.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-embeddings/bin/python scripts/embed_bluesky_million.py
julia --project=. examples/bluesky_million.jl
.venv-embeddings/bin/python scripts/bluesky_million_report.py
```

The data script uses the same pinned Hugging Face dataset and Model2Vec ONNX model as the earlier Bluesky runs. It scans 60,000 records from each of 20 shards, excludes blank text and duplicate/missing URIs, and selects a seeded sample without replacement. It batches tokenization and GPU inference and writes text and embeddings to `data/bluesky_1000000`. Its manifest records provenance, hashes, providers, timings, and normalization/batch consistency checks. Selection is uniform within this bounded eligible pool, not the full dataset. Repeated text with different URIs remains eligible.

Julia memory-maps the Float32 matrix and ingests columns in seed-192 shuffled order without making another full embedding matrix. Tree fitting uses one CPU BLAS thread. Parameters: split after 512 new arrivals since leaf creation; up to 64 retained examples per leaf; rank 4; depth limit 24; 10,000-leaf safety cap; fitting seed 918. The report includes the actual depth and leaf count so a reached cap can be identified. Splits use range-weighted random axes and reservoir-quartile thresholds; this remains a Mondrian-style adaptive heuristic rather than an exact Mondrian process.

## Memory interpretation

`tree_bytes` is `Base.summarysize(nodes)`: retained PPCA states, node bounds/routing metadata, and bounded example reservoirs. The state/reservoir breakdowns have their own small container overheads and are not strictly additive. This excludes runtime, garbage, input vectors, logs, and per-post metadata. The reservoir still stores up to 64 original Float32 embeddings per leaf; this is not a model with zero retained examples.

Process RSS and peak RSS come from Linux `/proc/self/status`, in bytes. They include Julia compilation/runtime, temporary allocations, shuffled indices, tracking records, and resident memory-mapped embedding pages. They should not be confused with model storage. The serialization includes moment-transfer error diagnostics as well as the tree.

The fit time includes initial validation, instrumentation, and checkpoint sizing, but excludes compilation warmup, final exact routing audit, serialization, and report generation. PPCA update and split timers are nested components of the fit, not independent end-to-end timings. GPU embedding time excludes tokenization, downloads, and writing artifacts; these are reported separately in the embedding manifest.

## Correctness and limitations

The run asserts preserved total mass, released internal-node state, finite leaf means, and consistent cut/leaf counts. It also reroutes every input through the final tree and reports total variation between exact empirical leaf masses and the transferred Gaussian estimates. This measures historical mass-assignment error, not full continuous density error or recommendation quality.

Conditional moment transfer models each parent as an untruncated Gaussian and does not integrate all inherited box constraints. Repeated transfers and low-rank recompression remain approximate. The archived full vectors are retained for reproducibility; this experiment does not claim compressed per-post reconstruction or retrieval without those archives.

Validation: `test/leaf_latent.jl` covers moment transfer and model behavior; `test/leaf_latent_tracking.jl` verifies instrumentation leaves topology and masses unchanged.
