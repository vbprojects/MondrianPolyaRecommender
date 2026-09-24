# Local post codecs under splits, merges and expiry

The selected centroid + 8-bit approach now has an [integrated ingestion implementation](Live_Ingestion.md): immutable decoder batches, separate PPCA density states, native platform IDs, and Arrow restart support. The measurements below describe the earlier comparative experiment, not the new ingestion service.

[Interactive comparison](../output/streaming_codecs/demo.html) · [Measured summary](../output/streaming_codecs/summary.txt) · [Raw results](../output/streaming_codecs/results.json)

This follows the [latent-migration stress test](Latent_Migration.md). That experiment kept a fixed cohort and showed that repeated low-rank reprojection can move or collapse historical posts. This experiment adds continuing arrivals and daily post expiry, measures the decoder versions that remain referenced, and compares local alternatives on synthetic streams and all 100,000 archived Bluesky embeddings.

The experimental Julia module is `src/StreamingCodecs.jl`; it is a storage layer attached to the actual `LeafLatentTrees` split/merge hooks. It does not change the existing reference tree, kernels, or interactive recommender. The tree's density model is identical across methods. This is a comparison of storing and retrieving actual posts, not a comparison of density estimates or generated Gaussian samples.

## Results and interpretation

The completed run tested **12 methods × 6 workloads × 3 seeds**, with **2,715 splits and 2,715 merges** and 3,960 measured checkpoints. It took approximately 255 seconds after warmup, including the expensive audit queries. All UUID and expiry checks passed. The binary reader and codec tests passed **18,785 assertions**, and the standalone viewer passed desktop/mobile control checks. [Validation record](../output/streaming_codecs/validation.json).

Selected Bluesky results, averaged across day-two-and-later grown-tree checkpoints and three seeds:

| Method | Packed bytes/post | Global midpoint recall@10 | Single-leaf midpoint recall@10 | Seed range of local recall |
|---|---:|---:|---:|---:|
| Float32 reference | 1,048.2 | 100% | 100% | 100–100% |
| Float16 coordinates | 536.2 | 99.8% | 99.8% | 99.5–100% |
| Centroid + 8-bit residual | 296.8 | 97.3% | 89.4% | 85.5–93.4% |
| Frozen batch PCA r16 | 300.5 | 99.7% | 90.0% | 85.2–95.7% |
| PPCA r4 + 4-bit residual | 234.8 | 88.9% | 78.4% | 75.6–83.3% |
| Adaptive rank / 8-bit fallback | 258.9 | 96.2% | 84.9% | 79.6–91.3% |
| Inherited basis + innovations | 102.8 | 95.9% | 83.8% | 78.0–91.1% |
| Frozen local PPCA r4 | 102.8 | 11.4% | 16.8% | 14.6–18.0% |
| Sequential PPCA r4 | 74.6 | 8.8% | 15.4% | 13.6–18.4% |

**Inherited bases are the strongest compact candidate among these tested settings.** They use about 10.2× less packed storage than the Float32 reference while retaining most global neighbors. The average latent rank is about 15.8, not four. Sharing reduces decoder storage from about 212 bytes/post for independent rank-16 bases to 15 bytes/post for the inherited variant, including local means and retained basis chains. The independent rank-16 control achieves better fidelity with a stronger fresh-batch fitter, at roughly three times the total storage. This comparison does not attribute every quality difference solely to basis sharing: their basis-fitting policies differ too.

For a stricter fidelity requirement, **Float16 is the strongest tested compressed option**. The two approximately 300-byte codecs offer a middle ground: independent PCA has higher global fidelity but costs about 18.4 ms per thousand arrivals to encode; centroid residuals cost about 0.7 ms in this implementation. Their mean local recall differs by only 0.6 percentage points, with overlapping seed ranges; this experiment does not establish a reliable local-retrieval superiority between them.

The most important unresolved problem is **routing around cut boundaries**. The inherited codec retrieves 95.9% of original neighbors when every retained post can compete, but 83.8% when candidates must come from the proposal's stored leaf. About 11.6% of its retained posts have a different region assignment from their original embeddings at grown-tree checkpoints. Centroid 8-bit residuals also lose local recall despite strong global recall. Better boundary handling, neighboring-leaf search, or selective original-vector refresh deserves a separate experiment. The current results should not be read as an end-to-end feed-quality guarantee.

Rank four alone is too aggressive for preserving individual Bluesky posts in this test. That does not establish that rank four is unsuitable for the tree's aggregate density model: every method here uses the same rank-four tree. Moment modeling and identity-linked post representation have different requirements.

The synthetic results show why there is no universal winner. On the stable rank-four manifold, inherited bases use 44.8 bytes/post and retain 99.7% of local midpoint neighbors. With sixteen similarly strong directions, plain 8-bit coordinates retain 99.1% at 60.5 bytes/post, while inherited bases cost 93.1 bytes/post and retain 94.7%. On the rotating long-retention stream, sequential rank-four codes retain only 19.1% of local midpoint neighbors, versus 95.8% for inherited bases and 95.5% for the adaptive policy. These are observed tradeoffs for the specified lifetimes, batching, rank caps and bit depths; decoder reuse and hyperparameters were not exhaustively optimized.

## Methods

| Method | Per-post representation | Shared state and lifecycle |
|---|---|---|
| Float32 reference | All original coordinates | No decoder; exact reference |
| Float16 coordinates | All coordinates rounded to Float16 | No decoder |
| 8-bit / 4-bit coordinates | All coordinates, symmetric uniform quantization | One Float32 scale per post, no learned decoder |
| Centroid + 8-bit / 4-bit residual | Quantized full-dimensional offset | Frozen Float32 mean per incoming leaf/batch group; per-post scale |
| Frozen local PPCA r4 | Up to four Float32 latent coordinates | Frozen copy of the live region mean and basis at encoding |
| Frozen batch PCA r16 | Up to sixteen Float32 latent coordinates | Independent fresh-batch mean and SVD basis per local group; rank-matched control for inherited bases |
| PPCA r4 + 4-bit residual | Float32 latent code plus quantized full-dimensional error | Frozen local decoder plus per-post residual scale |
| Adaptive rank / 8-bit fallback | Smallest eligible local low-rank code, otherwise 8-bit coordinates | Fresh-batch SVD; ranks 0, 2, 4, 8, 16; choose low rank only if centered reconstruction NRMSE is at most 0.1 and numeric payload is smaller |
| Inherited basis + innovations | Float32 coordinates in the inherited/extended space | Immutable shared basis chain; add at most four orthogonal directions per group, capped at 16; a new local mean per group |
| Sequential PPCA r4 | Repeatedly rewritten Float32 latent coordinates | Reproject old reconstructions to each child/merged decoder; migration baseline |

In two dimensions, rank four is capped at two. Quantization uses 15 symmetric values at four bits and 255 at eight bits, including an exact zero code. Packed four-bit coordinates really occupy half a byte each; odd final nibbles are padded. Per-post scales use the largest absolute coordinate/residual, so calibration needs no future observations. This differs from the earlier [offline codec experiment](Embedding_Codecs.md), which used per-component coordinate scales and a different identifier/index format. Their storage and recall numbers are not directly interchangeable.

The adaptive method and frozen rank-16 control deliberately use a stronger batch-local fitter than the live rank-four sketch, and their extra SVD time is charged to encoding. The adaptive reconstruction target is a heuristic, not a guarantee about retrieval recall. Small groups fall back to coordinate quantization when a decoder would cost more than it saves. This particular adaptive policy does not search all residual bit depths or optimize a recommendation metric. Frozen independent codecs create a decoder per incoming leaf/batch group; we have not tuned cross-batch decoder reuse for those baselines.

An inherited basis is shared by reference, not copied into every descendant. At a split, both children inherit the available basis hint. At a merge, the higher-rank child hint becomes the parent's starting hint (left wins ties); fresh arrivals can add directions from their residuals. Historical posts retain the decoder that encoded them. A new mean never changes an old decoder. The 16-direction cap is a deliberate limitation, and this method has no residual fallback. Every retained parent basis in the dependency chain is counted, including bases held by live region hints. Expiry clears unused hints and recursively unreferenced chains.

After a merge, the inherited basis can be shared across a broader region and its future descendants; this design does not guarantee an independent latent space for every leaf. Local means remain separate. The rank-16 control measures the cost and quality of keeping those bases independent.

## Common stream and tree

Each method sees exactly the same actual tree, trained on original incoming vectors. Tree statistics decay exponentially with half-life one. Observed-range random cuts use the existing conditional-Gaussian moment transfer. Sibling merges use the existing moment-preserving low-rank merge. Original historical embeddings never enter any codec update; they are retained in a separate audit array only.

The input is processed in microbatches. First, the common tree consumes a batch. Then each new post is encoded in the leaf containing its original embedding at the end of that batch. Centroids, adaptive SVD, and innovations may use that current batch's local points. This means the experiment permits batch latency and a temporary original-vector buffer. It is not an immediate single-post encoding service. Previously encoded posts participate in every intervening split and merge.

| Workload | Dimensions | Arrivals per seed | Batches per artificial day | Retained days |
|---|---:|---:|---:|---:|
| Moving two-cluster control | 2 | 20,480 | 4 × 512 | 2 |
| Stable rank-four manifold | 32 | 20,480 | 4 × 512 | 2 |
| Rotating rank-four manifold | 32 | 20,480 | 4 × 512 | 2 |
| Rotating, long retention | 32 | 20,480 | 4 × 512 | 8 |
| Sixteen similarly strong directions | 32 | 20,480 | 4 × 512 | 2 |
| Archived Bluesky posts | 256 | 100,000 | 8 × 2,500 | 2 |

There are three seeds per workload. Synthetic seeds change both data and tree; Bluesky seeds vary the tree and audit queries while preserving the archive's existing row order. The rotating manifold turns between two four-dimensional subspaces; it is contained in eight ambient directions over its complete trajectory.

Each ingestion batch spans 0.25 density half-lives. After each artificial day, a 16-half-life idle interval and threshold-three merge return the density tree to its root. Synthetic trees permit 16 leaves and split after effective arrival weight 32. Bluesky permits 32 leaves and splits at 96. All use reservoirs capped at 64. This deliberately induces structural churn. **The Bluesky clock is simulated, not reconstructed from post timestamps.**

At the start of a new day, records older than the retained day window expire. A live window contains up to 4,096 synthetic posts (16,384 for long retention) or 40,000 Bluesky posts. Expiration removes UUIDs, codes and their unreferenced decoders. It does not subtract the old exponential tail from the density model: content retention and density decay are separate policies. A final complete drain asserts that every decoder and inherited basis chain disappears.

Stable codecs update per-post region membership at each cut using their reconstructed split-axis coordinate; their stored values are not rewritten. This can route a point differently from the original. A newly encoded reconstruction may also cross its assigned region's boundary. Sequential PPCA additionally changes the reconstruction on every structural transition. These two errors are measured separately.

## Measurements

At every day end, both before and after the merge:

- Check every retained UUID appears exactly once, all reconstructions are finite, and the retention window is correct.
- Measure reconstruction NRMSE relative to the centered original cloud and retained variance.
- Compare exact original-space Euclidean nearest neighbors for 24 actual-post queries, excluding the query itself.
- Construct 24 arbitrary proposals by averaging two actual posts from the same original-data leaf. This stays inside the leaf's convex box. Measure global recall@10 and recall when candidates must come from that proposal's stored leaf. These are arithmetic midpoint proposals, not Fermat-path midpoints or Gaussian density samples.
- Check stored region membership against routing each original post through the current tree, and separately check how many reconstructions lie outside their stored leaf bounds.
- Evaluate four synthetic information-gain decisions, each with up to 32 original-vector location hypotheses and 96 candidate posts. The response model is `0.05 + 0.9 / (1 + squared_distance / scale)`, where scale is the median original hypothesis/candidate squared distance. The prior concentrates around one selected hypothesis. Only candidate representations change. Report selected-post agreement and regret in bits under original-vector scoring.

No feedback is collected from humans. No experiment here establishes recommendation relevance, localization speed, or performance of the complete Fermat-distance recommender. Keeping the density model common intentionally isolates the post-storage decision.

Summary metrics average day-two-and-later checkpoints within each seed, then average seeds equally. The long-retention workload is still filling its window during some summarized days; inspect its full trajectory as well. Reported ranges are seed minimum–maximum, not confidence intervals. Per-day queries change, so a fluctuating curve need not mean a frozen post's code changed. Near-equal or duplicate vectors can create nearest-neighbor ties; this audit does not apply a special tie-aware recall definition.

## Storage and timing accounting

Two storage measurements are reported:

1. **Packed payload:** actual numeric file layout, including 128-bit UUIDs, 64-bit current region IDs, block day/count/format headers, Float32 decoder means and bases, sharing references, per-post scales, and packed codes. Parent basis chains are counted once. Each seed-one Bluesky payload is written and its exact file size checked against the estimator.
2. **Reachable Julia store memory:** `Base.summarysize` of the live store, including matrices, array headers, vectors, dictionaries, basis hints and decoder objects. It is not process RSS or peak temporary allocation.

Shared density-tree memory is reported separately, including node bounds, sketches, reservoirs and the relevant clock/arrival metadata. The raw current input batch is reported separately. Text, a durable URI/UUID catalog, audit originals, output histories, and transient encoding/SVD workspaces are excluded. Thus neither payload size nor reachable store memory is a complete deployment RAM requirement. Metadata can dominate the two-dimensional control.

Experiment UUIDs encode the seed and original input row. They provide an exact, auditable identity mapping; they are not native Bluesky post identifiers. The existing aligned URI/text archive remains the external lookup. No source embeddings or posts were deleted.

Codec encoding, split/merge maintenance, and expiry times are measured separately. BLAS runs on one CPU thread; a disposable stream warms compilation, and codec order rotates to reduce consistent first/last-order effects. Timing includes allocations and GC and excludes audit retrieval. The inherited and adaptive methods use CPU SVD for fresh data; none of these new codecs are GPU kernels. Historical membership scans and sequential block fragmentation are unoptimized. Times characterize this implementation, not a lower bound on each approach or a production speed claim.

## Reproduce and verify

```sh
julia --project=. test/streaming_codecs.jl
julia --project=. examples/streaming_codec_experiment.jl
python3 scripts/streaming_codec_report.py
julia --project=. test/streaming_codec_outputs.jl
```

For a quick smoke test, use `julia --project=. examples/streaming_codec_experiment.jl --smoke`; it writes `smoke.json` without overwriting full results or binary payloads. The file format is a little-endian experiment artifact, not a durable production storage API. The independent test reader reconstructs each saved payload and compares its UUID/reconstruction hash to the in-memory encoding.

See the generated [summary](../output/streaming_codecs/summary.txt) for all workload results and the smallest measured codec exceeding 70% or 90% mean local midpoint recall. Those are descriptive choices among tested configurations, not universal quality thresholds.

The [density ablation](Density_Ablation.md) fixes centroid + 8-bit post storage and compares uniform, diagonal and PPCA density models using simulated binary-feedback localization.
