# Leaf-local compression and disk storage

[Interactive comparison](../output/disk_ablation/demo.html) · [Measured summary](../output/disk_ablation/summary.txt) · [All measurements](../output/disk_ablation/results.json) · [Pinned run configuration](../output/disk_ablation/config.json)

This is an isolated Julia disk-store prototype. It leaves `LiveIndex.jl`, the live ingestion configuration, and the deployed encoder choice unchanged. Its purpose is to select a post representation and physical layout before attempting million-post or seven-day scale.

## Measured representation results

The completed 100,000-post builds give the following means over three seeds. Disk totals include vectors, frozen decoders, text, identities, indexes, PPCA states and SQLite free pages. Recall uses 64 exact-neighbor queries per seed; it is a representation check, not a human relevance score.

| Format | Append MiB | Leaf MiB | Exact top-10 overlap | Leaf-assignment disagreement |
|---|---:|---:|---:|---:|
| Float32 | 238.70 | 244.60 | 100.00% | 0.00% |
| Float16 | 165.43 | 171.30 | 98.96% | 1.80% |
| Per-vector Int8 | 129.21 | 135.07 | 96.20% | 9.59% |
| Current centroid Int8 | 129.88 | 157.05 | 96.30% | 3.96% |
| Leaf-local centroid Int8 | 146.77 | 157.83 | 96.56% | 3.05% |
| Frozen leaf-local PCA, maximum rank 16 | 228.16 | 383.04 | 35.78% | 9.86% |

Float16 is the conservative format candidate: independently decodable vectors, high retrieval fidelity, and no decoder copies during structural changes. Per-vector Int8 is the smaller alternative. Centroid coding reduces boundary crossings relative to plain Int8, but local decoder overhead consumes much of its storage advantage over Float16.

Under the specified 1,024-post microbatches, strictly leaf-local fitting groups average only 10.22 posts. PCA's post-weighted retained rank averages 10.90; its mean and basis often cost more than the saved coordinates. Leaf relocation adds further frozen-decoder copies. This result argues against **this small-group frozen PCA policy**, not against retaining the separate rank-4 PPCA density model or testing larger fitting groups later.

All formats share identical tree cuts within each seed. “Leaf-assignment disagreement” counts original versus reconstructed vectors landing in different leaves; it does not mean that the tree topology changed. Exact-neighbor overlap can also penalize harmless tie changes, so the report includes tie-aware recall and cosine distortion.

Float16's tie-aware recall is 100% at the stated `1e-6` cosine-score tolerance, and its mean cosine distortion is `2.03e-8`. Its synthetic information-gain choice agreement is only 82.29%, but the average regret is `8.87e-10` bits: choosing a different nearly tied maximum is not evidence of meaningful preference-learning loss.

Regional restriction is a larger source of neighbor loss here than Float16 storage. Even Float32 retrieves only 11.20%, 26.51% and 51.88% of the global exact top ten when restricted to 1, 4 and 16 candidate leaves, respectively. This prototype is regional sampling, not an exact nearest-neighbor index. The viewer reports both global coverage and preservation within a region so those effects remain distinguishable.

## Layout recommendation and measured costs

For the intended workload with most data on disk, **Float16 plus leaf-oriented blocks** is the next prototype candidate. At four candidate leaves and a 256 MiB managed budget, the Linux-cache-eviction condition had p95 latency of **48.02 ms for leaf versus 75.99 ms for append**. Mean distinct vector frames fell from 19.71 to 11.00 per query, and Linux block reads fell from 330.03 to 258.60 KiB per returned post. Each condition here contains 24 queries pooled across three seeds, so these tail estimates remain exploratory.

The tradeoff matters: with continuously cached uniform traffic, p95 was **3.69 ms for append versus 13.48 ms for leaf** at the same leaf/cache budgets. Float16 compaction took **2.12 versus 19.86 seconds**, despite writing essentially the same 73.25 MiB of live vector payload. The leaf compactor's row-wise gathering/copying and metadata work need optimization; the timing difference is not explained simply by more vector bytes written. Keep append as the alternative for hot-cache or write-heavy workloads, and remeasure after optimizing compaction before a million-post trial.

Full-build Float16 throughput averaged 4,142 posts/s for append and 3,699 for leaf, including durable commits and excluding embedding inference. Across all representations/layouts/seeds, the slowest 32 MiB durable update probe processed 1,177 posts/s. The 100 posts/s target is therefore met at the tested corpus size, without establishing concurrent-service or weekly-scale throughput.

Final collection contains 36 real-data stores, 18 quality audits, 72 synthetic cases, 144 update probes and 78,624 measured queries. The synthetic cases executed 4,152 splits and 4,152 merges; encoded-value fingerprints, identity audits and common structural fixtures passed. The linear weekly storage estimates for Float16 are 97.71 GiB append and 101.17 GiB leaf, with the scaling limitations below. These are capacity estimates rather than a seven-day deployment test.

Final validation passed **795 Julia assertions** and all **168 viewer control combinations**, with desktop/mobile screenshots and no browser errors. See the [validation record](../output/disk_ablation/validation.json) and [source/configuration record](../output/disk_ablation/config.json). At a 32 MiB managed budget the runtime still needs hundreds of MiB beyond those managed buffers; the report's RSS columns are the relevant process-memory observations.

## Data and controls

The experiment uses the first 100,000 existing historical Bluesky records, preserving their original input order and native AT-URIs. Fresh embeddings use `TaylorAI/gte-tiny` at revision `4cc5e73d86a67c601897257b467187234aa3bca3`, Float32 model weights and inference, a 256-token limit, masked mean pooling, and normalized 384-dimensional Float32 output. TF32 is disabled. INT8 model-weight inference is not part of this experiment.

The preparation script writes aligned `posts.jsonl`, `embeddings.f32`, and a hash/version manifest under `data/gte_disk_ablation`. Embedding preparation is separate from storage workers. The measured preparation took 46.58 seconds on the workstation's RTX 4070 SUPER; checked CPU/GPU and singleton/batch outputs differed by at most `8.94e-8`. These are spot checks, not an all-post CPU comparison.

The storage experiment varies seeds 17, 29, and 43. For each seed all formats receive identical records, input microbatches, original-vector tree updates, cuts and query fixtures. A shared rank-4 PPCA model uses the existing `LeafLatentTrees` moment update, conditional split, and sibling-merge routines. The tree has at most 128 leaves, a 64-point reservoir per leaf, split threshold 256 effective arrivals, and a six-hour half-life. This remains the repository's adaptive tree approximation, not a new exact Mondrian-process implementation.

Arrival time is simulated at 100 posts/s; the real 100,000-post build spans 1,000 simulated seconds. Historical source timestamps are preserved as metadata. This avoids attributing chronological meaning to the archive's existing row order. Seven-day query eligibility is independent of the six-hour density half-life. The real-data comparison mostly exercises growth; repeated decay-driven merging and rolling expiry are tested in the synthetic cases.

## The two independent choices

| Representation | Numeric post data | Frozen shared decoder |
|---|---|---|
| Float32 | 1,536 bytes | None |
| Float16 | 768 bytes | None |
| Per-vector Int8 | 384 bytes + one 4-byte scale | None |
| Current centroid Int8 | 384-byte offset + one 4-byte scale | 384 Float32 mean values per group |
| Leaf-local centroid Int8 | Same offset format | Mean fitted only to one current leaf's new points |
| Frozen leaf-local PCA | Up to 16 Float32 coordinates | Local Float32 mean and basis |

New batches contain at most 1,024 posts. Codec groups contain at most 256. The current centroid control sorts new points by their current leaf and then chunks that sequence, matching the live index's cross-leaf packing policy. Strictly local centroid/PCA groups never pool original points across a current leaf boundary. Small groups are retained and charged their real decoder overhead; PCA rank is `min(16, dimension, group_size - 1)`. There is no storage-saving fallback that would conceal an inefficient PCA group.

The implementation uses symmetric Int8 values from -127 to 127, a per-vector maximum-absolute-value scale, and frozen Float32 parameters. PCA latent coordinates are Float32. Float16 is the high-fidelity compressed baseline, not an assumed winner. Codec headers and per-fragment copies of decoders count toward actual storage.

The “current centroid” control reproduces the live index's grouping policy and representation, using this prototype's CPU codec. It is not a byte-for-byte port of `CentroidPostKernels.jl`: the prototype computes means/residuals in Float32, while the live kernels use Float64 intermediates before storing Float32 parameters. All reported quality measurements use the actual prototype reconstructions. Bitwise preservation is required between layouts and structural migrations of each encoded representation, not between this prototype and the live kernel's encoder.

The **append layout** keeps arrival-batch frames and moves only posting references during structural changes. Its daily compaction streams original frames and copies their surviving rows together. It does not interleave rows from different decoders just to restore individual arrival order. The **leaf layout** streams and repacks affected leaves after each split/merge, and streams each leaf during compaction. Every resulting frame is self-contained. Mean and basis bytes may be duplicated across fragments. Ordered density rebuilds use the separate posting-sequence index in both layouts.

“Local” describes both a fitting decision and a physical-layout decision. A decoder fitted in one leaf can later be used by descendants. It is not refitted when its posts move. In leaf layout, each descendant's frame gets the required decoder copy; in append layout, descendant postings can reference the same historical frame. The current PPCA density state is separate from these frozen post decoders.

Leaf packing streams sequence-ordered postings and coalesces adjacent compatible frames. It does not globally regroup all nonadjacent fragments by decoder hash. Consequently a decoder can occur more than once within one leaf; these copies are counted. The observed overhead is the cost of this simple streaming policy, not a minimum achievable by a more sophisticated leaf compactor.

Original vectors train the common tree as each input batch arrives. Historical migration routes decoded vectors through the new cut. A lossy reconstruction can therefore belong to a different leaf from its original embedding; that disagreement is measured. Historical vector bytes never enter the encoder again during splitting, merging or compaction. PCA decoding uses a fixed accumulation order so changing a frame's row count cannot change reconstruction rounding.

Routing uses raw reconstructed coordinates, matching the existing codec experiments. Cosine ranking separately normalizes reconstructed vectors. PCA can change vector norm as well as direction; a normalized-PCA routing variant is not included here. Very small coordinate changes can also cross narrow tree cuts, so interpret membership disagreement alongside distortion, tie-aware recall and regional retrieval preservation.

## Storage interface and publication

The experiment is implemented in [DiskTreeStore.jl](../src/DiskTreeStore.jl) and [DiskCodecs.jl](../src/DiskCodecs.jl). Include `LeafLatentTree.jl` and `PostRecords.jl` before these modules; the example scripts do that automatically.

| Operation | Interface |
|---|---|
| Open/restart | `open_store(path; budget, create_config)` |
| Durable microbatch | `append_batch!(store, records, vectors; time, cursor)` |
| Native identity / aliases / sampled ID | `lookup(store, PostKey(...))`, `lookup(store, id)` |
| Ordered leaf traversal | `leaf_refs(store, leaf; after=sequence, limit=256)` |
| Bounded selected-row reads | `fetch_vectors(store, ids)`, `fetch_records(store, ids)` |
| Regional samples | `sample_ids(store, query; limit=20, leaf_budget=4)` |
| Pólya leaf masses / probabilities | `polya_weights(store; time, kappa=2)` |
| Delete | `delete_post!(store, key; revision_at, time)` |
| Expire / decay-driven merges | `expire!(store; time)`, `maintain!(store; time)` |
| Physical maintenance | `compact!(store)`, `collect_garbage!(store)` |
| Density reconstruction | `rebuild_density!(store)` |
| Accounting / invariant checks | `storage_stats(store)`, `audit!(store)` |

SQLite holds native identities, aliases, post revisions, addressable leaf slots, sequence indexes, block/file locations, routing nodes, serialized leaf states, replay cursors, and the committed generation. Text and codec frames are immutable files. Monotonically allocated node/post/block IDs are independent of Julia array positions. A merged parent keeps its stable ID; retired child IDs are never reused.

Random sampling first obtains candidate leaves from routing-box distances, then samples their indexed counts and dense posting slots. Slot deletion swaps in the last slot. It never scans a leaf to pick a random row and never uses `ORDER BY RANDOM()`. A separate sequence index supports ordered, bounded-buffer rebuilds. Selected-row reads fetch the referenced frame, not every codec associated with the leaf. Identity lookup is indexed and does not load a catalog into Julia.

Samples return stable internal IDs. Resolve an ID with `row = lookup(store, id)` and recover its cross-platform key with `PostRecords.key_from_bytes(row.kind, row.native)`. Text/vector fetches preserve the requested ID order; their result positions remain associated with those IDs.

`polya_weights` computes the same geometric-prior branch probabilities as the reference from compact node masses, including lazy exponential decay and a non-decaying prior. It does not load PPCA factors or reservoirs. The storage benchmark samples actual eligible posts by indexed counts in the selected region; it does not substitute Pólya mass for the number of addressable posts.

Mutations use one SQLite transaction. New files are flushed and fsynced, renamed, and their directory fsynced before the database commit publishes their references. SQLite runs in WAL mode with `synchronous=FULL`, mmap disabled, and a bounded page-cache target. After an exception the instance is deliberately poisoned: close and reopen rather than continuing with uncommitted in-memory routing state. Startup loads routing metadata and counters; heavy caches start empty. Orphan files from interrupted publication are removed by garbage collection. Frames include checksums and reject corruption.

This follows SQLite's [local-filesystem WAL constraints](https://www.sqlite.org/wal.html). The prototype holds an exclusive process ownership lock. It does not yet expose concurrent reader connections, an HTTP service, or parallel mutation. No lazy ancestor posting inheritance is implemented: structural changes stream the affected leaves and publish all posting changes together.

## Memory and I/O accounting

The four managed budgets are 32 MiB, 256 MiB, 1 GiB, and 4 GiB. Each reserves 16 MiB for bounded input, decoding, fitting, routing metadata and work buffers. The remainder is partitioned between a SQLite page-cache allowance, a leaf-state cache, and a vector-frame cache. The SQLite page target leaves headroom for page overhead. Actual SQLite cache bytes/hits/misses and Julia cache occupancy/evictions are reported. Individual oversized cached items are rejected.

These are accounting budgets, **not a hard whole-process memory limit or allocator sandbox**. Julia runtime, compiled code, transient garbage, allocator fragmentation and Linux's file cache are additional. A single-worker process can use substantially more RSS than its managed budget. Both current RSS and process high-water RSS are reported. Query phases for a case share a process; high-water RSS is cumulative, so it must not be interpreted as an independently measured peak for each budget. Garbage collection occurs between budget phases.

Workers stream at most 1,024 incoming vectors and 256 historical posting references at once. There is no full vector matrix or post dictionary in the storage worker. The separate quality evaluator deliberately loads reference and reconstructed matrices; its memory is excluded from storage-worker results. The resident routing table is small at 128 leaves; this experiment does not prove its cost for millions of leaves.

Query timings cover topology selection, slot sampling, selected PPCA-state loading, vector decoding, and text fetching. They exclude client-side ranking. Sampling tests use 1, 4 and 16 candidate leaves, 20 returned posts, and a separate bounded 256-post ranking pool. The ordinary workload includes dispersed queries, repeated topic-region queries and revisits after eviction. Duplicate requested IDs are rejected in a sample, and expired entries are filtered even before the daily expiry sweep.

The report distinguishes:

- **Application cold:** managed caches are cleared, but Linux's file cache is uncontrolled.
- **Immediate repeat:** identical request and random seed immediately after its first execution.
- **Mixed:** the 256-post candidate pool after the preceding query sequence.
- **Continuous uniform, topic burst, and cold-leaf revisit:** caches remain live across 360 requests per case/budget, including 1/4/16-leaf queries. Separate identity lookups are timed and checked against the fixtures. These sequences measure persistent-cache behavior rather than only a request immediately repeated after clearing caches.
- **File-cache eviction hint:** managed caches are cleared and Linux receives `POSIX_FADV_DONTNEED` for this store's files. This is advisory. Measured `/proc/self/io` `read_bytes`, rather than the hint itself, establishes that disk reads occurred.

Application read bytes include fetched vector frames, text and leaf states. Kernel disk bytes also include SQLite and readahead. These are different quantities. A decoded-row request can read a whole frame, making bytes per returned post much larger than the representation's bytes per stored post.

This workstation runs the experiment inside WSL2 on an ext4 virtual disk (`/dev/sdd`). `/proc/self/io` establishes Linux block-layer reads, not physical SSD/NAND reads. The eviction hint does not flush the Windows host cache, SSD/controller cache or other host-side layers. The reported cold condition is therefore **Linux file-cache eviction**, not a verified fully cold physical SSD. Recheck latency on the intended production filesystem before selecting hardware.

Full 100,000-post ingestion runs use 256 MiB. All four budgets also receive a durable 1,024-post update probe on disposable snapshots of the completed stores. Snapshot creation is outside the measurement. The snapshots copy SQLite and hard-link immutable payloads; they never modify the source payload files. This measures update sensitivity to budget without repeating the entire build four times. Shared tree/storage/query paths are warmed before collection; any remaining format-specific first-use work stays in the measured time. Embedding inference is excluded from every storage timing. Runs are serialized to avoid benchmarks contending with each other.

Headline ingestion includes reading the prepared input, constructing records, tree work, encoding, and durable commits. Raw results also separate storage-API time. The single-batch budget probes time the API with an already prepared batch.

Ingestion and query workloads are timed separately. These figures establish batch capacity on this 100,000-post prototype, not end-to-end latency for a concurrent service ingesting at 100/s while serving clients.

SQLite `dbstat` reports actual table/index pages separately from payload files. Total disk usage includes reusable free pages. Before/after compaction sizes and written payload bytes expose maintenance amplification; temporary old and new immutable files coexist until the new generation commits and garbage collection runs. The raw ingestion measurements also include kernel write bytes and PPCA state write volume.

## Quality and correctness

The separate reference evaluator measures exact cosine top-10 overlap for 64 fixed queries per seed, excluding each query's source post. It also reports a tie-aware variant, mean/p99 cosine distortion, raw coordinate RMSE, and complete leaf-assignment disagreement. For 1/4/16 candidate leaves it measures both preservation against Float32 regional retrieval and recall of global exact neighbors; restricting a tree region can lose global neighbors even for exact Float32 storage.

The synthetic information-gain proxy uses the existing `response_probability` and `query_information` functions with 32 fixed hypotheses, 96 candidate posts and 64 trials per seed. A fixed prior concentrates around one original hypothesis. The response distance is squared distance between normalized vectors, with a median-distance scale. Only candidate representations change. Choice agreement and original-scoring regret in bits are not real recommendation relevance, Fermat-distance accuracy, or user-localization speed.

Tests compare tree cuts and PPCA moments against the existing in-memory implementation, and compare bounded weighted rebuilds against independent replay. Other gates cover exact encoded/decoded preservation, identity/slot uniqueness, semantic edits, metadata-only edits, aliases, stale replay, unknown-ID deletion tombstones, strict query expiry, corruption rejection, forced eviction and empty-cache restart. Abrupt `_exit` injection runs in child processes before file publication, before database commit and after database commit.

The synthetic experiment has 2D moving clusters and a 32D embedding control, six formats, both layouts and three seeds: 72 cases. Each uses three grow/decay/merge cycles followed by nine simulated days of arrivals, expiry and physical payload compaction. Frozen-value fingerprints are checked across repeated structural changes. Decay changes density evidence; it never rewrites a stored code.

Synthetic data generation and sampling use separate random generators. Each daily maintenance fixture includes a 900-second idle interval at a 30-second half-life, returning the tree to its root before the next burst. Exact recorded cut/node sequences are compared across formats and layouts, so compression cannot silently change the structural workload being timed.

## Reproduce

```sh
julia --project=environments/storage -e 'using Pkg; Pkg.instantiate()'
.venv-embeddings/bin/python scripts/prepare_disk_ablation.py
python3 scripts/run_disk_ablation.py
python3 scripts/validate_disk_ablation.py
```

The Python embedding environment needs the versions recorded in the preparation manifest. The preparation script reuses a matching verified cache. The source records must already exist in `data/bluesky_100000/posts.jsonl`; it does not silently download or substitute a different corpus. Storage workers use the pinned [Julia environment](../environments/storage/Project.toml) and its Manifest. This run uses Julia 1.13.0.

`run_disk_ablation.py` executes 36 isolated build workers, 18 separate quality audits, synthetic stress cases, cache-budget update probes, continuous traffic, cold-I/O trials, and the report generator. Complete result cases are skipped on rerun. A completed build with interrupted query measurements resumes those queries. A store whose build did not finish is not silently overwritten; move that experiment directory aside before rerunning it. Individual Julia scripts accept `MONDRIAN_DISK_INPUT`, `MONDRIAN_DISK_STORE` and `MONDRIAN_DISK_OUTPUT` for isolated debugging.

Reports live in `output/disk_ablation`; disk stores live in ignored `data/disk_ablation`. The standalone HTML embeds aggregate measurements and requires no web server. Raw per-case JSON, query CSV, summary CSV, source hashes, model/input hashes and dependency versions remain available alongside it.

The final validation script runs the Julia invariants and abrupt-process-death tests after performance collection, verifies the source/input hashes, regenerates the report, and exercises all 168 viewer control combinations at desktop and mobile sizes. Browser validation uses Playwright; set `PLAYWRIGHT_MODULE` to its installed module path if the workstation's recorded default is unavailable. Its test logs and `validation.json` stay beside the measurements. The pinned configuration retains launch hashes separately from final validated source hashes and records the intervening correctness fixes.

## Limits before production

The store retains identity tombstones after payload expiry, and SQLite reuses rather than automatically shrinking freed database pages. That supports replay suppression but needs a bounded metadata-history and vacuum policy for indefinite operation. Decoder/state serialization needs versioned migrations; current Julia-serialized states are trusted local artifacts and continuation requires the recorded Julia version. The publication tests validate process-crash recovery, not power-loss behavior of every SSD/controller.

The bounded caches use a simple scan to select their least-recently-used victim. With many small frames, eviction CPU cost grows with the number of cached entries; an indexed LRU implementation is a useful optimization before larger tests. Timings compare this prototype's complete implementations rather than theoretical lower bounds for each representation or layout.

Text packs are reclaimed when their last active record expires or is deleted. Unlike vector frames, partially live text packs are not rewritten by this prototype's compactor. Daily expiry therefore reclaims wholly obsolete text packs, while a long-lived record or recent edit can delay reclaiming other bytes in the same pack. Temporary-maintenance figures report old/new vector payload coexistence and written bytes; they are not a high-frequency measurement of peak WAL, filesystem or SSD-internal space.

The seven-day capacity target is 60,480,000 active posts at 100/s. Weekly figures are simple extrapolations of the measured bytes/post, including current overhead. Decoder occupancy, tree size, tombstone history, B-tree height, page-cache behavior and maintenance I/O may change at that scale. These estimates do not validate weekly operation, and the low-dimensional stress cases cannot establish high-dimensional production quality. No production format, live migration, or encoder switch is applied automatically.
