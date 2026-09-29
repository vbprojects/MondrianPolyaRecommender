# Adapter profile

The main ingestion cost is in adapter record/index handling, serialization and statement dispatch. Keyword reads split much of their sampled execution between deserialization and native point reads. The foreground journal is a smaller contributor in this fixture than initially suspected. The storage implementation was left unchanged; raw stack profiles and allocation samples are included.

## Unprofiled baselines

Median of three sequential runs per case, each preceded by a full workload warmup and garbage collection. These timed runs have no sampling profiler enabled. All reported compilation/recompilation times were zero. Embedding inference and keyword extraction are excluded.

| Case | Time per workload | Julia allocation volume | GC time |
| --- | ---: | ---: | ---: |
| sqlite-ingestion | 1.165 s | 654.4 MiB | 0.044 s |
| rocksdb-ingestion | 3.756 s | 1,699.6 MiB | 0.099 s |
| sqlite-terms | 4.120 s | 1,919.5 MiB | 0.136 s |
| rocksdb-terms | 9.936 s | 4,463.6 MiB | 0.243 s |

Ingestion covers 2,048 synthetic normalized 384D posts in eight durable batches of 256, with Float16 storage, rank-4 PPCA, splits, prepared hashtag contributions and cursor commits. Throughput: SQLite 1,759 posts/s; RocksDB 545 posts/s. Every run passed the storage audit and produced the same topology, including across engines.

The terms workload makes six direct calls to the actual server term-ranking function, using vectors from the first six active IDs in the same 100,000-post historical GTE-tiny fixture. It bypasses HTTP, networking and the owner queue. Each request scans the full vocabulary of its routed leaf; this is not a cached label lookup. Fixed historical clock; no ingestion or inference runs concurrently. Both engines returned identical leaf labels/counts/scores and preserved the committed generation.

Mean time per request within the median six-query run: SQLite 687 ms; RocksDB 1656 ms. These are not request-latency percentiles.

**Allocation volume is cumulative Julia-managed allocation, not resident RAM or native RocksDB memory.** Native malloc activity is not captured by Julia’s allocation profiler. Most allocated objects can be reclaimed during the workload.

## Where sampled execution landed

The following shares assign each captured workload stack to its nearest recognized subsystem, starting at the executing end of the stack. Base/runtime helpers are charged to their recognized caller. Columns sum to 100% (rounding aside). They are sampled stack shares, not a precise wall-time or CPU-cycle accounting. Raw C and Julia frames are retained for inspection.

| Exclusive attribution | RocksDB ingestion | RocksDB terms |
| --- | ---: | ---: |
| Row/index handling and adapter iteration | 28.2% | 8.6% |
| Serialization/deserialization | 24.4% | 37.4% |
| Statement dispatch, parsing and projection | 15.5% | 0.0% |
| Key encoding | 6.1% | 4.8% |
| RocksDB binding/native path | 16.0% | 36.7% |
| Journal orchestration (exclusive) | 0.5% | 0.0% |
| Keyword aggregation and ranking | 2.2% | 12.5% |
| Density / linear algebra | 1.8% | 0.0% |
| Other tree / payload work | 5.2% | 0.0% |

The entire journal call path appears in 9.3% of ingestion samples. This **inclusive** figure includes nested native calls and serialization, and must not be added to the table above. The journal can also create later background compaction work that this number does not account for.

Within the native paths, the recorded symbols show:

| Inclusive call path | Ingestion samples | Terms samples |
| --- | ---: | ---: |
| Native RocksDB library frame | 15.6% | 34.4% |
| rocksdb_get call path | 10.1% | 31.2% |
| rocksdb_write call path | 4.8% | 0.0% |
| rocksdb_iter_* call path | 0.4% | 3.2% |

These paths overlap each other and the preceding categories. They count stacks containing the named library/API frame, not calls or exact native elapsed time. Native query cost is mainly on the point-read path in this capture.

## Allocation sites

Allocation profiling ran separately, sampling allocation events with probability 0.0001. The rows below show the largest shares of sampled bytes; rare large allocations can make these estimates noisy. Use the exact baseline totals above for allocation volume.

### RocksDB ingestion

2,943 sampled allocation events.

| Site | Share of sampled bytes |
| --- | ---: |
| `#conditional_children##1 (src/LeafLatentTree.jl:98)` | 9.3% |
| `Serializer (/cache/build/builder-amdci4-4/julialang/julia-ci/usr/share/julia/stdlib/v1.13/Serialization/src/Serialization.jl:27)` | 8.6% |
| `indexkeys (src/RocksMetadata.jl:225)` | 8.2% |
| `key (src/RocksMetadata.jl:41)` | 7.0% |
| `#visitrows#23 (src/RocksMetadata.jl:311)` | 4.1% |
| `normalize_row (src/RocksMetadata.jl:211)` | 3.6% |
| `#indexkeys##0 (src/RocksMetadata.jl:224)` | 3.2% |
| `part (src/RocksMetadata.jl:34)` | 2.9% |

### RocksDB terms

11,035 sampled allocation events.

| Site | Share of sampled bytes |
| --- | ---: |
| `Serializer (/cache/build/builder-amdci4-4/julialang/julia-ci/usr/share/julia/stdlib/v1.13/Serialization/src/Serialization.jl:27)` | 23.5% |
| `key (src/RocksMetadata.jl:41)` | 8.7% |
| `#special_select##0 (src/RocksMetadata.jl:438)` | 7.1% |
| `#get#51 (/home/coll/.julia/packages/RocksDB/BkM2k/src/midlevel.jl:632)` | 6.0% |
| `deserialize_tuple (/cache/build/builder-amdci4-4/julialang/julia-ci/usr/share/julia/stdlib/v1.13/Serialization/src/Serialization.jl:1010)` | 5.2% |
| `deserialize_datatype (/cache/build/builder-amdci4-4/julialang/julia-ci/usr/share/julia/stdlib/v1.13/Serialization/src/Serialization.jl:1567)` | 4.4% |
| `deserialize_datatype (/cache/build/builder-amdci4-4/julialang/julia-ci/usr/share/julia/stdlib/v1.13/Serialization/src/Serialization.jl:1562)` | 4.3% |
| `#terms#18 (server/src/Terms.jl:42)` | 4.2% |

## Interpretation and next step

1. **Benchmark compact, fixed-schema metadata encoding first.** Deserializing NamedTuple type information and symbols is prominent in both profiles. This concerns metadata rows, not Float16 vector compression. Use a versioned experimental store so existing records remain readable.
2. **Reduce repeated row/index construction and string-statement work on ingestion.** `indexkeys`, `normalize_row`, key construction and statement dispatch are visible hotspots. Typed operations and updating only affected indexes should be tested independently.
3. **Remove unnecessary point reads from common-term queries.** Code inspection shows the common ranking still fetches each term’s global count and computes log-lift, although its score uses only leaf support. The profile confirms point-read/deserialization cost, but does not quantify the speedup from removing this work. Distinctive ranking still needs the global data.
4. **Defer transaction-protocol redesign until those costs are measured again.** The journal’s inclusive share limits how much eliminating its foreground work alone could help this fixture. Its indirect background work and larger structural migrations still need separate measurements.

GC pause time is a small part of the baseline duration. Allocation churn matters here through object construction, type reconstruction and copying as well as collection; simply changing GC settings would not address most of that work.

The profiles do not establish a general B-tree/LSM winner. RocksDB compaction/flush threads are native background threads and are not fully captured by Julia’s profiler; OS-cache reads and waits can also appear under native calls. A native-call sample is not proof of physical SSD IO. We did not count fsync calls or attribute SSD write amplification in this experiment.

## Reproduce and inspect

Run from the repository root, sequentially (no competing benchmark processes):

```sh
julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl rocksdb ingestion
julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl sqlite ingestion
julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl rocksdb terms
julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl sqlite terms
python server/benchmark/adapter_profile_report.py
```

- Existing private fixtures: `data/rocks_backend_100k` and `data/server_http_pilot`. The script requires an existing committed store for terms. Synthetic ingestion uses temporary stores.
- Same 32 MiB managed store budget, 4 MiB metadata allowance, and warm OS caches as the earlier comparison. Four Julia default threads; BLAS one thread. Each case runs in a fresh process.
- Time profiling: nominal 1 ms sampling, workload-stack filter excludes unrelated/idle Julia threads. Instrumented durations are stored separately and are not benchmark throughput. Sampling cadence and GC/safepoints can bias shares.
- The initial ingestion captures were reclassified from their saved raw stacks to avoid attributing generic Julia serialization methods to the SQLite precompile image. The correction changes analysis only. Analysis and measured-source hashes are recorded.
- `*-cpu.speedscope.json`: time-sample call stacks, weighted by sample count. `*-alloc.speedscope.json`: sampled allocation stacks weighted by observed bytes. Load these files into a Speedscope-compatible viewer; they contain no post text or vectors.
- `*-cpu-flat.txt`: standard Julia profiler output, including C frames and other threads. `*.json`: baseline runs, category summaries, source hashes and correctness checks.

[Interactive summary](../../output/adapter_profile/demo.html). No storage format, transaction protocol or live configuration was changed by this experiment.
