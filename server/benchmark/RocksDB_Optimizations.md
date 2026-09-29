# RocksDB adapter optimizations

Implemented compact metadata rows, specialized hot operations, direct key construction, unchanged-index elimination, and a common-keyword query path that skips global-count reads and log-lift. The undo journal, durable commits, Float16 payloads and public API remain unchanged.

## Measured before and after

Median of three unprofiled sequential runs after workload warmup and GC, in separate processes. Median reported compilation time is zero. Nonzero compilation samples, retained in timing: sqlite-ingestion run 1: 9.6 ms. The before values are the saved adapter profile baselines; the after values are a fresh run with the same fixtures. This is a combined optimization result, not a causal ablation of each change.

| Workload | Before | After | Speedup | Allocated MiB before → after |
| --- | ---: | ---: | ---: | ---: |
| rocksdb-ingestion | 3.756 s | 1.963 s | 1.91× | 1,699.6 → 751.6 |
| sqlite-ingestion | 1.165 s | 1.289 s | 0.90× | 654.4 → 654.4 |
| rocksdb-terms | 9.936 s | 4.000 s | 2.48× | 4,463.6 → 1,563.7 |
| sqlite-terms | 4.120 s | 3.102 s | 1.33× | 1,919.5 → 1,351.6 |

RocksDB ingestion improves from 545 to 1,043 posts/s. SQLite now measures 1,589 posts/s. Common-term requests average 667 ms within the median six-request run, down from 1656 ms. These are not latency percentiles.

**Allocation volume is cumulative Julia allocation, not peak RAM.** Native memory allocation and physical SSD IO are not measured by this harness.

- Ingestion: 2,048 normalized synthetic 384D vectors, seed 82, eight durable 256-post batches, Float16, rank-4 PPCA, splits, prepared hashtags and committed source cursors. Embedding inference and keyword extraction are excluded.
- Terms: six direct server common-term calls against the same 100,000-post GTE-tiny fixture, five leaves, generation 102, fixed historical clock. The optimized RocksDB fixture is a new compact-format copy. These calls scan the full leaf vocabulary and bypass HTTP, owner queue and networking.
- Same 32 MiB managed budget with 4 MiB allocated to metadata; warm OS caches, no cache dropping. Four Julia default threads, one BLAS thread. Workstation: AMD Ryzen 7 7800X3D, 16 logical CPUs exposed by the host.
- SQLite also benefits from skipping the unnecessary common-term global join. Distinctive ranking still reads global counts and is correctness-tested, but has no speed claim from these timings.

## Existing records and storage

The new query path on the existing v1 RocksDB fixture takes 6.211 s for six requests, without rewriting records or upgrading its format marker. Compact rows take 4.000 s. This separates the practical read-only deployment case from a fresh compact store, although the two stores have different SST histories.

The new 100k copy occupies 195.1 MiB of metadata and 300.5 MiB including payloads at measurement time. Vector and text files total 105.4 MiB and were copied unchanged. Metadata sizes include engine files and depend on compaction history; this is not a normalized write-amplification or compression-ratio study.

The reader supports old serialized and new fixed-schema rows in the same store. The first write transaction upgrades the store format marker atomically; unchanged old rows are not automatically rewritten. Older binaries reject upgraded stores. Keep an offline pre-upgrade backup when rollback to an older binary is required. No live store was migrated or switched.

## Correctness

All before/after and SQLite/RocksDB topology and returned-term digests match. Every synthetic ingestion run passes the storage audit; every terms run preserves its committed generation. The import verifies identities, payload hashes, topology and posting integrity.

Recorded validation: 6,146 passing assertions across 9 suite runs. Coverage includes wire-format round trips, ordered key compatibility, mixed old/new records, rollback of format upgrades, forced eviction, structural changes, keywords, live ingestion, HTTP, and process-crash recovery. See [validation.json](../../output/rocks_optimized/validation.json) for individual suites.

## Recommendation

Use the optimized adapter for further RocksDB experiments, while retaining SQLite as the default until sustained ingestion, compaction and concurrent server workloads establish the tradeoff. Keyword queries still scan a leaf vocabulary: a bounded summary cache or maintained top-term summaries are separate future work. These measurements do not establish an inherent LSM advantage or a production capacity guarantee.

## Reproduce

Run from the repository root with the private 100k fixture available; use a new destination for the import:

```sh
julia --threads=4,1 --project=server server/bin/import_sqlite.jl --source data/server_http_pilot --destination data/rocks_optimized_100k --cache-mib 32
export MONDRIAN_PROFILE_OUTPUT=output/rocks_optimized
export MONDRIAN_PROFILE_ROCKS_STORE=data/rocks_optimized_100k
export MONDRIAN_PROFILE_BASELINE_ONLY=1
for workload in ingestion terms; do
  for backend in rocksdb sqlite; do
    julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl "$backend" "$workload"
  done
done
MONDRIAN_PROFILE_OUTPUT=output/rocks_optimized/legacy_rows MONDRIAN_PROFILE_ROCKS_STORE=data/rocks_backend_100k julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl rocksdb terms
julia --threads=4,1 --project=server server/test/rocks_optimized.jl
python server/benchmark/optimized_report.py
```

[Interactive comparison](../../output/rocks_optimized/demo.html) · [Original profiles](Adapter_Profile.md) · [Backend and compatibility details](../RocksDB.md)
