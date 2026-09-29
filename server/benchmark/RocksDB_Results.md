# Integrated RocksDB backend results

RocksDB is now selectable for the complete disk tree and HTTP server. The existing live store was not switched. This prototype passes lifecycle, recovery and cross-backend parity checks, but these full-backend measurements favor SQLite for ingestion and metadata size. The earlier metadata-only 6× result does not carry over to this implementation.

## 100,000 real posts

An explicit offline import copied the existing historical GTE-tiny fixture: 100,000 Float16 384D vectors, five leaves, 346,145 dictionary terms and 732,333 leaf/global term-count records. Payload hashes, topology and generation were preserved; no vectors were re-encoded. The destination has no SQLite metadata database.

Each engine served 12 requests per endpoint after warmup, sequentially by engine, with the same source vectors and historical clock. Connections are fresh for each request to avoid idle pool expiry across compilation warmup. Regional candidate sampling remains randomized. These are small, single-client, warm-OS-cache measurements, not cold-device results, a multi-seed performance ablation, or a latency SLO. The pilots did not run ingestion or inference, but background correctness validation was active during part of the workstation runs; timings are indicative rather than isolated performance estimates.

| Endpoint | SQLite p50 / p95 | RocksDB p50 / p95 |
| --- | ---: | ---: |
| /v1/tree | 6.6 / 7.6 ms | 6.5 / 7.7 ms |
| /v1/nearby | 322.3 / 366.5 ms | 327.1 / 372.0 ms |
| /v1/terms | 624.3 / 700.0 ms | 1712.3 / 1952.5 ms |

| Footprint at measurement | SQLite | RocksDB |
| --- | ---: | ---: |
| Metadata files, including engine logs | 113.2 MiB | 279.3 MiB |
| All store files | 218.6 MiB | 384.7 MiB |
| Process peak RSS, includes Julia/runtime and HTTP client | 996.6 MiB | 956.5 MiB |

All nearby responses returned 25 IDs and reads left the generation unchanged. Both engines share the same scattered vector blocks and full leaf-vocabulary ranking. Those costs remain regardless of metadata engine. Reported vector file reads may be served by the OS cache; they are not SSD device traffic.

The managed store budget was 32 MiB, including a 4 MiB database allowance, plus a separate 32 MiB server-buffer allowance. RocksDB uses a 2 MiB block cache (also for partitioned SST index/filter blocks), up to two 1 MiB memtables and a 512 KiB application write-buffer target. These allowances are not process RSS limits. Native usage is in the raw results.

RocksDB general compression is disabled. Verbose serialized records, ordered secondary indexes, WAL/compaction state and the undo journal affect its footprint. The RocksDB pilot received an explicit full SST rewrite before measurement to apply partitioned indexes/filters to the initially imported files; SQLite did not receive VACUUM. File sizes describe these actual stores, not an equal-reclamation or inherent B-tree versus LSM ratio.

## Integrated ingestion smoke measurement

2,048 seeded synthetic normalized 384D vectors, prepared hashtag contributions, eight durable batches of 256, Float16 storage and the common rank-4 PPCA tree. A 256-post warmup preceded each engine. Embedding inference and keyword preparation are excluded. Both produced the same 25-leaf tree.

| Engine | Durable storage/tree ingestion | Total measured time |
| --- | ---: | ---: |
| sqlite | 1507 posts/s | 1.359 s |
| rocksdb | 426 posts/s | 4.803 s |

Both clear 100 posts/s in this small fixture. This is one smoke run on a Ryzen 7 7800X3D host, with four Julia worker threads and BLAS limited to one thread; it does not establish sustained live capacity. Recovery tests ran separately as background validation on this workstation during the measurement period, so timing differences should be treated as indicative. The RocksDB path includes application statement dispatch, secondary-index maintenance and an undo journal absent from the earlier microbenchmark.

## Correctness

- RocksDB API: 172 assertions passed.
- RocksDB storage lifecycle: 753 assertions passed.
- RocksDB keyword lifecycle and adapter contracts: 1912 assertions passed.
- RocksDB ingestion runner: 826 assertions passed.
- RocksDB process crashes: 42 assertions passed.
- RocksDB source publication crashes: 18 assertions passed.
- Cross-backend parity, copy and interrupted recovery: 63 assertions passed.
- SQLite API regression: 172 assertions passed.
- SQLite storage regression: 753 assertions passed.
- SQLite keywords regression: 1912 assertions passed.

Checks cover source cursor publication, keyword counts, splits/merges, edits/deletion/expiry, compaction, restart/eviction, API semantics, exact decoded-vector parity and explicit import. Subprocess crashes cover open/committed transactions and interruption during recovery itself. These are process-kill tests, not physical power-loss or filesystem-fault certification.

A separate bounded combined-mode run collected two 64-event live Bluesky batches using the existing 256D encoder and Julia YAKE, answered HTTP health, and shut down cleanly. Reopening audited 117 active posts with 100% keyword coverage and 3 persisted cursor records. This is connectivity/restart validation, not a sustained ingestion benchmark. Raw records are in `live-smoke.json` and `live-restart.json`.

## Recommendation

Keep SQLite as the default while testing RocksDB explicitly. The native backend is functional; it is not currently an efficiency upgrade for this full workload. An initial unpartitioned SST configuration thrashed its tiny index/filter cache (terabytes of OS-cache reads); that run was stopped. Partitioned indexes and filters are now the default, keeping metadata fetches small. Next optimization candidates are typed operations in the hot path, compact record encoding, metadata compression and reducing journal/secondary-index writes. Vector locality and cached or incrementally maintained term rankings are separate shared query improvements.

RocksDB still writes a WAL, SST files, compactions and the application undo journal. Leaf locality does not eliminate write amplification. The journal is the current tradeoff for bounded-memory, crash-safe atomic structural updates under one store owner.

See [../RocksDB.md](../RocksDB.md) for run/import commands and the single-owner visibility contract. [Interactive measurements](../../output/rocks_backend/demo.html) and raw JSON are in `output/rocks_backend/`; configuration and both environments are pinned. Million-post, seven-day, sustained mixed-query/ingestion capacity and cold-device performance remain unmeasured.
