# Server and metadata-engine results

Historical metadata-only ablation: the server now supports both engines. See [RocksDB_Results.md](RocksDB_Results.md) for the integrated backend measurements and [../RocksDB.md](../RocksDB.md) for usage. No live index was switched automatically.

## Server pilot

The real fixture contains 100,000 Float16 384D GTE-tiny posts in 5 existing leaves. This is an offline historical space; live ingestion keeps its existing 256D encoder. Tests used one local client, warm OS caches, a 32 MiB managed store budget, and a separate 32 MiB server buffer allowance. Each endpoint received 32 measured requests after compilation warmup, using vectors from the first 32 active posts.

| Endpoint | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| /v1/tree | 6.3 ms | 6.4 ms | 7.4 ms |
| /v1/nearby | 270.8 ms | 309.8 ms | 321.3 ms |
| /v1/terms | 548.9 ms | 637.8 ms | 639.4 ms |

All nearby responses contained 25 real post IDs. Median per-request vector blocks touched: 188; median vector file bytes read: 42.8 MiB. These are application file reads, **not physical-device read measurements**. The OS cache was warm; small application caches still evicted blocks.

The partition pack transferred 5,863 JSON bytes; the optional PPCA pack was 259,888 bytes. Nearby JSON responses were approximately 128 KiB. Peak process RSS was 869 MiB, including Julia compilation/runtime and in-process HTTP clients; the managed budgets are not process limits.

Queries left the committed generation unchanged. The pilot used a fixed historical clock, so it does not measure simultaneous ingestion, source lag, daily expiry pauses or a latency SLO. Global-neighbor recall was not measured in this pilot: the endpoint deliberately samples a bounded regional pool.

## Metadata-engine ablation

Host: AMD Ryzen 7 7800X3D 8-Core Processor, julia version 1.13.0. Three seeds (17, 29, 43), 100,000 real identities and term contributions, and the existing 128-leaf layout. Engine processes ran sequentially with alternating engine order. Source fixtures were joined by post ID, native identity and content digest.

Both engines received the same key/value representation: identities, post locations, dense leaf posting slots, leaf/global term counts and a source cursor. SQLite used a WITHOUT ROWID key/value table, WAL and synchronous=FULL. RocksDB used sync=true WAL writes, no general compression, a 16 MiB block cache and 8 MiB write buffer. SQLite requested a 32 MiB page cache. These native cache settings do not enforce identical total memory usage.

Ingestion used durable batches of 1,024. After each batch, reads fetched 256 posting slots, 256 post locations and the first 25 term keys. This term-prefix read is **not** the server’s support-ranked vocabulary scan. Maintenance moved 128 posts to another leaf and back, then deleted 1,024 posts; each operation was one atomic batch. This exercises posting maintenance, not full density fitting or actual geometric split decisions.

Values below are medians across three runs; read percentile rows are medians of each run’s percentiles.

| Measure | SQLite | RocksDB |
| --- | ---: | ---: |
| Durable ingestion commit time | 7.40 s | 1.23 s |
| Commit-only ingestion throughput | 13,522 posts/s | 81,055 posts/s |
| Fixture update preparation + writes + interleaved reads | 9.00 s | 2.39 s |
| Metadata read p50 | 5.52 ms | 1.34 ms |
| Metadata read p95 | 6.82 ms | 1.99 ms |
| Ingestion OS write bytes, including settle | 2,233.7 MiB | 139.7 MiB |
| Final metadata files | 50.1 MiB | 26.8 MiB |
| Peak process RSS, including fixtures/JIT | 640.4 MiB | 663.6 MiB |

| Maintenance | SQLite | RocksDB |
| --- | ---: | ---: |
| 128-post relocation | 411.6 ms | 401.9 ms |
| Merge back | 18.3 ms | 15.0 ms |
| 1,024 deletions | 149.5 ms | 60.8 ms |

The first relocation includes first-call Julia compilation of the migration helper; it is not a steady-state split latency. Merge-back reuses that compiled helper. Neither includes PPCA or geometric tree work.

RocksDB’s commit-only throughput was 6.0× SQLite’s in this fixture. SQLite recorded 16.0× the ingestion OS write bytes. This supports evaluating an LSM backend, rather than assuming it will be faster.

OS write counters include checkpoint/manual compaction work at the end of ingestion. RocksDB received full manual compaction; SQLite received WAL checkpointing, **not VACUUM**. SQLite can reuse freed pages on later writes. Final file sizes should not be interpreted as equal physical-reclamation policies. OS counters are not SSD internal/NAND write amplification.

Correctness passed in all six runs: every remaining posting slot mapped back to its post/leaf, deleted identities were absent, leaf/global term counts matched reference counts, and committed cursors survived reopen. The final count digests matched across engines/seeds. These are clean-restart checks; the metadata microbenchmark does not constitute full RocksDB tree crash-recovery validation.

## Recommendation

The original recommendation was to implement and validate the full RocksDB backend before switching. That implementation is now available; consult [RocksDB_Results.md](RocksDB_Results.md) for its actual results. These simplified metadata numbers alone do not justify a live switch.

For query latency, also address vector block locality and repeated term-summary scans. The measured API costs are substantially larger than either engine’s bounded metadata-read timing. Replacing SQLite alone will not remove those costs.

## Scope and reproduction

- Metadata fixture preparation and term-count bookkeeping use in-memory reference structures. There is no embedding inference, vector/text payload IO, PPCA update, full keyword dictionary storage, retention chronology or long-running LSM steady state in the engine benchmark.
- The HTTP fixture has five broad leaves; the engine fixture uses 128. They answer different questions and must not be presented as interchangeable timings.
- The server suite covers API geometry, serialization, stale summaries, generation/RNG invariance, cursor pagination, concurrent HTTP, preparation failures, overload, queued deadlines, and compaction/restart. The existing ingestion crash suite additionally tests publication before files, before commit and after commit.
- A separate bounded combined-mode run collected live Bluesky batches, with vectors/keywords/source cursors committed together. It was a smoke test, not a sustained throughput benchmark.
- No cold-device, million-post, weekly-scale, concurrent-client capacity or real recommendation-quality claim follows from this run.

Run commands are in [../README.md](../README.md). [../../output/server/demo.html](../../output/server/demo.html) contains an interactive comparison. Raw runs, per-request HTTP timings and the pinned configuration live in `output/server/`. Julia dependency manifests are pinned in this folder and the server folder.
