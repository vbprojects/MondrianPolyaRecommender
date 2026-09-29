# RocksDB backend

The server and disk ingester can use RocksDB for all metadata. Vector/text payloads
remain immutable files. The tree, rank-4 PPCA, Float16 codecs, keyword extraction,
retention rules and public HTTP contract are shared with SQLite.

## Run a new index

```sh
julia --project=server -e 'using Pkg; Pkg.instantiate()'
julia --threads=4,1 --project=server server/bin/serve.jl --config server/config.rocksdb.toml
```

This configuration creates `data/live_rocks`, binds to localhost:8080 and uses the
existing live encoder/sources. `mode = "serve"` disables collection/inference but
keeps maintenance running. Store paths resolve relative to the server TOML.

The switch is `[storage] backend = "rocksdb"`. `"sqlite"` selects SQLite;
`"auto"` detects an existing store and defaults to SQLite for a new directory.
An explicit mismatch fails rather than creating a second database in the same
store. A directory containing both formats is rejected as ambiguous.

For the standalone ingester, use `[disk] backend = "rocksdb"` in its configuration
or `--metadata-backend rocksdb`. Its existing `--backend cpu` selects tree execution,
not the database engine. `examples/inspect_live.jl` recognizes either store type.
RocksDB.jl 1.0.0 is included in the server, ingestion and storage environments.

## Copy an existing SQLite index

```sh
julia --threads=4,1 --project=server server/bin/import_sqlite.jl \
  --source data/live_disk --destination data/live_rocks --cache-mib 256
```

The destination must be new, and the source must be closed by its ingester/server.
The copy acquires the existing writer lock, streams metadata in bounded pages,
copies and hashes payload files, verifies topology/generation/posting integrity,
and publishes the new directory by rename. It preserves post IDs and aliases,
serialized leaf states, source cursors, keyword contributions, generations and any
existing server store/embedding identifiers. It performs no extraction, embedding
inference, re-encoding or PPCA refitting.

The source remains available for rollback. Point the server configuration at the
new directory when choosing to switch. The copy represents the same logical store,
including its UUID: use one active deployment for it rather than allowing both
copies to diverge under the same public identity. Copying is offline and uses a
private staging directory; an interrupted copy can be restarted from the source.
It is not an online or resumable migration protocol.

## Layout and publication

`metadata.rocks/` contains the RocksDB database. It stores typed records and ordered
indexes for identities, leaf slots, sequence-ordered iteration, expiry, block/text
references, keyword dictionaries/contributions, counters and cursors. There is no
SQLite mirror or SQLite transaction journal in a RocksDB store. SQLite.jl remains
a dependency to support the other backend and the copy tool.

The current disk code's internal statement helpers dispatch to native RocksDB
operations. This adapter supports the application's operation vocabulary, not
arbitrary SQL. Unsupported joins/orderings fail explicitly. New storage operations
must have native implementations and backend parity tests.

Metadata format v2 uses fixed-schema binary rows: a version/schema header, null
bitmap, little-endian 64-bit numbers, and length-prefixed UTF-8 strings or byte
arrays. It avoids storing Julia datatype descriptions in each metadata row.
Vector/text files and opaque PPCA/runtime/contribution blobs retain their existing
encoding. Key bytes and ordering are unchanged.

The new reader accepts both v1 serialized rows and v2 binary rows, even within one
store. Opening an existing v1 store for reads does not upgrade it. The first write
transaction journals the v2 format marker together with new or changed records;
rollback restores the prior marker. Older server binaries reject the v2 marker.
Keep an offline backup before deploying if rollback to an older binary is needed.
Existing unchanged rows are not rewritten automatically, so full read-speed and
space gains require compact rows; the benchmark uses an explicit new copy.

Frequent inserts and scalar lookups have a finite set of prepared native handlers.
Key construction writes directly into one byte buffer, and row updates skip
unchanged secondary indexes/counters. Common keyword ranking scans local counts
and dictionary labels without reading per-term global counts; distinctive ranking
still uses global counts. Transaction durability and publication are unchanged.

One store owner and the existing exclusive writer lock protect visibility. Large
splits, merges, rebuilds and expiry passes can exceed a reasonable RAM write batch,
so publication uses a bounded buffer and a **disk-backed undo journal**:

1. Durably mark the transaction open.
2. Flush bounded batches of changed keys together with their original values.
   Subsequent reads inside the mutation see its staged changes.
3. The shared store code fsyncs immutable payloads and their directory, then writes
   the runtime generation and source cursor with the metadata changes.
4. Durably mark the metadata transaction committed.
5. Remove journal entries in bounded pages and clear the marker durably.

On reopen, an open transaction is rolled back; a committed transaction only needs
journal cleanup. Recovery finishes before the store is exposed. Recovery itself
can be interrupted and resumed. Tests kill subprocesses before payload publication,
before/after commit, during recovery and immediately after the RocksDB commit marker.

This is a single-owner transaction protocol. Independent native RocksDB readers
could observe intermediate keys and are not supported. HTTP readers go through
the owner. Adding concurrent snapshots would require a different visibility
protocol and leases for external payload files.

The journal adds writes and recovery work. The old key/value microbenchmark did
not include this protocol, full record indexes, PPCA or full keyword maintenance;
its throughput ratios are not predictions for this backend.

## Memory and maintenance

The database allowance is split between a block cache and at most two memtables.
SST indexes and filters are partitioned into approximately 4 KiB pieces and share the bounded block cache; open table handles are capped
at 128. The application write buffer targets 512 KiB and may exceed that by one
record. Its batches and undo values use the existing work-buffer reservation.
The binding's completed iterator references are periodically removed so they do
not accumulate forever.

`storage_stats` reports metadata backend/bytes, RocksDB cache/memtable/table-reader
usage, buffer peak, and journal flushes. SQLite-specific byte/cache fields are zero
or empty for RocksDB. Managed cache settings are not a Julia/native process RSS cap;
file readers, transient buffers, compiler/runtime and embedding workers must also
be measured.

Payload reclamation is unchanged and preserves encoded bytes. Removing obsolete
metadata creates RocksDB tombstones; physical SST reclamation is handled by its
compactor. This implementation initially disables general RocksDB compression to
keep the baseline explicit. Secondary indexes, opaque serialized state and
compaction state still affect storage and speed; consult the measured results.

## Validation and measurements

```sh
MONDRIAN_TEST_BACKEND=rocksdb julia --threads=4,1 --project=server server/test/runtests.jl
MONDRIAN_TEST_BACKEND=rocksdb julia --threads=4,1 --project=server test/disk_tree_store.jl
MONDRIAN_TEST_BACKEND=rocksdb julia --threads=4,1 --project=server test/disk_keywords.jl
MONDRIAN_TEST_BACKEND=rocksdb julia --threads=4,1 --project=server test/disk_ingestion_runner.jl
MONDRIAN_TEST_BACKEND=rocksdb julia --threads=4,1 --project=server test/disk_crash_process.jl
MONDRIAN_TEST_BACKEND=rocksdb julia --threads=4,1 --project=server test/disk_live_crash.jl
julia --threads=4,1 --project=server server/test/rocks_backend.jl
julia --threads=4,1 --project=server server/test/rocks_optimized.jl
julia --threads=4,1 --project=server server/benchmark/backend_ingestion.jl
```

Unset `MONDRIAN_TEST_BACKEND` to run the existing SQLite suites. The cross-backend
suite exercises repeated splits/merges, edits, deletion, expiry, keywords and copy
parity. The real-data pilot copies the existing 100,000-post GTE-tiny fixture and
serves it at its historical clock. See [benchmark/RocksDB_Results.md](benchmark/RocksDB_Results.md)
for measured results and limits.

The follow-up [adapter profile](benchmark/Adapter_Profile.md) identifies costs
before changing the implementation; reproduction and raw stack files are linked there.
The [optimization results](benchmark/RocksDB_Optimizations.md) measure the compact
format and optimized operations against those archived baselines, with an
[interactive comparison](../output/rocks_optimized/demo.html).

To repeat an HTTP pilot on an existing private fixture (no source collection):

```sh
MONDRIAN_SERVER_PILOT_STORE=data/rocks_backend_100k \
MONDRIAN_SERVER_PILOT_REQUESTS=12 \
MONDRIAN_SERVER_PILOT_OUTPUT=output/rocks_backend/http-rocksdb.json \
julia --threads=4,1 --project=server server/benchmark/http_pilot.jl
```

`server/test/live_rocks_config.toml` provides the separate bounded combined-mode
smoke fixture; run the server with that configuration and `--seconds 30`.
`python server/benchmark/rocks_report.py` rebuilds the report/viewer from saved
measurements. The 100k historical pilot is separate from live ingestion's current
256D encoder and does not change its embedding model.
