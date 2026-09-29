# Mondrian query server

Julia server for the disk-backed tree, with selectable SQLite and RocksDB metadata. It can ingest and serve together,
or serve an existing index with expiry and reclamation running in the background.
Clients retain their interest models, feedback and recommendation policies.

| Endpoint | Result |
| --- | --- |
| `GET /v1/tree` | Versioned cuts, partition probabilities and log densities. Optional `?view=ppca` adds local covariance factors. |
| `POST /v1/nearby` | Up to 25 real post identities, approximate regional cosine ranking, and vector offsets from the query. |
| `POST /v1/terms` | Frequencies for the containing leaf, common or distinctive ranking; new default configuration uses hashtags only. |
| `GET /healthz`, `GET /readyz` | Liveness and storage/background readiness. |

## Run

To inspect an already-running standalone SQLite ingestion without restarting it,
use the [live tree inspector](inspector/README.md). It provides a hierarchy,
probability treemap, on-demand leaf keywords/posts, and process memory readings at
`http://127.0.0.1:8091`.

For persistent Docker deployment on Lightsail or another Linux VM, see the
[deployment guide](../deploy/README.md): combined/serve-only images, Compose,
CPU model preparation, optional HTTPS, backups and an offline smoke test.

From the repository root:

```sh
julia --project=server -e 'using Pkg; Pkg.instantiate()'
julia --threads=4,1 --project=server server/bin/serve.jl --config server/config.example.toml
```

The example binds to `127.0.0.1:8080`, opens `data/live_disk_gte`, and runs combined
mode using the sources in `examples/live_sources.toml`. It **replaces** standalone
ingestion for that store: the writer lock prevents running both simultaneously.
Combined mode uses the existing Python embedding worker/environment and hashtag summaries. It uses pinned 384D GTE-tiny (float ONNX, 256-token masked mean pooling),
Float16 storage and the existing keyword policy. The original Qwen store remains
separate; resuming it requires the explicit Qwen source configuration.

Set `mode = "serve"` to disable source collection and inference. This requires an
existing disk store; it continues expiry and physical maintenance and needs no
Python embedding worker. Paths in the TOML resolve relative to that TOML file.
`--seconds 60` runs a bounded pilot; Ctrl-C shuts down the normal run.

Combined startup checks the keyword fingerprint and finishes any resumable backfill
before listening. A mismatched encoder or keyword policy fails explicitly; there
is no automatic migration from the Arrow index or a different embedding space.

```sh
curl http://127.0.0.1:8080/v1/tree
```

Take `embedding_space_id` and `embedding.dimension` from the tree response, then
send a JSON body to either vector endpoint:

```json
{"embedding_space_id":"ID_FROM_TREE","vector":[0.1,0.2],"limit":25}
```

The two-coordinate example is illustrative: provide **all** coordinates in the
advertised space. Nearby defaults to four candidate leaves and 256 sampled posts;
raise `leaf_budget` to at most 16 and `candidate_budget` to at most 1,024.
A query plus each returned `delta` reconstructs the stored vector approximately.
Nearby returns native identities, not post text or synthetic PPCA points.

## Design and limits

One bounded owner queue serializes store access for either backend, including commits, maintenance
and queries. Source networking and embedding inference happen outside that queue;
commit-time planning checks current identities again. Native keyword preparation
still runs inside the owner before its transaction. Source cursors publish with
the corresponding vectors, identities and keyword updates.

Daily reclamation copies one affected vector frame or 128 text records per owner
job, with durable progress and no re-encoding. Entirely live vector files are
skipped. GC unlinks only after publication while the owner excludes other readers.
Logical expiry, density rebuilds, splits and merges can still take a long single
transaction; request deadlines are not a guarantee of low tail latency during them.

Tree packs use ETags/gzip and disclose snapshot time. Term pagination is signed and
rejects changed generations. Aggregate queries return retryable errors when expiry
is overdue; nearby independently excludes expired candidates. API reads do not
advance generation, cursors or ingestion RNGs.

Default storage cache budget is 256 MiB plus a 32 MiB server buffer allowance.
These are not process RSS limits. HTTP admits eight requests; serialized replies
are capped at 1 MiB with the default allowance. Larger PPCA packs require increasing
`server_budget_mib`; the default partition pack is much smaller. Term ranking scans
one leaf's aggregate rows in bounded pages, so large vocabularies can be expensive.

Read [API.md](API.md) for wire details, [PLAN.md](PLAN.md) for status and remaining
work, and [benchmark/RESULTS.md](benchmark/RESULTS.md) for measured results.
The example defaults to SQLite for a new store. [RocksDB.md](RocksDB.md) describes
the selectable RocksDB backend, its transaction protocol, the separate
[configuration](config.rocksdb.toml), and the explicit SQLite copy command.
Existing indexes are not switched automatically.

[Adapter profiling](benchmark/Adapter_Profile.md) separates warmed ingestion and
keyword-query costs into adapter, serialization and native call paths, with raw
time/allocation stacks and an [interactive summary](../output/adapter_profile/demo.html).
The subsequent [optimization results](benchmark/RocksDB_Optimizations.md) compare
compact metadata and prepared operations against those original measurements.

## Validate and benchmark

```sh
julia --threads=4,1 --project=server server/test/runtests.jl
julia --threads=4,1 --project=server server/benchmark/http_pilot.jl
julia --project=server/benchmark -e 'using Pkg; Pkg.instantiate()'
python server/benchmark/run_ablation.py
```

The HTTP pilot copies the existing 100k keyword fixture to `data/server_http_pilot`,
using a separate SQLite database and immutable hardlinked payloads. The metadata
benchmark uses real identities, term contributions and 128-leaf postings from the
existing fixtures. Both leave the live index unchanged. Full fixture paths and
scope are recorded in the benchmark documentation.
