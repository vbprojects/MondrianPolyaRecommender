# Disk-backed live ingestion with hashtag summaries

The default live path now combines Float16 post vectors, immutable payload files,
SQLite identity/posting indexes, rank-4 leaf PPCA, exponential decay and
hashtag summaries. Routing and density factors use the existing disk
tree; this path performs tree/storage work on CPU. GPU embedding inference remains
optional.

The default live encoder is now `TaylorAI/gte-tiny` at revision
`4cc5e73d86a67c601897257b467187234aa3bca3`: float ONNX inference, 256-token
truncation including special tokens, masked mean pooling, normalized 384D Float32
outputs, then Float16 storage. Transformer microbatches are capped at 32 posts;
ONNX CPU inference uses two intra-op threads and one inter-op thread.

The original Qwen index remains in `data/live_disk`. GTE uses the separate
`data/live_disk_gte` store and fresh source cursors. Existing posts are not
re-embedded automatically. A configured/stored model mismatch fails before
keyword backfill or source ingestion; changing a TOML dimension alone is insufficient.
To resume Qwen explicitly:

```sh
LIVE_CONFIG=examples/live_sources.qwen.toml ./scripts/run_overnight.sh 12h
```

The GTE bridge was checked against an explicit Float32 PyTorch reference on 36
mixed-length inputs including Unicode and truncation cases. Maximum absolute
coordinate error was `1.05e-7`; this is inference parity, not a retrieval-quality
benchmark. See `test/test_live_gte.py` and `test/live_gte.jl` for protocol,
Float16 persistence/restart, and wrong-encoder rejection checks. PyTorch is only
needed for the reference test, not live inference.

## Run

From the repository root:

```sh
julia --project=environments/ingestion -e 'using Pkg; Pkg.instantiate()'
# Existing embedding/verification Python environment is still required.
# Native keyword extraction does not require Python YAKE.
julia --project=environments/ingestion examples/ingest_live.jl \
  --config examples/live_sources.toml --max-events 256 --seconds 10

# Continuous ingestion, detached and time-limited.
./scripts/run_overnight.sh 12h
tail -f data/live_runs/latest/ingest.log

# Interactive read-only observer: leave ingestion running.
julia --threads=2,1 --project=server server/bin/inspect_live.jl \
  --store data/live_disk_gte --run-dir data/live_runs/latest --port 8091
# Open http://127.0.0.1:8091

# The older CLI inspector still requires stopping the writer first.
julia --project=environments/ingestion examples/inspect_live.jl data/live_disk_gte
```

The [live inspector guide](../server/inspector/README.md) explains the linked
hierarchy/treemap, leaf terms and examples, memory readings, and snapshot limits.

The default configuration uses `data/live_disk_gte`, seven-day query eligibility,
six-hour density/keyword half-lives, a 256 MiB managed storage budget and
append-oriented Float16 blocks. It enables Bluesky. The existing source
configuration also supports Mastodon, ActivityPub, Nostr and Threads.
The cache budget excludes Julia runtime, the embedding worker, and source buffers.

Use `--cache-mib 32` to force cache pressure. Use `--embedding-device cuda`
(or `EMBEDDING_DEVICE=cuda` with the overnight launcher) for GPU inference.
The disk path rejects `--backend cuda`; existing device kernels remain available
through the legacy Arrow runner.

## Publication and maintenance

Each bounded source batch first selects the latest valid event per identity,
checks aliases/deletion ownership, extracts keywords and embeds changed text.
Exact duplicate replay performs neither inference nor extraction. Metadata refreshes
reuse cached keyword contributions when their content/tags/language digest matches.
Hashtag-only edits can change terms without changing vector bytes or content age.

One SQLite transaction publishes post changes, keyword contributions, tree changes,
logical expiry and the acknowledged source cursor. Payloads are flushed before
publication. Preparation failure leaves the cursor unchanged; a transaction failure
terminates the writer, which must reopen the store before continuing. Recovery
replays from the last committed cursor. No process-wide post catalog is loaded.

Quiet/failed source rounds also expire records and decay/merge the tree. Daily
physical reclamation is separate: vector compaction copies frozen bytes, and
bounded text compaction copies surviving records out of partially obsolete files.
An interrupted reclamation is retried on restart. Logical expiry excludes posts
and their keyword contributions before physical reclamation. SQLite free pages are
reused; compaction does not promise a smaller SQLite file.

Daily compaction remains proportional to retained data and can pause source
collection. This is the disk prototype wired into ingestion, not a weekly-scale
throughput guarantee. Tombstones currently remain on disk; a bounded tombstone
retention/GC policy is still separate work.

## Keywords and access

`[keywords] enabled=true` with `method="hashtags"` selects `HashtagsOnlyExtractor()`.
Up to four distinct hashtags from platform metadata and inline text contribute
once per post. Unicode/case normalization and existing decayed common/distinctive
rankings are retained. Posts without hashtags contribute no terms; processed
coverage does not mean every post has a label.

`method="julia_yake"` remains available for experiments and legacy stores. An omitted
method retains the legacy YAKE default. Extractor fingerprints prevent silently
mixing policies. Backfill uses bounded 128-post pages and does not recompute embeddings.

A store with a different keyword fingerprint fails with an explicit rebuild error.
For an intentional switch during ingestion startup, pass `--keyword-rebuild` once. Interruption is recoverable:
restart **without** that flag to continue the remaining backfill. Do not leave the
flag on a recurring launch command.


For an offline switch on collected data, stop ingestion, set the desired method
in the config, then run:

```sh
julia --threads=2,1 --project=server server/bin/rebuild_keywords.jl \
  --store data/live_disk_gte --config examples/live_sources.toml --rebuild
```

This SQLite utility starts neither source collection nor an embedding worker.
It checks a streaming digest of posts, vectors' block references, leaf states,
routing, file registrations and source cursors before/after migration. Interrupted
runs resume with the same command **without `--rebuild`**. Partial coverage is
visible in the inspector. It uses the store's committed clock, preserving content
age; physical database shrinkage is separate from this logical policy change.

Existing retrieval interfaces apply directly:

```julia
include("examples/disk_live_modules.jl")
using .DiskTreeStore
s = open_store("data/live_disk_gte")
try
    leaf = first(n.id for n in values(s.nodes) if n.axis == 0)
    summary = leaf_keywords(s, leaf; ranking=:distinctive, limit=10)
    # With a query vector from the same pinned encoder:
    # selected = sample_ids(s, query; limit=20, leaf_budget=4)
    # posts = fetch_records(s, selected.ids)
    # vectors = fetch_vectors(s, selected.ids)
finally
    close_store(s)
end
```

The inspector reports storage, keyword coverage and common labels per leaf.
It acquires the store's writer lock; no HTTP serving layer is introduced here.

## Legacy checkpoints

Existing `data/live/CURRENT` Arrow stores are preserved. Use the original index
configuration and `--storage arrow --store data/live` to continue them, or invoke
`examples/ingest_arrow_live.jl` directly. Do not apply the new seven-day default
configuration to a legacy one-day checkpoint.

Without an explicit storage choice/configuration, an existing Arrow root is
detected. The default supplied config explicitly selects disk storage and refuses
an Arrow root. There is no automatic Arrow-to-SQLite migration. The overnight
launcher defaults to disk; select `LIVE_STORAGE=arrow`, `LIVE_STORE=data/live`
and a matching legacy `LIVE_CONFIG` to use it for an old checkpoint.

See [original live source/Arrow documentation](Live_Ingestion.md) for source
coverage, credentials and identity details.

## Validation

```sh
julia --project=environments/ingestion test/disk_ingestion_runner.jl
julia --project=environments/ingestion test/disk_live_crash.jl
julia --project=environments/storage test/disk_tree_store.jl
julia --project=environments/storage test/disk_keywords.jl
julia --project=environments/ingestion test/ingestion_runner.jl
```

The integration and regression suites passed **3,520 assertions**, including
process exits before files, before commit and after commit. A real-embedding
mixed-platform replay retained 66 posts across two leaves with 100% keyword
coverage. Replaying the same source after restart left the generation unchanged.
Two bounded live Bluesky runs received 320 messages and retained 301 posts across
two leaves, with full keyword coverage after restart. Raw logs and
configuration are in [output/live_disk](../output/live_disk/).

These are correctness/connectivity tests, not sustained 100 posts/s or seven-day
capacity measurements. The native extractor's separate throughput benchmark
does not establish the throughput of embedding, durable ingestion or compaction.

## Combined ingestion and query server

The implementation in [server/README.md](../server/README.md) can replace the
standalone ingester for the same disk store while exposing tree packs, approximate
nearby post IDs/vector deltas, and combined keyword/hashtag frequencies. Use its
`combined` mode; do not run two writers against the same directory. `serve` mode
runs queries and maintenance without source collection or embedding inference.
The server keeps the current encoder and keyword policy. [RocksDB is now selectable](../server/RocksDB.md), with an explicit offline copy
command for existing SQLite stores. The live store is not migrated automatically.
