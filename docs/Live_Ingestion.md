# Live ingestion

The current default is the [disk-backed live pipeline](Disk_Live_Ingestion.md):
Float16 vectors, rank-4 PPCA, SQLite publications and per-leaf hashtag
summaries, using `data/live_disk_gte`. The overnight launcher uses this path.
The live default is pinned 384D GTE-tiny with float ONNX inference; the original
Qwen store remains at `data/live_disk` with `examples/live_sources.qwen.toml`.

A [native Rust CPU INT8 worker](../native/gte-worker/README.md) is available through
`examples/live_sources.rust.toml`. It performs embedding and Nostr verification without
a Python process. It uses a distinct embedding identity and a new
`data/live_disk_gte_int8` store; existing FP32/Qwen configurations keep their Python backend.

The sections below document the **legacy Arrow backend** and shared source/identity
semantics. For those commands, invoke `examples/ingest_arrow_live.jl` directly
(or add `--storage arrow`) and use the configuration matching the original store.
The new `examples/live_sources.toml` selects disk storage and seven-day retention.

## Legacy Arrow checkpoints

The Julia ingestion path combines the adaptive tree and leaf PPCA density with **independent frozen centroid + 8-bit residual post codes**. It saves native post identities, text, metadata, codecs, tree state, and source cursors together in Arrow IPC files. CPU is the default; the numerical kernels also run on CUDA. This is a bounded ingestion prototype, with live source readers and restart support, not a claim of complete platform coverage or production firehose throughput.

## Run it

Use the isolated environment; the original dependency-light demos are unchanged:

```sh
julia --project=environments/ingestion -e 'using Pkg; Pkg.instantiate()'
# Reuse the CPU or GPU embedding environment from docs/Bluesky_Embeddings.md.
# Only if no embedding environment exists, prepare a fresh CPU environment:
# python3 -m venv .venv-embeddings
# .venv-embeddings/bin/pip install -r scripts/requirements-embeddings.txt
.venv-embeddings/bin/pip install -r scripts/requirements-live.txt

# One bounded public Bluesky subscription; resumes the existing Arrow checkpoint.
julia --project=environments/ingestion examples/ingest_live.jl \
  --config examples/live_sources.toml --max-events 256 --seconds 10

# Deterministic source fixtures: actual pinned embeddings, no API credentials.
julia --project=environments/ingestion examples/ingest_live.jl \
  --replay test/fixtures/live_sources.jsonl --config test/fixtures/live_config.toml \
  --max-events 16 --store data/live_fixture

# Inspect the catalog, memory, cuts, and portable decoded-vector hash.
julia --project=environments/ingestion examples/inspect_live.jl data/live_fixture
.venv-embeddings/bin/python scripts/check_live_arrow.py data/live_fixture
```

The legacy default ONNX bridge uses `futur/Qwen3-Embedding-0.6B-model2vec-onnx` pinned to revision `a3f60a85a2570e7976ea7f2d6505ad3e5152a96f`: normalized 256D Float32 embeddings. A long-lived Python worker provides the existing ONNX runtime and vetted BIP-340 verification via `coincurve`; ingestion, tree updates, codecs, and storage are Julia. Embeddings and Nostr verification share a serialized worker protocol. The worker rejects oversized batches and invalid vectors; it does not log post text.

`--backend cuda` selects GPU tree/codec kernels. `--embedding-device cuda` independently selects GPU ONNX inference using the previously configured GPU Python environment. Neither is required. Defaults are one round, CPU kernels and CPU embeddings. `--rounds N` repeats bounded rounds; `--loop` explicitly runs continuously until interrupted. Source readers take turns, keeping memory bounded and applying backpressure while embedding and saving. No background service is installed.

The source configuration is in [examples/live_sources.toml](../examples/live_sources.toml). Uncomment and configure only the sources you want. Tokens are read from named environment variables and are neither placed in checkpoint cursors nor printed in diagnostics. Give each subscription a stable, distinct `name`; use a new name when changing its platform, server, filter, or protocol version. A checkpoint pins its model and index configuration; use a new store to change them.

### Overnight run

After installing the dependencies above, the Linux launcher runs detached for 12 hours by default, with CPU tree kernels and embeddings, 1,024-event collection budgets, and the existing configuration's sources:

```sh
./scripts/run_overnight.sh 12h
tail -f data/live_runs/latest/ingest.log
```

It resumes `data/live`, saves timestamped logs under `data/live_runs`, and prints the PID for stopping early. Closing the terminal does not stop it; the workstation must remain awake. The default configuration enables **Bluesky only**. Enable other configured sources before launching. Do not change source filters or index settings in the existing store during a run.

For the previously configured GPU embedding environment, use `EMBEDDING_DEVICE=cuda ./scripts/run_overnight.sh 12h`; `TREE_BACKEND=cuda` separately selects GPU tree kernels. `LIVE_CONFIG`, `LIVE_STORE`, and `LIVE_LOG_DIR` override paths, resolved relative to the repository. See `./scripts/run_overnight.sh --help` for all options. The script does not install packages or repeatedly restart a failed writer; source read retries remain handled by the Julia runner.

At the duration limit, GNU `timeout` sends TERM to Julia and its embedding worker, with forced cleanup after 60 seconds if needed. This can abandon the current uncommitted batch; the last published Arrow checkpoint and its matching source cursor remain the restart point. Re-running the launcher resumes that store. Full snapshots still rewrite all retained data, so this overnight run measures the current prototype's behavior rather than guaranteeing full-network capture. After it stops, inspect the results with `julia --project=environments/ingestion examples/inspect_live.jl data/live`.

## Identity: native keys outside the tree, small integers inside it

The unique external key is `(identifier_kind::UInt8, native_id)`. The index allocates a monotonically increasing `UInt64` post ID and never reuses it, including after expiry. Tree memberships, code rows, and client candidate references use this integer. It occupies **8 bytes per reference**, or 8 MB per million references, before container overhead. These integers are unique within one persistent index; cross-instance protocols would also need a store identity. The native ID is kept once in the catalog, plus any proven aliases; a hash alone would not let us retrieve the original post.

| Source | Native identity retained | Storage and identity rule |
|---|---|---|
| ActivityPub / Mastodon | Full ActivityPub **object** `id`; Mastodon API `status.uri` | UTF-8 bytes. Mastodon's `status.id` is server-local; keep `(API origin, local ID)` only as an alias. The surrounding `Create` activity has a different identity. |
| Threads | API media object's `id` | Opaque string, preserved exactly, including leading zeroes. Keep `shortcode` and permalink as metadata. |
| Nostr | Event `id` | 32 raw SHA-256 bytes on disk rather than 64 hex characters. The same event from two relays has one identity. This reader indexes ordinary kind-1 notes and kind-5 deletion requests, not replaceable/addressable event families. |
| Bluesky | `at://<DID>/app.bsky.feed.post/<rkey>` | Lossless UTF-8; use DID rather than mutable handle. The record key alone is not globally unique. Keep CID/revision separately in metadata. |

These rules follow [ActivityPub object identifiers](https://www.w3.org/TR/activitypub/#obj-id), [Mastodon Status fields](https://docs.joinmastodon.org/entities/Status/), [Meta's Threads media API](https://www.postman.com/meta/threads/request/34203612-416ad6d2-214e-415a-bfe4-98c959fe25ad), [Nostr NIP-01](https://github.com/nostr-protocol/nips/blob/master/01.md), and the [AT-URI specification](https://atproto.com/specs/at-uri-scheme). AT Protocol explicitly discourages treating TID-shaped record keys as globally unique integers: [record key rules](https://atproto.com/specs/record-key).

`+` is legal in generic URI paths, so it can appear in an ActivityPub object ID ([RFC 3986](https://www.rfc-editor.org/rfc/rfc3986#section-2.2)). An `id+platform` string can still be parsed unambiguously by splitting at the **last** plus if platform labels cannot contain plus, but separate typed columns avoid delimiter and escaping concerns. Preserve native values; do not lowercase arbitrary URL paths, remove query strings, or convert opaque decimal-looking IDs to numbers.

The transport platform and identifier namespace are separate. A Mastodon-delivered object still has ActivityPub identity. A Threads API ID and a federated ActivityPub object ID can be aliases only when an authoritative mapping establishes that they identify the same object. This code does not infer cross-platform equivalence from identical text, usernames, or URLs that merely look related.

## What the kernels and index retain

There are three independent kinds of state:

1. **Tree geometry and Pólya masses:** cuts, child links, bounds, clocks, split/merge counters, and decayed evidence.
2. **Density:** rank-4 PPCA state per active leaf, with mass, mean, scatter diagonal, low-rank directions and eigenvalues. Existing bounded raw-vector reservoirs support future cuts.
3. **Actual posts:** immutable batch centroid, one Float32 scale per post, and one byte per coordinate. Post ID and current region membership are separate arrays.

For a retained post, reconstruction is `centroid + scale × signed_residual`. Residual integers are in `[-127,127]`, stored as bytes `q + 127`; 255 is invalid. The scale is the maximum absolute residual divided by 127, with a positive Float32 floor. Away from floating-point rounding, each coordinate's quantization error is at most half that post's scale. Small Float32 roundoff remains. Decoding does not renormalize vectors; this keeps reconstruction and routing consistent.

New posts are sorted by leaf and packed into bounded centroid blocks. A block may span leaves, preventing one mostly empty decoder per leaf in a small microbatch. Decoders are frozen after encoding. A split or merge reroutes coded posts directly through the new cuts and changes only memberships; it never projects old post codes into a new PPCA basis or repeatedly requantizes them. Expiry slices surviving bytes and releases empty blocks. Very small batches and partially expired blocks still amortize centroid storage poorly; no fixed compression ratio is promised.

`CentroidPostKernels.jl` adds portable kernels for centroid fitting, quantization, reconstruction, routing directly from codes, candidate distances without a dense reconstruction matrix, and weighted PPCA updates. The existing tree kernels continue to handle routing, decay, Pólya allocation, split initialization and merge moments. Host code still coordinates topology, identity maps, Arrow I/O, source reading, and the existing thin-SVD merge work. This is not a wholly GPU-resident data structure.

Scheduled fitting now preserves `_initial` when merging is enabled. Previously that path could discard an earlier fitted batch instead of continuing it. Continuation validates dimensions and the relevant density/reservoir configuration. CPU/GPU parity tests include continued arrivals after decay-induced merges.

## Time, edits, deletion, and retention

Each microbatch uses one monotone ingestion clock for density updates. Source creation, revision, and observation timestamps are stored separately. **Decay starts when a content version enters this index**, not at the source's claimed creation timestamp. A delayed replay does not reproduce an event-time simulation. Existing retained contributions decay by `exp2(-elapsed / half_life)`; incoming admitted versions have unit weight.

Duplicate delivery and unchanged text do not add density evidence or trigger another embedding. Newer metadata snapshots may refresh metadata; interaction counts remain explicitly observed snapshots. Edits reuse the internal post ID, remove the old code, and encode the new text. Observed edits that remove indexed text or public visibility retire the old indexed representation. Older source revisions cannot overwrite newer ones; deleted records cannot be revived by an older or equal revision.

A truncated PPCA sketch cannot exactly subtract an old vector's contribution. For an edit, deletion, or expiry, the implementation rebuilds each affected leaf from its **retained decoded posts**, in original insertion order, using current exponential weights. This removes the old post's modeled contribution but is approximate because of quantization, rank truncation, and prior cut/membership approximations. It may change density even for surviving posts. We explicitly rebase split evidence during this rebuild. Rebuild cost is proportional to retained posts in the affected leaves, not constant per deletion.

At the first batch/maintenance cycle of each UTC day, posts older than `retention_seconds` are removed from text and code storage. With a 24-hour retention setting and daily sweeping, physical retention can approach 48 hours between sweeps. Quiet or failed source rounds still run maintenance. Decay does not by itself delete post records. Identity tombstones outlive content to prevent replay resurrection; the CLI keeps them for at least seven days and the configured retention window. Adjust that horizon if replaying older data. IDs and decoder IDs are never reused.

Nostr events must pass canonical event-hash and Schnorr signature verification. A deletion is accepted only when its signer matches the known original author, including notes and deletions delivered in the same batch. Deletions targeting unknown Nostr events are ignored; this prototype does not maintain an authenticated pending-deletion archive. ActivityPub intake reads configured trusted public outboxes; it is not a public inbox or HTTP-signature verification implementation.

## Source capabilities and limits

| Reader | What it reads | Resume and coverage |
|---|---|---|
| Bluesky | Jetstream v2 commit messages for `app.bsky.feed.post`, including updates/deletes; explicit legacy mode also supported | Saves cursor with the batch. Inclusive replay is deduplicated. Uses the required `xrpc.v1.json` WebSocket subprotocol on v2. The upstream lookback window is bounded; no automatic archive gap recovery. |
| Nostr | Configured relay, kinds 1 and 5, with real signature verification | Recent `since` query plus live messages. Advances the timestamp checkpoint only after EOSE, with overlap on reconnect. Relay limits/filtering can omit history; this is not a complete network archive. |
| Mastodon | Public timeline polling; optional authenticated WebSocket updates/deletes | REST uses `min_id` to page forward. Initial attachment begins with a bounded recent page. Streaming has no durable replay cursor; disconnected edits/deletes can be missed. Public access depends on the instance. |
| ActivityPub | Configured actor's public outbox, paginated, with inline or dereferenced objects | Saves page/offset when bounded work stops, revisits the head after finishing. Outboxes are not immutable archives; polling can miss transient activities. No universal ActivityPub firehose. |
| Threads | Official keyword search, `RECENT`, fixed time windows and pagination | Requires an application/token with the relevant Threads permissions. Reconstructs requests with opaque pagination cursors; does not follow token-bearing next URLs. Search results are not a complete firehose and do not provide a deletion stream. |

References: [Jetstream](https://bsky.network/docs/jetstream/), [Mastodon public timeline](https://docs.joinmastodon.org/methods/timelines/#public), [Mastodon streaming](https://docs.joinmastodon.org/methods/streaming/), [ActivityPub outbox](https://www.w3.org/TR/activitypub/#outbox), [Nostr deletion](https://github.com/nostr-protocol/nips/blob/master/09.md), [Meta Threads keyword search](https://www.postman.com/meta/threads/request/34203612-b3b2c12a-7ce6-4d86-a3c6-6d31e3b66ea1).

Readers bound frame/page sizes and request durations. The CLI retries source reads three times with bounded backoff, logs redacted failures, and continues the other configured sources. A malformed recognized event fails its source batch without advancing its cursor; quarantine/reconciliation is future work. Historical source coverage and deletion completeness must be evaluated separately from correct local replay.

## Arrow schema and recovery

Each completed `generation-*` directory contains explicit Arrow tables:

| Table | Purpose |
|---|---|
| `posts` | Stable UInt64 ID; identifier kind/native bytes; current text, author, timestamps, tags, metadata, status |
| `aliases` | Secondary `(kind, native bytes) → post ID` mappings |
| `codes` | Post ID, decoder ID, current region, Float32 scale, residual byte list |
| `decoders` | Decoder ID and shared Float32 centroid |
| `nodes` | Geometry, reservoirs, clocks, split evidence |
| `states` | Per-leaf PPCA moments and factors |
| `cursors` | Source cursors and maintenance day |
| `metadata` | Schema version, model/configuration, counters, two RNG states |
| `checksums` | SHA-256 integrity hashes of all preceding tables |

Nostr IDs occupy 32 bytes in the native-ID byte list. Other native IDs are UTF-8 in the same column. Separate kind columns avoid mixed string/binary object containers. Arrow tables use Zstandard compression. Repeated platform strings compress; an explicit dictionary of domains/DIDs could reduce catalog memory further, but is not required for correctness and is not yet implemented.

Files and directory entries are flushed before atomically replacing `CURRENT` on a local POSIX filesystem. The old generation remains authoritative if writing fails before publication. Source cursors are part of the same snapshot as the data they acknowledge. The CLI holds an OS `flock` for single-writer ownership; a crash releases it automatically. It retains two completed generations. Incomplete staging directories left by crashes are not automatically deleted. A process interrupted after publication may resume from the newly published generation.

If a kernel or storage operation fails during a commit, **discard that in-memory index and restart from `CURRENT`**. The CLI terminates that writer rather than continuing a partially mutated state. Low-level `ingest!` is not an in-memory rollback transaction. Snapshot integrity is checked before restore. All post, code and model arrays are readable by other Arrow implementations; only the two exact-continuation RNG blobs use Julia serialization, and restore requires the same Julia version. Only restore trusted local checkpoints.

Snapshots currently rewrite the entire retained dataset and perform an audit. This costs O(retained data) per checkpoint, with temporary Arrow allocation and compression, so the implementation is suitable for integration experiments rather than a measured production-rate service. Append-only segments, compact posting lists, and a database/WAL are the next storage optimization. Normal reader calls also scan retained IDs for regional candidate sampling; they do not claim indexed or exact kNN.

## Retrieve actual post records

```julia
include("examples/live_modules.jl")
using .LiveIndex, .ArrowPostStore, .PostRecords
index = restore("data/live")
post = lookup(index, PostKey(:bluesky, "at://did:plc:EXAMPLE/app.bsky.feed.post/KEY"))
# For an existing active post:
vector = embedding(index, post.id)
ids = candidates(index, vector; limit=20, leaf_budget=4)
records = [index.posts[id].record for id in ids]
```

`candidates` probes a bounded number of nearby leaf boxes and returns a bounded random sample of actual retained post IDs. The caller can rank those posts using decoded coordinates, source metadata, or client-specific preferences. It does not generate a Gaussian point and pretend that point is a real post. Kernels also expose `centroid_distances` for scoring selected codes. A server/client API, browser snapshot export, Fermat graph, and ranking model are separate work.

## Validation

```sh
julia --project=environments/ingestion test/centroid_post_kernels.jl --gpu
julia --project=environments/ingestion test/live_index.jl
julia --project=environments/ingestion test/source_adapters.jl --embed
julia --project=environments/ingestion test/live_sources.jl
julia --project=environments/ingestion test/ingestion_runner.jl
julia --project=environments/kernels test/leaf_tree_kernels.jl --gpu
julia --project=environments/kernels test/scheduled_leaf_kernels.jl --gpu
julia --project=environments/kernels test/kernel_sibling_merging.jl --gpu
```

The checks cover CPU/GPU numerical parity, weighted moments against direct formulas, immutable codes through splits/merges, continuation, edits/deletions, duplicate replay, within-batch alias/deletion ordering, expiry and tombstone GC, exact Arrow continuation of RNG-driven cuts, injected checkpoint failures, checksum corruption, native-ID round trips, authentic Nostr signatures, pinned ONNX inference, source request/cursor behavior, and actual local HTTP/WebSocket transport. The independent Python reader joins every identity/code/decoder row, reconstructs every vector, and verifies routing without reading Julia RNG serialization.

The completed run passed **28,705 assertions** across eight suites. The mixed-platform fixture produced identical CPU/GPU decoded vectors; the maximum difference in its low-rank scatter factors was `2.22e-16`. A bounded live Bluesky run continued across restarts to **2,220 retained posts, 6 leaves and 5 cuts**; its full checkpoint was **1,133,754 bytes**, including text and metadata. Reachable Julia index memory was **3,411,195 bytes**, excluding the runtime, embedding worker/model, device buffers and temporary work. Bounded reservoirs retained 384 raw vectors; the full per-post embedding archive was not retained. A separate live Nostr run retained 10 verified notes. Mastodon/ActivityPub authentication and absent Threads credentials limited those platforms to adapter/transport fixtures.

See the [measured summary](../output/live_ingestion/summary.txt), [validation record](../output/live_ingestion/validation.json), and [independent Arrow reports](../output/live_ingestion/). The live tests are bounded connectivity/integration checks, not a throughput benchmark. Tests do not establish completeness of upstream data, recommendation quality, or semantic density calibration.
