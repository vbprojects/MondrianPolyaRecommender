# Live tree inspector

From the repository root:

```sh
julia --threads=2,1 --project=server server/bin/inspect_live.jl \
  --store data/live_disk_gte --run-dir data/live_runs/latest --port 8091
```

Open **http://127.0.0.1:8091**. Stop the inspector with Ctrl-C; ingestion continues.
The service deliberately binds to localhost. For a remote workstation use SSH
port forwarding, e.g. `ssh -L 8091:127.0.0.1:8091 workstation`.

This observes the current **SQLite-backed** run alongside its existing writer.
It does not acquire the store's writer lock, run expiry, advance cursors, sample
using ingestion RNGs, or change metadata. RocksDB observation needs an endpoint
inside its existing store owner; this separate-process inspector explicitly
rejects a RocksDB-only store. Use the same Julia version as the writer because
runtime and PPCA state use Julia serialization.

## Reading the view

- **Hierarchy:** actual node identities and parent/child relationships. Select a
  cut to see its embedding dimension and threshold and zoom its subtree.
- **Treemap:** each leaf's area is its Pólya probability, using the existing
  geometric prior and concentration 2. Placement expresses ancestry, not distance
  in embedding space. Tiny leaves remain selectable through the hierarchy.
- **Leaf details:** counts, decayed evidence mass, log density, PPCA rank and
  reservoir size. PPCA signal fraction is the modeled low-rank variance fraction,
  not an accuracy score. Log density is the tree's uniform-within-leaf density,
  not the local PPCA likelihood. Internal nodes show average subtree density.
- **Labels:** common and distinctive combined YAKE/hashtag terms, with raw counts
  and decayed support. Distinctiveness uses the same smoothed global comparison
  as the store. Labels appear in the map after visiting a leaf; a refresh does
  not scan every leaf's vocabulary.
- **Examples:** up to six evenly spaced posting slots. These are real stored
  texts, not nearest neighbors or necessarily evidence for a displayed keyword.
- **Live history:** active-post count and observed split/merge transitions since
  opening the page, keeping at most 240 observations. Multiple structural changes
  between polls can be missed; this is not an event log.

Tree polling is every five seconds, selected-leaf refresh at most once per
15 seconds automatically. Refresh requests both immediately; Pause suspends
automatic polling (an already-running request can still finish).

## Consistency, resource limits, and scope

Each API response uses a short SQLite WAL read transaction, opened with
`mode=ro`, verified with `sqlite3_db_readonly`, and additionally protected with
`query_only`. Tree and leaf responses carry a committed generation and timestamp.
Leaf requests can require a generation; a change returns HTTP 409 and the viewer
retries against a fresh tree. A summary retained from an older generation is
labeled accordingly. Statistics use the **committed clock**, without pretending
the writer has already processed wall-clock decay or expiry.

The observer serializes database requests, uses an 8 MiB SQLite page-cache
target per read connection, disables mmap, retains one tree response for two
seconds, and streams keyword candidates into two top-ten lists. It reads no
vector corpus: only routing nodes, aggregate rows, one selected leaf state and
at most six text records. Node snapshots are capped below 8,192 nodes; individual
text reads at 1 MiB and PPCA decoding at 2 MiB. SQLite has a four-second progress
deadline and 250 ms busy timeout. The SQL deadline does not bound Julia JIT,
deserialization or filesystem calls. First startup/request is slower from JIT.

These limits are not a hard process-RAM cap. The inspector has its own Julia
runtime and RSS. Displayed ingestion/embedding RSS and thread information come
from Linux `/proc` descendants of `runner.pid`, independently of the committed
database snapshot. Missing/exited processes show dashes. Store file sizes include
registered files awaiting reclamation and do not mean active vector bytes alone.

Readers can delay WAL checkpointing while a transaction is open. Immutable
payload files do not have cross-process reader leases: concurrent physical
reclamation can make an example unavailable after its metadata was read. The
inspector skips these examples and reports the number unavailable; refresh retries.
It cannot see uncommitted ingestion buffers, live cache-hit counters or the entire
history of splits and merges. Public post text is displayed literally, without
executing HTML; external links accept only HTTP(S).

## Validation

```sh
julia --threads=2,1 --project=server server/test/live_inspector.jl
```

The fixture keeps a writer open while verifying probabilities/masses and keyword
rankings against the existing implementation, read-only enforcement, interruption,
generation conflicts after a write, missing payload handling, and observer restart.
Browser checks cover leaf selection, example rendering, mobile layout and refresh.
