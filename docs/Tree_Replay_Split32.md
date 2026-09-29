# Retained-post tree replay

This experiment replays the stopped GTE-tiny collection into a new disk store:

| Setting | Original | Replay |
| --- | ---: | ---: |
| `split_after` | 256 | 32 |
| `max_leaves` | 128 | 1024 |
| `max_depth` | 24 | 64 |
| `merge_below` | 8 | 2 |

Other storage/tree settings come from the original persisted configuration,
including rank-4 PPCA, Float16 vectors, append layout, random seed, retention,
reservoir size, and half-life. Hashtag-only summaries retain their own half-life.

```sh
julia --threads=2,1 --project=server server/bin/replay_tree.jl \
  data/live_disk_gte data/live_disk_gte_split32 \
  output/tree_replay_split32/results.json
```

The destination must not already exist. The original store is opened read-only,
with its writer lock held to exclude ingestion or maintenance during the replay.
The script does not connect to social platforms or start embedding inference.
Ingestion remains stopped, and the default live configuration remains unchanged.

The same replay can use a tenfold higher leaf ceiling while retaining the other
settings:

```sh
julia --threads=2,1 --project=server server/bin/replay_tree.jl \
  data/live_disk_gte data/live_disk_gte_split32_10k \
  output/tree_replay_split32_10k/results.json --max-leaves 10240
```

The ceiling does not preallocate leaves. Both replays use the same 256 MiB
managed storage budget, including a 60 MiB decoded-state cache. Routing metadata
is resident and grows with actual node count; leaf PPCA/reservoir states are
evicted to SQLite. Wider access can therefore increase I/O without requiring
simultaneous residency of every leaf state. Process RSS also includes Julia,
temporary allocations, and the replay's separate source-reader buffers.

The optional server `view=ppca` export assembles factor copies across all leaves;
its existing size guard rejects oversized packs. That bulk export differs from
normal regional access and the inspector's one-leaf-at-a-time detail reads.

This is a replay of **retained active content versions**, not the original source
event log. It processes them in saved sequence order, at their original
`density_at` timestamps, and advances to the source's final committed clock.
Bounded pages can subdivide an original batch. Earlier edited versions, deleted
posts, and empty collection rounds cannot be reproduced from this snapshot.
Results therefore compare a retained-post replay with the original live tree;
they are not a controlled four-parameter-only ablation of the full event stream.

Saved Float16 coordinates are decoded and reused, rather than recomputed with
GTE-tiny. Re-encoding to Float16 must preserve every coordinate exactly. Text,
metadata, primary platform identities and aliases are copied; internal numeric
post IDs and tree-node IDs are new. Source cursors and tombstones are not copied
to the experimental destination. It is intended for offline inspection.

Validation checks active counts, decoded vectors, content digests, content ages,
global hashtag counts and decayed supports, as well as the destination's complete
identity/posting/routing audit. A small fixture also checks alias preservation,
exclusion of deleted records, and unchanged source generation/topology:

```sh
julia --threads=2,1 --project=server server/test/replay_tree.jl
```

The [raw report](../output/tree_replay_split32/results.json) contains the exact
configuration, leaf-size distributions, split/merge counters, storage statistics,
validation results and a growth trajectory. [Summary](../output/tree_replay_split32/summary.md).

To inspect the replay:

```sh
julia --threads=2,1 --project=server server/bin/inspect_live.jl \
  --store data/live_disk_gte_split32 --port 8091
```

To compare both trees simultaneously, run a second inspector on port 8092 with
`--store data/live_disk_gte`. No source ingestion is required.
