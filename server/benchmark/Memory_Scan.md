# Ingestion memory review

Reviewed the running SQLite ingestion path on 2026-09-29: collection, embedding
bridge, native Julia YAKE, caches, statements, structural cleanup and reclamation.
No live process was stopped, debug-attached or modified. A small private replay
ran at lower CPU priority alongside it; this can still affect throughput.

**No persistent storage-side leak was confirmed. There is an unbounded WebSocket
receive queue and two temporary-memory risks.** This short scan does not certify
an overnight run as leak-free and is not a full native heap profile.

## Findings

1. **Unbounded WebSocket queue — highest priority.** Pinned HTTP.jl 2.7.1 creates
   `Channel{Union{String,Vector{UInt8}}}(Inf)` in `http_websockets.jl:196`.
   Its background reader enqueues without an aggregate byte/item bound
   (`_enqueue_message!`, line 286). [Our collector](../../src/LiveSources.jl:37)
   limits consumed frames and elapsed time. The 1 MiB frame limit does not bound
   the pending queue. Bursts or slow parsing can accumulate messages.
   Closing each session limits its lifetime, but not a firm peak.
   A direct internal-queue probe accepted 20,000 ~517-byte messages, retaining
   10,708,854 bytes despite a 1 KiB frame limit. This was not a network test or
   evidence that the live run reached that queue size. Add transport-level
   byte/message limits with backpressure or explicit overflow/reconnect behavior,
   preserving the last consumed cursor.

2. **Batches lack an aggregate work-memory budget.** Collection counts events,
   whose text/tags/metadata sizes vary. [Normalization](../../src/SourceAdapters.jl:22)
   accepts up to 64 KiB text plus selected metadata. The
   [worker](../../scripts/live_embedding_worker.py:55) checks serialized request
   size, but tokenizes the entire accepted batch without truncation. Token lists
   and ONNX intermediates can exceed text size substantially. These temporary
   allocations can leave allocator high-water memory behind. Add a source byte
   budget and token-budgeted inference microbatches without changing text or
   embedding semantics.

3. **Reclamation lists the whole payload directory.**
   [collect_garbage!](../../src/DiskTreeStore.jl:775) uses `readdir(s.root)`.
   This filename array grows with file count even though database work is paged.
   Use bounded directory iteration or a reclamation manifest before large-scale
   deployment. This is a temporary scale-dependent spike, not retained history.

## Bounded structures

- State/frame caches evict; reservoirs and routing-node counts have configured
  caps. Splits remove parent state; merges remove child nodes/state/cache entries.
  Merge cleanup was reviewed statically; this replay recorded no merges.
- Statement strings are a finite internal vocabulary on these paths. Diagnostic
  metric keys and request-tracking sets are bounded.
- YAKE dictionaries are per document; the extractor keeps no post history.
  The keyword cache holds one bounded result and is invalidated on publication.
- Payload streams close at publication/error, and reads use scoped file handles.
  SQLite statements are reused and closed with the store.
- Python retains its model/session and last batch's locals intentionally.
  Native inference allocator retention was not heap-profiled.
- RocksDB iterator cleanup and bounded pending maps were inspected, but the
  current run and this replay use SQLite.

## Measurements

Thirteen live samples over about 118 seconds covered growth from 9,278 to 14,828
committed active posts:

| Process | RSS first → last | File descriptors | Threads |
| --- | ---: | ---: | ---: |
| Julia | 1,105.8 → 1,109.5 MiB | 26–27 | 11 |
| Python | 505.9 → 517.2 MiB | 7 | 19 |

The Python increase needs longer observation to distinguish warmup, larger
batches and a slow leak. No descriptor/thread accumulation appeared. No forced
GC was applied to the live process.

The private replay processed 6,144 synthetic 256D posts in 96 batches with
Float16, rank-4 PPCA, native keywords and short retention/decay. The managed
budget was 32 MiB, with frame cache reduced to 512 KiB to force early eviction.
Expiry held active population at 576 after warmup; full GC ran at checkpoints.

- Batch 32 → 96: live Julia heap 43,532,840 → 43,568,384 bytes, about 41.5 MiB;
  routing nodes 9 → 11.
- Statements stayed at 52, descriptors at 22 and metric keys at 14.
- Frame cache: 15 entries / 494,520 bytes; 81 evictions.
- After clearing caches: live heap 42,891,080 bytes; SQLite cache 13,664 bytes;
  RSS still about 1,022 MiB.
- Small request-tracking growth is bounded by its 4,096-ID cap. SQLite's page
  cache was warming and is separate from Julia's live heap.

High RSS with much smaller stable live heap is consistent with JIT/runtime,
native libraries/caches and allocator-retained pages. It does not attribute every
byte or exclude all native leaks.

## Reproduction and next step

From repository root:

```sh
julia --threads=1 --project=environments/ingestion server/benchmark/memory_scan.jl
```

The script creates/removes a private temporary store. Raw results:
[live samples](../../output/memory_scan/live.json),
[replay](../../output/memory_scan/replay.json),
[queue probe](../../output/memory_scan/websocket_queue.json).

Prioritize bounding the transport queue and aggregate inference work, then log
RSS/PSS, GC live heap, cache occupancy, batch bytes/tokens and descriptor counts
over hours, including a stable retained-post population. No runtime fixes or
automatic restarts were applied by this review.
