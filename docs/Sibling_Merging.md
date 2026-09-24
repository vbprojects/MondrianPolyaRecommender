# Sibling merging for the decayed leaf tree

The reference Julia implementation and the KernelAbstractions CPU/GPU implementations can now remove old cuts. Merging is opt-in; existing experiments keep their previous behavior unless `merge_below` is positive.

```julia
include("src/LeafLatentTree.jl")
using .LeafLatentTrees

fit = fit_leaf_tree(X;
    timestamps=seconds,
    half_life=6*3600,
    configs=[(:ppca, 4)],
    split_after=128,
    merge_below=16,
    merge_interval=3600)

# Explicit maintenance during a period with no arrivals.
# Assign the return value: arrays mutate, while clocks/counters are returned.
fit = merge_tree!(fit; time=fit.time + 24*3600)
```

`merge_below` is the **combined effective observation mass of two siblings**, not their density, lifetime arrival count, or Pólya probability. A merge happens when both children are leaves and their combined mass is at most this threshold. A weak leaf paired with a busy sibling keeps its cut. The threshold must be nonnegative and strictly below `split_after`; enabled automatic merging requires a finite half-life and a positive finite interval.

Maintenance first advances all leaves to the same time. It then visits parents bottom-up. Merging two children can make their parent eligible to merge with its own sibling in the same pass. Even a deep tree can collapse to its root after sufficient inactivity. The root itself remains.

During fitting, maintenance happens before the first arrival at least `merge_interval` after the previous maintenance clock, and once at the end of the supplied stream. The default interval is one half-life. There is no background timer. `leaf_probabilities(fit; time=...)` remains a nonmutating query; call `merge_tree!` to change topology during complete silence.

## What a merged leaf stores

For sibling weights `mL`, `mR`, means `muL`, `muR`, and stored diagonal central scatters `sL`, `sR`:

```text
m       = mL + mR
delta   = muR - muL
mu      = muL + (mR/m) * delta
between = mL * (mR/m)
s       = sL + sR + between * delta.^2
```

This preserves total mass, first moment, and the stored scatter diagonal, up to floating-point error. A completely extinct pair becomes a zero-mass sketch.

The low-rank scatter uses a thin factor:

```text
F = [UL * sqrt(lambdaL), UR * sqrt(lambdaR), sqrt(between) * delta]
```

A thin SVD retains the leading `rank` singular directions and their squared singular values. This is the best rank-limited approximation of the *retained child scatters plus their between-mean scatter*. It cannot reconstruct correlations already discarded by a child. The emitted PPCA covariance, including its isotropic residual, remains an approximation. No full `dimension × dimension` covariance is created. With fixed rank, the factor workspace scales linearly with dimension and the SVD work is approximately proportional to `dimension × rank²`.

The parent keeps at most `reservoir_cap` of the newest **retained** examples across both children, ordered by original arrival ID. These IDs handle equal timestamps correctly. This is a bounded FIFO policy; a merge cannot retrieve older examples that children already discarded. New split evidence resets to zero, so the merged leaf needs fresh arrivals before splitting again.

The implementation removes child slots and compacts node indices, clocks, and reservoir IDs. Freed leaf capacity can be used for future splits. Any external cache keyed by node index must be rebuilt after a merge. `on_merge(row, count, nodes)` receives the compacted tree, and `on_checkpoint` includes cumulative `split_count` and `merge_count`. Diagnostic split-error history remains cumulative and is separate from live tree storage. Post files and source embedding archives are not deleted.

## Kernel execution

```julia
include("src/LeafTreeKernels.jl")
using .LeafTreeKernels
using CUDA
CUDA.allowscalar(false)
k = KernelTreeBackend(CUDA.CuArray) # use KernelTreeBackend() for CPU

fit = fit_kernel_tree(k, X;
    timestamps=seconds, half_life=6*3600,
    configs=[(:ppca, 4)], split_after=128,
    merge_below=16, merge_interval=3600)
fit = kernel_merge_tree!(k, fit; time=fit.time + 24*3600)
```

Both `execution=:event` and the default scheduled execution support merging. Event execution invokes the kernel operations as observations arrive. Scheduled execution divides the input into epochs ending before maintenance boundaries. It plans and computes an epoch, merges using actual decayed statistics, and only then plans the next epoch. Each leaf still processes its observations in their original order. This avoids planning future cuts from missing or placeholder merge statistics.

The device kernels compute decay and weighted merge means, scatter diagonals, and factor assembly. Thin SVD, bounded-reservoir selection, and topology compaction run on the CPU. The portable eigensolver and external LAPACK/cuSOLVER event variants were tested. Observers and intermediate callbacks use the event path to expose materialized prefix statistics. The public fitting API still consumes a supplied batch. The newer [live ingestion wrapper](Live_Ingestion.md) supplies durable continuation, frozen post codes that survive merges, and Arrow checkpoints, and fixes preservation of earlier batches in scheduled fitting with merging enabled.

## Measured behavior

Open [the interactive comparison](../output/sibling_merging/demo.html), [summary](../output/sibling_merging/summary.txt), or [raw results](../output/sibling_merging/results.json).

On 10,500 synthetic 2D observations with a one-minute half-life, topic changes, and a 15-minute silence:

| Policy | Final leaves | Live node storage | Cuts created | Cuts merged |
|---|---:|---:|---:|---:|
| Decay only | 70 | 76.7 KiB | 69 | 0 |
| Decay + merging | 20 | 21.6 KiB | 75 | 56 |

Both policies use a split threshold of 24 and a 128-leaf ceiling; merging uses threshold 3 and a one-minute interval. Merging changes subsequent routing and therefore the sequence of cuts. This measures forgetting and storage behavior, not recommendation quality.

Three warmed full-fit trials per backend on this workstation:

| Workload | Reference Julia | Scheduled KA CPU | Scheduled KA GPU |
|---|---:|---:|---:|
| 10,500 2D points, frequent maintenance | 0.0162 s | 0.0155 s | 0.2909 s |
| 12,000 real 256D embeddings, synthetic burst/silence clock | 0.2093 s | 0.1064 s | 0.2374 s |

The real-data stress test uses existing Bluesky embeddings with a **synthetic clock**, not their original posting times. Frequent small maintenance epochs increase kernel launch and transfer overhead; the earlier no-merging speedups do not apply automatically. Every timed fit matched reference topology, retained examples, and split/merge counts. The largest checked probability/mean/covariance discrepancy was about `4.4e-14`.

Advancing maintenance by 30 additional half-lives collapsed both benchmark trees to one leaf. The 256D live model, including index metadata, shrank from approximately 1,077 KiB to 51 KiB. Live model size excludes the input embeddings, diagnostic histories, allocator caches, and Julia/CUDA process memory; reclaimed objects do not imply an immediate drop in process RSS.

## Reproduce and verify

```bash
julia --project=. test/leaf_sibling_merging.jl
julia --project=environments/kernels test/kernel_sibling_merging.jl --gpu
julia --project=environments/kernels examples/sibling_merging.jl
python3 scripts/sibling_merging_report.py
```

The new reference tests check weighted moment conservation against the original observations, rank-limited scatter reconstruction, multi-level collapse, finite behavior at complete extinction, storage reduction, capacity reuse, FIFO ordering, normalized probabilities, and validation errors. The kernel tests compare both drivers and both eigensolver families on CPU and NVIDIA GPU, including routing and sampling after compaction. Existing decay, tracking, tree-kernel, and scheduled-kernel regression tests are also rerun; logs are in `output/sibling_merging/`.

Final validation: **28,500 assertions passed**, including 2,466 new merging assertions and 26,034 existing regression assertions. The browser check found no JavaScript errors, rendered all six benchmark rows, and exercised the timeline slider. [Validation record](../output/sibling_merging/validation.json).
