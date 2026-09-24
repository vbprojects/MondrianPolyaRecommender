# KernelAbstractions backend for the leaf-local Mondrian–Pólya tree

`src/LeafTreeKernels.jl` supplies kernels for tree traversal, online moment updates, exponential decay, reservoir replacement, split-axis widths and threshold preparation, reservoir partitioning, conditional child moments and covariance-operator products, sibling moment/factor combination, subtree mass reduction, Pólya branch probabilities, and bounded Gaussian proposals.

The backend applies to the current leaf-local adaptive tree in `LeafLatentTree.jl`, with or without decay. The older histogram, ancestor-residual, and historical-owner MPPCA implementations are separate experiments and are not ported by this module.

The [live ingestion wrapper](Live_Ingestion.md) adds centroid/residual encoding, direct coded routing/distances, weighted PPCA rebasing, native post identity, and Arrow checkpoints. It also fixes scheduled continuation with sibling merging enabled. That wrapper provides the resumable insertion layer; the low-level fitting API described below remains batch-oriented.

## Run

An isolated environment keeps GPU packages out of the original toy package:

```bash
julia --project=environments/kernels -e 'using Pkg; Pkg.instantiate()'
julia --project=environments/kernels test/leaf_tree_kernels.jl --gpu
julia --project=environments/kernels examples/kernel_tree_comparison.jl
```

Omit `--gpu` to run CPU tests only. Include the reference module before the kernel module:

```julia
include("src/LeafLatentTree.jl")
include("src/LeafTreeKernels.jl")
using .LeafLatentTrees, .LeafTreeKernels

cpu = KernelTreeBackend() # scheduled fused kernels
fit = fit_kernel_tree(cpu, X;
    timestamps=times, half_life=1.0,
    configs=[(:ppca, 1)], split_after=64)
prob = kernel_probabilities(cpu, fit; time=fit.time + 1.0)
ids = kernel_route(cpu, fit.nodes, X)
```

For NVIDIA, the optimized path uses scheduled fused kernels:

```julia
using CUDA
CUDA.allowscalar(false)
gpu = KernelTreeBackend(CUDA.CuArray)
fit = fit_kernel_tree(gpu, X;
    timestamps=times, half_life=1.0, configs=[(:ppca, 4)])
```

Omitting `eigensolver` selects scheduled execution with the portable cyclic-Jacobi kernel (an analytic 2×2 update for the 2D/rank-1 case), tested against the library paths. Supplying an external solver selects the original event driver. For example `KernelTreeBackend(CUDA.CuArray; eigensolver=C -> CUDA.cuSOLVER.syevjBatched!('V', 'U', C))` retains library-backed event execution. `execution=:event` can also explicitly select that driver. A replacement solver accepts an `(r+1) × (r+1) × batch` device array and returns ascending eigenvalues and matching eigenvector columns on the same backend. Inactive dimensions are padded with -1 eigenvalues below the covariance spectrum, so rank-deficient startup models share the same batch shape. The update retains at most the requested rank. Eigenvector signs are arbitrary: comparisons use means, scatter, and covariance, rather than requiring equal basis entries.

## Library investigation

The suggested [JuliaCon talk](https://pretalx.com/juliacon-2026/talk/3MV3BW/) describes [KernelAbstractionsPseudospectra.jl](https://github.com/dan123222123/KernelAbstractionsPseudospectra.jl). It uses a CPU Schur/QZ reduction followed by portable batched inverse-Lanczos/triangular-solve work. Its public pseudospectrum API is not the small symmetric full-eigenpair operation needed by this update. Its separation of host orchestration and batched device work is nevertheless applicable.

CUDA.jl's cuSOLVER wrapper exposes [`syevjBatched!`](https://github.com/JuliaGPU/CUDA.jl), corresponding to [NVIDIA's batched symmetric eigensolver](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-syevjbatched). This is directly applicable to rank-4 updates (at most 5×5 matrices). CPU LAPACK and cuSOLVER are supported interchangeable solvers; the portable Jacobi path provides a backend-independent option. No dependency on the pseudospectra package is necessary.

## Execution model and limitations

The default scheduled driver plans the exact same topology and reservoir contents on CPU, then evaluates the moment dependency graph in depth order. This is valid because this implementation chooses cuts from reservoir geometry and effective arrival counts, independently of the PPCA statistics. Each region processes its assigned observations in original order; it transfers its final pre-cut moments before its children begin. No observations from after a cut are used to estimate its pre-cut moments.

Input data, indexed arrival lists, and model buffers persist across kernel launches. Each level exchanges a single packed state buffer. Independent regions run concurrently; 256D updates use cooperating 32-thread GPU workgroups and shared-memory scratch. The 2D/rank-1 specialization keeps its recurrence in registers. Decay factors are computed once from each region's actual observation times. CPU launches use allocation-free inner recurrences. Julia continues to handle allocation of tree nodes, seeded random cut selection, reservoir updates, and the rare conditional-moment split recompression.

This optimization operates on the full input batch supplied to `fit_kernel_tree`, as the reference API does. It is not a new resumable, per-post online service. Intermediate callbacks require materialized prefix states and therefore select the existing event driver. No callback sees placeholder planning statistics. This callback path preserves behavior but retains its older per-event overhead.

The original event driver is retained for custom external eigensolvers and diagnostics. It packs/transfers/synchronizes at operation boundaries. `update_sketches!` still batches independent sketches one observation each; the new scheduled fit instead fuses each region's complete assigned sequence. The external eigensolver benchmarks below describe the original event driver, not the optimized default.

All split decisions, time units, prior strengths and compression ranks remain unchanged. Returned objects work with the existing reference and kernel query/sampling APIs. Query buffers no longer make redundant host copies before routing. A conservative shared-memory budget selects the global-scratch kernel when a requested dimension/rank combination is too large for the cooperative path.

All current kernel arithmetic uses Float64. CPU and NVIDIA CUDA have been executed and validated. AMD, Intel, and Apple backends have not been tested; devices without Float64 support need a separate precision-generalization pass. KernelAbstractions syntax alone is not a guarantee that this implementation works on every device.

`kernel_sample_leaf` uses the same generated normal candidates as the reference and rejects outside leaf bounds. It generates the whole candidate budget before selection, so it advances its RNG further than the early-stopping reference. Use independent seeded RNGs per sampling request. Accepted points and the first successful stopping index are comparable, but subsequent RNG state is not identical.

Optional sibling merging is supported by both event and scheduled execution. Set `merge_below` and `merge_interval` as described in [Sibling merging](Sibling_Merging.md). The scheduled driver plans only until the next maintenance boundary, executes the numeric segments, merges using their actual decayed mass, and then plans the next epoch. Merge decisions do not use placeholder statistics. Merge moment/factor assembly and decay use kernels; thin SVD, compaction, and topology edits remain on the CPU. Callbacks, including `on_merge`, use the event driver.

The inherited halfspace split approximation remains. Merging preserves stored mass, first moment and scatter diagonal, while recompressing low-rank correlations. Reservoirs keep the newest retained examples across siblings. Daily source-post expiration is separate.

## Validation

`test/leaf_tree_kernels.jl` exercises four variants: KA CPU with portable Jacobi and LAPACK, and KA NVIDIA with portable Jacobi and cuSOLVER. Tests cover 2D/rank 1, 12D/rank 4, 256D/rank 4, diagonal moments, extreme forgetting, full adaptive trees with static and decayed streams, identical cuts/routing/reservoirs, covariance agreement, normalized Pólya queries including complete decay, and seeded rejection sampling. Reference tests continue to run without GPU dependencies.

`examples/kernel_tree_comparison.jl` runs all three reference/CPU-library/GPU-library implementations on 10,500-point 2D streams at three decay settings and 2,000 real 256D Bluesky vectors. Results go to `output/kernel_tree/results.json`.

Measured outcome: all comparison runs produced identical topology. Maximum covariance discrepancy was 7.36e-15 and maximum Pólya probability TV was 1.37e-15. All 19,112 kernel assertions and 162 reference assertions passed. These are seeded finite tests, not a proof for all streams.

The 10,500-point one-minute 2D run took 0.020 s in reference Julia, 0.190 s with KA CPU/LAPACK, and 7.021 s with KA NVIDIA/cuSOLVER. The 2,000-point 256D run took 0.035 / 0.869 / 3.494 s, respectively. These expose the current transfer/allocation overhead and do not demonstrate a GPU speed advantage. [Comparison report](../output/kernel_tree/demo.html).

## Optimized validation and speed

Run the optimized benchmark independently of test jobs:

```bash
julia --threads=1 --project=environments/kernels examples/kernel_optimized_speed.jl
python3 scripts/kernel_optimized_report.py
julia --project=environments/kernels test/scheduled_leaf_kernels.jl --gpu
```

The scheduled tests cover three seeds, ranks 0/1/4/8, diagonal/PPCA/low-rank-diagonal states, multiple model configurations per tree, static and decayed observations, silent gaps, degenerate constant streams, and callback prefix consistency. The benchmark compares complete fits (planning, allocation, transfers and final state included), rotates trial order, excludes warmup and audits, and tests the two previous live workloads plus static 2D and larger real-data streams. [Optimized report](../output/kernel_optimized/demo.html).

Prior benchmark with merging disabled: scheduled CPU/GPU fitting exceeded reference Julia on all five tested workloads. On 100,000 static 256D posts: reference 1.504 s, scheduled CPU 0.936 s, GPU 0.535 s. On 89,798 chronological decayed posts: 1.489 / 0.876 / 0.553 s. Small 2D GPU gains were modest (1.11–1.18×), while CPU gains were 2.39–4.77×. All these full-fit comparisons preserved topology and reservoirs; worst numerical errors were below 1e-13. No claim is made for untested workloads or the retained per-event callback/external-solver mode.

Final optimized validation: 19,112 kernel assertions, 6,760 scheduling assertions (including the 1024D global-scratch fallback), and 162 original assertions passed. Test logs are retained in `output/kernel_optimized/`.


## Sibling merging validation

```bash
julia --project=. test/leaf_sibling_merging.jl
julia --project=environments/kernels test/kernel_sibling_merging.jl --gpu
julia --project=environments/kernels examples/sibling_merging.jl
python3 scripts/sibling_merging_report.py
```

The new tests exercise collapse, capacity reuse, retained arrival IDs, multiple configurations, silent gaps, extinction, routing and probability queries after compaction, portable CPU/GPU execution, and LAPACK/cuSOLVER event execution. The [new report](../output/sibling_merging/demo.html) measures fits with merging enabled separately from the prior no-merging speed benchmark. Small, frequently maintained batches have more transfer/launch overhead.
