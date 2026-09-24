# Live tree speed benchmark

[Open the report](../output/kernel_speed/demo.html). The [run log](../output/kernel_speed/run.log) records each trial as it finishes. This compares the existing implementations without introducing optimizations.

```bash
# Requires data/bluesky_100000 and chronological row references from the decay demo.
python3 scripts/prepare_leaf_decay.py
julia --threads=1 --project=environments/kernels examples/kernel_speed.jl
python3 scripts/kernel_speed_report.py
```

Two full-fit workloads run three times on each backend:

- 10,500 synthetic 2D arrivals across 120 simulated minutes, with a one-minute half-life, moving topics and a 15-minute silence. PPCA rank 1, split threshold 24 effective arrivals.
- 10,000 real 256D Bluesky embeddings, selected as the first 10,000 chronologically ordered eligible posts from the existing 100k archive. Six-hour half-life and PPCA rank 4, split threshold 128 effective arrivals.

Both use seed 82, up to 256 leaves, depth limit 16, and reservoir capacity 32. Each trial starts a fresh model on identical input. Each backend is warmed on the first 1,000 observations, compilation is excluded, and the median/minimum/maximum are saved. Models use Float64, and input embeddings are Float32. One Julia thread and one BLAS thread are used. Results are a workstation measurement with other applications present, not an isolated performance laboratory.

Full-fit timings include input validation, host allocation, packing, transfers, synchronization, updates, cuts, and final lazy-decay materialization. Loading data, embedding inference, subsequent correctness audits, and pre-trial garbage collection are excluded. Garbage collection during the timed fit is included and reported. `host_allocated_bytes` is allocation traffic, not peak retained memory or GPU allocation traffic.

Every measured fit checks identical cut topology and Pólya-probability agreement with the reference. After each workload, a separate fixed-tree routing benchmark runs five repetitions: 105,000 synthetic queries and all 100,000 archived Bluesky vectors. This includes query packing, transfers and output collection; node IDs must exactly match reference routing. Routing throughput is not streaming-ingestion throughput.

The current KernelAbstractions driver executes each arrival in reference order and transfers/synchronizes small arrays around each operation. GPU timing measures this overhead as well as the kernels. Persistent state and batches of independent leaves are still future work; this benchmark makes no claim about their attainable speed.
