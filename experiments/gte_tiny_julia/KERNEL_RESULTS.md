# GTE-tiny: Julia kernel follow-up

The dedicated Julia INT8 implementation now has CPU and CUDA kernel backends. **The CPU kernels preserve the original Julia vectors on every evaluated fixture and retrieval input. The optional polynomial math variant improves throughput by 4–5.7× over the original, but ONNX INT8 CPU is still about 2× faster.** CUDA reaches 314 texts/s at batch 8 with close numerical agreement. This is progress toward parity, not a claim of exact ONNX equivalence or production readiness.

Measured 2026-10-27 on a Ryzen 7 7800X3D and RTX 4070 SUPER (12 GB), Julia 1.13.0 and ONNX Runtime 1.30.0. CPU inference uses two threads. These are local results; an L4 has not been measured.

## Implementation

The project pins **KernelAbstractions 0.9.42 and CUDA 6.4.0**, matching the versions already used in `environments/kernels`. The original [GteTiny.jl](GteTiny.jl) remains unchanged as a numerical reference. All variants use the same pinned model, exported quantized weights, tokenizer, 256-token limit, mean pooling and normalized 384D output described in [README.md](README.md).

- [KernelGte.jl](KernelGte.jl): packed CPU dense kernels using AVX-512 VNNI (`vpdpbusd`), with exact INT32 accumulation and zero-point correction. A KernelAbstractions CPU implementation provides the portable fallback. VNNI is selected only when the Linux x86 CPU reports the necessary features.
- [CPUKernels.jl](CPUKernels.jl): quantization, packed query/key attention, contiguous value reductions, softmax and GELU kernels, plus scratch buffers reused across layers within each batch. Standard math is the default. An explicit `cpu-polynomial` option approximates the exponential used in softmax and GELU; it must not be assumed equivalent on unseen inputs.
- [GPUKernels.jl](GPUKernels.jl): CUDA kernels for embedding lookup, normalization, dense layers, attention and pooling. Dense products use CUDA's integer `dp4a` instruction. The implementation does not use Tensor Cores or cuBLAS matrix multiplication. Weights and transformer tensors stay on the GPU during inference; quantization extrema return scalars to the host, which computes scales and launches kernels. Tokenization and final L2 normalization run on the CPU.

There is no ONNX execution engine, Python bridge or external matrix multiplication library in these Julia inference paths. CUDA still requires NVIDIA's driver/runtime. Python/ONNX is used only to prepare artifacts and measure the comparison baseline. The kernels implement this GTE-tiny architecture, not arbitrary ONNX graphs.

## Throughput

Same 16-text panel, batches of 1 and 8, five measured whole-panel passes after warmup. Values are texts divided by median elapsed time. Timing includes tokenization, padding, dynamic quantization, inference, pooling, normalization, and GPU transfers/synchronization where applicable. Loading, compilation and weight upload are excluded. Julia backend order rotates between passes; ONNX runs separately without rotation. These are small-panel measurements, not service load tests.

### CPU

| Backend | Batch 1, texts/s | Batch 8, texts/s |
| --- | ---: | ---: |
| Original Julia INT8 | 20.05 | 8.98 |
| Julia KA + VNNI, standard math | 59.39 | 33.27 |
| Julia KA + VNNI, polynomial math | **82.88** | **51.26** |
| ONNX Runtime INT8 | 200.01 | 109.91 |
| ONNX Runtime FP32 | 92.93 | 49.59 |

Standard math improves throughput 2.96× / 3.70× over the original. Polynomial math improves it 4.13× / 5.71×, reaching 41% / 47% of ONNX INT8 throughput. These CPU gains depend on AVX-512 VNNI; the portable fallback has correctness coverage but no reported throughput measurement.

CPU cumulative allocations per whole-panel pass fall from 286 MB to 61 MB at batch 1, and from 636 MB to 125 MB at batch 8. These are allocation totals, not peak memory or model size. Scratch reuse currently spans layers within a batch, not multiple calls. Larger batches pad short texts to the longest input, so they can reduce throughput on this mixed-length panel.

### GPU

| Backend | Batch 1, texts/s | Batch 8, texts/s |
| --- | ---: | ---: |
| Julia KA + CUDA INT8 DP4A | **106.53** | **314.00** |
| ONNX Runtime INT8, CUDA + CPU fallback | 90.40 | 90.30 |
| ONNX Runtime FP32, CUDA | 501.01 | 1,297.49 |

The published INT8 graph is a poor GPU baseline in this environment: a separate instrumented two-text run records 48 `MatMulInteger` and 48 `DynamicQuantizeLinear` executions on CPU, plus two other CPU nodes and 144 transfer nodes. The FP32 graph has only two CPU node executions. These are execution counts, not time percentages.

Julia beats this mixed CPU/GPU INT8 baseline by 1.18× / 3.48×, but ONNX FP32 CUDA remains 4.70× / 4.13× faster. The INT8 result therefore does not establish competitive GPU performance against an optimized implementation. Host synchronization for quantization, serial reductions within some kernels, allocations and kernel launches remain optimization candidates.

## Numerical and retrieval agreement

**136 kernel assertions passed**, covering portable and VNNI dense arithmetic, full-range byte inputs, zero points, token tails, attention masks and odd lengths, polynomial approximation bounds, complete embeddings, repeated calls and workspace reuse. GPU checks run with scalar array indexing disabled. The original baseline previously passed 627 assertions, including 585 tokenizer fixtures; the optimized variants share that unchanged tokenizer.

Across five full-model numerical fixtures:

- Standard CPU and polynomial CPU return exactly equal Float32 vectors to the original Julia implementation on these inputs. Their minimum cosine against ONNX INT8 is **0.9998188**.
- CUDA's minimum cosine against ONNX INT8 is **0.9993967**; maximum absolute difference against the original Julia vectors is **0.004644** or less.
- The validation gate requires cosine at least 0.999 against both references on these fixtures. This threshold is not a guarantee for arbitrary inputs or the larger retrieval screen.

The follow-up also re-encodes all 30 queries and 423 candidate documents from the original reduced retrieval screen. Both CPU variants exactly match the original Julia vectors on those inputs. Minimum corresponding-vector cosine against ONNX INT8 is **0.998762 for CPU and 0.9987158 for CUDA**, below the five-fixture 0.999 gate. The retrieval benchmark records this broader-screen agreement rather than enforcing that fixture-specific threshold.

| Reduced screen | Queries / documents | Julia CPU, either math mode | Julia CUDA | ONNX INT8 | ONNX FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NanoNQ | 10 / 139 | 0.96509 | 0.96509 | 0.96509 | 0.96509 |
| NanoHotpotQA | 10 / 146 | 0.94416 | 0.94556 | 0.94556 | 0.92263 |
| NanoMSMARCO | 10 / 138 | 0.95000 | 0.95000 | 0.95000 | 0.96309 |

Values are NDCG@10 with identical candidate IDs, query batches and relevance labels. CUDA matches ONNX INT8's aggregate score on all three reduced screens; CPU retains the small Hotpot difference already present in the original implementation. GPU vectors are not exactly equal to CPU or ONNX vectors. Dynamic quantization can amplify small floating-point differences when values cross rounding thresholds.

These small candidate pools include all positives plus the first 128 sorted corpus IDs. They are easier than full NanoBEIR and do not establish quality parity. The full NanoBEIR and Twitter AP evaluations remain outstanding. The polynomial variant's observed equality also does not imply mathematical or universal bitwise equivalence.

## Reproduce and use

Run from the repository root. Preparation and model downloads follow [README.md](README.md). The new Julia validation and benchmark scripts use `output/gte_tiny_julia/`; unlike the original scripts, they do not accept an alternate artifact directory. Match the thread count in all three commands.

```sh
julia --project=experiments/gte_tiny_julia -e 'using Pkg; Pkg.instantiate()'

# Fresh ONNX CPU and CUDA timings; CUDA needs an available CUDA execution provider.
.venv-embeddings/bin/python experiments/gte_tiny_julia/kernel_reference.py --threads 2

julia --startup-file=no --threads=2 --project=experiments/gte_tiny_julia \
  experiments/gte_tiny_julia/kernel_validate.jl
julia --startup-file=no --threads=2 --project=experiments/gte_tiny_julia \
  experiments/gte_tiny_julia/kernel_benchmark.jl
```

For a machine without CUDA, append `--cpu-only` to each of the reference, validation and benchmark commands. Standard CPU inference does not load CUDA. Validation saves source and fixture hashes; benchmarking rejects changes to those files until validation is rerun. The model loader separately checks weight payload hashes.

```sh
# Choose cpu, cpu-polynomial, or cuda. No Python process is used for inference.
julia --startup-file=no --threads=2 --project=experiments/gte_tiny_julia \
  experiments/gte_tiny_julia/kernel_embed.jl cpu output/gte_tiny_julia/model \
  'Julia kernel parity.' 'Another sentence.'
```

The CLI returns JSON with the selected backend and normalized vectors. For Julia callers, include `KernelGte.jl`, construct `KernelGte.KernelModel(GteTiny.Model(bundle))`, and call `KernelGte.encode(model, texts; batch_size=8)`. Use `polynomial_math=true` explicitly to enable the approximation. For CUDA, include `GPUKernels.jl`, construct `GPUKernels.GPUModel(reference)`, and call `GPUKernels.encode`. These remain single-caller experimental APIs; concurrent server integration has not been tested.

Generated artifacts stay ignored under `output/gte_tiny_julia/`: `kernel-validation.json`, `kernel-reference.json`, `kernel-benchmark.json`, per-backend vectors in `kernel-vectors/`, and ONNX provider profiles. The benchmark JSON retains raw timing samples, allocation/GC measurements, per-query scores and source/fixture hashes. CPU and CUDA standalone CLI smoke checks both return finite normalized 384D vectors.

## Next parity targets

1. Trace the first divergent quantization boundary against ONNX on the existing Hotpot inputs, then expand numerical and retrieval coverage beyond the reduced screen. Keep standard math as the default during that investigation.
2. Profile remaining CPU time and optimize the measured bottleneck while preserving standard-math vector equality. The present speed gap is still approximately 2.1–2.4× even with polynomial math.
3. On GPU, measure device-resident quantization scales, fused reductions and buffer reuse; separately test an INT8 Tensor Core dense kernel. Compare against FP32 CUDA as well as the mixed INT8 baseline. Repeat on an L4 before making L4 claims.

Library references: [KernelAbstractions documentation](https://juliagpu.github.io/KernelAbstractions.jl/stable/) and [CUDA.jl kernel programming](https://cuda.juliagpu.org/stable/development/kernel/). Version pins deliberately follow the repository's existing kernel environment.
