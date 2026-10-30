# Quantized GTE-tiny in Julia

For the native ONNX Runtime route, see the [Rust worker and Julia bridge](../../native/gte-worker/README.md). It removes Python from inference while retaining ONNX Runtime's optimized CPU kernels. This directory retains the pure-Julia implementation and kernel experiments.

This experiment runs the pinned `TaylorAI/gte-tiny` quantized model with a dedicated Julia implementation. Tokenization, embedding lookup, six transformer layers, attention, pooling and normalization run in Julia. There is no ONNX interpreter, Python bridge, transformer framework or BLAS matrix multiplication in the inference path. Julia's normal runtime and standard math/Unicode facilities remain in use; JSON3 reads the bundle metadata and tokenizer vocabulary.

**The kernel follow-up is now implemented and measured:** [results and commands](KERNEL_RESULTS.md). It uses the repository's existing KernelAbstractions/CUDA versions, with packed AVX-512 VNNI CPU kernels and a CUDA INT8 path. The optional CPU math variant is 4–5.7× faster than the original implementation on the measured panel; ONNX INT8 CPU remains about 2× faster. CUDA reaches 314 texts/s at batch 8 on an RTX 4070 SUPER. Numerical agreement is close, but exact ONNX parity and general performance parity remain open.

The original `GteTiny.jl` remains the correctness baseline. Its initial measurements appear below; the follow-up report contains fresh comparisons. Both implementations are experiments, not production replacements. JSON3 is sufficient for the original inference code; the expanded project also installs KernelAbstractions and CUDA for the new variants.

## Model and implementation

The checkpoint is `TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3`: 384 hidden dimensions, six layers, twelve 32-dimensional attention heads, and a 1,536-dimensional feed-forward intermediate. The experiment uses a 256-token cap, masked mean pooling, and L2-normalized 384D Float32 output.

`prepare.py` checks the published quantized ONNX graph's SHA-256, extracts only the tensors required by this architecture into column-major binary files, and generates Python/ONNX reference fixtures. It is one-time preparation/reference tooling, not an inference dependency. The Julia loader validates every tensor checksum. Downloaded weights and generated fixtures stay under ignored `output/gte_tiny_julia/`.

The implementation preserves this graph's quantization scheme:

- UINT8 embedding tables with scalar scales and zero points; selected rows are dequantized on lookup.
- INT8 dense weights, dynamically quantized UINT8 activations, and INT32 accumulation in Julia SIMD loops.
- Quantized query/key and probability/value attention products, not just quantized dense layers. Activation ranges span the entire tensor, including the batch and padded positions, as in the source graph.
- Float32 residuals, layer normalization (`epsilon=1e-12`), softmax and GELU. GELU uses the Abramowitz–Stegun erf approximation; this is a numerical approximation, not bitwise equivalence to the reference's erf implementation.

WordPiece supports the checkpoint's uncased BERT normalization, accents, CJK splitting, punctuation, special tokens, truncation and right padding. The API accepts single texts, not paired sequences. Batch sizes are bounded to 1–32. Mixed batches can change quantization scales and embeddings; singleton/batched bitwise invariance is not promised.

## Run

From the repository root, using Julia 1.13.0 for the recorded environment:

```sh
julia --project=experiments/gte_tiny_julia -e 'using Pkg; Pkg.instantiate()'

# Existing repository Python environment, or install requirements-reference.txt in a venv.
.venv-embeddings/bin/python experiments/gte_tiny_julia/prepare.py

julia --startup-file=no --threads=2 --project=experiments/gte_tiny_julia \
  experiments/gte_tiny_julia/validate.jl
julia --startup-file=no --threads=2 --project=experiments/gte_tiny_julia \
  experiments/gte_tiny_julia/benchmark.jl
```

Preparation downloads the pinned model if necessary. Retrieval inputs default to the existing benchmark data in `output/gte_tiny_screen/bced8b63d83cdd78/data/`; pass `--data PATH` to use another copy of that notebook-09-format data. Without those files, preparation still produces numerical/tokenizer fixtures and a fallback timing panel, but no retrieval results. `--queries` and `--distractors` control screen size; defaults are ten queries and the first 128 sorted corpus IDs plus all relevant documents per dataset. These intentionally small candidate pools are easier than full NanoBEIR.

Use a separate `--output PATH` for alternate runs; pass that same directory as the positional argument to both Julia scripts. Match Julia's thread count to `prepare.py --threads` (default two). Benchmarking checks that the implementation and input artifacts still match a successful validation run. The prototype's static threaded kernels are intended for a single main-thread caller, not concurrent server requests.

After preparation, embedding needs only Julia and the model bundle, without Python, network access or reference fixtures:

```sh
julia --startup-file=no --threads=2 --project=experiments/gte_tiny_julia \
  experiments/gte_tiny_julia/embed.jl output/gte_tiny_julia/model \
  'Julia trees and probability.' 'An unrelated recipe for pancakes.'
```

The command prints JSON with normalized vectors. In Julia, use `include("GteTiny.jl")`, `using .GteTiny`, `model = Model(bundle_path)`, and `encode(model, texts; batch_size=8)`; columns correspond to input texts.

## Initial measurements

Recorded 2026-10-27 on the local Ryzen 7 7800X3D, Julia 1.13.0 and ONNX Runtime 1.30.0, with two inference threads each. These are CPU measurements, not L4 results.

**Validation: 627 assertions passed.** All 585 tokenizer cases match Hugging Face, including the actual retrieval and timing inputs. Checks cover integer accumulation/zero points, seven hidden-state boundaries, empty and repeated text, mixed-length padding, 256-token truncation, Unicode, deterministic repeats, normalization, and invalid inputs. The standalone CLI also ran with an empty executable search path, without a Python worker.

On the three-text layer fixture, maximum absolute hidden-state error is at most `2.39e-6`. On the five numerical fixtures, minimum Julia/ONNX INT8 embedding cosine is **0.9998188** and maximum absolute vector error is **0.00298**. Across the larger retrieval screen, the minimum corresponding query/document cosine is **0.998762**. These results establish close numerical agreement, not bitwise parity; threshold-sensitive dynamic quantization and floating-point reductions warrant further investigation.

| Backend | Batch 1, texts/s | Batch 8, texts/s |
| --- | ---: | ---: |
| Julia INT8 prototype | 19.25 | 9.05 |
| ONNX Runtime INT8 | 190.54 | 92.15 |
| ONNX Runtime FP32 | 76.82 | 42.15 |

Timing includes tokenization, padding, inference, pooling and normalization on the same 16 real texts. Values use the median of three whole-panel passes after warmup. Loading and first compilation are excluded. Julia measured 0.98 s for model loading and 1.33 s for its first one-text encode inside the process; these are not shell-to-result startup times. Backend order was not rotated, so this is an initial performance screen. Larger batches pad short texts to the longest sequence, explaining why throughput can fall on this mixed panel.

Exported weight payloads total **22.66 MB**; the loaded Julia model object measures **24.31 MB**, including vocabulary and auxiliary arrays. Neither number is peak process RSS. Cumulative allocations per 16-text pass were about 286 MB at batch 1 and 636 MB at batch 8; buffer reuse is an optimization target.

| Reduced retrieval screen | Queries / documents | Julia INT8 NDCG@10 | ONNX INT8 | ONNX FP32 |
| --- | ---: | ---: | ---: | ---: |
| NanoNQ | 10 / 139 | 0.96509 | 0.96509 | 0.96509 |
| NanoHotpotQA | 10 / 146 | 0.94416 | 0.94556 | 0.92263 |
| NanoMSMARCO | 10 / 138 | 0.95000 | 0.95000 | 0.96309 |

These deterministic reduced-corpus checks use identical candidate IDs, query batches and binary relevance labels for all backends. They are implementation checks, **not comparable to the full retrieval results in [the encoder report](../../encoder_experiments_colab/BENCHMARK_RESULTS.md)**. Thirty queries do not establish quantized-model quality or significance. Twitter AP and the full NanoBEIR corpora have not been evaluated here.

The [kernel follow-up](KERNEL_RESULTS.md) implements weight packing, integer kernels and buffer reuse while preserving this reference. Remaining work includes closing the ONNX numerical/performance gaps and running the full quality evaluation. A sysimage can separately reduce startup compilation; its effect on warm throughput must be measured.

## Artifacts and sources

- [Julia model and tokenizer](GteTiny.jl), [validation](validate.jl), [benchmark](benchmark.jl), [CLI](embed.jl), [preparation](prepare.py).
- Local generated [validation details](../../output/gte_tiny_julia/validation.json) and [benchmark samples and per-query scores](../../output/gte_tiny_julia/benchmark.json). These files are intentionally not committed.
- [Pinned model](https://huggingface.co/TaylorAI/gte-tiny/tree/4cc5e73d86a67c601897257b467187234aa3bca3), [ONNX dynamic quantization semantics](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html), and [integer matrix multiplication semantics](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html).
