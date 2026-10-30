# Native Rust GTE-tiny results

Recorded 2026-10-27 on Linux x86-64, Ryzen 7 7800X3D, Rust 1.97.1 and ONNX Runtime 1.30.0. Both paths use two inference threads and the same pinned INT8 model and byte-identical native runtime library. The Rust worker achieves approximately the existing ONNX CPU throughput while removing Python from deployed inference.

| Backend | Batch 1, texts/s | Batch 8, texts/s |
| --- | ---: | ---: |
| Rust worker, including protocol round-trip | 201.81 | 100.86 |
| Python, in-process ONNX reference | 193.10 | 93.24 |

These are medians of five alternating whole-panel passes after warmup on the same 16 texts used in the Julia kernel experiment. Rust timing includes JSON/pipe transport, base64 decoding and output checks in the reference client; Python calls ONNX directly. Both include tokenization, padding, pooling and L2 normalization. Loading and compilation are excluded. Larger batches introduce padding work on this mixed-length panel. The small observed advantage is not evidence of universally faster Rust inference: both use the same ONNX kernels, and this is one local timing run.

## Parity and integration

- **585 tokenizer fixtures match exactly**, including Unicode, empty text, truncation and the retrieval inputs.
- **49 embeddings across six test cases** match fresh Python ONNX outputs with maximum absolute error `1.044e-7` or less. Cases include empty text, mixed lengths, long/truncated inputs and a 36-text request spanning multiple internal batches.
- **453 retrieval embeddings** (30 queries and 423 candidate documents) match stored ONNX INT8 references with maximum absolute error `1.789e-7` or less. All per-query NDCG@10 values match, not only dataset averages.
- Positive BIP340 signatures generated independently with Python/coincurve pass Rust verification, including Unicode, control characters and escaping. Tampered content, signatures and malformed fields fail.
- Invalid requests preserve worker usability; oversized frames terminate with bounded buffering. Corrupt pinned assets fail before the ready handshake.
- The worker ran with an empty executable search path and no Python module path. Julia bridge tests also passed with the Python executable deliberately set to `/no/python`.
- Rust unit tests and Clippy passed; **16 Julia native-bridge assertions** and **14 existing FP32 bridge/persistence assertions** passed. A local disk fixture replay and restart succeeded using Rust, retaining 66 active records and the same generation on replay continuation.

| Reduced retrieval screen | Queries / documents | Rust and ONNX INT8 NDCG@10 |
| --- | ---: | ---: |
| NanoNQ | 10 / 139 | 0.96509 |
| NanoHotpotQA | 10 / 146 | 0.94556 |
| NanoMSMARCO | 10 / 138 | 0.95000 |

These reduced candidate pools are implementation checks, not full NanoBEIR quality evaluation. Float32 pooling/normalization reduction order can produce small differences even with identical hidden states; universal bitwise equality is not claimed. CUDA was not tested or enabled for this worker.

## Size and memory

| Deployment component | Size |
| --- | ---: |
| Stripped Rust worker, including tokenizer and signature verification | 4.64 MiB |
| Native ONNX Runtime shared library | 27.72 MiB |
| Quantized ONNX model | 21.83 MiB |
| Tokenizer JSON | 0.68 MiB |
| Total above, excluding system libraries and notices | **54.86 MiB** |

The measured bundle has no Python interpreter, Python bindings, CUDA provider or CUDA toolkit. ONNX Runtime is dynamically loaded; its size is additional to the Rust executable. This full worker's size is not comparable to the earlier minimal C/Rust version-printing probes.

One cached-filesystem process-start-to-ready observation was **88 ms**, including checksums and model loading. This is not a cold-storage startup benchmark. Linux worker RSS/high-water readings after the timing panel were about **80.5 MiB at batch 1** and **192.1 MiB at batch 8**; these include ONNX allocator arenas and shared mappings, exclude the Julia/Python parent, and are not proportional-memory measurements or worst-case bounds.

Generated evidence is in ignored `output/gte_tiny_rust/validation.json`: raw timing samples, per-query scores, absolute vector errors, source/fixture/binary/runtime hashes and memory readings. [Build, integration and reproduction instructions](README.md) describe the separate INT8 embedding identity and new-store configuration.
