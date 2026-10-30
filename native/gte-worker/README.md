# Native GTE-tiny worker

A persistent Rust worker for the pinned GTE-tiny **INT8 CPU** model. It uses ONNX Runtime's native library through `ort`, Hugging Face's Rust tokenizer, masked mean pooling and L2 normalization. It also implements the existing bridge's Nostr/BIP340 verification operation using `secp256k1`. Inference needs no Python, PyTorch, Julia subprocess, network access or model downloads.

The Julia application communicates through the existing JSON-lines bridge. The worker loads and checks the model/tokenizer before emitting `ready`, then reuses one ONNX session. Input tensors are INT64, truncation is 256 tokens with right padding, and output is 384D Float32. CPU inference defaults to two threads and batches of eight. Tokenizer parallelism is disabled to avoid an extra thread pool.

This is a runnable native integration, not a replacement for the existing FP32/Qwen indexes. INT8 has a separate model identifier:

```text
TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3;onnx-int8-ceab9964bcdb
```

Existing configurations retain their Python backend. Selecting this identity automatically selects Rust. A different embedding identity requires a new store; existing index compatibility checks reject mixing it with FP32 or Qwen vectors. Dynamic quantization depends on batch composition and padding, so retain the default batch policy for a given deployment.

## Build and prepare

From the repository root, on Linux:

```sh
cargo build --release --locked --manifest-path native/gte-worker/Cargo.toml
.venv-embeddings/bin/python native/gte-worker/prepare.py
```

The first command requires a Rust toolchain (declared minimum 1.92; tested with 1.97.1). Cargo.lock pins dependencies. `ort` is pinned to `2.0.0-rc.15` with the **API 30** feature, dynamic loading, and no automatic runtime downloads. The tokenizer is pinned to `0.23.2`. No HTTP client or GPU provider is enabled in the worker.

Preparation is a one-time Python standard-library script. It obtains the pinned model and tokenizer from the local Hugging Face cache or downloads them, verifies their SHA-256 values, and copies the native runtime from the existing ONNX Python installation. It does not copy Python bindings or CUDA providers. To use a separately obtained trusted Linux runtime instead:

```sh
python3 native/gte-worker/prepare.py --ort-library /absolute/path/libonnxruntime.so.1.30.0
```

The validated runtime is ONNX Runtime 1.30.0. The binding requests API 30; compatibility with another runtime build should be checked with the parity suite. The bundle lives under ignored `output/gte_tiny_rust/`, with the following deployment files:

- `model_quantized.onnx` (full expected SHA-256: `ceab9964bcdb19ad013a2fe44bc1ddb3657ab0af1f48cddc349014d6085762ca`).
- `tokenizer.json` (expected SHA-256: `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0`).
- `libonnxruntime.so` and the target system's ordinary native library dependencies.

`bundle.json` records checksums and sizes for inspection; the worker independently checks the pinned model and tokenizer against compiled constants. Runtime binaries must come from a trusted distribution. Preserve the applicable model, ONNX Runtime and dependency licenses/notices when redistributing a bundle. This change does not add a new Docker image or change the existing Docker deployment.

## Run

```sh
printf '%s\n' '{"op":"embed","texts":["Julia and Rust.","Another topic."]}' '{"op":"close"}' | \
  native/gte-worker/target/release/mondrian-gte-worker --bundle output/gte_tiny_rust
```

`--threads N` and `--batch-size N` configure CPU inference; batch sizes are bounded to 1–32. `--ort-library FILE` overrides the bundled runtime path. This implementation accepts `--device cpu` only. GPU and alternate models still use the existing Python path.

From Julia, using an environment containing JSON3:

```julia
include("src/EmbeddingBridge.jl")
worker = EmbeddingBridge.Worker(model=EmbeddingBridge.GTE_INT8_MODEL)
try
    vectors = EmbeddingBridge.embed(worker, ["Julia and Rust.", "Another topic."])
    # 384 × 2 Matrix{Float32}; columns correspond to inputs.
finally
    close(worker)
end
```

Set `MONDRIAN_EMBEDDING_RUST` to override the executable and `MONDRIAN_GTE_BUNDLE` to override the bundle directory. The bridge also accepts `rust_worker`, `bundle`, `threads` and `batch_size` keywords. These settings apply to the native worker. The old `MONDRIAN_EMBEDDING_PYTHON` setting still applies to legacy models.

For disk ingestion, [live_sources.rust.toml](../../examples/live_sources.rust.toml) selects the new model and `data/live_disk_gte_int8`. After preparing the bundle, a local fixture replay can exercise the complete path without contacting a source:

```sh
julia --project=environments/ingestion examples/ingest_live.jl --storage disk \
  --config examples/live_sources.rust.toml --store output/gte_rust_fixture_index \
  --replay test/fixtures/live_sources.jsonl --max-events 16
```

For the server, select this ingestion configuration and a new store in your server TOML. Its bridge also automatically selects Rust by model identity. Source collection and other application components have their own dependencies; removing Python from the encoder is not a claim that every repository command is Python-free.

## Protocol

One request/response per line. Standard output contains only protocol JSON; startup failures go to stderr and exit nonzero before `ready`.

- `embed`: `texts` contains 1–1024 UTF-8 strings of at most 64 KiB each. Empty strings are supported as BERT special-token inputs. Responses include the model identity, dimension, count, CPU provider and base64-encoded little-endian Float32 vectors, contiguous by text. An empty list is rejected by the worker; the Julia API returns a correctly shaped empty matrix without a request.
- `verify_nostr`: `events` contains at most 1024 events. Responses contain booleans in input order. Verification checks the canonical event hash and BIP340 signature; malformed events return false. Integer timestamps must fit `u64`.
- `tokenize`: diagnostic operation returning padded token IDs and attention masks, used by the parity suite.
- `close`: acknowledges and exits cleanly.

Frames are bounded to 8 MiB including the newline. An oversized frame terminates the worker; malformed JSON or ordinary invalid requests receive an error and leave it available for subsequent requests. Errors do not include submitted text. The Julia bridge serializes requests with its existing lock.

## Validation and measurements

```sh
cargo test --locked --manifest-path native/gte-worker/Cargo.toml
cargo clippy --locked --manifest-path native/gte-worker/Cargo.toml --all-targets -- -D warnings
julia --startup-file=no --project=experiments/gte_tiny_julia test/rust_gte.jl
.venv-embeddings/bin/python native/gte-worker/validate.py
```

The Python command is independent reference tooling, not part of inference. It uses the existing embedding/live Python environment (NumPy, ONNX Runtime, tokenizers and coincurve) and fixtures from [the Julia experiment](../../experiments/gte_tiny_julia/README.md). Run that experiment's preparation first if the fixtures are absent. It checks that Python and Rust load byte-identical runtime libraries, compares token IDs and embeddings, exercises signed Nostr events and invalid requests, checks startup rejection of corrupt assets and oversized frames, and benchmarks the same 16-text panel.

Validation launches the native worker with an empty executable search path and no Python module path. Timing uses five alternating passes after warmup, comparing the complete native worker round-trip (pipes, JSON, base64 and output checks included) against in-process Python ONNX inference. No compilation or model loading enters warm throughput. Generated `validation.json` records raw samples, parity results, process memory, startup, source hashes, binary hashes and bundle hashes. See [RESULTS.md](RESULTS.md) for the recorded run.

The original [pure-Julia kernel experiment](../../experiments/gte_tiny_julia/KERNEL_RESULTS.md) remains available for research. The native worker delegates transformer kernels to ONNX Runtime.
