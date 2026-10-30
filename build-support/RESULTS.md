# PackageCompiler application results

Validated locally on Linux x86-64 with Julia **1.13.0**, PackageCompiler **2.4.5**,
and Julia CPU target `generic`. The build includes SQLite, native Julia YAKE, and
the pinned Rust GTE-tiny INT8 encoder with CPU ONNX Runtime 1.30.0.

The resulting executables are:

- `output/mondrian-app/bin/mondrian-server`
- `output/mondrian-app/bin/mondrian-tool`

Both use the same bundled Julia image. Neither requires a Julia installation,
Python, the repository, or a populated Julia package depot at runtime.

## Measurements

| Measurement | Observed result |
| --- | --- |
| Complete uncompressed bundle on disk | About **518 MiB** |
| Transfer archive (`output/mondrian-app-linux-x86_64.tar.gz`) | About **216 MiB** |
| `mondrian-server --help`, three processes | **0.616–0.649 s** |
| Serve-mode readiness, 256D fixture | **0.715 s**; restart **0.741 s** |
| Serve-mode readiness, 384D INT8 fixture | **0.758 s**; restart **0.732 s** |
| Server RSS after HTTP queries | **454–470 MiB** |
| Compiler process RSS during image generation | About **8 GiB** |

Startup measurements use an already-built bundle and warm filesystem caches on
the development machine, with two Julia worker threads, one interactive thread,
and one BLAS thread. The stores contain only 32 posts. RSS covers the serve-mode
server without an active embedding worker; these figures are not a production
capacity test or a comparison against the earlier full ingestion pilot.

The first build took roughly eleven minutes, including the filtered Julia base
image. Rebuilding with that base image cached took roughly four minutes. Build on
a machine with sufficient memory and transfer the whole bundle to the host.

## Validation

`python3 build-support/smoke.py output/mondrian-app` passed with the application
copied to an unrelated temporary directory, its generated build-source directory
hidden, executable `PATH` empty, and an empty Julia depot:

- Core toy-model demo: normalized density and two observations.
- Bundled Rust encoder: two finite, normalized 384D vectors.
- Combined ingestion: local mock Mastodon feed, HTML normalization, native
  embeddings, and two posts committed to a new SQLite store.
- Synthetic 256D and actual INT8 384D stores: 32 posts written and audited.
- Health, readiness, tree export, PPCA export, nearby results, and keyword counts.
- SIGINT shutdown and restart, retaining the same store identity.
- Source Julia reads a compiled-written store; compiled executables read and serve
  a source-written store. Keeping the existing `Main.*` module identities preserves
  Julia Serialization compatibility for these fixtures on the tested Julia version.

The raw results are in `output/mondrian-app-smoke.json`; build dependency versions
and configuration are under `output/mondrian-app/share/mondrian/`.

The ingestion test exposed an existing HTTP-reader bug: `json_get` forced every
response into a dictionary, rejecting Mastodon's top-level arrays. It now accepts
both objects and arrays. All **23** tests in `test/live_sources.jl` passed, including
new array-response coverage and the existing WebSocket test.

The image hook replayed 750 precompile statements; 25 references unavailable in
the image-build process were skipped. Runtime compilation remains enabled. This
build does not establish zero-JIT execution, portability to other libc versions,
cross-version store compatibility, optional RocksDB support, or live public-feed
coverage. Use the [build instructions](README.md) to reproduce and retest on the
deployment platform.
