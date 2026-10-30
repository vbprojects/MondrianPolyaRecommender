# Standalone Julia applications

This recipe uses PackageCompiler to build two executables sharing one compiled
Julia image:

- `mondrian-server`: the existing HTTP server, with serve and combined ingestion modes.
- `mondrian-tool`: embeddings, store auditing, offline fixtures, and the core toy-model demo.

The default bundle contains SQLite, native Julia YAKE, the Rust GTE-tiny INT8 worker,
its model/tokenizer, and CPU ONNX Runtime. Deployment requires neither an installed
Julia nor Python. **Distribute the entire output directory**, including its `lib`,
`share`, and `libexec` directories; the launchers alone are not standalone binaries.
Julia's runtime, garbage collector, and JIT remain part of the application.

## GitHub Actions build

[Build standalone application](../.github/workflows/build-app.yml) builds the default
Linux x86-64 SQLite/INT8 bundle on an `ubuntu-24.04` GitHub-hosted runner. It runs
for relevant pull requests and changes pushed to `main`, or manually through
**Actions → Build standalone application → Run workflow** once the workflow is
on the default branch.

The workflow installs Julia 1.13.0, Rust 1.97.1, and Python 3.12, downloads the
checksum-verified model/tokenizer, and obtains CPU ONNX Runtime 1.30.0 from its
Python wheel. Python and its bindings are not shipped. Julia dependencies,
PackageCompiler's base image, and Rust build dependencies are cached. Image
compilation uses one code-generation thread, with 4 GiB additional swap on runners
with less than 12 GiB RAM.

After worker and transport tests, compilation, and the full relocation/ingestion
smoke test, download `mondrian-app-linux-x86_64-<commit>` from the run's **Artifacts**.
It contains the application tarball and a SHA-256 checksum. The archive preserves
executable permissions and shared-library symlinks. Build logs are uploaded
separately even on failure; artifacts expire after 14 days. This workflow builds
artifacts only: it does not deploy a server or publish a GitHub release.

```sh
# After extracting the downloaded Actions artifact:
sha256sum -c mondrian-app-linux-x86_64.tar.gz.sha256
tar -xzf mondrian-app-linux-x86_64.tar.gz
./mondrian-app/bin/mondrian-server --help
```

Push the build-support files, native worker sources and `Cargo.lock`, server
environment, and required fixtures together. The stopword list in
`julia-yake/data/stopwords_en.txt` and signed test events in
`test/fixtures/nostr_signed.json` are small source inputs explicitly retained by
`.gitignore`; model weights and generated bundles remain ignored.

## Build

Run from the repository root on Linux with Julia 1.13, a C compiler/linker, and the
Rust toolchain from [the worker instructions](../native/gte-worker/README.md).
Prepare the encoder assets once if `output/gte_tiny_rust` is absent:

```sh
.venv-embeddings/bin/python native/gte-worker/prepare.py
```

Python is only needed for this preparation helper and the optional smoke test.
The builder uses its own pinned PackageCompiler environment and leaves the root
and server projects unchanged:

```sh
julia --startup-file=no --project=build-support/compiler -e 'using Pkg; Pkg.instantiate()'
JULIA_NUM_PRECOMPILE_TASKS=2 julia --startup-file=no --project=build-support/compiler build-support/build.jl
```

The result is `output/mondrian-app`. A first build also compiles a filtered Julia
base image and can take ten minutes or more. The local compiler process reached
about 8 GiB RSS; build on a development machine and copy the bundle to a smaller
server. Intermediate sources and precompile
statements are retained under `output/mondrian-build-*` for diagnosis. Once the
build finishes, these directories are not needed at runtime.

Build options:

| Option | Effect |
| --- | --- |
| `--output DIR` | Choose the bundle directory. |
| `--force` | Replace an existing output bundle. |
| `--without-encoder` | Omit the Rust executable/model/runtime; use for serving existing stores and synthetic fixtures. |
| `--with-rocksdb` | Include the server's pinned RocksDB dependency as well as SQLite. |
| `--cpu-target TARGET` | Julia CPU target; defaults to `generic` for portability. |

The default SQLite/encoder build is the validation target. Optional RocksDB builds
need their own smoke run. The recipe is native compilation, not cross-compilation:
build for the target operating system, architecture, and compatible libc. `generic`
controls Julia code generation; it does not make native dependencies universal.
The target still needs ordinary Linux C/C++ system libraries and CA certificates
for HTTPS sources. It does not need the Julia, Rust, or C compiler toolchains.
See [the measured build results](RESULTS.md) for size, startup, and validation scope.

To package the directory for transfer after validation:

```sh
tar -czf output/mondrian-app-linux-x86_64.tar.gz -C output mondrian-app
```

Extract the archive on a compatible Linux x86-64 host and run the executables
under its `mondrian-app/bin` directory. Keep persistent store data outside the
bundle when replacing it with a new build.

## Run

```sh
output/mondrian-app/bin/mondrian-tool demo
output/mondrian-app/bin/mondrian-tool embed 'Julia and Rust.' 'Another topic.'

# An offline 32-post index using real GTE-tiny INT8 embeddings:
output/mondrian-app/bin/mondrian-tool fixture output/compiled-fixture --embed
output/mondrian-app/bin/mondrian-tool audit output/compiled-fixture/store
output/mondrian-app/bin/mondrian-server \
  --config output/compiled-fixture/serve.toml --julia-args --threads=2,1
```

The fixture command refuses an existing store. Without `--embed` it creates
synthetic vectors for testing only. The server listens on `127.0.0.1:8080` in the
generated fixture configuration. Stop with Ctrl-C; `--seconds N` supports bounded
runs. Julia runtime options go after `--julia-args`.

For live ingestion, edit the generated `share/mondrian/sources.toml` and
`share/mondrian/server.toml`, then run:

```sh
output/mondrian-app/bin/mondrian-server \
  --config output/mondrian-app/share/mondrian/server.toml --julia-args --threads=2,1
```

Live mode contacts the configured public sources. Configuration paths are relative
to the configuration file. Set the store to a persistent writable directory when
deploying; the generated default puts it in `../var/mondrian` beside the bundle.
Combined mode accepts the pinned Rust GTE INT8 identity and CPU device;
serving an existing SQLite FP32/Qwen index does not need its original encoder.
Changing embedding identity still requires a new store.

The bundled worker and assets are resolved relative to the relocated executable.
`MONDRIAN_EMBEDDING_RUST` and `MONDRIAN_GTE_BUNDLE` remain explicit overrides.
BLAS uses one thread; the encoder uses the bridge's two-thread default.

## Why the build hook exists

The server currently includes modules into `Main`, and serialized indexes contain
types such as `Main.PostRecords.PostKey`. Moving those definitions under a new
application module would change the persistence contract. `load_runtime.jl` and
`image.jl` load the original modules into `Main` during image creation, while the
small `MondrianCLI` package provides PackageCompiler's `Cint` entry points.
No source file is included when a deployed executable starts.

The workload covers the toy model, actual native embeddings, SQLite writes and
reopening, keyword extraction, auditing, HTTP tree export, nearby queries, terms,
and PPCA export. Its trace is replayed after loading the script modules in the
image hook. Other code paths can still compile on first use. This is a deployment
and startup improvement; it does not promise a lower steady-state memory budget
or different inference throughput.

## Verify a bundle

```sh
python3 build-support/smoke.py output/mondrian-app
```

The smoke test copies the bundle to a temporary unrelated path, hides its generated
build sources, clears executable `PATH`, and uses an empty Julia depot. It exercises
embeddings, both 256D synthetic and 384D INT8 fixtures, HTTP queries, restart, and
clean interrupt shutdown. A loopback mock Mastodon feed also exercises actual
collection, HTML parsing, Rust embedding, and committing two posts in combined mode.
With a host Julia available it also checks store
serialization in both directions between source scripts and compiled executables.
The JSON report is written beside the bundle. Host Julia/Python are test tools,
not dependencies of the deployed processes.

PackageCompiler's [application documentation](https://julialang.github.io/PackageCompiler.jl/stable/apps.html)
and [API reference](https://julialang.github.io/PackageCompiler.jl/stable/refs.html)
describe the runtime bundle, entry-point contract, CPU targets, and stdlib filtering.
