"""Exercise a relocated PackageCompiler bundle with no Julia/Python on PATH.

Python is test tooling only. The deployed application does not require it.
The generated build-stage directory is temporarily hidden during the test.
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tomllib
import urllib.request
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--julia", default=shutil.which("julia"), help="Source-store interoperability test")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    metadata = tomllib.loads((bundle / "share/mondrian/build.toml").read_text())
    stage = Path(metadata["source_stage"])
    hidden = stage.with_name(stage.name + ".smoke-hidden")
    # Only hide a generated build directory, never a source checkout.
    hide = stage.is_dir() and stage.parent == ROOT / "output" and stage.name.startswith("mondrian-build-")
    if hide and hidden.exists():
        raise RuntimeError(f"Refusing existing destination {hidden}")
    results = {"encoder": metadata["encoder"], "source_stage_hidden": hide}
    with tempfile.TemporaryDirectory(prefix="mondrian-relocated-") as directory:
        work = Path(directory)
        app = work / "app"
        shutil.copytree(bundle, app, symlinks=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("JULIA_", "MONDRIAN_", "PYTHON", "LD_"))}
        env.update(PATH="", HOME=str(work), JULIA_DEPOT_PATH=str(work / "empty-depot"),
                   JULIA_LOAD_PATH="@stdlib", JULIA_NUM_THREADS="2,1", OPENBLAS_NUM_THREADS="1")
        tool = app / "bin/mondrian-tool"
        server = app / "bin/mondrian-server"

        def run(argv, **kwargs):
            result = subprocess.run([str(x) for x in argv], cwd=work, env=env,
                                    text=True, capture_output=True, timeout=180, **kwargs)
            if result.stderr:
                print(result.stderr, flush=True)
            result.check_returncode()
            return result.stdout

        def call(*argv):
            return json.loads(run([tool, *argv]))

        def check_http(config, dimension):
            config.write_text(config.read_text().replace("port = 8080", "port = 0"))
            start = time.monotonic()
            with tempfile.TemporaryFile(mode="w+") as errors:
                proc = subprocess.Popen([str(server), "--config", str(config)], cwd=work,
                                        env=env, text=True, stdout=subprocess.PIPE, stderr=errors)
                try:
                    with selectors.DefaultSelector() as selector:
                        selector.register(proc.stdout, selectors.EVENT_READ)
                        while True:
                            if not selector.select(max(0, 120 - (time.monotonic() - start))):
                                raise TimeoutError("compiled server did not become ready")
                            line = proc.stdout.readline()
                            if not line:
                                raise RuntimeError(f"compiled server exited: {proc.poll()}")
                            status = json.loads(line)
                            if status.get("server") == "ready":
                                break
                    elapsed = time.monotonic() - start
                    base = f"http://127.0.0.1:{status['port']}"
                    # Ignore system proxy settings for this loopback-only fixture.
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

                    def request(path, body=None):
                        data = None if body is None else json.dumps(body).encode()
                        req = urllib.request.Request(base + path, data, {"Content-Type": "application/json"})
                        with opener.open(req, timeout=60) as response:
                            return json.load(response)

                    assert request("/healthz")["status"] == "ok"
                    assert request("/readyz")["status"] == "ready"
                    tree = request("/v1/tree")
                    assert tree["embedding"]["dimension"] == dimension
                    assert sum(leaf["post_count"] for leaf in tree["leaves"]) == 32
                    query = {"embedding_space_id": tree["embedding_space_id"],
                             "vector": [1.] + [0.] * (dimension - 1)}
                    near = request("/v1/nearby", query)
                    assert near["returned"] == 25
                    assert len({item["identity"]["native_id"] for item in near["items"]}) == 25
                    assert all(len(item["delta"]) == dimension for item in near["items"])
                    terms = request("/v1/terms", query)
                    assert terms["covered"] == 32
                    assert any(term["term"] == "julia" and term["count"] == 32 for term in terms["terms"])
                    request("/v1/tree?view=ppca")
                    memory = {}
                    status_file = Path(f"/proc/{proc.pid}/status")
                    if status_file.exists():
                        for line in status_file.read_text().splitlines():
                            if line.startswith(("VmRSS:", "VmHWM:")):
                                key, value, _ = line.split()
                                memory[key.rstrip(":")] = int(value)
                    proc.send_signal(signal.SIGINT)
                    assert proc.wait(timeout=30) == 0
                    return {"startup_seconds": elapsed, "store_id": tree["store_id"],
                            "dimension": dimension, "memory_kib": memory}
                except BaseException:
                    errors.seek(0)
                    print(errors.read())
                    raise
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    proc.stdout.close()

        if hide:
            stage.rename(hidden)
        try:
            samples = []
            for _ in range(3):
                start = time.monotonic()
                assert "--config" in run([server, "--help"])
                samples.append(time.monotonic() - start)
            results["help_seconds"] = samples
            demo = call("demo")
            assert abs(demo["integral"] - 1) < 1e-6 and demo["observations"] == 2
            if metadata["encoder"]:
                embedding = call("embed", "Julia and Rust.", "Another topic.")
                assert embedding["dimension"] == 384 and len(embedding["vectors"]) == 2
                for vector in embedding["vectors"]:
                    assert len(vector) == 384 and all(math.isfinite(x) for x in vector)
                    assert abs(sum(x*x for x in vector) - 1) < 1e-5
                # Exercise the compiled collector -> Rust -> persistent-store path.
                # All requests stay on loopback; no public source is contacted.
                timestamp = datetime.now(timezone.utc).isoformat()
                class Feed(BaseHTTPRequestHandler):
                    def do_GET(self):
                        query = parse_qs(urlparse(self.path).query)
                        rows = [] if "min_id" in query else [
                            {"id": str(i), "uri": f"https://fixture.invalid/posts/{i}",
                             "created_at": timestamp, "content": f"<p>Julia compiled ingestion example {i}.</p>",
                             "visibility": "public", "tags": [{"name": "Julia"}]}
                            for i in (2, 1)]
                        data = json.dumps(rows).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)

                    def log_message(self, *args):
                        pass

                feed = ThreadingHTTPServer(("127.0.0.1", 0), Feed)
                thread = threading.Thread(target=feed.serve_forever, daemon=True)
                thread.start()
                combined = work / "combined"
                combined.mkdir()
                (combined / "sources.toml").write_text(
                    'poll_interval = 0.1\n[index]\ndimension = 384\n'
                    f'embedding_model = {json.dumps(embedding["model"])}\n'
                    'max_leaves = 1\n[keywords]\nmethod = "hashtags"\n'
                    '[[sources]]\nname = "local-fixture"\nkind = "mastodon"\n'
                    f'origin = "http://127.0.0.1:{feed.server_port}"\n')
                (combined / "server.toml").write_text(
                    'mode = "combined"\nstore = "store"\ningestion_config = "sources.toml"\n'
                    'embedding_device = "cpu"\nsource_seconds = 2\nmax_events = 32\n'
                    '[http]\nhost = "127.0.0.1"\nport = 0\n'
                    '[storage]\nbackend = "sqlite"\nmanaged_budget_mib = 32\n')
                try:
                    logs = run([server, "--config", combined / "server.toml", "--seconds", "20"])
                    assert any(json.loads(line).get("received", 0) == 2 for line in logs.splitlines()), logs
                    assert call("audit", combined / "store")["posts"] == 2
                    results["combined_ingestion_posts"] = 2
                finally:
                    feed.shutdown()
                    thread.join()
                    feed.server_close()
            dimensions = [256, 384] if metadata["encoder"] else [256]
            results["http"] = []
            for dimension in dimensions:
                fixture = work / f"fixture-{dimension}"
                call("fixture", fixture, *(["--embed"] if dimension == 384 else []))
                assert call("audit", fixture / "store")["posts"] == 32
                first = check_http(fixture / "serve.toml", dimension)
                second = check_http(fixture / "serve.toml", dimension)
                assert first["store_id"] == second["store_id"]
                results["http"].append({"first": first, "restart": second})
            if args.julia:
                # Source code opens a store written by the executable.
                source_env = os.environ.copy()
                source_env.update(JULIA_NUM_THREADS="2,1", OPENBLAS_NUM_THREADS="1")
                def source(*argv):
                    return subprocess.run([args.julia, "--startup-file=no", f"--project={ROOT / 'server'}", *argv],
                                          cwd=ROOT, env=source_env, capture_output=True, text=True,
                                          check=True, timeout=180)
                source("-e", 'include(ARGS[1]); s=DiskTreeStore.open_store(ARGS[2];require_existing=true); '
                       'try @assert DiskTreeStore.audit!(s)==32 finally DiskTreeStore.close_store(s) end',
                       str(ROOT / "server/src/MondrianServer.jl"), str(work / f"fixture-{dimensions[-1]}/store"))
                # Executable opens a store written by source scripts.
                fixture = work / "source-fixture"
                source(str(ROOT / "server/docker/smoke_fixture.jl"), str(fixture), "sqlite")
                assert call("audit", fixture / "store")["posts"] == 32
                check_http(fixture / "smoke.toml", 256)
                results["serialization_both_directions"] = True
            results["passed"] = True
        finally:
            if hide:
                hidden.rename(stage)
    report = bundle.parent / (bundle.name + "-smoke.json")
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
