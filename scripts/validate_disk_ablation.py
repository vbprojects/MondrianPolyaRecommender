#!/usr/bin/env python3
"""Run after performance collection; correctness work must not compete with it."""
import ast
import hashlib
import json
import pathlib
import platform
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output/disk_ablation"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tests = []
    for script in ["test/disk_tree_store.jl", "test/disk_crash_process.jl"]:
        command = ["julia", "--project=environments/storage", script]
        log = logs / (pathlib.Path(script).stem + "-validation.log")
        start = time.monotonic()
        with log.open("w") as f:
            subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        groups = re.findall(r"^([^|\n]+)\|\s+(\d+)\s+(\d+)\s+[^\n]+$", log.read_text(), re.M)
        assert groups and all(a == b for _, a, b in groups), log
        tests.append(dict(command=command, assertions=sum(int(a) for _, a, _ in groups),
                          seconds=time.monotonic() - start, log=str(log.relative_to(OUT))))
        print(script, tests[-1]["assertions"], "passed", flush=True)

    manifest = json.loads((ROOT / "data/gte_disk_ablation/manifest.json").read_text())
    verified = {}
    for name, expected in manifest["sha256"].items():
        verified[name] = sha(ROOT / "data/gte_disk_ablation" / name)
        assert verified[name] == expected, name
    verified["source"] = sha(ROOT / "data/bluesky_100000/posts.jsonl")
    assert verified["source"] == manifest["contract"]["source_sha256"]

    for script in ROOT.glob("scripts/*disk_ablation*.py"):
        ast.parse(script.read_text(), filename=str(script))
    subprocess.run(["node", "--check", "test/disk_ablation_viewer.cjs"], cwd=ROOT, check=True)

    paths = [ROOT / p for p in ["src/DiskTreeStore.jl", "src/DiskCodecs.jl",
             "src/LeafLatentTree.jl", "src/PostRecords.jl", "test/disk_tree_store.jl",
             "test/disk_crash_process.jl", "test/disk_ablation_viewer.cjs",
             "environments/storage/Project.toml", "environments/storage/Manifest.toml"]]
    paths += list(ROOT.glob("examples/disk_ablation*")) + list(ROOT.glob("scripts/*disk_ablation*.py"))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
    config = json.loads((OUT / "config.json").read_text())
    config.setdefault("launch_sources", config["sources"])
    config["sources"] = hashes
    config["quality_protocol_version"] = 3
    config["embedding_details"] = dict(pooling="masked mean", normalization="L2",
                                        model_weights="Float32", outputs="Float32", tf32=False)
    config["source_history"] = [
        "Launch hashes are retained separately from final validated source hashes.",
        "During collection, atomic empty-store bootstrap and an existing-alias batch guard were added. "
        "These do not change the measured fresh-post encoding, tree update or query paths.",
        "An indexed reverse lookup from internal ID to native platform identity and comment clarifications were also added; "
        "the measured traffic uses the existing native-key lookup path.",
        "All 18 quality audits use protocol 3, which excludes the source post from regional candidates "
        "and counts empty decoded regions as zero preservation.",
        "Performance collection is serialized; the final validation suite runs afterward."]
    config["host"] = dict(platform=platform.platform(),
        cpu=next(line.split(":", 1)[1].strip() for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")),
        physical_memory_bytes=int(re.search(r"MemTotal:\s+(\d+)", pathlib.Path("/proc/meminfo").read_text())[1])*1024,
        filesystem=subprocess.check_output(["findmnt", "-T", str(ROOT), "-no", "SOURCE,FSTYPE,TARGET"], text=True).strip(),
        cold_io_scope="Linux file-cache eviction hint and block-layer read_bytes; WSL2 host/device caches uncontrolled",
        julia=subprocess.check_output(["julia", "--version"], text=True).strip())
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(["python3", "scripts/disk_ablation_report.py"], cwd=ROOT, check=True)
    subprocess.run(["node", "test/disk_ablation_viewer.cjs"], cwd=ROOT, check=True)
    report = dict(tests=tests, assertions=sum(t["assertions"] for t in tests),
                  verified_input_hashes=verified, source_hashes=hashes,
                  browser=json.loads((OUT / "browser_validation.json").read_text()),
                  benchmark_checks=json.loads((OUT / "summary.json").read_text())["checks"])
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Validation complete:", report["assertions"], "Julia assertions and 168 browser combinations.")


if __name__ == "__main__":
    main()
