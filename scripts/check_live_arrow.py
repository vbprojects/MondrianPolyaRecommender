#!/usr/bin/env python3
"""Independent Arrow reader: validate identity joins, decode and route every post.
Reads no Julia serialization and does not require the original embeddings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.ipc as ipc

def validate(root):
    root = Path(root)
    name = (root / "CURRENT").read_text().strip()
    if not name.startswith("generation-") or "/" in name or "\\" in name:
        raise ValueError("invalid checkpoint pointer")
    path = root / name
    def table(name):
        return ipc.open_file(path / f"{name}.arrow").read_all().to_pylist()
    for row in table("checksums"):
        assert hashlib.sha256((path / row["file"]).read_bytes()).hexdigest() == row["sha256"]
    metadata = table("metadata")[0]
    config = json.loads(metadata["config_json"])
    dimension = config["dimension"]
    posts = {p["post_id"]: p for p in table("posts")}
    keys = {}
    for p in posts.values():
        key = (p["kind"], bytes(p["native_id"]))
        assert key not in keys
        keys[key] = p["post_id"]
        if p["kind"] == 3:
            assert len(key[1]) == 32
        else:
            key[1].decode("utf-8")
        assert bool(p["has_record"]) == (p["status"] == 1)
    for a in table("aliases"):
        key = (a["kind"], bytes(a["native_id"]))
        assert a["post_id"] in posts and key not in keys
        keys[key] = a["post_id"]
    centers = {c["decoder_id"]: np.array(c["center"], dtype=np.float64) for c in table("decoders")}
    nodes = {n["node_id"]: n for n in table("nodes")}
    codes = table("codes")
    seen = set()
    decoded = {}
    for c in codes:
        pid = c["post_id"]
        assert pid in posts and posts[pid]["status"] == 1 and pid not in seen
        seen.add(pid)
        q = np.array(c["residual"], dtype=np.int16)
        assert len(q) == dimension and np.all((q >= 0) & (q <= 254))
        vector = (centers[c["decoder_id"]] + float(c["scale"]) * (q - 127)).astype("<f4")
        assert np.isfinite(vector).all()
        node = 1
        visited = set()
        while nodes[node]["axis"]:
            assert node not in visited
            visited.add(node)
            n = nodes[node]
            node = n["left"] if vector[n["axis"]-1] < n["cut"] else n["right"]
        assert node == c["region"]
        decoded[pid] = vector
    assert seen == {p for p, row in posts.items() if row["status"] == 1}
    assert not posts or metadata["next_id"] > max(posts)
    for state in table("states"):
        assert nodes[state["node_id"]]["axis"] == 0 and state["mass"] >= 0
        assert all(np.isfinite(state[name]).all() for name in ("mean", "scatter", "basis", "eigenvalues"))
    digest = hashlib.sha256()
    for pid in sorted(decoded):
        digest.update(pid.to_bytes(8, "little"))
        digest.update(decoded[pid].tobytes())
    return dict(active_posts=len(codes), catalog_posts=len(posts), identities=len(keys),
                decoder_blocks=len(centers), dimension=dimension, leaves=sum(n["axis"] == 0 for n in nodes.values()),
                splits=metadata["split_count"], merges=metadata["merge_count"],
                checkpoint_bytes=sum(p.stat().st_size for p in path.glob("*.arrow")),
                decoded_sha256=digest.hexdigest(), platforms=sorted({p["platform"] for p in posts.values() if p["has_record"]}),
                cursors=len(table("cursors")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = json.dumps(validate(args.store), indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report + "\n")
    print(report)
