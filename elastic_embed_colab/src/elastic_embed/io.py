import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def records(cfg):
    from .config import run_dir
    root = run_dir(cfg)
    manifest = json.loads((root / "corpus_manifest.json").read_text())
    if checksum(root / "corpus.jsonl") != manifest["corpus_sha256"]:
        raise ValueError("Corpus checksum mismatch; regenerate dependent caches")
    return [json.loads(line) for line in (root / "corpus.jsonl").open()], manifest


def collate(rows, pad_id, device="cpu"):
    n = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        t = r["input_ids"]
        ids[i, :len(t)] = torch.tensor(t)
        mask[i, :len(t)] = 1
    return ids.to(device), mask.to(device)
