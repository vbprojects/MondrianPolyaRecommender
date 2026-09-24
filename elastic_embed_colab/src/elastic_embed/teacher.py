import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import identity, run_dir, teacher_layers
from .io import atomic_json, checksum, collate, records
from .model import last_valid, prefix


def load_teacher(cfg, attn_implementation="sdpa"):
    from transformers import AutoModel
    precision = cfg["teacher"]["precision"]
    if precision not in ("bfloat16", "float32"):
        raise ValueError("Teacher supports explicit BF16 or FP32; no implicit quantization/FP16")
    if precision == "bfloat16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Teacher BF16 needs a supporting GPU; explicitly select float32/offload or another GPU")
    args = dict(revision=cfg["teacher"]["revision"], torch_dtype=getattr(torch, precision),
                device_map=cfg["teacher"]["device_map"], attn_implementation=attn_implementation)
    if cfg["teacher"].get("max_memory"):
        args["max_memory"] = cfg["teacher"]["max_memory"]
    model = AutoModel.from_pretrained(cfg["teacher"]["name"], **args).eval()
    if (model.config.hidden_size, model.config.num_hidden_layers, model.config.vocab_size) != (4096, 36, 151665):
        raise ValueError("Unexpected teacher configuration; revisit layer mapping")
    return model


def cache_contract(cfg, corpus, kind):
    if corpus["tokenizer_revision"] != cfg["teacher"]["revision"]:
        raise ValueError("Corpus tokenizer revision differs from teacher")
    return {"kind": kind, "corpus_id": corpus["id"], "teacher": cfg["teacher"],
            "pooling": "last-valid-token", "prompt": "", "quantization": None,
            "normalization": "slice384_then_l2_then_fp16", "preprocessing": corpus["preprocessing"],
            "tokenizer_revision": corpus["tokenizer_revision"], "cache_config": cfg["cache"]}


def open_cache(path, contract, resume):
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "manifest.json"
    key = identity(contract)
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m["id"] != key:
            raise ValueError("Incompatible cache identity; choose a new run_dir")
        if not resume:
            raise FileExistsError("Cache exists: pass --resume")
        for shard in m["shards"]:
            for name, sha in shard["checksums"].items():
                if checksum(path / name) != sha:
                    raise ValueError(f"Corrupt completed shard: {name}")
        return m
    m = {"id": key, "contract": contract, "shards": [], "complete": False}
    atomic_json(manifest_path, m)
    return m


def write_shard(path, manifest, start, rows, arrays):
    names = {}
    for name, array in arrays.items():
        filename = f"{start:09d}.{name}.npy"
        tmp = path / (filename + ".tmp")
        with tmp.open("wb") as f:
            np.save(f, array, allow_pickle=False)
        tmp.replace(path / filename)
        names[filename] = checksum(path / filename)
    meta_name = f"{start:09d}.rows.json"
    atomic_json(path / meta_name, [{"id": r["id"], "text_sha256": r["text_sha256"]} for r in rows])
    names[meta_name] = checksum(path / meta_name)
    manifest["shards"].append({"start": start, "count": len(rows), "metadata": meta_name, "checksums": names})
    # Manifest replacement is the completion marker; orphan files get overwritten on resume.
    atomic_json(path / "manifest.json", manifest)


def cache_embeddings(cfg, resume=False):
    rows, corpus = records(cfg)
    root = run_dir(cfg)
    report = json.loads((root / "runtime_report.json").read_text())
    if not report.get("reference_passed") or (cfg["model"]["backend"] == "fla" and not report.get("fla_passed")):
        raise RuntimeError("Run runtime-check successfully for the chosen training backend before teacher caching")
    path = root / "teacher_cache"
    m = open_cache(path, cache_contract(cfg, corpus, "sentence"), resume)
    if m["complete"]:
        print("Teacher cache is already complete")
        return
    teacher = load_teacher(cfg)
    device = teacher.get_input_embeddings().weight.device
    start = sum(s["count"] for s in m["shards"])
    size, batch = cfg["cache"]["shard_size"], cfg["teacher"]["batch_size"]
    clock = time.perf_counter()
    for offset in range(start, len(rows), size):
        subset = rows[offset:offset + size]
        targets, audits = [], []
        for j in range(0, len(subset), batch):
            ids, mask = collate(subset[j:j + batch], corpus["pad_id"], device)
            with torch.inference_mode():
                h = teacher(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                pooled = last_valid(h.to(device), mask).float()
                targets.append(prefix(pooled, 384).cpu().numpy().astype(np.float16))
                if offset < cfg["cache"]["audit_examples"]:
                    audits.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        arrays = {"targets": np.concatenate(targets)}
        if audits:
            arrays["audit4096"] = np.concatenate(audits)
        write_shard(path, m, offset, subset, arrays)
        print(f"Teacher cache {offset + len(subset)}/{len(rows)}; {time.perf_counter() - clock:.1f}s", flush=True)
    m["complete"] = True
    m["elapsed_this_session_seconds"] = time.perf_counter() - clock
    atomic_json(path / "manifest.json", m)


class TargetCache:
    def __init__(self, path, rows, expected_corpus=None):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if not self.manifest["complete"]:
            raise ValueError("Teacher cache incomplete")
        if expected_corpus and self.manifest["contract"]["corpus_id"] != expected_corpus:
            raise ValueError("Cache belongs to a different corpus")
        self.index, self.arrays = {}, []
        for shard in self.manifest["shards"]:
            for name, sha in shard["checksums"].items():
                if checksum(self.path / name) != sha:
                    raise ValueError(f"Cache checksum failed: {name}")
            meta = json.loads((self.path / shard["metadata"]).read_text())
            array = np.load(self.path / f"{shard['start']:09d}.targets.npy", mmap_mode="r")
            self.arrays.append(array)
            for j, r in enumerate(meta):
                if r["id"] in self.index:
                    raise ValueError("Duplicate sample ID in cache")
                self.index[r["id"]] = (len(self.arrays) - 1, j, r["text_sha256"])
        for r in rows:
            if r["id"] not in self.index or self.index[r["id"]][2] != r["text_sha256"]:
                raise ValueError(f"Sample ID/text mismatch: {r['id']}")

    def batch(self, rows, device):
        vectors = [self.arrays[self.index[r["id"]][0]][self.index[r["id"]][1]] for r in rows]
        return torch.tensor(np.stack(vectors), dtype=torch.float32, device=device)


def cache_hidden(cfg, resume=False):
    rows, corpus = records(cfg)
    root = run_dir(cfg)
    bridge_path = root / "bridge.pt"
    bridge = torch.load(bridge_path, map_location="cpu", weights_only=True)["P"]
    rows = sorted((r for r in rows if r["split"] == "train"), key=lambda r: identity([cfg["seed"], r["id"]]))[:cfg["recovery"]["examples"]]
    contract = cache_contract(cfg, corpus, "recovery")
    layers = teacher_layers(cfg)
    contract.update(bridge_sha256=checksum(bridge_path), recovery=cfg["recovery"],
                    states="raw post-block residual, before next norm; final block before model.norm", layers=layers)
    path = root / "recovery_cache"
    m = open_cache(path, contract, resume)
    if m["complete"]:
        return
    teacher = load_teacher(cfg)
    device = teacher.get_input_embeddings().weight.device
    start = sum(s["count"] for s in m["shards"])
    slots = cfg["recovery"]["positions"]
    # Small shards and hooks avoid retaining every teacher hidden layer.
    for offset in range(start, len(rows), 32):
        subset = rows[offset:offset + 32]
        hidden, positions, validity = [], [], []
        for r in subset:
            ids, mask = collate([r], corpus["pad_id"], device)
            pos = torch.linspace(0, len(r["input_ids"]) - 1, min(slots, len(r["input_ids"]))).round().long().unique()
            valid = torch.arange(slots) < len(pos)
            pos = torch.cat([pos, pos[-1:].expand(slots - len(pos))])
            captured = {}
            def hook_for(j):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[j] = (h[0, pos.to(h.device)].float() @ bridge.to(h.device)).cpu()
                return hook
            hooks = [teacher.layers[j - 1].register_forward_hook(hook_for(j)) for j in layers]
            try:
                with torch.inference_mode():
                    teacher(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                for hook in hooks:
                    hook.remove()
            hidden.append(torch.stack([captured[j] for j in layers]).numpy().astype(np.float16))
            positions.append(pos.numpy())
            validity.append(valid.numpy())
        write_shard(path, m, offset, subset, {"hidden": np.stack(hidden), "positions": np.stack(positions), "valid": np.stack(validity)})
        print(f"Recovery cache {offset + len(subset)}/{len(rows)}", flush=True)
    m["complete"] = True
    atomic_json(path / "manifest.json", m)
