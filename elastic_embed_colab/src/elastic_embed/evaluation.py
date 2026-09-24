from collections import Counter
import json
import numpy as np
import torch
from .config import run_dir
from .io import atomic_json, collate, records
from .model import prefix
from .teacher import TargetCache
from .training import latest_checkpoint, load_student, precision_context


def metrics(s, t):
    n = len(s)
    errors = 1 - (s * t).sum(-1)
    result = {"n": n, "mean_cosine": float(1 - errors.mean()),
              "cosine_error_p50": float(errors.quantile(.5)), "cosine_error_p95": float(errors.quantile(.95))}
    if n > 1:
        gs, gt = s @ s.T, t @ t.T
        mask = ~torch.eye(n, dtype=torch.bool)
        result["similarity_mse"] = float((gs[mask] - gt[mask]).square().mean())
        gs.fill_diagonal_(-float("inf"))
        gt.fill_diagonal_(-float("inf"))
        for k in (10, 50):
            actual_k = min(k, n - 1)
            a, b = gs.topk(actual_k).indices, gt.topk(actual_k).indices
            result[f"neighbor_recall@{k}"] = float((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean())
            result[f"neighbor_actual_k@{k}"] = actual_k
    return result


def evaluate(cfg):
    root = run_dir(cfg)
    rows, corpus = records(cfg)
    rows = [r for r in rows if r["split"] == cfg["evaluation"]["split"]]
    # Round-robin language coverage before applying the evaluation cap.
    buckets = {}
    for row in rows:
        buckets.setdefault(row["language"], []).append(row)
    rows = []
    for i in range(max(map(len, buckets.values()), default=0)):
        rows.extend(b[i] for b in buckets.values() if i < len(b))
        if len(rows) >= cfg["evaluation"]["max_examples"]:
            break
    rows = rows[:cfg["evaluation"]["max_examples"]]
    if not rows:
        raise ValueError("No held-out evaluation records")
    cache = TargetCache(root / "teacher_cache", rows, corpus["id"])
    model, _ = load_student(cfg, latest_checkpoint(cfg), cfg["model"]["backend"])
    model.to(cfg["training"]["device"]).eval()
    vectors = {d: [] for d in cfg["model"]["depths"]}
    batch = cfg["evaluation"]["batch_size"]
    for start in range(0, len(rows), batch):
        ids, mask = collate(rows[start:start + batch], corpus["pad_id"], cfg["training"]["device"])
        with torch.inference_mode(), precision_context(cfg):
            outputs = model(ids, mask)
        for d, z in outputs.items():
            vectors[d].append(z.float().cpu())
    vectors = {d: torch.cat(v) for d, v in vectors.items()}
    targets = cache.batch(rows, "cpu")
    result = {"split": cfg["evaluation"]["split"], "coverage": dict(Counter(r["language"] for r in rows)),
              "grid": {}, "per_language": {}, "teacher_truncation_audit": {},
              "limitations": ["No external STS/retrieval benchmark adapter yet", "No release thresholds set",
                              "Exact dedup only; no near-duplicate exclusion", "Neighbor recall uses this held-out candidate pool",
                              "No Mondrian routing-boundary audit yet"]}
    for l, z in vectors.items():
        for d in cfg["model"]["dims"]:
            key = f"{l}/{d}"
            s, t = prefix(z, d), prefix(targets, d)
            result["grid"][key] = metrics(s, t)
            result["grid"][key]["full_to_exit_cosine_gap"] = float((1 - (s * prefix(vectors[12], d)).sum(-1)).mean())
            result["per_language"][key] = {}
            for lang in result["coverage"]:
                idx = torch.tensor([i for i, r in enumerate(rows) if r["language"] == lang])
                result["per_language"][key][lang] = metrics(s[idx], t[idx])
            result["grid"][key]["macro_language_cosine"] = float(np.mean([m["mean_cosine"] for m in result["per_language"][key].values()]))
            # Parallel-pair matching across different languages when metadata exists.
            similarities = s @ s.T
            hits = []
            for i, r in enumerate(rows):
                if not r.get("parallel_group_id"):
                    continue
                candidates = [j for j, other in enumerate(rows) if other["language"] != r["language"]]
                if not any(rows[j].get("parallel_group_id") == r["parallel_group_id"] for j in candidates):
                    continue
                best = candidates[int(similarities[i, candidates].argmax())]
                hits.append(rows[best].get("parallel_group_id") == r["parallel_group_id"])
            result["grid"][key]["cross_language_parallel_top1"] = float(np.mean(hits)) if hits else None
            result["grid"][key]["parallel_queries"] = len(hits)
    for path in sorted((root / "teacher_cache").glob("*.audit4096.npy")):
        full = torch.from_numpy(np.load(path)).float()
        cache384 = torch.from_numpy(np.load(path.with_name(path.name.replace("audit4096", "targets")))).float()
        for d in cfg["model"]["dims"]:
            p, q = prefix(full, d), prefix(cache384, d)
            result["teacher_truncation_audit"].setdefault(str(d), []).append({
                "shard": path.name, "n": len(full), "full_vs_prefix_similarity_mse": float((full @ full.T - p @ p.T).square().mean()),
                "fp16_cache_max_abs_error": float((p - q).abs().max()), "fp16_cache_min_cosine": float((p * q).sum(-1).min())})
    atomic_json(root / "evaluation.json", result)
    print(f"Wrote {root / 'evaluation.json'} ({len(rows)} held-out texts)")
