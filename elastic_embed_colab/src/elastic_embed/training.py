import contextlib
import json
import math
import resource
import time
import numpy as np
import torch
from .config import identity, mohawk_enabled, run_dir, teacher_layers
from .io import atomic_torch, checksum, collate, records, restore_rng, rng_state, seed_all
from .losses import distillation, recovery_loss
from .model import Student
from .teacher import TargetCache, cache_contract


def precision_context(cfg):
    device = cfg["training"]["device"]
    precision = cfg["training"]["precision"]
    if precision == "bfloat16":
        if not device.startswith("cuda") or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 training requires a supported CUDA GPU; choose float32 explicitly")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision != "float32":
        raise ValueError("Use bfloat16 or float32; FP16/scaler path has not been validated")
    return contextlib.nullcontext()


def load_student(cfg, path, backend=None):
    state = torch.load(path, map_location="cpu", weights_only=False)  # Only load your own trusted checkpoints.
    model_cfg = dict(state["model_config"])
    if backend:
        model_cfg["backend"] = backend
    model = Student(model_cfg)
    model.load_state_dict(state["model"], strict=True)
    return model, state


def latest_checkpoint(cfg):
    root = run_dir(cfg)
    for name in ("matryoshka", "prefix", "full-depth", "recovery"):
        p = root / f"{name}.pt"
        if p.exists():
            return p
    return root / "initialized.pt"


class RecoveryCache:
    def __init__(self, cfg, corpus_id):
        root = run_dir(cfg)
        path = root / "recovery_cache"
        m = json.loads((path / "manifest.json").read_text())
        if not m["complete"] or m["contract"]["corpus_id"] != corpus_id or m["contract"]["bridge_sha256"] != checksum(root / "bridge.pt"):
            raise ValueError("Incomplete or incompatible recovery cache")
        if m["contract"].get("layers") != teacher_layers(cfg):
            raise ValueError("Recovery cache teacher layer mapping differs from initialization")
        self.id, self.index, self.arrays = m["id"], {}, []
        for shard in m["shards"]:
            for name, sha in shard["checksums"].items():
                if checksum(path / name) != sha:
                    raise ValueError(f"Corrupt recovery shard {name}")
            arrays = {k: np.load(path / f"{shard['start']:09d}.{k}.npy", mmap_mode="r") for k in ("hidden", "positions", "valid")}
            self.arrays.append(arrays)
            for j, row in enumerate(json.loads((path / shard["metadata"]).read_text())):
                self.index[row["id"]] = len(self.arrays) - 1, j, row["text_sha256"]

    def batch(self, rows, device):
        for r in rows:
            if self.index[r["id"]][2] != r["text_sha256"]:
                raise ValueError("Recovery sample text hash mismatch")
        return [torch.tensor(np.stack([self.arrays[self.index[r["id"]][0]][k][self.index[r["id"]][1]] for r in rows]), device=device) for k in ("hidden", "positions", "valid")]


class BatchStream:
    """Deterministic shuffled stream; sampler state is serializable mid-epoch."""
    def __init__(self, rows, seed, cache=None, neighbor_fraction=0, neighbor_pool=4096):
        self.rows, self.seed, self.cache = rows, seed, cache
        self.epoch, self.position = 0, 0
        self.neighbor_fraction, self.neighbor_pool = neighbor_fraction, neighbor_pool
        if not rows:
            raise ValueError("No training samples")
        self._shuffle()

    def _shuffle(self):
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.rows))

    def next(self, count):
        indices = []
        for _ in range(count):
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self._shuffle()
            indices.append(int(self.order[self.position]))
            self.position += 1
        rows = [self.rows[i] for i in indices]
        n_neighbors = min(count - 1, int(count * self.neighbor_fraction))
        if n_neighbors and self.cache:
            rng = np.random.default_rng(self.seed + self.epoch * len(self.rows) + self.position)
            candidate_ids = rng.choice(len(self.rows), min(self.neighbor_pool, len(self.rows)), replace=False)
            candidates = [self.rows[int(i)] for i in candidate_ids if self.rows[int(i)]["id"] not in {r["id"] for r in rows}]
            if candidates:
                targets = torch.nn.functional.normalize(self.cache.batch(candidates, "cpu"), dim=-1)
                anchor = torch.nn.functional.normalize(self.cache.batch(rows[:1], "cpu"), dim=-1)
                best = (targets @ anchor.T).flatten().argsort(descending=True)[:n_neighbors].tolist()
                rows[-len(best):] = [candidates[i] for i in best]
        return rows

    def state(self):
        return {"epoch": self.epoch, "position": self.position}

    def restore(self, state):
        self.epoch, self.position = state["epoch"], state["position"]
        self._shuffle()


def train(cfg, stage, resume=False):
    if stage in ("mohawk-matrix", "mohawk-block"):
        from .mohawk import train_alignment
        return train_alignment(cfg, stage, resume)
    root, tc = run_dir(cfg), cfg["training"]
    all_rows, corpus = records(cfg)
    cache = TargetCache(root / "teacher_cache", all_rows, corpus["id"])
    if cache.manifest["id"] != identity(cache_contract(cfg, corpus, "sentence")):
        raise ValueError("Teacher cache configuration differs from this run")
    hidden_cache = RecoveryCache(cfg, corpus["id"]) if stage == "recovery" else None
    rows = [r for r in all_rows if r["split"] == "train" and (hidden_cache is None or r["id"] in hidden_cache.index)]
    steps = cfg["recovery"]["steps"] if stage == "recovery" else tc[stage.replace("-", "_") + "_steps"]
    output = root / f"{stage}.pt"
    predecessors = {"recovery": ["initialized"], "full-depth": ["recovery", "initialized"],
                    "prefix": ["full-depth"], "matryoshka": ["prefix"]}
    if mohawk_enabled(cfg):
        predecessors["recovery"] = ["mohawk-block"]
        predecessors["full-depth"] = ["recovery"]
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; pass --resume or use another run_dir")
    if resume and output.exists():
        source = output
    else:
        choices = [root / f"{name}.pt" for name in predecessors[stage]]
        source = next((p for p in choices if p.exists()), None)
        if source is None:
            raise FileNotFoundError(f"Run predecessor stage first: {predecessors[stage]}")
    seed_all(cfg["seed"])
    model, initial = load_student(cfg, source, cfg["model"]["backend"])
    if initial.get("stage") and initial["step"] < initial["total_steps"] and source != output:
        raise ValueError("Predecessor stage is incomplete; resume it first")
    if mohawk_enabled(cfg):
        if initial.get("teacher_layers") != teacher_layers(cfg) or initial.get("bridge_sha256") != checksum(root / "bridge.pt"):
            raise ValueError("Predecessor checkpoint uses a different bridge/layer mapping")
        if not initial.get("mohawk_cache_id"):
            raise ValueError("MOHAWK enabled but predecessor checkpoint has no alignment lineage")
        if initial.get("config", {}).get("mohawk") != cfg["mohawk"]:
            raise ValueError("MOHAWK settings changed since predecessor checkpoint")
    device = tc["device"]
    model.to(device).train()
    # Invoke validation now, before allocating the optimizer.
    with precision_context(cfg):
        pass
    transferred = initial.get("transferred", [])
    groups = {}
    for name, p in model.named_parameters():
        decay = p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)
        lr = tc["transferred_lr"] if name in transferred else tc["new_lr"]
        key = lr, decay
        groups.setdefault(key, []).append(p)
    optimizer = torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": tc["weight_decay"] if decay else 0} for (lr, decay), ps in groups.items()])
    warmup = max(1, int(steps * tc["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, 1.) * max(0., (steps - s) / max(1, steps - warmup)))
    sampler = BatchStream(rows, cfg["seed"], cache, tc["neighbor_fraction"] if stage == "matryoshka" else 0, tc["neighbor_pool"])
    contract = identity({"cfg": cfg, "stage": stage, "cache": cache.manifest["id"], "hidden": hidden_cache.id if hidden_cache else None})
    step, examples, tokens = 0, 0, 0
    if source == output:
        if initial["contract"] != contract:
            raise ValueError("Checkpoint/config/cache identity changed; resume requires the same configuration")
        optimizer.load_state_dict(initial["optimizer"])
        scheduler.load_state_dict(initial["scheduler"])
        sampler.restore(initial["sampler"])
        restore_rng(initial["rng"])
        step, examples, tokens = initial["step"], initial["examples"], initial["tokens"]
    micro = tc["microbatch"]
    if micro < 1 or tc["effective_batch"] < micro:
        raise ValueError("Require 1 <= microbatch <= effective_batch")
    clock = time.perf_counter()
    def save():
        atomic_torch(output, {"model": model.state_dict(), "model_config": model.cfg, "transferred": transferred,
                             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": None,
                             "rng": rng_state(), "sampler": sampler.state(), "contract": contract,
                             "stage": stage, "step": step, "total_steps": steps, "examples": examples, "tokens": tokens,
                             "config": cfg, "corpus_manifest": corpus, "teacher_cache_id": cache.manifest["id"],
                             "teacher_layers": initial.get("teacher_layers"), "bridge_sha256": initial.get("bridge_sha256"),
                             "mohawk_cache_id": initial.get("mohawk_cache_id"),
                             "loss_schedule": {"shallow_ramp": min(1., step / max(1, steps * .1))}})
    while step < steps:
        if stage == "recovery":
            for name, p in model.named_parameters():
                p.requires_grad_(name not in transferred or step >= cfg["recovery"]["freeze_steps"])
        optimizer.zero_grad(set_to_none=True)
        consumed, total_loss, metrics = 0, 0., {}
        while consumed < tc["effective_batch"]:
            count = min(micro, tc["effective_batch"] - consumed)
            batch = sampler.next(count)
            ids, mask = collate(batch, corpus["pad_id"], device)
            targets = cache.batch(batch, device)
            if stage == "recovery":
                depths, dims = list(range(1, 13)), [384]
                # Internal recovery needs all 12 blocks, including non-deployment exits.
                saved_depths = model.cfg["depths"]
                model.cfg["depths"] = depths
                try:
                    with precision_context(cfg):
                        outputs, states = model(ids, mask, depths, return_hidden=True)
                finally:
                    model.cfg["depths"] = saved_depths
                ht, pos, valid = hidden_cache.batch(batch, device)
                internal = recovery_loss(states, ht, pos.long(), valid.bool())
                sentence, metrics = distillation({12: outputs[12]}, targets, batch, dims, cfg)
                loss = (1 - step / max(1, steps)) * internal + cfg["recovery"]["anchor_weight"] * sentence
                metrics["recovery"] = float(internal.detach())
            else:
                depths = cfg["model"]["depths"] if stage == "matryoshka" else [12]
                dims = [384] if stage == "full-depth" else cfg["model"]["dims"]
                with precision_context(cfg):
                    outputs = model(ids, mask, depths)
                loss, metrics = distillation(outputs, targets, batch, dims, cfg, min(1., (step + 1) / max(1, steps * .1)))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at step {step}; last atomic checkpoint remains usable")
            (loss * count / tc["effective_batch"]).backward()
            total_loss += float(loss.detach()) * count / tc["effective_batch"]
            examples += count
            tokens += int(mask.sum())
            consumed += count
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % tc["log_every"] == 0 or step == 1:
            log = {"stage": stage, "step": step, "loss": total_loss, "grad_norm": float(grad_norm),
                   "examples": examples, "nonpadding_tokens": tokens, "seconds_this_session": time.perf_counter() - clock,
                   "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                   "last_microbatch_metrics": metrics}
            with (root / "training.jsonl").open("a") as f:
                f.write(json.dumps(log) + "\n")
            print(f"{stage} {step}/{steps} loss={total_loss:.5f} examples={examples}", flush=True)
        if step % tc["checkpoint_every"] == 0 or step == steps:
            save()
