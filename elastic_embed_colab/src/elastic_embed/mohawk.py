"""MOHAWK adaptation for a narrower, shallower KDA embedding student.

1. Align Frobenius-normalized, head-mean token mixers on short prefixes.
2. Independently fit attention/whole-block outputs on projected teacher inputs.
3. Hand off to whole-student recovery and native-prefix embedding distillation.

This is not the paper's same-width, head-by-head Phi-to-Mamba conversion.
The frozen P is an internal bridge, never a rotation of embedding targets.
"""
from collections import Counter, defaultdict
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from .config import identity, mohawk_enabled, run_dir, teacher_layers
from .io import atomic_json, atomic_torch, checksum, collate, records, restore_rng, rng_state, seed_all
from .model import scan
from .teacher import load_teacher, open_cache, write_shard

CACHE_VERSION = "qwen-kda-mohawk-v1"
STATE_FIELDS = ("inputs", "attention", "outputs", "norm1", "norm2")


def select_calibration(rows, split, count, seed):
    """Deterministic language round-robin, with disjoint existing corpus splits."""
    buckets = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            buckets[row["language"]].append(row)
    for values in buckets.values():
        values.sort(key=lambda r: identity([seed, r["id"]]))
    selected = []
    for i in range(max(map(len, buckets.values()), default=0)):
        for language in sorted(buckets):
            if i < len(buckets[language]):
                selected.append(buckets[language][i])
                if len(selected) == count:
                    return selected
    return selected


def alignment_contract(cfg, corpus):
    mh = cfg["mohawk"]
    return {"version": CACHE_VERSION, "corpus_id": corpus["id"], "teacher": cfg["teacher"],
            "tokenizer_revision": corpus["tokenizer_revision"], "bridge_sha256": checksum(run_dir(cfg) / "bridge.pt"),
            "layers": teacher_layers(cfg), "seed": cfg["seed"],
            "cache_settings": {k: mh[k] for k in ("examples", "validation_examples", "max_length", "matrix_length", "shard_size")},
            "input_view": "contiguous input_ids prefix from position zero, no state carry between documents",
            "states": "raw pre-block residual and post-attention-o_proj/post-block/norm outputs, each @ frozen P",
            "matrix": "head-mean teacher softmax after QK norms/RoPE/causal mask; top-left matrix_length prefix",
            "attention_backend": "eager", "storage_dtype": "float16", "projection_dtype": "float32"}


def capture_alignment(teacher, bridge, layers, ids, mask, matrix_length):
    """Hooks reduce one document immediately; never retain all 4096-wide layers."""
    if ids.shape[0] != 1 or not bool(mask.all()):
        raise ValueError("Alignment capture expects one unpadded, nonempty document")
    captured, hooks, projections = {}, [], {}
    def project(h):
        key = str(h.device)
        if key not in projections:
            projections[key] = bridge.to(h.device)
        return (h[0].float() @ projections[key]).cpu()
    def input_hook(layer):
        def hook(module, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            captured[layer, "inputs"] = project(h)
        return hook
    def output_hook(layer, field):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer, field] = project(h)
            if field == "attention":
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise RuntimeError("Teacher did not expose attention matrices; require pinned Qwen eager backend")
                captured[layer, "matrix"] = output[1][0, :, :matrix_length, :matrix_length].float().mean(0).cpu()
        return hook
    try:
        for layer in layers:
            block = teacher.layers[layer - 1]
            hooks.append(block.register_forward_pre_hook(input_hook(layer), with_kwargs=True))
            for module, field in ((block.self_attn, "attention"), (block, "outputs"),
                                  (block.input_layernorm, "norm1"), (block.post_attention_layernorm, "norm2")):
                hooks.append(module.register_forward_hook(output_hook(layer, field)))
        with torch.inference_mode():
            # Eager self_attn returns weights to its hook even without collecting
            # output_attentions on the outer model (which would retain all layers).
            teacher(input_ids=ids, attention_mask=mask, use_cache=False, output_attentions=False)
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.stack([captured[layer, name] for layer in layers]) for name in (*STATE_FIELDS, "matrix")}


def cache_alignment(cfg, resume=False):
    if not mohawk_enabled(cfg):
        raise ValueError("Enable mohawk before creating its calibration cache")
    root, mh = run_dir(cfg), cfg["mohawk"]
    rows, corpus = records(cfg)
    if corpus["tokenizer_revision"] != cfg["teacher"]["revision"]:
        raise ValueError("Corpus and teacher tokenizer revisions differ")
    train_rows = select_calibration(rows, "train", mh["examples"], cfg["seed"])
    validation_rows = select_calibration(rows, "validation", mh["validation_examples"], cfg["seed"])
    if not train_rows or not validation_rows:
        raise ValueError("MOHAWK requires nonempty train and validation calibration splits")
    selected = train_rows + validation_rows
    n, width = cfg["model"]["layers"], cfg["model"]["width"]
    size = len(selected) * (len(STATE_FIELDS) * n * mh["max_length"] * width * 2 + n * mh["matrix_length"]**2 * 2 + mh["max_length"] * 8)
    print(f"MOHAWK cache upper-bound arrays: {size / 1e9:.2f} GB, {len(train_rows)} train / {len(validation_rows)} validation", flush=True)
    if size > mh["max_cache_gb"] * 1e9:
        raise ValueError("MOHAWK cache exceeds max_cache_gb; explicitly reduce examples/length or increase the storage budget")
    contract = alignment_contract(cfg, corpus)
    path = root / "mohawk_cache"
    manifest = open_cache(path, contract, resume)
    if manifest["complete"]:
        return
    P = torch.load(root / "bridge.pt", map_location="cpu", weights_only=True)["P"]
    if P.shape != (4096, width):
        raise ValueError("Unexpected Qwen internal bridge shape")
    teacher = load_teacher(cfg, attn_implementation="eager")
    device = teacher.get_input_embeddings().weight.device
    start = sum(s["count"] for s in manifest["shards"])
    clock = time.perf_counter()
    for offset in range(start, len(selected), mh["shard_size"]):
        subset = selected[offset:offset + mh["shard_size"]]
        shape = (len(subset), n, mh["max_length"], width)
        arrays = {field: np.zeros(shape, dtype=np.float16) for field in STATE_FIELDS}
        arrays["matrix"] = np.zeros((len(subset), n, mh["matrix_length"], mh["matrix_length"]), np.float16)
        arrays["input_ids"] = np.full((len(subset), mh["max_length"]), corpus["pad_id"], np.int64)
        arrays["lengths"] = np.zeros(len(subset), np.int64)
        for i, row in enumerate(subset):
            view = dict(row, input_ids=row["input_ids"][:mh["max_length"]])
            ids, mask = collate([view], corpus["pad_id"], device)
            length = ids.shape[1]
            result = capture_alignment(teacher, P, teacher_layers(cfg), ids, mask, mh["matrix_length"])
            for field in STATE_FIELDS:
                value = result[field].numpy().astype(np.float16)
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite FP16 {field} target for {row['id']}")
                arrays[field][i, :, :length] = value
            matrix_n = min(length, mh["matrix_length"])
            arrays["matrix"][i, :, :matrix_n, :matrix_n] = result["matrix"].numpy().astype(np.float16)
            arrays["input_ids"][i, :length] = view["input_ids"]
            arrays["lengths"][i] = length
        write_shard(path, manifest, offset, subset, arrays)
        print(f"MOHAWK cache {offset + len(subset)}/{len(selected)}; {time.perf_counter() - clock:.1f}s", flush=True)
    manifest.update(complete=True, coverage={split: dict(Counter(r["language"] for r in group))
                                           for split, group in (("train", train_rows), ("validation", validation_rows))},
                    array_bytes_upper_bound=size)
    atomic_json(path / "manifest.json", manifest)


class AlignmentCache:
    def __init__(self, cfg, rows, corpus):
        path = run_dir(cfg) / "mohawk_cache"
        manifest = json.loads((path / "manifest.json").read_text())
        if not manifest["complete"] or manifest["id"] != identity(alignment_contract(cfg, corpus)):
            raise ValueError("Incomplete or incompatible MOHAWK cache; recreate with this corpus/bridge/settings")
        self.id, self.arrays, self.index = manifest["id"], [], {}
        lookup = {r["id"]: r for r in rows}
        for shard in manifest["shards"]:
            for file, sha in shard["checksums"].items():
                if checksum(path / file) != sha:
                    raise ValueError(f"Corrupt MOHAWK shard: {file}")
            arrays = {k: np.load(path / f"{shard['start']:09d}.{k}.npy", mmap_mode="r")
                      for k in (*STATE_FIELDS, "matrix", "input_ids", "lengths")}
            self.arrays.append(arrays)
            for i, meta in enumerate(json.loads((path / shard["metadata"]).read_text())):
                row = lookup.get(meta["id"])
                if row is None or row["text_sha256"] != meta["text_sha256"] or row["id"] in self.index:
                    raise ValueError("MOHAWK sample ID/text mismatch or duplicate")
                view = row["input_ids"][:cfg["mohawk"]["max_length"]]
                if int(arrays["lengths"][i]) != len(view) or not np.array_equal(arrays["input_ids"][i, :len(view)], view):
                    raise ValueError("MOHAWK token view mismatch")
                if row["split"] not in ("train", "validation"):
                    raise ValueError("Test data cannot be used for MOHAWK calibration")
                self.index[row["id"]] = len(self.arrays) - 1, i
        expected = [r["id"] for split, count in (("train", cfg["mohawk"]["examples"]), ("validation", cfg["mohawk"]["validation_examples"]))
                    for r in select_calibration(rows, split, count, cfg["seed"])]
        if set(expected) != set(self.index):
            raise ValueError("MOHAWK cache sample coverage differs from selection contract")

    def batch(self, rows, layer, device, stage):
        entries = [(self.arrays[self.index[r["id"]][0]], self.index[r["id"]][1]) for r in rows]
        lengths = [int(a["lengths"][i]) for a, i in entries]
        maximum = max(lengths)
        if stage == "mohawk-matrix":
            maximum = min(maximum, entries[0][0]["matrix"].shape[-1])
        fields = ("inputs", "matrix") if stage == "mohawk-matrix" else STATE_FIELDS
        result = {}
        for field in fields:
            values = [a[field][i, layer, :maximum, :maximum] if field == "matrix" else a[field][i, layer, :maximum] for a, i in entries]
            result[field] = torch.tensor(np.stack(values), device=device, dtype=torch.float32)
        result["mask"] = torch.arange(maximum, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        return result


def mixer_coefficients(mixer, x, mask):
    """Exact latent KDA v→o coefficients for fixed q/k/gates, before output norm.

    Identity probes materialize the causal mixing matrix with autograd. The
    value projection/convolution and output gate/norm are fitted in stage 2.
    Works on either backend's parameters; orientation itself uses FP32 Torch.
    """
    heads = mixer.A_log.numel()
    head_dim = mixer.dt_bias.numel() // heads
    qk = []
    for name in ("q", "k"):
        z = getattr(mixer, name + "_proj")(x) * mask.unsqueeze(-1)
        conv = getattr(mixer, name + "_conv1d")
        z = F.conv1d(F.pad(z.transpose(1, 2), (conv.weight.shape[-1] - 1, 0)),
                     conv.weight, conv.bias, groups=z.shape[-1])
        z = F.silu(z.transpose(1, 2)).reshape(x.shape[0], x.shape[1], heads, head_dim).float()
        qk.append(z * torch.rsqrt(z.square().sum(-1, keepdim=True) + 1e-6))
    g = (mixer.f_proj(x).float() + mixer.dt_bias.float()).reshape_as(qk[0])
    decay = (-mixer.A_log.float().exp().view(1, 1, heads, 1) * F.softplus(g)).exp()
    beta = mixer.b_proj(x).float().sigmoid()
    probes = torch.eye(x.shape[1], device=x.device).view(1, x.shape[1], 1, x.shape[1]).expand(x.shape[0], -1, heads, -1)
    return scan(qk[0], qk[1], probes, decay, beta, mask).permute(0, 2, 1, 3)


def matrix_loss(actual, target, mask):
    # Head means avoid inventing a 32→6 head correspondence. Normalize global
    # Frobenius scale because q/k L2 normalization constrains KDA coefficients.
    valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    valid = valid & torch.ones_like(valid).tril()
    a = (actual.float().mean(1) * valid).flatten(1)
    t = (target.float() * valid).flatten(1)
    return (F.normalize(a, dim=-1) - F.normalize(t, dim=-1)).square().sum(-1).mean()


def relative_mse(actual, target, mask):
    valid = mask.unsqueeze(-1).float()
    error = ((actual.float() - target.float()).square() * valid).sum((1, 2))
    energy = (target.float().square() * valid).sum((1, 2)).clamp_min(1e-6)
    return (error / energy).mean()


def layer_objective(block, batch, stage, cfg):
    x, mask = batch["inputs"], batch["mask"]
    normalized = block.norm1(x)
    if stage == "mohawk-matrix":
        loss = matrix_loss(mixer_coefficients(block.mixer, normalized, mask), batch["matrix"], mask)
        return loss, {"matrix": float(loss.detach())}
    attention = block.mixer(normalized, attention_mask=mask.long())[0]
    residual = x + attention
    hidden = block.norm2(residual)
    output = residual + block.down(F.silu(block.gate(hidden)) * block.up(hidden))
    attn_loss = relative_mse(attention, batch["attention"], mask)
    block_loss = relative_mse(output, batch["outputs"], mask)
    # Learn norm gains functionally in the projected space, on teacher inputs.
    norm_loss = (relative_mse(normalized, batch["norm1"], mask) +
                 relative_mse(block.norm2(x + batch["attention"]), batch["norm2"], mask)) / 2
    mh = cfg["mohawk"]
    loss = mh["attention_weight"] * attn_loss + mh["block_weight"] * block_loss + mh["norm_weight"] * norm_loss
    return loss, {"attention": float(attn_loss.detach()), "block": float(block_loss.detach()), "norm": float(norm_loss.detach())}


def set_trainable(model, stage, step, cfg):
    orientation_parts = ("q_proj.", "k_proj.", "q_conv1d.", "k_conv1d.", "f_proj.", "b_proj.", "A_log", "dt_bias")
    for name, p in model.named_parameters():
        enabled = False
        if name.startswith("blocks."):
            local = name.split(".", 2)[2]
            if stage == "mohawk-matrix":
                enabled = local.startswith(tuple("mixer." + part for part in orientation_parts))
            else:
                enabled = local.startswith(("mixer.", "norm1.", "norm2.")) or step >= cfg["mohawk"]["block_freeze_ffn_steps"]
        p.requires_grad_(enabled)


def alignment_validation(model, cache, rows, stage, cfg):
    from .training import precision_context
    was_training = model.training
    model.eval()
    per_language, per_layer = defaultdict(list), defaultdict(list)
    try:
        with torch.no_grad():
            for row in rows:
                total = 0.
                for j, block in enumerate(model.blocks):
                    batch = cache.batch([row], j, cfg["training"]["device"], stage)
                    if stage == "mohawk-matrix":
                        loss, _ = layer_objective(block, batch, stage, cfg)
                    else:
                        with precision_context(cfg):
                            loss, _ = layer_objective(block, batch, stage, cfg)
                    per_layer[str(j + 1)].append(float(loss))
                    total += float(loss) / len(model.blocks)
                per_language[row["language"]].append(total)
    finally:
        model.train(was_training)
    means = {k: float(np.mean(v)) for k, v in per_language.items()}
    return {"examples": len(rows), "coverage": dict(Counter(r["language"] for r in rows)),
            "per_language_loss": means, "macro_language_loss": float(np.mean(list(means.values()))),
            "per_layer_loss": {k: float(np.mean(v)) for k, v in per_layer.items()}}


def train_alignment(cfg, stage, resume=False):
    from .training import BatchStream, load_student, precision_context
    if stage not in ("mohawk-matrix", "mohawk-block") or not mohawk_enabled(cfg):
        raise ValueError("MOHAWK must be enabled with a matrix or block stage")
    root, mh, tc = run_dir(cfg), cfg["mohawk"], cfg["training"]
    rows, corpus = records(cfg)
    cache = AlignmentCache(cfg, rows, corpus)
    training_rows = [r for r in rows if r["split"] == "train" and r["id"] in cache.index]
    validation_rows = [r for r in rows if r["split"] == "validation" and r["id"] in cache.index]
    if not training_rows or not validation_rows:
        raise ValueError("MOHAWK needs both train and held-out validation records")
    output = root / f"{stage}.pt"
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; pass --resume")
    source = output if resume and output.exists() else root / ("initialized.pt" if stage == "mohawk-matrix" else "mohawk-matrix.pt")
    seed_all(cfg["seed"])
    model, initial = load_student(cfg, source, cfg["model"]["backend"])
    if initial.get("stage") and source != output and initial["step"] < initial["total_steps"]:
        raise ValueError("Complete MOHAWK matrix alignment before block alignment")
    if initial.get("bridge_sha256") != checksum(root / "bridge.pt") or initial.get("teacher_layers") != teacher_layers(cfg):
        raise ValueError("Initialization bridge/layer mapping missing or changed; create a fresh initialized checkpoint")
    if {k:v for k,v in model.cfg.items() if k != "backend"} != {k:v for k,v in cfg["model"].items() if k != "backend"}:
        raise ValueError("Student architecture differs from initialized checkpoint")
    model.to(tc["device"]).train()
    with precision_context(cfg):
        pass
    transferred = initial.get("transferred", [])
    groups = {}
    # Keep potentially unfrozen FFNs in optimizer groups from the start so resume
    # has the same group layout across the freeze boundary. No grads => no decay.
    for name, p in model.named_parameters():
        if not name.startswith("blocks."):
            continue
        lr = tc["transferred_lr"] if name in transferred else tc["new_lr"]
        decay = tc["weight_decay"] if p.ndim > 1 and not getattr(p, "_no_weight_decay", False) else 0
        groups.setdefault((lr, decay), []).append(p)
    optimizer = torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=decay) for (lr, decay), ps in groups.items()])
    steps = mh["matrix_steps" if stage == "mohawk-matrix" else "block_steps"]
    warmup = max(1, int(steps * tc["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, 1.) * max(0., (steps - s) / max(1, steps - warmup)))
    sampler = BatchStream(training_rows, cfg["seed"])
    contract = identity({"cfg": cfg, "stage": stage, "cache": cache.id})
    step, examples, tokens = 0, 0, 0
    history = []
    if source == output:
        if initial["contract"] != contract:
            raise ValueError("MOHAWK resume configuration/cache changed")
        optimizer.load_state_dict(initial["optimizer"])
        scheduler.load_state_dict(initial["scheduler"])
        sampler.restore(initial["sampler"])
        restore_rng(initial["rng"])
        step, examples, tokens = initial["step"], initial["examples"], initial["tokens"]
        history = initial["validation_history"]
    micro = tc["microbatch"]
    if not 1 <= micro <= mh["effective_batch"]:
        raise ValueError("Require microbatch <= mohawk.effective_batch")
    if not history:
        history.append({"step": 0, **alignment_validation(model, cache, validation_rows, stage, cfg)})
    clock = time.perf_counter()
    while step < steps:
        set_trainable(model, stage, step, cfg)
        optimizer.zero_grad(set_to_none=True)
        consumed, total_loss, metrics = 0, 0., {}
        while consumed < mh["effective_batch"]:
            count = min(micro, mh["effective_batch"] - consumed)
            batch_rows = sampler.next(count)
            # Independent blocks consume teacher inputs. Sequential backward
            # releases each block's activations before moving to the next one.
            for j, block in enumerate(model.blocks):
                batch = cache.batch(batch_rows, j, tc["device"], stage)
                if stage == "mohawk-matrix":
                    loss, details = layer_objective(block, batch, stage, cfg)
                else:
                    with precision_context(cfg):
                        loss, details = layer_objective(block, batch, stage, cfg)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite {stage} loss at step {step}, block {j + 1}")
                weight = count / (mh["effective_batch"] * len(model.blocks))
                (loss * weight).backward()
                total_loss += float(loss.detach()) * weight
                metrics[str(j + 1)] = details
            examples += count
            tokens += int(batch["mask"].sum())
            consumed += count
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % tc["log_every"] == 0 or step == 1:
            with (root / "training.jsonl").open("a") as f:
                f.write(json.dumps({"stage": stage, "step": step, "loss": total_loss, "grad_norm": float(norm),
                                    "examples": examples, "nonpadding_tokens": tokens,
                                    "seconds_this_session": time.perf_counter() - clock, "last_microbatch_per_layer": metrics}) + "\n")
            print(f"{stage} {step}/{steps} loss={total_loss:.5f}", flush=True)
        if step % tc["checkpoint_every"] == 0 or step == steps:
            history.append({"step": step, **alignment_validation(model, cache, validation_rows, stage, cfg)})
            state = {"model": model.state_dict(), "model_config": model.cfg, "transferred": transferred,
                     "teacher_layers": teacher_layers(cfg), "bridge_sha256": checksum(root / "bridge.pt"),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": None,
                     "rng": rng_state(), "sampler": sampler.state(), "contract": contract,
                     "stage": stage, "step": step, "total_steps": steps, "examples": examples, "tokens": tokens,
                     "config": cfg, "corpus_manifest": corpus, "mohawk_cache_id": cache.id,
                     "validation_history": history, "loss_schedule": {"ffn_frozen": stage == "mohawk-matrix" or step < mh["block_freeze_ffn_steps"]}}
            atomic_torch(output, state)
            atomic_json(root / f"{stage}_report.json", {"stage": stage, "cache_id": cache.id, "validation": history,
                        "limitations": ["Reduced-width/depth MOHAWK adaptation", "Head-mean, scale-normalized matrix orientation",
                                        "Alignment metrics do not establish downstream generalization"]})
