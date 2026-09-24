"""Portable standard-operator ONNX Loop fallback; performance must be measured."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .config import identity, run_dir
from .data import tokenizer
from .io import atomic_json, checksum
from .model import prefix, validate_inputs
from .training import latest_checkpoint, load_student


class ExportProfile(nn.Module):
    def __init__(self, model, depth):
        super().__init__()
        self.model, self.depth = model, depth

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask, (self.depth,))[self.depth]


def export_module(module, args, path, names, opset=17):
    torch.onnx.export(module.eval(), args, str(path), input_names=names, output_names=["readout"],
                      dynamic_axes={name: {0: "batch", 1: "sequence"} for name in names} | {"readout": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    import onnx
    onnx.checker.check_model(str(path))


def session_with_trace(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(path).with_suffix(".trace"))
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def finish_trace(session):
    path = session.end_profiling()
    events = json.loads(Path(path).read_text())
    providers = sorted({e.get("args", {}).get("provider") for e in events if e.get("args", {}).get("provider")})
    return {"trace": path, "assigned_providers": providers}


def export(cfg):
    root = run_dir(cfg)
    checkpoint = latest_checkpoint(cfg)
    model, state = load_student(cfg, checkpoint, "reference")
    model.eval()
    destination = root / "onnx"
    destination.mkdir(exist_ok=True)
    tok = tokenizer(cfg)
    tok.save_pretrained(destination / "tokenizer")
    profiles = {}
    for depth in cfg["export"]["depths"]:
        path = destination / f"depth-{depth}.onnx"
        ids = torch.ones(1, cfg["export"]["example_length"], dtype=torch.long)
        export_module(ExportProfile(model, depth), (ids, torch.ones_like(ids)), path, ["input_ids", "attention_mask"], cfg["export"]["opset"])
        session = session_with_trace(path)
        checks = []
        for batch, length in ((1, 1), (2, 7), (1, 65)):
            ids = torch.randint(0, model.cfg["vocab_size"], (batch, length))
            mask = torch.ones_like(ids)
            if batch > 1:
                mask[-1, max(1, length // 2):] = 0
            with torch.no_grad():
                expected = model(ids, mask, (depth,))[depth]
            actual = torch.from_numpy(session.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0])
            error = float((actual - expected).abs().max())
            cosines = {str(d): float((prefix(actual, d) * prefix(expected, d)).sum(-1).min()) for d in model.cfg["dims"]}
            checks.append({"batch": batch, "length": length, "max_abs_error": error, "prefix_min_cosine": cosines})
            if error > cfg["runtime"]["onnx_atol"] or min(cosines.values()) < .9999:
                raise RuntimeError(f"ONNX parity failed at depth {depth}: {checks[-1]}")
        profiles[str(depth)] = {"file": path.name, "sha256": checksum(path), "bytes": path.stat().st_size,
                                "parity": checks, **finish_trace(session)}
        print(f"Exported depth {depth}: {path.stat().st_size / 1e6:.1f} MB", flush=True)
    manifest = {"checkpoint_sha256": checksum(checkpoint), "tokenizer_revision": cfg["teacher"]["revision"],
                "preprocessing": "strip-sentence-prefix-v1", "prompt": "", "pooling": "last-valid-token", "quantization": None,
                "dims": model.cfg["dims"], "max_length": 512, "default_length": 256,
                "opset": cfg["export"]["opset"], "operator_domain": "ai.onnx", "lowering": "standard-operator Loop, FP32",
                "profiles": profiles, "duplicated_initializer_storage": True,
                "release_status": "experimental; semantic thresholds not established",
                "untested": ["ARM", "WASM", "WebGPU", "quantization"]}
    manifest["id"] = identity(manifest)
    atomic_json(destination / "manifest.json", manifest)


class Encoder:
    def __init__(self, directory, max_length=256):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory / "tokenizer", padding_side="right")
        self.sessions = {}
        self.ort = ort
        if max_length not in (256, 512):
            raise ValueError("Only the 256/512 profiles are defined")
        self.max_length = max_length

    def representation_id(self, depth=12, truncate_dim=384, normalize=True):
        return identity({"package": self.manifest["id"], "depth": depth, "dim": truncate_dim,
                         "normalize": normalize, "max_length": self.max_length})

    def encode(self, texts, depth=12, truncate_dim=384, normalize=True):
        from .data import canonical_span
        if str(depth) not in self.manifest["profiles"] or truncate_dim not in self.manifest["dims"]:
            raise ValueError("Untrained or unexported depth/dimension")
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, truncate_dim), np.float32)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Empty/all-padding text inputs are rejected")
        texts = [canonical_span(t, self.tokenizer, self.max_length)[0] for t in texts]
        batch = self.tokenizer(texts, padding=True, return_tensors="np")
        if depth not in self.sessions:
            info = self.manifest["profiles"][str(depth)]
            path = self.directory / info["file"]
            if checksum(path) != info["sha256"]:
                raise ValueError("Export checksum mismatch")
            self.sessions[depth] = self.ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ids, mask = batch["input_ids"].astype(np.int64), batch["attention_mask"].astype(np.int64)
        validate_inputs(torch.from_numpy(ids), torch.from_numpy(mask))
        z = self.sessions[depth].run(None, {"input_ids": ids, "attention_mask": mask})[0][:, :truncate_dim]
        if normalize:
            z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
        return z


def benchmark(cfg):
    from .io import records
    root = run_dir(cfg)
    rows, _ = records(cfg)
    texts = [r["text"] for r in rows if r["split"] == cfg["evaluation"]["split"]][:cfg["evaluation"]["batch_size"]]
    if not texts:
        raise ValueError("No held-out texts for benchmark")
    results = []
    for depth, dim in zip(cfg["model"]["depths"], cfg["model"]["dims"]):
        clock = time.perf_counter()
        encoder = Encoder(root / "onnx")
        encoder.encode(texts, depth, dim)
        cold = time.perf_counter() - clock
        times = []
        for _ in range(cfg["evaluation"]["benchmark_repeats"]):
            clock = time.perf_counter()
            encoder.encode(texts, depth, dim)
            times.append(time.perf_counter() - clock)
        results.append({"depth": depth, "dim": dim, "batch": len(texts), "cold_seconds": cold,
                        "p50_seconds": float(np.quantile(times, .5)), "p95_seconds": float(np.quantile(times, .95)),
                        "texts_per_second": len(texts) / float(np.mean(times)), "provider": "CPUExecutionProvider",
                        "includes_tokenization_normalization": True})
    atomic_json(root / "benchmark.json", {"results": results, "untested": ["ARM", "WASM", "WebGPU"]})
