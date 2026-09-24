"""Partial weight warm-start, not a softmax-to-KDA conversion."""
import json
import torch
from .config import run_dir, teacher_layers
from .io import atomic_json, atomic_torch, checksum, seed_all
from .model import Student


class TeacherTensors:
    def __init__(self, cfg):
        from huggingface_hub import hf_hub_download
        self.download = lambda file: hf_hub_download(cfg["teacher"]["name"], file, revision=cfg["teacher"]["revision"])
        config = json.load(open(self.download("config.json")))
        if (config["num_hidden_layers"], config["hidden_size"], config["vocab_size"]) != (36, 4096, 151665):
            raise ValueError("Unexpected teacher shape")
        self.index = json.load(open(self.download("model.safetensors.index.json")))["weight_map"]
        self.resolved_names = {}

    def tensor(self, name):
        # Qwen3-Embedding stores the bare backbone names; causal-LM wrappers
        # commonly add "model.". Resolve only this known prefix, never suffixes.
        canonical = name.removeprefix("model.")
        candidates = (name, canonical, "model." + canonical)
        resolved = next((key for key in candidates if key in self.index), None)
        if resolved is None:
            raise KeyError(f"Teacher checkpoint has no tensor {name!r}; tried "
                           f"{list(dict.fromkeys(candidates))}. "
                           f"Example available keys: {list(self.index)[:5]}")
        from safetensors import safe_open
        with safe_open(self.download(self.index[resolved]), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(resolved)
        self.resolved_names[name] = resolved
        return tensor


@torch.no_grad()
def initialize(cfg):
    root = run_dir(cfg)
    if (root / "initialized.pt").exists():
        raise FileExistsError("Initialization already exists; use separate run_dir per ablation")
    seed_all(cfg["seed"])
    model = Student(cfg["model"])
    mode = cfg["initialization"]["mode"]
    if mode not in ("random", "table", "table-ffn"):
        raise ValueError("Unknown initialization mode")
    transferred, report = [], {"mode": mode, "teacher": cfg["teacher"], "transferred": []}
    if mode != "random":
        source = TeacherTensors(cfg)
        embedding = source.tensor("embed_tokens.weight")
        n = min(len(embedding), cfg["initialization"]["svd_rows"])
        sample = embedding[torch.randperm(len(embedding))[:n]].float()
        # Uncentered randomized SVD; deterministic uniform vocabulary sample, no output-space rotation.
        _, _, P = torch.svd_lowrank(sample, q=cfg["model"]["width"], niter=4)
        del sample
        # Deterministic sign convention for a reproducible internal basis.
        signs = P[P.abs().argmax(0), torch.arange(P.shape[1])].sign()
        P *= signs
        projected = torch.empty(len(embedding), P.shape[1])
        for i in range(0, len(embedding), 2048):
            projected[i:i + 2048] = embedding[i:i + 2048].float() @ P
        del embedding
        gram = projected.T @ projected
        _, basis = torch.linalg.eigh(gram)
        B = basis[:, -cfg["model"]["input_rank"]:].flip(-1)
        A = projected @ B
        model.table.weight.copy_(A)
        model.input_proj.weight.copy_(B)
        error = 1 - float(A.square().sum() / projected.square().sum())
        atomic_torch(root / "bridge.pt", {"P": P})
        report.update(projection="uncentered randomized SVD; uniform vocabulary sample; seed locked", svd_rows=n,
                      factorization_relative_squared_error=error, projected_rms=float(projected.square().mean().sqrt()))
        transferred += ["table.weight", "input_proj.weight"]
        del A, projected
        if mode == "table-ffn":
            for i, block in enumerate(model.blocks):
                layer = teacher_layers(cfg)[i] - 1
                name = f"layers.{layer}.mlp."
                gate, up, down = [source.tensor(name + part + "_proj.weight") for part in ("gate", "up", "down")]
                # Joint row/column weight RMS saliency, with one shared channel set.
                score = gate.float().square().mean(1).sqrt() * up.float().square().mean(1).sqrt() * down.float().square().mean(0).sqrt()
                selected = torch.argsort(score, descending=True, stable=True)[:cfg["model"]["ffn_width"]]
                block.gate.weight.copy_(gate[selected].float() @ P)
                block.up.weight.copy_(up[selected].float() @ P)
                block.down.weight.copy_(P.T @ down[:, selected].float())
                report.setdefault("channel_indices", {})[str(layer)] = selected.tolist()
                transferred += [f"blocks.{i}.{part}.weight" for part in ("gate", "up", "down")]
                del gate, up, down
                print(f"Transferred FFN {i + 1}/12", flush=True)
        report["teacher_tensor_names"] = source.resolved_names
    report["transferred"] = transferred
    report["fresh"] = [n for n, _ in model.named_parameters() if n not in transferred]
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["teacher_layers"] = teacher_layers(cfg)
    report["attention_transfer"] = "KDA defaults; teacher mixer behavior is learned in MOHAWK alignment"
    atomic_json(root / "initialization_report.json", report)
    atomic_torch(root / "initialized.pt", {"model": model.state_dict(), "model_config": cfg["model"], "transferred": transferred,
                                          "teacher_layers": teacher_layers(cfg),
                                          "bridge_sha256": checksum(root / "bridge.pt") if mode != "random" else None})
