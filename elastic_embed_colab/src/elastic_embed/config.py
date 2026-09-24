from pathlib import Path
import copy
import hashlib
import json
import re
import yaml

TEACHER_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
FLA_REVISION = "954438d1fcb5e1bb05c22f9908de9c5c2df74ae5"


def teacher_layers(cfg):
    """One-based teacher block numbers, shared by transfer and all recovery stages."""
    layers = cfg.get("mohawk", {}).get("teacher_layers", list(range(3, 37, 3)))
    if len(layers) != cfg["model"]["layers"] or sorted(set(layers)) != layers or not all(1 <= n <= 36 for n in layers):
        raise ValueError("teacher_layers must contain one increasing Qwen block number per student block")
    return layers


def mohawk_enabled(cfg):
    return cfg.get("mohawk", {}).get("enabled", False)


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "extends" in cfg:
        base = load_config(path.parent / cfg.pop("extends"))
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
            return a
        cfg = merge(copy.deepcopy(base), cfg)
    m = cfg["model"]
    if (m["width"], m["layers"], m["heads"], m["head_dim"]) != (384, 12, 6, 64):
        raise ValueError("The baseline requires width=384, layers=12, heads=6, head_dim=64")
    if m["backend"] not in ("reference", "fla"):
        raise ValueError("backend must be reference or fla")
    if cfg["teacher"]["name"] != "Qwen/Qwen3-Embedding-8B":
        raise ValueError("This runner is locked to the 8B embedding teacher")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["teacher"]["revision"]):
        raise ValueError("Use an immutable teacher commit SHA")
    if cfg["scale"] not in ("smoke", "pilot", "main"):
        raise ValueError("Choose an explicit scale")
    if cfg["data"]["max_examples"] > {"smoke": 20000, "pilot": 200000, "main": 1000000}[cfg["scale"]]:
        raise ValueError("Example budget exceeds chosen scale")
    teacher_layers(cfg)
    if mohawk_enabled(cfg):
        mh = cfg["mohawk"]
        if cfg["initialization"]["mode"] == "random":
            raise ValueError("MOHAWK needs the frozen Qwen bridge: choose table/table-ffn, or disable mohawk for random-init")
        for key in ("examples", "validation_examples", "shard_size", "max_length", "matrix_length", "matrix_steps", "block_steps", "effective_batch"):
            if mh[key] < 1:
                raise ValueError(f"mohawk.{key} must be positive")
        if not 1 <= mh["matrix_length"] <= mh["max_length"] <= 512:
            raise ValueError("Require matrix_length <= max_length <= 512")
        if mh["block_freeze_ffn_steps"] < 0 or mh["norm_weight"] < 0 or mh["attention_weight"] <= 0 or mh["block_weight"] <= 0:
            raise ValueError("Invalid MOHAWK freeze schedule/loss weights")
    return cfg


def run_dir(cfg):
    p = Path(cfg["run_dir"]).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p
