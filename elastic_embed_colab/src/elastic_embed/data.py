"""Bounded streaming ingestion and exact deduplication, with group-safe splits."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import yaml
from .config import identity, run_dir
from .io import atomic_json, checksum

PREPROCESSING = "strip-sentence-prefix-v1"


def tokenizer(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["teacher"]["name"], revision=cfg["teacher"]["revision"], padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if max(tok.get_vocab().values()) >= cfg["model"]["vocab_size"]:
        raise ValueError("Tokenizer IDs exceed configured student vocabulary")
    return tok


def quotas(sources, budget, minimum, english_cap):
    languages = list(dict.fromkeys(s["language"] for s in sources))
    if budget < minimum * len(languages):
        raise ValueError("Budget cannot satisfy minimum language counts")
    counts = {l: sum(s["available_count"] for s in sources if s["language"] == l) for l in languages}
    if any(v <= 0 for v in counts.values()):
        raise ValueError("Every source needs a positive available_count estimate")
    weights = {l: np.sqrt(v) for l, v in counts.items()}
    result = dict.fromkeys(languages, minimum)
    cap = int(budget * english_cap)
    if "en" in languages and minimum > cap:
        raise ValueError("English minimum exceeds cap")
    for _ in range(budget - sum(result.values())):
        eligible = [l for l in languages if l != "en" or result[l] < cap]
        if not eligible:
            raise ValueError("English-only corpus cannot satisfy English cap; set it explicitly to 1")
        l = min(eligible, key=lambda l: (result[l] - minimum + 1) / weights[l])
        result[l] += 1
    return result, {l: float(w / sum(weights.values())) for l, w in weights.items()}


def canonical_span(text, tok, limit):
    text = text.strip()
    ids = tok(text, add_special_tokens=True)["input_ids"]
    original = len(ids)
    if original > limit:
        ends = [m.end() for m in re.finditer(r"[.!?。！？](?:\s|$)", text)]
        lo, hi, best = 0, len(ends), None
        while lo < hi:
            mid = (lo + hi) // 2
            candidate = text[:ends[mid]].rstrip()
            if len(tok(candidate, add_special_tokens=True)["input_ids"]) <= limit:
                best, lo = candidate, mid + 1
            else:
                hi = mid
        text = best if best else tok.decode(ids[:limit], skip_special_tokens=True).rstrip()
        ids = tok(text, add_special_tokens=True)["input_ids"]
        while len(ids) > limit:
            text = text[:-1]
            ids = tok(text, add_special_tokens=True)["input_ids"]
    return text, ids, original


def split_groups(rows, seed, validation, test):
    # Union groups transitively, including duplicate spans from different documents.
    parent = list(range(len(rows)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [("text", row["text_sha256"])]
        for name in ("source_document_id", "duplicate_group_id", "parallel_group_id", "conversation_id"):
            if row.get(name):
                namespace = row["source_dataset"] if name in ("source_document_id", "conversation_id") else "global"
                keys.append((namespace, name, row[name]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    labels = {}
    for i, row in enumerate(rows):
        root = find(i)
        labels[root] = min(labels.get(root, row["id"]), row["id"])
    output, hashes = [], set()
    for i, row in enumerate(rows):
        group = labels[find(i)]
        u = int(identity([seed, group])[:12], 16) / 16**12
        row["split"] = "test" if u < test else "validation" if u < test + validation else "train"
        row["split_group"] = group
        if row["text_sha256"] not in hashes:
            output.append(row)
            hashes.add(row["text_sha256"])
    return output


def prepare(cfg):
    root = run_dir(cfg)
    if (root / "corpus_manifest.json").exists():
        raise FileExistsError("Corpus already exists; use a new run_dir to change data")
    sources = yaml.safe_load(Path(cfg["data"]["sources"]).read_text())["sources"]
    required = {"language", "script", "available_count", "license", "terms_reference"}
    for s in sources:
        if not required <= s.keys():
            raise ValueError(f"Missing source fields: {required - s.keys()}")
    quota, weights = quotas(sources, cfg["data"]["max_examples"], cfg["data"]["min_per_language"], cfg["data"]["english_cap"])
    tok = tokenizer(cfg)
    excluded = set()
    if cfg["data"].get("exclude_jsonl"):
        for line in Path(cfg["data"]["exclude_jsonl"]).open():
            raw = json.loads(line)["text"]
            for limit in (cfg["data"]["default_length"], cfg["data"]["long_length"]):
                text, _, _ = canonical_span(raw, tok, limit)
                excluded.add(identity(text))
    rows, stats, locked = [], defaultdict(Counter), []
    for source in sources:
        s = dict(source)
        lang = s["language"]
        # Divide language quota among its sources by their declared availability.
        peers = [p for p in sources if p["language"] == lang]
        weights_source = np.array([p["available_count"] for p in peers], dtype=float)
        alloc = np.floor(quota[lang] * weights_source / weights_source.sum()).astype(int)
        alloc[:quota[lang] - int(alloc.sum())] += 1
        wanted = int(alloc[peers.index(source)])
        if "path" in s:
            s["revision"] = checksum(s["path"])
            stream = (json.loads(line) for line in Path(s["path"]).open())
            dataset = str(Path(s["path"]))
        else:
            from datasets import load_dataset, get_dataset_config_names
            from huggingface_hub import HfApi
            s["revision"] = HfApi().dataset_info(s["dataset"], revision=s.get("revision", "main")).sha
            if s["config"] not in get_dataset_config_names(s["dataset"], revision=s["revision"]):
                raise ValueError(f"Unavailable source configuration: {s['dataset']} / {s['config']}")
            stream = load_dataset(s["dataset"], s["config"], revision=s["revision"], split=s.get("split", "train"), streaming=True)
            stream = stream.shuffle(seed=cfg["seed"], buffer_size=min(10000, max(100, wanted)))
            dataset = s["dataset"]
        locked.append(s)
        accepted = 0
        for i, raw in enumerate(stream):
            if accepted >= wanted:
                break
            if i >= max(wanted * 20, 1000):
                break
            text = raw.get(s.get("text_field", "text"), "")
            if not isinstance(text, str) or not text.strip():
                stats[lang]["discarded_empty"] += 1
                continue
            long = int(identity(text)[:8], 16) / 16**8 < cfg["data"]["long_fraction"]
            limit = cfg["data"]["long_length" if long else "default_length"]
            text, ids, original = canonical_span(text, tok, limit)
            if not text or not ids or identity(text) in excluded:
                stats[lang]["discarded_empty_or_benchmark"] += 1
                continue
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            row = {"id": text_hash, "text": text, "text_sha256": text_hash, "input_ids": ids,
                   "language": lang, "script": s["script"], "source_dataset": dataset,
                   "source_revision": s["revision"], "source_id": str(raw.get("id", i)),
                   "source_document_id": str(raw.get("source_document_id", raw.get("id", i))),
                   "duplicate_group_id": raw.get("duplicate_group_id"), "parallel_group_id": raw.get("parallel_group_id"),
                   "conversation_id": raw.get("conversation_id"), "source_license": s["license"],
                   "terms_reference": s["terms_reference"], "token_count": len(ids),
                   "tokenizer_revision": cfg["teacher"]["revision"], "preprocessing_version": PREPROCESSING}
            rows.append(row)
            accepted += 1
            stats[lang]["before_dedup"] += 1
            stats[lang]["truncated"] += original > len(ids)
            stats[lang]["long_257_512"] += len(ids) > 256
            stats[lang]["tokens"] += len(ids)
        if accepted < wanted:
            raise ValueError(f"{lang}/{dataset}: only {accepted}/{wanted} usable records; adjust source budget explicitly")
        print(f"Prepared {lang}: {accepted} from {dataset}", flush=True)
    rows = split_groups(rows, cfg["seed"], cfg["data"]["validation_fraction"], cfg["data"]["test_fraction"])
    counts = Counter(r["language"] for r in rows)
    if any(counts[l] < cfg["data"]["min_per_language"] for l in quota):
        raise ValueError(f"Post-dedup minimum language representation failed: {dict(counts)}")
    if counts["en"] > len(rows) * cfg["data"]["english_cap"] + 1:
        raise ValueError("Post-dedup English cap exceeded; adjust input sources")
    tmp = root / "corpus.jsonl.tmp"
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(root / "corpus.jsonl")
    manifest = {"sources": locked, "teacher": cfg["teacher"]["name"], "tokenizer_revision": cfg["teacher"]["revision"],
                "preprocessing": PREPROCESSING, "seed": cfg["seed"], "data_config": cfg["data"],
                "counts": dict(counts), "sampling_weights": weights, "stats": dict(stats),
                "split_counts": dict(Counter(r["split"] for r in rows)), "pad_id": tok.pad_token_id,
                "corpus_sha256": checksum(root / "corpus.jsonl"),
                "limitations": ["Exact dedup/exclusion only; near duplicates not detected", "No automatic curated/parallel sources"]}
    manifest["id"] = identity(manifest)
    atomic_json(root / "corpus_manifest.json", manifest)
