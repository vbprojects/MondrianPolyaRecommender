"""Aggregate measured streaming-codec results; no scientific computations in JS."""
import json
import statistics as stats
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "output/streaming_codecs"
data = json.loads((out / "results.json").read_text())
methods = list(dict.fromkeys(r["method"] for r in data["records"]))
cases = list(dict.fromkeys(r["case"] for r in data["records"]))
metrics = ["recall10", "proposal_recall10", "local_proposal_recall10", "nrmse", "ig_regret_bits",
           "ig_choice_agreement", "membership_error", "outside_fraction", "variance_ratio", "mean_rank"]
aggregates = []
for case in cases:
    for phase in ("grown", "merged"):
        for method in methods:
            # Equal seed weighting, all checkpoints from day two onward. Temporal
            # observations within seeds are not independent confidence intervals.
            rs = [r for r in data["records"] if r["case"] == case and r["method"] == method
                  and r["phase"] == phase and r["day"] >= 2]
            seeds = sorted({r["seed"] for r in rs})
            a = dict(case=case, phase=phase, method=method, checkpoints=len(rs))
            for key in metrics + ["packed_per_post", "runtime_per_post", "decoder_per_post", "metadata_per_post"]:
                def value(r):
                    columns = dict(packed_per_post="packed_bytes", runtime_per_post="runtime_bytes",
                                   decoder_per_post="decoder_bytes", metadata_per_post="metadata_bytes")
                    return r[columns[key]] / r["posts"] if key in columns else r[key]
                vals = [stats.mean(value(r) for r in rs if r["seed"] == seed) for seed in seeds]
                a[key] = dict(mean=stats.mean(vals), minimum=min(vals), maximum=max(vals))
            run = [r for r in data["runs"] if r["case"] == case and r["method"] == method]
            for kind in ("encode", "maintenance", "expiry"):
                a[kind + "_ms_per_1000"] = stats.mean(r[kind + "_seconds"] * 1e6 / r["arrivals"] for r in run)
            a["peak_runtime_bytes"] = max(r["peak_runtime_bytes"] for r in run)
            a["peak_packed_bytes"] = max(r["peak_packed_bytes"] for r in run)
            a["total_ms_per_1000"] = sum(a[k + "_ms_per_1000"] for k in ("encode", "maintenance", "expiry"))
            aggregates.append(a)

lines = ["Local embedding codecs under continuing arrivals, splits, merges and expiry",
         "12 methods; 6 workloads; 3 seeds each. Bluesky: all 100,000 posts per seed, 256 dimensions.",
         "Artificial burst/idle clock; Bluesky archive order is not chronological.",
         "Means use equally weighted seed means across day >=2 grown-state checkpoints.",
         "Packed bytes include shared decoders, scales, UUID128, UInt64 membership and block headers.",
         "RAM is reachable Julia store memory; tree, input batches, text and audit storage are separate.", ""]
for case in cases:
    lines.append(case)
    for a in [a for a in aggregates if a["case"] == case and a["phase"] == "grown"]:
        m = lambda k: a[k]["mean"]
        lines.append(f"  {a['method']}: packed {m('packed_per_post'):.1f} B/post; RAM {m('runtime_per_post'):.1f} B/post; "
                     f"recall {m('recall10'):.1%}; midpoint recall {m('proposal_recall10'):.1%}; "
                     f"local midpoint {m('local_proposal_recall10'):.1%}; NRMSE {m('nrmse'):.4f}; "
                     f"membership error {m('membership_error'):.2%}; IG regret {m('ig_regret_bits'):.6f} bits; "
                     f"encode/maintain {a['encode_ms_per_1000']:.1f}/{a['maintenance_ms_per_1000']:.1f} ms per 1,000 arrivals")
    lines.append("")
chosen = []
for case in cases:
    candidates = [a for a in aggregates if a["case"] == case and a["phase"] == "grown"]
    for threshold in (.9, .7):
        eligible = [a for a in candidates if a["local_proposal_recall10"]["mean"] >= threshold]
        best = min(eligible, key=lambda a: a["packed_per_post"]["mean"])
        chosen.append(dict(case=case, minimum_recall=threshold, method=best["method"],
                           packed_per_post=best["packed_per_post"]["mean"],
                           recall=best["local_proposal_recall10"]["mean"]))
        lines.append(f"Smallest measured codec at >= {threshold:.0%} local midpoint recall / {case}: "
                     f"{best['method']} ({best['packed_per_post']['mean']:.1f} B/post).")
lines += ["", "These are choices among tested settings using measured mean recall, not validated universal thresholds.",
          "Recall compares exact Euclidean neighbors; midpoint proposals are within-leaf arithmetic midpoints, not Fermat paths.",
          "Synthetic IG does not validate human recommendations. The shared density model is unchanged across codecs.",
          "All UUID inventories passed; complete expiry released every decoder and inherited basis chain.",
          "Latent-only methods can lose discarded directions even when their codes are immutable.",
          "Timings include Julia allocation/GC and exclude audits; this is an unoptimized CPU experiment."]
data["aggregates"] = aggregates
data["choices"] = chosen
(out / "aggregates.json").write_text(json.dumps(aggregates, indent=2) + "\n")
(out / "summary.txt").write_text("\n".join(lines) + "\n")
template = (root / "examples/streaming_codec_viewer.html").read_text()
(out / "demo.html").write_text(template.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
print("\n".join(lines))
