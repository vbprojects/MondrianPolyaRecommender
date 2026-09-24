import json,html
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/kernel_tree';rows=json.loads((out/'results.json').read_text())
lines=['Kernel tree comparison: reference / KA CPU+LAPACK / KA NVIDIA+cuSOLVER','Times include per-event allocation, transfers, and synchronization; compilation warmup excluded.','']
for r in rows:
    lines.append(f"{r['case']} | {r['method']} | {r['leaves']} leaves | {r['seconds']:.3f}s | identical cuts: {r['identical_topology']} | probability TV: {r['probability_tv']:.3g} | max mean error: {r['max_leaf_mean_error']:.3g} | max covariance error: {r['max_covariance_error']:.3g}")
summary='\n'.join(lines);(out/'summary.txt').write_text(summary)
body='<h1>KernelAbstractions tree: parity checks</h1><p>Same arrival order, seeds, and split thresholds. All comparisons assert identical topology and errors below 10⁻⁶. Kernel paths cover routing, moments, reservoirs, cut preparation, child conditional moments, subtree masses, probabilities and sampling. Julia coordinates node allocation and split recompression.</p><table><tr><th>Stream</th><th>Implementation</th><th>Leaves</th><th>Seconds</th><th>Probability TV vs reference</th><th>Max covariance error</th></tr>'
for r in rows:body+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['case'],r['method'],r['leaves'],f"{r['seconds']:.3f}",f"{r['probability_tv']:.3g}",f"{r['max_covariance_error']:.3g}"])+'</tr>'
body+='</table><p>This is a correctness-oriented hybrid backend. Per-event buffer transfers dominate GPU time; it does not yet use a persistent device store. CPU and NVIDIA were tested with Float64. Other devices and Float32 are unvalidated. The portable Jacobi eigensolver also passes the unit tests; this table uses external LAPACK/cuSOLVER libraries.</p><p><a href="summary.txt">Summary</a> · <a href="results.json">JSON</a></p>'
body='<p><a href="../kernel_optimized/demo.html">New: optimized scheduled-backend speed results</a>. The measurements below use the earlier event-by-event driver.</p>'+body
(out/'demo.html').write_text('<!doctype html><meta charset="utf-8"><title>Kernel tree comparison</title><style>body{max-width:1200px;margin:40px auto;padding:20px;font:16px system-ui;background:#111827;color:#eee}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #475569;text-align:left}a{color:#7dd3fc}</style>'+body)
print(summary)
