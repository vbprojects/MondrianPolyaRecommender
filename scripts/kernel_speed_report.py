"""Render measured full-driver speed and separate batched-routing results."""
import html
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'output/kernel_speed'
data=json.loads((out/'results.json').read_text());rows=data['records'];h=data['hardware']
lines=['Live kernel tree speed benchmark',f"CPU: {h['cpu']}",f"GPU: {h['gpu']}",f"Julia {h['julia']}; Julia threads={h['julia_threads']}; BLAS threads={h['blas_threads']}",h['timing'],'Median of 3 full fits or 5 routing repetitions.\n']
body='<h1>Measured tree speed on this workstation</h1><p>'+html.escape(' · '.join([h['cpu'],h['gpu'],'Julia '+h['julia']]))+'</p>'
body+='<p>One Julia thread and one BLAS thread. Models use Float64. Timings exclude compilation, warmup, disk loading, and embedding inference. Packing, transfers, validation, allocation and synchronization are included. Independent identical-stream trials rebuild each tree from scratch. This is not an isolated machine; other applications may affect timing.</p>'
for operation in ['full tree fit','batched routing']:
    body+='<h2>'+html.escape(operation.capitalize())+'</h2><table><tr><th>Stream</th><th>Backend</th><th>Inputs</th><th>Median seconds</th><th>Min–max</th><th>Inputs/s</th><th>Time / reference</th></tr>'
    for r in rows:
        if r['operation']!=operation:continue
        baseline=next(x for x in rows if x['operation']==operation and x['case']==r['case'] and x['method']=='Reference Julia')
        ratio=r['median_seconds']/baseline['median_seconds']
        fields=[r['case'],r['method'],f"{r['units']:,}",f"{r['median_seconds']:.5f}",f"{r['min_seconds']:.5f}–{r['max_seconds']:.5f}",f"{r['units_per_second']:,.0f}",f'{ratio:.1f}×']
        body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in fields)+'</tr>'
        lines.append(' | '.join([operation]+fields))
    body+='</table>'
worst=max(r['max_probability_tv'] for r in rows)
lines+=['',f'All runs passed topology/routing equivalence; maximum probability TV versus reference: {worst:.3g}.','Batched routing is only a fixed-tree query benchmark; it is not full streaming ingestion.','The current kernel driver transfers and synchronizes small arrays per event. Results measure that driver, not the potential of a persistent/batched GPU design.']
body+='<p>All measured fits produced identical topology. Maximum probability TV against the reference: '+f'{worst:.3g}'+'. Batched routing returned identical node IDs.</p><p>The full tree driver currently transfers and synchronizes small arrays per event. It has no persistent device state or batched ingestion scheduler. Routing batches use a fixed completed tree; their throughput does not describe online tree fitting. No speed advantage is implied by portability.</p><p><a href="results.json">Full results, per-trial times and host allocation counts</a> · <a href="summary.txt">Text summary</a> · <a href="run.log">Live run log</a></p>'
(out/'summary.txt').write_text('\n'.join(lines))
body='<p><a href="../kernel_optimized/demo.html">New: optimized scheduled-backend speed results</a>. The measurements below use the earlier event-by-event driver.</p>'+body
(out/'demo.html').write_text('<!doctype html><meta charset="utf-8"><title>Live tree speed benchmark</title><style>body{max-width:1400px;margin:35px auto;padding:20px;font:16px system-ui;background:#111827;color:#eee}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #475569}a{color:#7dd3fc}</style>'+body)
print('\n'.join(lines))
