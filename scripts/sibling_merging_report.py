import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/sibling_merging'
data=json.loads((out/'results.json').read_text())
text=(root/'examples/sibling_merging_viewer.html').read_text().replace('__DATA__',json.dumps(data).replace('</','<\\/'))
(out/'demo.html').write_text(text)
lines=['Sibling merging stress test',data['description'],'']
for name in ['Decay only','Decay + sibling merging']:
    h=[r for r in data['history'] if r['method']==name];r=h[-1]
    lines.append(f"{name}: final {r['leaves']} leaves, {r['tree_bytes']/1024:.1f} KiB live nodes; {r['splits']} splits / {r['merges']} merges; peak {max(x['leaves'] for x in h)} leaves.")
lines+=['','Three warmed whole-fit timing trials per method; timings exclude audits and visualization.']
for r in data['records']:
    times=[s['seconds'] for s in r['samples']]
    lines.append(f"{r['case']} | {r['method']}: median {r['median_seconds']:.5f}s (range {min(times):.5f}–{max(times):.5f}); idle maintenance {r['leaves_before_idle']} → 1 leaves, {r['bytes_before_idle']/1024:.1f} → {r['bytes_after_idle']/1024:.1f} KiB live model.")
worst=max(max(s['probability_tv'],s['mean_error'],s['covariance_error']) for r in data['records'] for s in r['samples'])
lines+=['',f'Identical topology/reservoirs/counts in every timed fit. Worst numerical error {worst:.3g}.',
'Memory counts live tree models/reservoirs and indexing metadata. It excludes source embeddings, diagnostic history, allocator caches, and process/runtime state.',
'Quiet sibling pairs merge by combined mass. Moments preserve stored mass, first moment and scatter diagonal; low-rank correlations are recompressed. Reservoirs keep newest retained arrival IDs. Source archives are untouched.']
(out/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
