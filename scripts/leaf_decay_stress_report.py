import json, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/leaf_decay_stress';data=json.loads((out/'results.json').read_text())
lines=[data['description'],'','Mean across three seeds (checkpoint metrics averaged within each run):']
for method in ['No decay','5-minute half-life','1-minute half-life']:
    rows=[r for r in data['records'] if r['method']==method]
    ends=[r for r in rows if r['posts']==105000]
    assert len(ends)==3
    lines.append(f"{method}: final leaves {statistics.mean(r['leaves'] for r in ends):.1f}; final memory {statistics.mean(r['tree_bytes'] for r in ends)/2**20:.3f} MiB; mean mass TV {statistics.mean(r['mass_tv'] for r in rows):.4f}; mean recent-mean distance {statistics.mean(r['recent_mean_distance'] for r in rows):.4f}; final effective mass {statistics.mean(r['mass'] for r in ends):.1f}")
lines+=['',f"Maximum relative mass conservation error: {max(r['mass_relative_error'] for r in data['records']):.3g}",f"Maximum weighted mean conservation error: {max(r['mean_error'] for r in data['records']):.3g}",'No merging, pruning, daily expiration, or GPU optimization. Memory does not shrink just because weights fade.']
summary='\n'.join(lines);data['summary']=summary;(out/'summary.txt').write_text(summary)
viewer=(root/'examples/leaf_decay_stress_viewer.html').read_text();(out/'demo.html').write_text(viewer.replace('__DATA__',json.dumps(data).replace('<','\\u003c')))
print(summary)
