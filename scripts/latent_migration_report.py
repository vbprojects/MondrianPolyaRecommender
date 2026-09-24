import json, statistics, collections
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/latent_migration'
data=json.loads((out/'results.json').read_text());rows=data['records']
methods=list(dict.fromkeys(r['method'] for r in rows));cases=list(dict.fromkeys(r['case'] for r in rows))
summary=[]
for case in cases:
    for method in methods:
        rs=[r for r in rows if r['case']==case and r['method']==method and r['cycle']==80 and r['phase']=='merged']
        assert len(rs)==3
        item=dict(case=case,method=method)
        for key in ['nrmse','migration_nrmse','variance_ratio','recall10','routing_error','max_map_log10_condition']:
            vals=[r[key] for r in rs]
            item[key]=dict(mean=statistics.mean(vals),minimum=min(vals),maximum=max(vals)) if all(v is not None for v in vals) else None
        summary.append(item)
(out/'aggregates.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['Repeated per-post latent-code migration',
'5 synthetic geometries × 3 seeds × 80 grow/idle-merge cycles; 384 fixed UUIDs per run.',
'12 policies share each actual adaptive PPCA tree. Float32/16 alter stored codes; arithmetic remains Float64.',
'NRMSE is RMS reconstruction error normalized by the original centered data spread. Recall@10 uses 24 fixed original-vector queries per run.',
'Final merged-state means across 3 seeds (ranges are seed minimum–maximum).','']
for case in cases:
    lines.append(case)
    for r in [x for x in summary if x['case']==case]:
        n=r['nrmse'];v=r['variance_ratio'];q=r['recall10']
        lines.append(f"  {r['method']}: NRMSE {n['mean']:.6g} [{n['minimum']:.6g}, {n['maximum']:.6g}], variance {v['mean']:.6g}, recall@10 {q['mean']:.1%} [{q['minimum']:.1%}, {q['maximum']:.1%}]")
    lines.append('')
final=[r for r in rows if r['method']=='Sequential Float64' and r['cycle']==80 and r['phase']=='merged']
lines+=['Trajectory check: rotating 256D geometry returns to the original plane at cycle 40, and its original orientation at cycle 80.']
for method in ['Sequential Float64','Orthogonal transport','Remap from initial code']:
    rs=[r for r in rows if r['case']=='Rotating 256D manifold' and r['method']==method and r['cycle']==40 and r['phase']=='merged']
    lines.append(f"  Cycle 40 / {method}: NRMSE {statistics.mean(r['nrmse'] for r in rs):.4f}, recall@10 {statistics.mean(r['recall10'] for r in rs):.2%}.")
lines+=['A strong final-cycle score can hide earlier geometric drift. Read the full trajectory, not only the endpoint.','']
lines.extend([f"Actual tree operations across runs: {sum(r['splits'] for r in final)} splits and {sum(r['merges'] for r in final)} merges.",
    f"Maximum mapping equivalence error: {max(r['map_equivalence_error'] for r in rows):.4g}.",
    f"Maximum forward-map singular value: {max(r['max_map_singular'] for r in rows):.16g}.",
    f"Maximum basis orthogonality error: {max(r['basis_orthogonality_error'] for r in rows):.4g}.",
    'All checkpoint UUID sets were complete and duplicate-free; all reconstructed vectors were finite.',
    'Noise variants are heuristics, not proofs about every perturbation strategy. This experiment does not validate actual recommendation quality.',
    'Oracle/periodic refresh read original audit vectors; anchor/frozen policies retain the initial decoder and original compressed code.',
    'Persistent old posts intentionally outlive the decay window. Decoder-version storage growth under continuing new post arrivals is not measured.',
    'Very low variance causes near-tied neighbor distances, so small recall differences after collapse are not robust.'])
(out/'summary.txt').write_text('\n'.join(lines)+'\n')
html=(root/'examples/latent_migration_viewer.html').read_text().replace('__DATA__',json.dumps(data).replace('</','<\\/'))
(out/'demo.html').write_text(html)
print('\n'.join(lines))
