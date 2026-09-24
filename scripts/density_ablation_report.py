import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

root=Path(__file__).resolve().parents[1];out=root/'output/density_ablation'
data=json.loads((out/'results.json').read_text());groups=defaultdict(list)
for r in data['trials']:groups[r['case'],r['world'],r['radius'],r['method']].append(r)
summary=[]
for (case,world,radius,method),rs in groups.items():
    successes=[r for r in rs if r['correct']]
    a=dict(case=case,world=world,radius=radius,method=method,trials=len(rs),
           success=st.mean(r['correct'] for r in rs),false_stop=st.mean(r['false_stop'] for r in rs),
           exhausted=st.mean(r['exhausted'] for r in rs),mean_used=st.mean(r['used'] for r in rs),
           mean_correct=st.mean(r['stop'] for r in successes) if successes else None,
           capped_correct=st.mean(r['capped_correct'] for r in rs),
           brier=st.mean(r['brier'] for r in rs),selection_ms=st.mean(r['selection_ms'] for r in rs),
           mean_candidates=st.mean(r['mean_candidates'] for r in rs),fallback_fraction=st.mean(r['fallback_fraction'] for r in rs))
    seedrates=[st.mean(r['correct'] for r in rs if r['seed']==seed) for seed in (1,2,3)]
    a['seed_success_range']=[min(seedrates),max(seedrates)]
    a['success_curve']=[sum(r['correct'] and 0<r['stop']<=i for r in rs)/len(rs) for i in range(81)]
    a['false_curve']=[sum(r['false_stop'] and 0<r['stop']<=i for r in rs)/len(rs) for i in range(81)]
    summary.append(a)

def paired(a,b):
    aa={(r['seed'],r['trial']):r for r in a};bb={(r['seed'],r['trial']):r for r in b}
    assert aa.keys()==bb.keys()
    differences={s:[(int(aa[s,t]['correct'])-int(bb[s,t]['correct']),
                     aa[s,t]['capped_correct']-bb[s,t]['capped_correct'],
                     int(aa[s,t]['false_stop'])-int(bb[s,t]['false_stop'])) for t in range(1,101)] for s in (1,2,3)}
    rng=random.Random(512);samples=[[],[],[]]
    for _ in range(1000):
        selected=[v for s in (1,2,3) for v in rng.choices(differences[s],k=100)]
        for k in (0,1,2):samples[k].append(st.mean(x[k] for x in selected))
    result={}
    for k,name in enumerate(('success_difference','capped_difference','false_difference')):
        values=sorted(samples[k]);mean=st.mean(v[k] for ss in differences.values() for v in ss)
        result[name]=dict(mean=mean,ci95=[values[25],values[974]])
    return result

comparisons=[]
for key,rows in groups.items():
    case,world,radius,method=key
    if method=='PPCA rank 4':continue
    result=paired(groups[case,world,radius,'PPCA rank 4'],rows)
    comparisons.append(dict(case=case,world=world,radius=radius,against=method,**result))
direct_comparisons=[]
for (case,world,radius,method),rows in groups.items():
    if method!='Actual posts + PPCA density':continue
    for other in ('Actual posts; no Gaussian','Actual posts + diagonal density'):
        result=paired(rows,groups[case,world,radius,other])
        direct_comparisons.append(dict(case=case,world=world,radius=radius,against=other,**result))
diagnostics=[]
for case in dict.fromkeys(r['case'] for r in data['diagnostics']):
    for method in dict.fromkeys(r['method'] for r in data['diagnostics']):
        rs=[r for r in data['diagnostics'] if r['case']==case and r['method']==method]
        a=dict(case=case,method=method)
        for key in rs[0]:
            if key in ('case','method','seed','codec_sha256'):continue
            a[key]=st.mean(r[key] for r in rs)
        diagnostics.append(a)
geometry=[]
for case in dict.fromkeys(r['case'] for r in data['geometry']):
    for method in ('uniform','diagonal','ppca'):
        rs=[r for r in data['geometry'] if r['case']==case and r['method']==method]
        geometry.append(dict(case=case,method=method,**{k:st.mean(r[k] for r in rs) for k in rs[0] if k not in ('case','method','seed')}))
data['summary']=summary;data['comparisons']=comparisons;data['direct_comparisons']=direct_comparisons;data['density_summary']=diagnostics;data['geometry_summary']=geometry
(out/'aggregates.json').write_text(json.dumps(dict(summary=summary,comparisons=comparisons,direct_comparisons=direct_comparisons,density=diagnostics,geometry=geometry),indent=2)+'\n')
with (out/'trials.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(data['trials'][0]));writer.writeheader();writer.writerows(data['trials'])
lines=['Within-leaf density ablation with one shared centroid + 8-bit post codec',
       '4 workloads x 3 seeds x 2 radii x (two proposal-only settings with 4 methods, one complete setting with 6 methods) x 100 trials = 33,600 trials.',
       'First 90%-confidence stop; correct only within independent true-geometry radius. Budget 80 responses.',
       'Failure-penalized responses = first correct stop, otherwise 81 (not an uncensored mean).',
       'Simulator geometry: Euclidean or independent held-out kNN density/Fermat graph. All feedback synthetic.',
       'Shared static scaffold, exact Pólya region masses, fresh online fits in final leaves. No dynamic mass-transfer ablation.','']
for case in dict.fromkeys(a['case'] for a in summary):
    for world in dict.fromkeys(a['world'] for a in summary):
        for radius in (.05,.10):
            lines.append(f'{case} / {world} / radius {radius}')
            for r in summary:
                if (r['case'],r['world'],r['radius'])!=(case,world,radius):continue
                correct='n/a' if r['mean_correct'] is None else f"{r['mean_correct']:.1f}"
                lines.append(f"  {r['method']}: correct {r['success']:.1%}, false {r['false_stop']:.1%}, exhausted {r['exhausted']:.1%}; "
                             f"consumed {r['mean_used']:.1f}, correct-only {correct}, penalized {r['capped_correct']:.1f}; "
                             f"scored {r['mean_candidates']:.1f}/response, fallback {r['fallback_fraction']:.1%}")
            lines.append('')
lines+=['Paired PPCA minus actual-post sampling, strict radius, complete density/Fermat setting:']
for c in comparisons:
    if c['world']=='Fermat: density + proposals' and c['radius']==.05 and c['against']=='Actual posts; no Gaussian':
        v=c['success_difference'];cost=c['capped_difference']
        lines.append(f"  {c['case']}: success delta {v['mean']:+.1%}, stratified-bootstrap 95% interval [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]; "
                     f"penalized-response delta {cost['mean']:+.2f} [{cost['ci95'][0]:+.2f}, {cost['ci95'][1]:+.2f}].")
lines+=['','Direct-post sampling held fixed: PPCA density minus alternative density, strict radius:']
for c in direct_comparisons:
    if c['radius']!=.05:continue
    v=c['success_difference'];false=c['false_difference'];cost=c['capped_difference']
    lines.append(f"  {c['case']} / versus {c['against']}: success delta {v['mean']:+.1%} [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]; "
                 f"false-stop delta {false['mean']:+.1%} [{false['ci95'][0]:+.1%}, {false['ci95'][1]:+.1%}]; "
                 f"penalized-response delta {cost['mean']:+.2f} [{cost['ci95'][0]:+.2f}, {cost['ci95'][1]:+.2f}].")
lines+=['','Bootstrap resamples paired trials within each fixed seed; intervals do not represent new-corpus/landscape uncertainty.',
        'The direct-post baseline uses uniform tree density in the complete setting. Proposal-only settings share reference geometry.',
        'Local density PDFs are box-truncated; diagonal normalization analytic, PPCA normalization estimated with 4096 Monte Carlo draws.',
        'All methods share the same per-case codec SHA256 and post identities. No original embedding files were changed.']
(out/'summary.txt').write_text('\n'.join(lines)+'\n')
template=(root/'examples/density_ablation_viewer.html').read_text()
(out/'demo.html').write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')))
print('\n'.join(lines))
