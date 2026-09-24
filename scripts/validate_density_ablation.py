import hashlib,json,math
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/density_ablation'
d=json.loads((out/'results.json').read_text());rows=d['trials'];assert len(rows)==33600
assert len(d['traces'])==112 and len(d['visuals'])==4
paired=defaultdict(list);settings=defaultdict(list)
for r in rows:
 assert 1<=r['target']<=64 and 0<=r['stop']<=80
 assert r['used']==(r['stop'] or 80)
 assert r['capped_correct']==(r['stop'] if r['correct'] else 81)
 assert int(r['correct'])+int(r['false_stop'])+int(r['exhausted'])==1
 assert r['correct']==(r['stop']>0 and r['error']<=r['radius'])
 assert 0<=r['brier']<=1 and math.isfinite(r['logloss'])
 assert 1<=r['mean_candidates']<=32 and 0<=r['fallback_fraction']<=1
 paired[r['case'],r['seed'],r['trial'],r['radius'],r['world']].append(r)
 settings[r['case'],r['seed'],r['world'],r['radius'],r['method']].append(r)
for key,rs in paired.items():
 assert len({r['target'] for r in rs})==1
 assert len(rs)==(6 if key[-1]=='Fermat: density + proposals' else 4)
for rs in settings.values():assert len(rs)==100
for trace in d['traces']:
 h=trace['history'];assert len(h)==80
 assert len({r['query'] for r in h})==80
 assert all(abs(sum(r['belief'])-1)<1e-12 and min(r['belief'])>=0 for r in h)
 first=next((r['step'] for r in h if r['certainty']>=.9),0)
 assert first==trace['stop']
 if first:assert trace['correct']==(h[first-1]['error']<=trace['radius'])
for case in {r['case'] for r in d['diagnostics']}:
 for seed in (1,2,3):
  rs=[r for r in d['diagnostics'] if r['case']==case and r['seed']==seed]
  assert len(rs)==6 and len({r['codec_sha256'] for r in rs})==1
  assert len({(r['training_posts'],r['leaves'],r['codec_bytes']) for r in rs})==1
  assert all(0<r['min_truncation_acceptance']<=1 for r in rs)
base=[r for r in rows if not r['method'].startswith('Actual posts +')]
keys=('case','seed','world','radius','method','trial','target','stop','correct','used','capped_correct','error')
digest=hashlib.sha256(json.dumps([tuple(r[k] for k in keys) for r in base],separators=(',',':')).encode()).hexdigest()
original=(out/'original_outcome_sha256.txt').read_text().strip();assert digest==original
browser=json.loads((out/'browser_validation.json').read_text());assert not browser['errors']
tests=(out/'tests.log').read_text();assert '57' in tests and 'Fail' not in tests and 'ERROR' not in tests
paths=['src/DensityAblation.jl','examples/density_ablation.jl','examples/density_ablation_viewer.html',
       'scripts/density_ablation_report.py','test/density_ablation.jl','test/density_ablation_viewer.cjs',
       'output/density_ablation/results.json','output/density_ablation/demo.html']
v=dict(trials=len(rows),landscapes=12,static_codec_identity=True,paired_targets=True,
       posterior_and_stopping_audit=True,original_28800_outcomes_unchanged=True,original_outcomes_sha256=digest,
       unit_tests=tests,browser=browser,sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
(out/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({k:val for k,val in v.items() if k not in ('sha256','unit_tests')},indent=2))
