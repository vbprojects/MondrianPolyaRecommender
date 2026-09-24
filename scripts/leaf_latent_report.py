"""Held-out evaluation/report for leaf-only Gaussian tree representations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json,hashlib
from pathlib import Path
import numpy as np
from proposal_experiment import ig,stats
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/leaf_latent'

def main():
    data=json.loads((OUT/'experiment.json').read_text())
    X=np.load(ROOT/'data/bluesky_100000/embeddings.npy');train=np.array(data['train_rows']);test=np.array(data['test_rows']);targets=np.array(data['target_rows'])
    assert len(train)==80000 and len(test)==20000 and not np.intersect1d(train,test).size
    catalog=X[train];H=X[targets];banks={};diagnostics={}
    for method in data['proposals']:
        name=method['name'];bank=[];distances=[];owners_out=[]
        for h,points in enumerate(method['points']):
            if not points:bank.append(np.array([],int));continue
            P=np.asarray(points,np.float32);bounds=data['target_bounds'][h]
            assert np.all(P>=np.array(bounds['lo'])-1e-6) and np.all(P<=np.array(bounds['hi'])+1e-6)
            indices=np.argmax(P@catalog.T,axis=1);bank.append(train[indices]);distances.extend(np.linalg.norm(P-catalog[indices],axis=1))
            owners_out.extend(np.any((catalog[indices]<bounds['lo'])|(catalog[indices]>=bounds['hi']),axis=1))
        banks[name]=bank
        diagnostics[name]=dict(mean_snap_distance=float(np.mean(distances)) if distances else None,
            retrieved_outside_owner=float(np.mean(owners_out)) if owners_out else None,
            unique_posts=int(len(set(np.concatenate(bank)))) if any(len(b) for b in bank) else 0)
    rng=np.random.default_rng(551);bank=[]
    for bounds in data['target_bounds']:
        eligible=train[np.all((catalog>=bounds['lo'])&(catalog<bounds['hi']),axis=1)]
        assert len(eligible)>0
        bank.append(rng.choice(eligible,8,replace=len(eligible)<8))
    banks['local_actual_posts']=bank
    allids=np.unique(np.concatenate([b for v in banks.values() for b in v]))
    positions={int(row):i for i,row in enumerate(allids)}
    indexed={name:[np.array([positions[int(x)] for x in rows]) for rows in bank] for name,bank in banks.items()}
    distances=np.sqrt(np.maximum(0,2-2*X[allids]@H.T))
    # Scale fixed from held-out target pair distances, not each method's proposals.
    pair=np.sqrt(np.maximum(0,2-2*H@H.T));scale=float(np.median(pair)*.25)
    L=.05+.9/(1+(distances/scale)**2)
    beliefs=np.random.default_rng(994).dirichlet(np.full(64,.3),size=100)
    values={name:[] for name in banks};unique={name:[] for name in banks}
    rng=np.random.default_rng(448)
    for belief in beliefs:
        hs=np.searchsorted(np.cumsum(belief),rng.random(32));u=rng.random(32)
        for name,bank in indexed.items():
            chosen=[bank[h][min(len(bank[h])-1,int(v*len(bank[h])))] for h,v in zip(hs,u) if len(bank[h])]
            selected=np.unique(chosen).astype(int)
            values[name].append(float(ig(belief,L[selected]).max()/np.log(2)) if len(selected) else 0.)
            unique[name].append(len(selected))
    result=dict(metrics=data['metrics'],sampling=diagnostics,
        proposal_ig={name:dict(information_bits=stats(v),mean_unique_candidates=float(np.mean(unique[name]))) for name,v in values.items()},
        paired_ig_minus_diagonal={name:stats(np.array(v)-values['diagonal_r0']) for name,v in values.items() if name!='diagonal_r0'},
        configs=data['configs'],fit_wall_seconds=data['fit_wall_seconds'],feedback_scale=scale,
        input_manifest_sha256=hashlib.sha256((ROOT/'data/bluesky_100000/manifest.json').read_bytes()).hexdigest())
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    rows=['Leaf-only latent Gaussian experiment: 80000 training / 20000 held-out posts.',
          'name | training model MiB incl tree/reservoir | compact Float32 decoder KiB | test cosine | sampling acceptance | mean proposal IG bits']
    for m in data['metrics']:
        rows.append(f"{m['name']} | {m['model_bytes']/2**20:.3f} | {m['decoder_float32_bytes']/1024:.1f} | {m['test_cosine']:.5f} | {m['acceptance']:.4f} | {result['proposal_ig'][m['name']]['information_bits']['mean']:.5f}")
    (OUT/'summary.txt').write_text('\n'.join(rows)+'\n')
    table=''.join(f"<tr><td>{m['name']}</td><td>{m['leaves']}</td><td>{m['model_bytes']/2**20:.2f}</td><td>{m['decoder_float32_bytes']/1024:.1f}</td><td>{m['update_seconds']:.2f} / {m['split_seconds']:.2f}</td><td>{m['test_cosine']:.4f}</td><td>{100*m['acceptance']:.1f}%</td><td>{m['failed_regions']}</td><td>{result['proposal_ig'][m['name']]['information_bits']['mean']:.4f}</td></tr>" for m in data['metrics'])
    comparisons=''.join(f"<li>{name}: {s['mean']:+.4f} bits, 95% interval [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]</li>" for name,s in result['paired_ig_minus_diagonal'].items())
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Leaf latent representations</title><style>body{font:16px system-ui;background:#111827;color:#eef3ff;max-width:1200px;margin:35px auto;padding:0 20px}p,li{line-height:1.6;color:#bfcddd}table{border-collapse:collapse;font-size:14px;width:100%}td,th{text-align:left;border-bottom:1px solid #415065;padding:10px}a{color:#8ed0ff}.scroll{overflow:auto}</style><h1>New latent models after each tree split</h1><p>80,000 training posts · 20,000 held-out posts · shared adaptive topology · only active leaves retain Gaussian models · no full covariance</p><p>Children inherit halfspace-conditioned Gaussian moments, then fit their own compact bases. Parent models and reservoirs are released. Reservoirs guide cuts and child initialization. Sampling uses rejection, not clipping.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Leaves</th><th>Training model MiB</th><th>Decoder KiB*</th><th>Updates / splits (s)</th><th>Test reconstruction cosine</th><th>Acceptance</th><th>Failed regions / 64</th><th>Proposal IG bits</th></tr></thead><tbody>__TABLE__</tbody></table></div><p>*Decoder is a calculated Float32 parameter payload, excluding shared tree, reservoirs, per-post codes and IDs. Training model memory includes the tree and bounded reservoirs. Reconstruction is unquantized; this is not yet a vector codec benchmark.</p><h2>Paired proposal information versus diagonal Gaussians</h2><ul>__COMPARISONS__</ul><p>100 fixed beliefs over 64 held-out user locations, 32 proposals per belief. Rejected/failed proposal regions receive no substitute. Accepted samples are mapped to original training posts, then scored using a common chord-distance response model. Bootstrap intervals are conditional on this one tree and sample bank.</p><p>Conditional moment formulas apply to an untruncated parent Gaussian and a single new cut. They do not exactly account for all inherited box constraints. Rejection sampling respects the final box but the probability-mass transfer remains approximate. Rank-eight subspaces are not necessarily better than rank four for sampling.</p><p><a href="results.json">Full metrics</a> · <a href="summary.txt">Summary</a> · <a href="../../docs/Leaf_Latent_Experiment.md">Protocol and limitations</a></p></html>'''
    (OUT/'demo.html').write_text(html.replace('__TABLE__',table).replace('__COMPARISONS__',comparisons))
    print('\n'.join(rows),flush=True)
if __name__=='__main__':main()
