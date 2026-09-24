"""Paired active-localization experiment with shared geometry and fixed budgets."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json,time,csv,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/proposal_experiment'
METHODS=['tree_uniform','micro_diagonal','mppca','local_real_posts','random_posts']

def ent(x):
    x=np.clip(x,1e-12,1-1e-12)
    return -x*np.log(x)-(1-x)*np.log1p(-x)

def ig(p,L):return ent(L@p)-ent(L)@p

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def stats(values):
    v=np.asarray(values);rng=np.random.default_rng(411)
    means=np.mean(v[rng.integers(len(v),size=(2000,len(v)))],axis=1)
    return dict(mean=float(v.mean()),ci95=[float(x) for x in np.quantile(means,[.025,.975])])

def main():
    start=time.perf_counter();model=json.loads((OUT/'models.json').read_text())
    X=np.load(ROOT/'data/bluesky_100000/embeddings.npy'); targets=np.array(model['rows']);H=X[targets];K=len(H)
    lo=np.array(model['bounds'],np.float32)[:,:256];hi=np.array(model['bounds'],np.float32)[:,256:]
    mu=np.array(model['means'],np.float32);B=np.array(model['bases'],np.float32)
    var=np.array(model['variances'],np.float32);noise=np.array(model['noise'],np.float32);mass=np.array(model['mass'])
    rng=np.random.default_rng(819);bank_size=64
    # For each region, identical mixture draws and Gaussian innovations isolate covariance geometry.
    proposals={m:[] for m in METHODS[:3]}; banks={}; diagnostics={}
    for h in range(K):
        ids=np.array(model['components'][h]);ids=rng.choice(ids,bank_size,p=mass[ids]/mass[ids].sum())
        eps=rng.normal(size=(bank_size,256));z=rng.normal(size=(bank_size,4))
        gauss=mu[ids]+np.sqrt(noise[ids,None])*eps+np.einsum('nrd,nr->nd',B[ids],z*np.sqrt(np.maximum(0,var[ids]-noise[ids,None])))
        diag=noise[ids,None]+np.einsum('nrd,nr->nd',B[ids]**2,np.maximum(0,var[ids]-noise[ids,None]))
        diagonal=mu[ids]+np.sqrt(diag)*eps
        proposals['tree_uniform'].append(rng.uniform(lo[h],hi[h],size=(bank_size,256)))
        proposals['micro_diagonal'].append(diagonal)
        proposals['mppca'].append(gauss)
    # Equal projection to owning box. All retrieval uses original vectors and
    # Euclidean distance (norms of corpus posts are all one), no codec distortion.
    for method,raw in proposals.items():
        mapping_start=time.perf_counter()
        raw=np.asarray(raw,np.float32); outside=np.mean(np.any((raw<lo[:,None])|(raw>hi[:,None]),axis=2))
        P=np.clip(raw,lo[:,None],np.nextafter(hi[:,None],lo[:,None])).reshape(-1,256)
        snapped=[];gaps=[];norms=[]
        for first in range(0,len(P),128):
            batch=P[first:first+128]; scores=batch@X.T;scores[:,targets]=-np.inf
            ids=np.argmax(scores,axis=1);snapped.extend(ids)
            gaps.extend(np.linalg.norm(batch-X[ids],axis=1));norms.extend(np.linalg.norm(batch,axis=1))
        banks[method]=np.array(snapped).reshape(K,bank_size)
        diagnostics[method]=dict(outside_before_clipping=float(outside),mean_snap_distance=float(np.mean(gaps)),
            mean_proposal_norm=float(np.mean(norms)),retrieved_outside_owner=float(np.mean(np.any((X[np.array(snapped)].reshape(K,bank_size,256)<lo[:,None])|(X[np.array(snapped)].reshape(K,bank_size,256)>hi[:,None]),axis=2))),unique_posts=int(len(set(snapped))),
            mean_unique_per_region=float(np.mean([len(set(r)) for r in banks[method]])),mapping_seconds=time.perf_counter()-mapping_start)
        print('Mapped',method,diagnostics[method],flush=True)
    for method in METHODS[3:]:
        banks[method]=np.empty((K,bank_size),int)
        for h in range(K):
            if method=='local_real_posts':
                eligible=np.flatnonzero(np.all((X>=lo[h])&(X<=hi[h]),axis=1))
                eligible=eligible[~np.isin(eligible,targets)]
            else:eligible=np.setdiff1d(np.arange(len(X)),targets)
            assert len(eligible)>0
            banks[method][h]=rng.choice(eligible,bank_size,replace=len(eligible)<bank_size)
    allids=np.unique(np.concatenate([v.ravel() for v in banks.values()]))
    index={int(v):i for i,v in enumerate(allids)}
    banks={m:np.array([[index[int(i)] for i in row] for row in bank]) for m,bank in banks.items()}
    # Shared evaluations: chord distance and density-weighted graph distance.
    chord=np.sqrt(np.maximum(0,2-2*(X[allids]@H.T)))
    D=np.sqrt(np.maximum(0,2-2*(H@H.T)));np.fill_diagonal(D,0)
    near=np.argsort(D,axis=1)[:,1:9]
    radius=np.take_along_axis(D,near[:,-1:],axis=1).ravel()
    density=np.clip((np.median(radius)/np.maximum(radius,1e-6))**2,.1,10)
    graph=np.full_like(D,np.inf);np.fill_diagonal(graph,0)
    cost=D/(density[:,None]*density[None,:])**.125
    for i in range(K):
        for j in near[i]:graph[i,j]=graph[j,i]=cost[i,j]
    used={0}
    while len(used)<K:
        source=list(used);dest=[i for i in range(K) if i not in used]
        i,j=np.unravel_index(np.argmin(D[np.ix_(source,dest)]),(len(source),len(dest)))
        a,b=source[i],dest[j];graph[a,b]=graph[b,a]=cost[a,b];used.add(b)
    for k in range(K):graph=np.minimum(graph,graph[:,k,None]+graph[None,k,:])
    anchor=np.argsort(chord,axis=1)[:,:3]
    fermat=np.min(chord[np.arange(len(chord))[:,None],anchor][:,:,None]/density[anchor][:,:,None]**.25+graph[anchor],axis=1)
    trials=60;steps=60;budget=32;results=[];paired={};curves={};frozen={};scales={};frozen_pairs={}
    rng=np.random.default_rng(993)
    beliefs=rng.dirichlet(np.full(K,.3),size=100)
    for world,dist in [('chord',chord),('fermat_graph',fermat)]:
        scale=float(np.median(dist)*.25);scales[world]=scale;L=.05+.9/(1+(dist/scale)**2)
        # Frozen belief experiment: same posterior and draws for every method.
        frng=np.random.default_rng(119)
        observed={m:[] for m in METHODS};unique={m:[] for m in METHODS};regret={m:[] for m in METHODS}
        for p in beliefs:
            hs=np.searchsorted(np.cumsum(p),frng.random(budget));bs=frng.integers(bank_size,size=budget)
            best=float(ig(p,L).max()) # oracle over the shared finite union only
            for m in METHODS:
                ids=np.unique(banks[m][hs,bs]);value=float(ig(p,L[ids]).max())
                observed[m].append(value/np.log(2));unique[m].append(len(ids));regret[m].append((best-value)/np.log(2))
        frozen[world]={m:dict(best_ig_bits=stats(observed[m]),oracle_regret_bits=stats(regret[m]),mean_unique_candidates=float(np.mean(unique[m]))) for m in METHODS}
        frozen_pairs[world]={m:stats(np.array(observed['mppca'])-observed[m]) for m in METHODS if m!='mppca'}
        for method in METHODS:
            simulation_start=time.perf_counter()
            trialrows=[];trace=[]
            for trial in range(trials):
                gen=np.random.default_rng(4000+trial);truth=int(gen.integers(K));uniforms=gen.random(steps)
                p=np.full(K,1/K);seen=set();hit=None;likes=0;gains=[];history=[]
                for step in range(steps):
                    # Shared random numbers, but method-specific posterior after feedback.
                    qgen=np.random.default_rng(100000+trial*steps+step)
                    hs=np.searchsorted(np.cumsum(p),qgen.random(budget));bs=qgen.integers(bank_size,size=budget)
                    ids=np.unique(banks[method][hs,bs]);ids=np.array([i for i in ids if int(i) not in seen],int)
                    if len(ids)==0:
                        # Exhausted proposal draw consumes interaction budget but no feedback.
                        history.append(float(p[truth]));continue
                    values=ig(p,L[ids]);chosen=int(ids[np.argmax(values)]);seen.add(chosen)
                    positive=uniforms[step]<L[chosen,truth];likes+=positive
                    gains.append(float(values.max()/np.log(2)))
                    p*=L[chosen] if positive else 1-L[chosen];p/=p.sum()
                    if hit is None and p[truth]>=.5:hit=step+1
                    history.append(float(p[truth]))
                trialrows.append(dict(trial=trial,truth=truth,success=hit is not None,first_hit=hit,
                    censored_steps=hit if hit is not None else steps+1,final_true_mass=float(p[truth]),
                    map_correct=int(np.argmax(p))==truth,likes=int(likes),feedback_count=len(seen),mean_ig_bits=float(np.mean(gains)) if gains else 0))
                trace.append(history)
            paired[world,method]=trialrows;curves[world+'_'+method]=np.mean(trace,axis=0).tolist()
            results.append(dict(world=world,method=method,simulation_seconds=time.perf_counter()-simulation_start,success_rate=stats([r['success'] for r in trialrows]),
                capped_steps=stats([r['censored_steps'] for r in trialrows]),final_true_mass=stats([r['final_true_mass'] for r in trialrows]),
                map_accuracy=stats([r['map_correct'] for r in trialrows]),mean_feedback_count=float(np.mean([r['feedback_count'] for r in trialrows]))))
            print(world,method,'success',results[-1]['success_rate']['mean'],'final mass',results[-1]['final_true_mass']['mean'],flush=True)
    comparisons={}
    for world in ['chord','fermat_graph']:
        pp=paired[world,'mppca']
        comparisons[world]={m:stats([a['final_true_mass']-b['final_true_mass'] for a,b in zip(pp,paired[world,m])]) for m in METHODS if m!='mppca'}
    with (OUT/'trials.csv').open('w') as f:
        fields=['world','method']+list(next(iter(paired.values()))[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for (world,method),rows in paired.items():
            for r in rows:w.writerow(dict(world=world,method=method,**r))
    report=dict(seed=804,proposal_seed=819,trial_seed_base=4000,likelihood_scales=scales,
        model_sha256=hashlib.sha256((OUT/'models.json').read_bytes()).hexdigest(),
        frozen_paired_ig_difference_mppca_minus=frozen_pairs,
        input_manifest_sha256=hashlib.sha256((ROOT/'data/bluesky_100000/manifest.json').read_bytes()).hexdigest(),trials=trials,steps=steps,proposal_budget=budget,bank_per_region=bank_size,hypotheses=K,
        retrieved_corpus=len(X),distinct_candidate_union=len(allids),diagnostics=diagnostics,frozen=frozen,
        sequential=results,paired_final_mass_difference_mppca_minus=comparisons,curves=curves,seconds=time.perf_counter()-start)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(OUT/'banks.npz',post_rows=allids,target_rows=targets,**banks)
    lines=['Paired proposal experiment; exact vectors, fixed 32-proposal budget, 60 trials x 60 steps per method/world.',
        'Confidence interval: trial bootstrap; first-hit failures assigned 61 (capped score, not mean uncensored stopping time).',
        'world | method | success at 0.5 posterior mass | mean final true mass | MAP accuracy']
    for r in results:lines.append(f"{r['world']} | {r['method']} | {r['success_rate']['mean']:.3f} | {r['final_true_mass']['mean']:.3f} | {r['map_accuracy']['mean']:.3f}")
    assert all(np.isfinite(v).all() for v in [chord,fermat])
    assert not np.isin(allids,targets).any()
    assert all(0<=r['final_true_mass']<=1 for rows in paired.values() for r in rows)
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('Completed in',report['seconds'],'seconds',flush=True)

if __name__=='__main__':main()
