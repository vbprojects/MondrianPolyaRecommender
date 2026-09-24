"""Follow-up: identical MPPCA proposals, retrieval in component-local PPCA charts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json,time,csv,hashlib
from pathlib import Path
import numpy as np
from proposal_experiment import ent,ig,stats
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'output/proposal_experiment'
OUT=ROOT/'output/proposal_knn'
METHODS=['mppca','latent_1nn','latent_5nn','ppca_metric_1nn','local_latent_5nn','local_original_1nn','local_real_posts']

def latent_coordinates(points,mean,basis,variance,noise):
    # E[z|x] under x=mean+Wz+epsilon, W=U sqrt(lambda-noise).
    return ((points-mean)@basis.T)*(np.sqrt(np.maximum(0,variance-noise))/variance)

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    model=json.loads((PREVIOUS/'models.json').read_text())
    previous=json.loads((PREVIOUS/'results.json').read_text())
    X=np.load(ROOT/'data/bluesky_100000/embeddings.npy');targets=np.array(model['rows']);H=X[targets];K=len(H)
    lo=np.array(model['bounds'],np.float32)[:,:256];hi=np.array(model['bounds'],np.float32)[:,256:]
    mu=np.array(model['means'],np.float32);B=np.array(model['bases'],np.float32)
    var=np.array(model['variances'],np.float32);noise=np.array(model['noise'],np.float32);mass=np.array(model['mass'])
    rng=np.random.default_rng(819);bank_size=64;raw=[];component_ids=[]
    for h in range(K):
        ids=np.array(model['components'][h]);ids=rng.choice(ids,bank_size,p=mass[ids]/mass[ids].sum())
        eps=rng.normal(size=(bank_size,256));z=rng.normal(size=(bank_size,4))
        gauss=mu[ids]+np.sqrt(noise[ids,None])*eps+np.einsum('nrd,nr->nd',B[ids],z*np.sqrt(np.maximum(0,var[ids]-noise[ids,None])))
        rng.uniform(lo[h],hi[h],size=(bank_size,256)) # preserve previous proposal RNG stream
        raw.append(gauss);component_ids.extend(ids)
    raw=np.asarray(raw,np.float32);P=np.clip(raw,lo[:,None],np.nextafter(hi[:,None],lo[:,None])).reshape(-1,256)
    component_ids=np.array(component_ids);old=np.load(PREVIOUS/'banks.npz')
    assert np.array_equal(targets,old['target_rows'])
    banks={m:old['post_rows'][old[m]] for m in ('mppca','local_real_posts')}
    for m in METHODS[1:-1]:banks[m]=np.empty((K,bank_size),int)
    diag={m:[] for m in METHODS[1:4]};mapping_start=time.perf_counter()
    X64=X.astype(np.float64);xnorm=np.sum(X64**2,axis=1)
    choose=np.random.default_rng(991).integers(5,size=len(P))
    eligibility=[]
    for h in range(K):
        ok=np.all((X>=lo[h])&(X<hi[h]),axis=1);ok[targets]=False
        eligibility.append(np.flatnonzero(ok))
        assert ok.sum()>=5
    for component in np.unique(component_ids):
        positions=np.flatnonzero(component_ids==component);proposals=P[positions].astype(np.float64)
        basis=B[component].astype(np.float64);mean=mu[component].astype(np.float64)
        variance=var[component].astype(np.float64);sigma=float(noise[component])
        Z=latent_coordinates(X64,mean,basis,variance,sigma)
        Q=latent_coordinates(proposals,mean,basis,variance,sigma)
        d=np.maximum(0,np.sum(Q*Q,axis=1)[:,None]+np.sum(Z*Z,axis=1)[None,:]-2*Q@Z.T)
        d[:,targets]=np.inf
        nearest=np.argpartition(d,4,axis=1)[:,:5]
        for j,pos in enumerate(positions):
            ids=nearest[j];ids=ids[np.lexsort((ids,d[j,ids]))]
            banks['latent_1nn'].ravel()[pos]=ids[0]
            banks['latent_5nn'].ravel()[pos]=ids[choose[pos]]
            eligible=eligibility[pos//bank_size]
            local=eligible[np.argpartition(d[j,eligible],4)[:5]]
            local=local[np.lexsort((local,d[j,local]))]
            banks['local_latent_5nn'].ravel()[pos]=local[choose[pos]]
        # Full PPCA Mahalanobis distance: latent differences plus off-subspace
        # differences penalized by isotropic noise. This needs original vectors.
        euclidean=np.maximum(0,np.sum(proposals**2,axis=1)[:,None]+xnorm[None,:]-2*proposals@X64.T)
        for j,pos in enumerate(positions):
            eligible=eligibility[pos//bank_size]
            banks['local_original_1nn'].ravel()[pos]=eligible[np.argmin(euclidean[j,eligible])]
        projX=X64@basis.T;projQ=proposals@basis.T
        delta=projQ[:,None,:]-projX[None,:,:]
        orthogonal=np.maximum(0,euclidean-np.sum(delta**2,axis=2))
        mahal=orthogonal/sigma+np.sum(delta**2/variance,axis=2)
        mahal[:,targets]=np.inf
        banks['ppca_metric_1nn'].ravel()[positions]=np.argmin(mahal,axis=1)
    diagnostics={}
    for m,bank in banks.items():
        snapped=X[bank];distance=np.linalg.norm(P.reshape(K,bank_size,256)-snapped,axis=2)
        diagnostics[m]=dict(mean_snap_distance=float(distance.mean()),
            outside_before_clipping=float(np.mean(np.any((raw<lo[:,None])|(raw>hi[:,None]),axis=2))),
            retrieved_outside_owner=float(np.mean(np.any((snapped<lo[:,None])|(snapped>hi[:,None]),axis=2))),
            unique_posts=int(len(np.unique(bank))),mean_unique_per_region=float(np.mean([len(set(row)) for row in bank])))
    # Baseline proposals must reproduce the old embedding-space snapping.
    scores=P[::97]@X.T;scores[:,targets]=-np.inf
    assert np.array_equal(np.argmax(scores,axis=1),banks['mppca'].ravel()[::97])
    print('Mapped PPCA charts in',time.perf_counter()-mapping_start,'seconds',flush=True)
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
        scale=previous['likelihood_scales'][world];scales[world]=scale;L=.05+.9/(1+(dist/scale)**2)
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
    local_comparisons={world:{m:stats([a['final_true_mass']-b['final_true_mass'] for a,b in zip(paired[world,'local_latent_5nn'],paired[world,m])]) for m in ('mppca','local_original_1nn','local_real_posts')} for world in ('chord','fermat_graph')}
    with (OUT/'trials.csv').open('w') as f:
        fields=['world','method']+list(next(iter(paired.values()))[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for (world,method),rows in paired.items():
            for r in rows:w.writerow(dict(world=world,method=method,**r))
    report=dict(paired_local_latent_minus=local_comparisons,seed=804,proposal_seed=819,retrieval_k=[1,5],scale_source="previous frozen experiment",trial_seed_base=4000,likelihood_scales=scales,
        model_sha256=hashlib.sha256((PREVIOUS/'models.json').read_bytes()).hexdigest(),
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
