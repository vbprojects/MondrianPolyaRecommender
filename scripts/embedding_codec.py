"""Frozen-codebook codec benchmark; invoked by Julia. No original data changed."""
import json
import hashlib
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/embedding_codec'

def quantize(values, groups, bits, label):
    # Symmetric ranges calibrated on the corpus being encoded; offline codec.
    k=int(groups.max())+1; dims=values.shape[1]
    scales=np.zeros((k,dims),np.float32)
    np.maximum.at(scales,groups,np.abs(values))
    scales=np.maximum(scales,1e-9)
    levels=(1<<bits)-1
    codes=np.rint((values/scales[groups]+1)*(.5*levels)).clip(0,levels).astype(np.uint8)
    flat=codes.ravel(); per=8//bits
    padded=np.pad(flat,(0,(-len(flat))%per)).reshape(-1,per)
    packed=np.zeros(len(padded),np.uint8)
    for j in range(per): packed|=padded[:,j]<<(j*bits)
    path=OUT/(label+'.codes');packed.tofile(path)
    scales.tofile(OUT/(label+'.scales.f32'))
    # Decode from actual serialized payload rather than uncompressed codes.
    stored=np.fromfile(path,np.uint8)
    recovered=np.stack([(stored>>(j*bits))&levels for j in range(per)],axis=1).ravel()[:len(flat)].reshape(codes.shape)
    assert np.array_equal(recovered,codes)
    decoded=(recovered.astype(np.float32)*(2/levels)-1)*scales[groups]
    return decoded, {'codes':path.stat().st_size,'scales':scales.nbytes}

def decode(name, rows=None):
    """Decode a saved variant, optionally only selected zero-based post rows.
    Returned vectors are raw reconstructions; normalize for cosine scoring.
    """
    all_groups=np.fromfile(OUT/'components.u16',np.uint16)
    n=len(all_groups)
    rows=np.arange(n) if rows is None else np.asarray(rows,dtype=np.int64)
    if np.any(rows<0) or np.any(rows>=n): raise ValueError('row outside archive')
    groups=all_groups[rows]
    means=np.fromfile(OUT/'means.f32',np.float32).reshape(-1,256)
    def unpack(label,bits,dims):
        scales=np.fromfile(OUT/(label+'.scales.f32'),np.float32).reshape(-1,dims)
        packed=np.memmap(OUT/(label+'.codes'),dtype=np.uint8,mode='r')
        offsets=rows[:,None]*dims+np.arange(dims)
        per=8//bits; levels=(1<<bits)-1
        code=(packed[offsets//per] >> ((offsets%per)*bits))&levels
        return (code.astype(np.float32)*(2/levels)-1)*scales[groups]
    if name.startswith('centroid_residual_'):
        bits=int(name.rsplit('_',1)[1].replace('bit',''))
        return means[groups]+unpack(f'centroid_{bits}bit',bits,256)
    basis=np.fromfile(OUT/'bases.f32',np.float32).reshape(-1,4,256)
    if name.startswith('ppca_rank4_latent8_tail'):
        bits=int(name.rsplit('tail',1)[1]); z=unpack('ppca_latent_8bit',8,4)
        return means[groups]+np.einsum('nrd,nr->nd',basis[groups],z)+unpack(f'ppca_tail_{bits}bit',bits,256)
    if name.startswith('ppca_rank4_latent_'):
        bits=int(name.rsplit('_',1)[1].replace('bit','')); z=unpack(f'ppca_latent_{bits}bit',bits,4)
        return means[groups]+np.einsum('nrd,nr->nd',basis[groups],z)
    raise ValueError('unknown codec')

def norm(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def entropy(p):
    p=np.clip(p,1e-8,1-1e-8)
    return -p*np.log(p)-(1-p)*np.log1p(-p)

def main():
    X=np.load(ROOT/'data/bluesky_100000/embeddings.npy')
    n,d=X.shape
    means=np.fromfile(OUT/'means.f32',np.float32).reshape(-1,d)
    basis=np.fromfile(OUT/'bases.f32',np.float32).reshape(-1,4,d)
    groups=np.fromfile(OUT/'components.u16',np.uint16)
    assert len(groups)==n and basis.shape[0]==len(means)
    base=means[groups]; residual=X-base
    latent=np.einsum('nrd,nd->nr',basis[groups],residual)
    rng=np.random.default_rng(49)
    queries=rng.choice(n,100,replace=False)
    exact=X[queries]@X.T
    exact[np.arange(len(queries)),queries]=-np.inf
    top=np.argpartition(-exact,10,axis=1)[:,:10]
    oracle=np.argmax(exact,axis=1)
    # Identical synthetic interest posteriors and proposal pools across codecs.
    sessions=[]
    for _ in range(50):
        h=rng.choice(n,128,replace=False); candidates=rng.choice(n,256,replace=False)
        p=rng.dirichlet(np.full(128,.3))
        L=.05+.9/(1+np.maximum(0,2-2*(X[candidates]@X[h].T))/.25)
        ig=entropy(L@p)-entropy(L)@p
        sessions.append((h,candidates,p,ig))
    common=sum((OUT/f).stat().st_size for f in ['means.f32','owners.i32','components.u16','post_ids.u64'])
    results=[]
    def evaluate(name,Y,files,extra):
        checkrows=np.arange(0,n,97)
        np.testing.assert_allclose(decode(name,checkrows),Y[checkrows],atol=1e-7,rtol=1e-6)
        before=Y.copy(); Y=norm(Y)
        scores=X[queries]@Y.T;scores[np.arange(len(queries)),queries]=-np.inf
        approx=np.argpartition(-scores,10,axis=1)[:,:10]
        recall=np.mean([len(set(a)&set(b))/10 for a,b in zip(top,approx)])
        chosen=np.argmax(scores,axis=1)
        regrets=[]; agrees=[]
        for h,c,p,ig in sessions:
            L=.05+.9/(1+np.maximum(0,2-2*(Y[c]@X[h].T))/.25)
            gain=entropy(L@p)-entropy(L)@p
            pick=int(np.argmax(gain));regrets.append(float(ig.max()-ig[pick]));agrees.append(pick==int(np.argmax(ig)))
        total=common+sum(files.values())+extra
        results.append(dict(name=name,total_bytes=total,bytes_per_post=total/n,compression=(X.nbytes+8*n)/total,
            rmse=float(np.sqrt(np.mean((before-X)**2))),mean_cosine=float(np.mean(np.sum(Y*X,axis=1))),
            recall10=float(recall),top1_agreement=float(np.mean(chosen==oracle)),
            mean_top1_cosine_regret=float(np.mean(exact[np.arange(100),oracle]-exact[np.arange(100),chosen])),
            ig_choice_agreement=float(np.mean(agrees)),mean_ig_regret_bits=float(np.mean(regrets)/np.log(2))))
        print(json.dumps(results[-1]),flush=True)
    for bits in (1,2,4,8):
        dec,files=quantize(residual,groups,bits,f'centroid_{bits}bit')
        evaluate(f'centroid_residual_{bits}bit',base+dec,files,0)
    for bits in (4,8):
        z,files=quantize(latent,groups,bits,f'ppca_latent_{bits}bit')
        low=base+np.einsum('nrd,nr->nd',basis[groups],z)
        evaluate(f'ppca_rank4_latent_{bits}bit',low,files,basis.nbytes)
        # Optional residual captures variation that rank-4 PPCA cannot retain.
        if bits==8:
            for tailbits in (1,2,4):
                tail,tailfiles=quantize(X-low,groups,tailbits,f'ppca_tail_{tailbits}bit')
                evaluate(f'ppca_rank4_latent8_tail{tailbits}',low+tail,
                         {**files,**{'tail_'+k:v for k,v in tailfiles.items()}},basis.nbytes)
    # Equal budget ceilings; no padding or artificially discarding byte savings.
    budgets=[]
    for mib in (8,12,20,36):
        item={'budget_mib':mib}
        for family,prefix in [('centroid','centroid'),('ppca','ppca')]:
            eligible=[r for r in results if r['name'].startswith(prefix) and r['total_bytes']<=mib*2**20]
            # Report all feasible configs, do not tune selection on evaluation scores.
            item[family]=[r['name'] for r in eligible]
        budgets.append(item)
    report={'count':n,'dimensions':d,'components':len(means),'rank':4,'results':results,'budget_options':budgets,
        'scope':'Transductive codec benchmark: existing tree/codebook and quantizer ranges use this corpus. Query posts also come from this corpus; self retrieval excluded. Not unseen-data generalization.',
        'shared_assignment':'Both families use identical nearest-centroid assignments among component owners on original tree paths; not original soft PPCA responsibilities or the previous standalone microcluster fit.',
        'storage':'Includes Float32 means/scales/bases as applicable, UInt16 component ID, UInt64 post ID, Int32 component owner. Excludes shared tree, text, file-system overhead and JSON metadata. Model training scatter/noise not required by decoder.',
        'evaluation':'100 fixed exact queries against all 100000 reconstructed posts; 50 fixed synthetic posteriors, 128 exact hypotheses and 256 decoded candidates. Cosine-distance binary likelihood, not previous Fermat graph. Normalized reconstructions for cosine scoring; RMSE before normalization.',
        'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.suffix in ('.codes','.f32','.u16','.u64','.i32')}}
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    lines=['Frozen local embedding codec comparison','method | bytes/post (incl shared decoder and IDs) | recall@10 | mean cosine | IG agreement | IG regret bits']
    for r in results: lines.append(f"{r['name']} | {r['bytes_per_post']:.2f} | {r['recall10']:.3f} | {r['mean_cosine']:.4f} | {r['ig_choice_agreement']:.2f} | {r['mean_ig_regret_bits']:.6f}")
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
