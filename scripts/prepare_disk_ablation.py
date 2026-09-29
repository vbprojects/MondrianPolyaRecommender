"""Fixed GTE-tiny Float32 reference embeddings for the disk codec ablation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'TaylorAI/gte-tiny'
REVISION = '4cc5e73d86a67c601897257b467187234aa3bca3'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--keywords', action='store_true', help='Prepare versioned YAKE sidecar alongside embeddings')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--output', type=Path, default=ROOT/'data/gte_disk_ablation')
    args = p.parse_args()
    assert 0 < args.count <= 100000 and args.batch_size > 0
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'data/bluesky_100000/posts.jsonl'
    contract = dict(model=MODEL, revision=REVISION, max_length=256, dimension=384,
                    count=args.count, source_sha256=sha(source))
    if (out/'manifest.json').exists():
        previous = json.loads((out/'manifest.json').read_text())
        if previous['contract'] == contract and all(sha(out/k)==v for k,v in previous['sha256'].items()):
            if args.keywords:
                from prepare_keywords import prepare
                prepare(out/'posts.jsonl',out/'keywords')
            print('Verified existing GTE reference:', out); return
        raise RuntimeError('Existing output has a different or damaged contract; use a new --output')
    posts = [json.loads(l) for l in source.open()][:args.count]
    assert len(posts)==args.count and len({x['uri'] for x in posts})==args.count
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).float().to(args.device).eval()

    @torch.inference_mode()
    def encode(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(args.device)
        output = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).to(output.dtype)
        z = (output*mask).sum(1)/mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(z, p=2, dim=1).cpu().numpy().astype('<f4')

    encode([p['text'] for p in posts[:args.batch_size]])
    path = out/'embeddings.f32.tmp'; start = time.perf_counter()
    with path.open('wb') as f:
        for i in range(0, len(posts), args.batch_size):
            X = encode([p['text'] for p in posts[i:i+args.batch_size]])
            assert X.shape[1]==384 and np.isfinite(X).all()
            assert np.max(abs(np.linalg.norm(X,axis=1)-1))<2e-6
            X.tofile(f)
            if i % (args.batch_size*100)==0:
                print(f'{i+len(X):,}/{len(posts):,}', flush=True)
    elapsed = time.perf_counter()-start
    path.replace(out/'embeddings.f32')
    X = np.memmap(out/'embeddings.f32', dtype='<f4', mode='r', shape=(len(posts),384))
    one = np.vstack([encode([p['text']]) for p in posts[:8]])
    batching_error = float(np.max(abs(one-X[:8])))
    assert batching_error<2e-5
    if args.device=='cuda':
        model.to('cpu'); args.device='cpu'
        cpu_error=float(np.max(abs(encode([p['text'] for p in posts[:8]])-X[:8])))
        assert cpu_error<2e-5
    else:
        cpu_error=0.
    # The original chronological anomalies remain metadata, not the experiment clock.
    with (out/'posts.jsonl').open('w') as f:
        for i,p in enumerate(posts):
            f.write(json.dumps(dict(id=i+1,uri=p['uri'],text=p['text'],author=p.get('author',''),
                                   source_created_at=p.get('created_at','')),ensure_ascii=False)+'\n')
    manifest=dict(contract=contract,encode_seconds=elapsed,batch_size=args.batch_size,
                  singleton_max_error=batching_error,cpu_max_error=cpu_error,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
                  sha256={k:sha(out/k) for k in ['posts.jsonl','embeddings.f32']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.keywords:
        from prepare_keywords import prepare
        prepare(out/'posts.jsonl',out/'keywords')
    print('GTE reference validated:',json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
