"""Bounded-memory, two-pass sampling and GPU embedding of pinned shard prefixes."""
import json, random, time
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from embed_bluesky import ROOT, DATASET, DATASET_REV, MODEL, MODEL_REV, fetch_prefix, make_session, pack, sha


def eligible(paths):
    seen = set()
    for path in paths:
        with path.open() as f:
            for number, line in enumerate(f, 1):
                row = json.loads(line)
                uri = row.get('uri')
                if not uri or uri in seen or not isinstance(row.get('text'), str) or not row['text'].strip():
                    continue
                seen.add(uri)
                yield dict(row, source_file=path.name, source_record=number)


def main():
    started = time.perf_counter()
    count, prefix, batch_size = 1_000_000, 60_000, 256
    out = ROOT/'data/bluesky_1000000'; out.mkdir(parents=True, exist_ok=True)
    paths, provenance = [], []
    for n in range(1, 21):
        name, rows, info = fetch_prefix(n, prefix)
        del rows
        paths.append(ROOT/'.cache/bluesky'/DATASET_REV/f'prefix_{prefix}'/name)
        provenance.append(info)
        print(f'Cached shard {n}/20', flush=True)
    available = sum(1 for _ in eligible(paths))
    if available < count: raise ValueError(f'Only {available} eligible posts')
    chosen = set(random.Random(42).sample(range(available), count))
    model = hf_hub_download(MODEL, 'onnx/model.onnx', revision=MODEL_REV)
    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, 'tokenizer.json', revision=MODEL_REV))
    tokenizer.no_padding(); tokenizer.no_truncation()
    session = make_session(model, 'cuda')
    session.run(['embeddings'], pack([tokenizer.encode('warm up', add_special_tokens=False).ids]))
    result = np.lib.format.open_memmap(out/'embeddings.npy', mode='w+', dtype='<f4', shape=(count,256))
    timing = {'tokenization_seconds':0., 'batch_encoding_seconds':0.}
    cursor = 0; batch = []; max_norm_error = 0.; singleton_error = 0.
    with (out/'posts.jsonl').open('w') as posts, (out/'embeddings.f32').open('wb') as rawfile:
        def flush():
            nonlocal cursor, max_norm_error, singleton_error
            t=time.perf_counter(); ids=[e.ids for e in tokenizer.encode_batch([r['text'] for r in batch], add_special_tokens=False)]
            timing['tokenization_seconds']+=time.perf_counter()-t
            t=time.perf_counter(); raw=session.run(['embeddings'],pack(ids))[0].astype(np.float32)
            timing['batch_encoding_seconds']+=time.perf_counter()-t
            if cursor==0:
                single=np.concatenate([session.run(['embeddings'],pack([x]))[0] for x in ids[:8]])
                singleton_error=float(np.max(np.abs(single-raw[:8])))
                assert singleton_error < .002
            norms=np.linalg.norm(raw,axis=1)
            assert np.isfinite(raw).all() and (norms>0).all()
            vectors=np.asarray(raw/norms[:,None],dtype='<f4')
            max_norm_error=max(max_norm_error,float(np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))))
            result[cursor:cursor+len(batch)]=vectors; vectors.tofile(rawfile)
            for j,(row,tokens) in enumerate(zip(batch,ids)):
                row.update(embedding_row=cursor+j,token_count=len(tokens)); posts.write(json.dumps(row,ensure_ascii=False)+'\n')
            cursor+=len(batch); batch.clear()
            if cursor%25600==0: print(f'Embedded {cursor}/{count}',flush=True)
        for i,row in enumerate(eligible(paths)):
            if i not in chosen: continue
            batch.append(row)
            if len(batch)==batch_size: flush()
        if batch: flush()
    assert cursor==count
    result.flush(); (out/'shape.txt').write_text(f'{count} 256\n')
    manifest=dict(dataset=DATASET,dataset_revision=DATASET_REV,model=MODEL,model_revision=MODEL_REV,count=count,dimensions=256,
        sampling=dict(seed=42,scanned=prefix*20,eligible=available,prefixes=provenance,policy='Uniform without replacement over eligible unique-URI nonblank posts in bounded shard prefixes, emitted in shard order; not uniform over full corpus.'),
        providers=session.get_providers(),batch_size=batch_size,**timing,normalized_norm_max_error=max_norm_error,batch_singleton_max_abs_error=singleton_error,
        pipeline_seconds=time.perf_counter()-started,artifact_sha256={n:sha(out/n) for n in ['posts.jsonl','embeddings.npy','embeddings.f32','shape.txt']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__': main()
