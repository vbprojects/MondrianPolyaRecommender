"""Pinned public-data preview and ONNX bridge; invoked by examples/embed_bluesky.jl."""
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATASET = 'withalim/bluesky-posts'
DATASET_REV = '45b2e60e2bf6ec6fb21db1fe4a5346b64bcf96f1'
MODEL = 'futur/Qwen3-Embedding-0.6B-model2vec-onnx'
MODEL_REV = 'a3f60a85a2570e7976ea7f2d6505ad3e5152a96f'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    # File iteration splits on actual record newlines, unlike str.splitlines(),
    # which also splits literal Unicode separators inside valid JSON strings.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fetch_prefix(number, prefix_records=1000):
    name = f'posts_{number:03d}.jsonl'
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REV}/{name}'
    cache_dir = ROOT / '.cache' / 'bluesky' / DATASET_REV
    if prefix_records != 1000:
        cache_dir /= f'prefix_{prefix_records}'
    cache = cache_dir / name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        # Bounded shard prefix; no full corpus download.
        for attempt in range(3):
            try:
                records = []
                with urllib.request.urlopen(url, timeout=60) as response:
                    for line in response:
                        if not line.strip():
                            continue
                        json.loads(line)  # Never execute or interpret post content as instructions.
                        records.append(line.rstrip(b'\r\n') + b'\n')
                        if len(records) == prefix_records:
                            break
                if len(records) != prefix_records:
                    raise ValueError(f'Short shard prefix: {name}')
                temp = cache.with_suffix('.tmp')
                temp.write_bytes(b''.join(records))
                temp.replace(cache)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt+1)
    records = read_jsonl(cache)
    if len(records) != prefix_records:
        raise ValueError(f'Invalid prefix cache: {cache}')
    return name, records, {'file': name, 'records': len(records), 'prefix_sha256': sha(cache), 'url': url}


def sample_posts(count, seed, prefix_records=1000):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        prefixes = list(pool.map(lambda n: fetch_prefix(n, prefix_records), range(1, 21)))
    eligible = []
    seen = set()
    skipped = {'empty_text': 0, 'missing_uri': 0, 'duplicate_uri': 0}
    for name, rows, _ in prefixes:
        for row_number, row in enumerate(rows, 1):
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                skipped['empty_text'] += 1
                continue
            uri = row.get('uri')
            if not uri:
                skipped['missing_uri'] += 1
                continue
            if uri in seen:
                skipped['duplicate_uri'] += 1
                continue
            seen.add(uri)
            eligible.append(dict(row, source_file=name, source_record=row_number))
    if count > len(eligible):
        raise ValueError('Requested count exceeds eligible bounded sample pool')
    chosen = random.Random(seed).sample(eligible, count)
    for i, row in enumerate(chosen):
        row['embedding_row'] = i
    return chosen, {'prefixes': [p[2] for p in prefixes], 'scanned': 20 * prefix_records, 'records_per_shard': prefix_records,
                    'eligible': len(eligible), 'skipped': skipped, 'seed': seed,
                    'policy': f'Seeded sample without replacement from first {prefix_records} records of each of 20 shards; unique URI and nonblank text. Not uniform over the corpus.'}


def pack(ids):
    lengths = np.array([len(x) for x in ids], dtype=np.int64)
    if np.any(lengths == 0):
        raise ValueError('Empty token sequence')
    return {'input_ids': np.concatenate([np.asarray(x, np.int64) for x in ids]),
            'offsets': np.r_[np.int64(0), np.cumsum(lengths[:-1])].astype(np.int64)}


def make_session(model_path, device="auto", profile_prefix=None):
    available = ort.get_available_providers()
    cuda = device == "cuda" or (device == "auto" and "CUDAExecutionProvider" in available)
    if cuda:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA requested but unavailable; install the GPU requirements")
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, options, providers=providers)
    if cuda and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA initialization failed; refusing silent CPU-only fallback")
    return session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--prefix-records', type=int, help='Records per shard; default max(1000, ceil(count/10))')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'bluesky_1000')
    args = parser.parse_args()
    if args.count <= 0 or args.batch_size <= 0:
        parser.error('count and batch-size must be positive')
    prefix_records = args.prefix_records if args.prefix_records is not None else max(1000, (args.count+9)//10)
    if prefix_records <= 0:
        parser.error('prefix-records must be positive')
    pipeline_start = time.perf_counter()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('Sampling pinned dataset shard prefixes...', flush=True)
    posts, sampling = sample_posts(args.count, args.seed, prefix_records)
    print(f'Selected {len(posts)} posts from {sampling["eligible"]} eligible records.', flush=True)
    paths = {name: hf_hub_download(MODEL, name, revision=MODEL_REV)
             for name in ['onnx/model.onnx', 'tokenizer.json', 'config.json']}
    config = json.loads(Path(paths['config.json']).read_text())
    tokenizer = Tokenizer.from_file(paths['tokenizer.json'])
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenize_start = time.perf_counter()
    token_ids = [e.ids for e in tokenizer.encode_batch([r['text'] for r in posts], add_special_tokens=False)]
    tokenize_seconds = time.perf_counter()-tokenize_start
    print(f'Tokenized {len(posts)} posts in {tokenize_seconds:.2f}s; starting {args.device} inference.', flush=True)
    session = make_session(paths['onnx/model.onnx'], args.device)
    if [x.name for x in session.get_inputs()] != ['input_ids', 'offsets']:
        raise ValueError('Unexpected ONNX input contract')
    session.run(['embeddings'], pack(token_ids[:args.batch_size]))  # warmup excluded
    start = time.perf_counter()
    raw = np.empty((len(posts), config['hidden_dim']), dtype=np.float32)
    for i in range(0, len(posts), args.batch_size):
        stop = min(i+args.batch_size, len(posts))
        raw[i:stop] = session.run(['embeddings'], pack(token_ids[i:stop]))[0]
        if (i//args.batch_size+1) % 200 == 0:
            print(f'Embedded {stop}/{len(posts)} posts', flush=True)
    encode_seconds = time.perf_counter()-start
    if raw.shape != (args.count, config['hidden_dim']) or not np.isfinite(raw).all():
        raise ValueError('Invalid embedding output')
    norms = np.linalg.norm(raw, axis=1)
    if np.any(norms <= 0):
        raise ValueError('Zero embeddings')
    # ONNX returns FP16 normalized embeddings. Renormalize after converting to
    # Float32 so downstream dot products have a consistent unit-length contract.
    embeddings = np.asarray(raw / norms[:, None], dtype='<f4')
    # Validate batching against singleton inference.
    singleton = np.concatenate([session.run(['embeddings'], pack([ids]))[0] for ids in token_ids[:8]]).astype(np.float32)
    batch_error = float(np.max(np.abs(singleton-raw[:8])))
    if batch_error > .002:
        raise ValueError('Batch/singleton disagreement')
    # Independent pooling check from exported embedding weights, not another
    # call through the same ONNX graph. SIF/PCA are already in these weights.
    graph = onnx.load(paths['onnx/model.onnx'])
    tables = [onnx.numpy_helper.to_array(x) for x in graph.graph.initializer
              if len(x.dims) == 2 and x.dims[1] == config['hidden_dim']]
    if len(tables) != 1:
        raise ValueError('Cannot identify static embedding table for verification')
    table = tables[0]
    reference = np.stack([table[ids].astype(np.float32).mean(axis=0) for ids in token_ids[:8]])
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    pooling_error = float(np.max(np.abs(reference-embeddings[:8])))
    if pooling_error > .003:
        raise ValueError(f'Pooling mismatch: {pooling_error}')
    for row, ids in zip(posts, token_ids):
        row['token_count'] = len(ids)
    post_path = out/'posts.jsonl'
    post_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in posts), encoding='utf-8')
    np.save(out/'embeddings.npy', embeddings, allow_pickle=False)
    embeddings.tofile(out/'embeddings.f32')  # C rows == Julia dimensions×posts columns
    (out/'shape.txt').write_text(f'{len(posts)} {embeddings.shape[1]}\n')
    # The small demo inspector uses exact all-pairs neighbors. Cap its size:
    # 100k posts would require a 40 GB similarity matrix before sorting.
    artifacts = ['posts.jsonl', 'embeddings.npy', 'embeddings.f32', 'shape.txt']
    if len(posts) <= 5000:
        similarities = embeddings @ embeddings.T
        np.fill_diagonal(similarities, -np.inf)
        nearest = np.argsort(-similarities, axis=1)[:, :min(5, len(posts)-1)]
        neighbors = [{'row': i, 'neighbors': [{'row': int(j), 'cosine': float(similarities[i,j])} for j in nearest[i]]} for i in range(len(posts))]
        (out/'neighbors.json').write_text(json.dumps(neighbors), encoding='utf-8')
        viewer = (ROOT/'examples'/'bluesky_viewer.html').read_text()
        data = json.dumps({'posts': posts, 'neighbors': neighbors, 'dimensions': embeddings.shape[1], 'provider': session.get_providers()[0]}, ensure_ascii=True).replace('<', '\\u003c')
        (out/'inspect.html').write_text(viewer.replace('__DATA__', data), encoding='utf-8')
        artifacts += ['neighbors.json', 'inspect.html']
    else:
        for stale in ['neighbors.json', 'inspect.html']:
            (out/stale).unlink(missing_ok=True)
    years = [r.get('created_at', '')[:4] for r in posts]
    manifest = {'dataset': DATASET, 'dataset_revision': DATASET_REV, 'model': MODEL,
                'model_revision': MODEL_REV, 'sampling': sampling, 'count': len(posts),
                'dimensions': embeddings.shape[1], 'dtype': 'little-endian float32',
                'layout': 'post-major in embeddings.f32; Julia reshape as (dimensions, posts)',
                'text_processing': 'Original text; no language filtering, prefix, truncation, padding or special-token insertion.',
                'normalization': 'ONNX FP16 output cast to Float32 and L2-renormalized',
                'raw_onnx_norm_range': [float(norms.min()), float(norms.max())],
                'normalized_norm_max_error': float(np.max(np.abs(np.linalg.norm(embeddings, axis=1)-1))),
                'batch_singleton_max_abs_error': batch_error, 'independent_pooling_max_abs_error': pooling_error,
                'tokenization_seconds': tokenize_seconds, 'pipeline_seconds_before_manifest': time.perf_counter()-pipeline_start,
                'inspector_generated': len(posts) <= 5000,
                'batch_encoding_seconds': encode_seconds, 'batch_size': args.batch_size,
                'warmup_batches_excluded': 1, 'requested_device': args.device, 'max_tokens': max(map(len, token_ids)),
                'duplicate_text_count': len(posts)-len({r['text'] for r in posts}),
                'created_at_year_counts': {y: years.count(y) for y in sorted(set(years))},
                'providers': session.get_providers(), 'onnxruntime_version': ort.__version__,
                'versions': {p: importlib.metadata.version(p) for p in ['numpy','onnx','tokenizers','huggingface_hub']},
                'model_file_sha256': {n: sha(p) for n,p in paths.items()},
                'artifact_sha256': {n: sha(out/n) for n in artifacts}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['count','dimensions','batch_encoding_seconds','max_tokens','duplicate_text_count','created_at_year_counts','batch_singleton_max_abs_error','independent_pooling_max_abs_error']}, indent=2), flush=True)
    print(f'Artifacts: {out}', flush=True)


if __name__ == '__main__':
    main()
