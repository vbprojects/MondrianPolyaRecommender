"""Compare warmed CPU/CUDA inference on the same saved posts, including transfers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from embed_bluesky import MODEL, MODEL_REV, ROOT, Tokenizer, hf_hub_download, make_session, np, pack, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'data/bluesky_1000_gpu')
    args = parser.parse_args()
    posts = read_jsonl(args.directory/'posts.jsonl')
    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, 'tokenizer.json', revision=MODEL_REV))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    ids = [e.ids for e in tokenizer.encode_batch([p['text'] for p in posts], add_special_tokens=False)]
    model = hf_hub_download(MODEL, 'onnx/model.onnx', revision=MODEL_REV)
    results = {'count': len(posts), 'batch_size': 64, 'repeats': 3,
               'timing_scope': 'Warmed inference including packing, host/device transfers and output concatenation; excludes tokenization, session initialization and normalization.'}
    outputs = {}
    for device in ['cpu', 'cuda']:
        session = make_session(model, device)
        session.run(['embeddings'], pack(ids[:64]))
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            outputs[device] = np.concatenate([session.run(['embeddings'], pack(ids[i:i+64]))[0]
                                             for i in range(0, len(ids), 64)]).astype(np.float32)
            elapsed.append(time.perf_counter()-start)
        results[device] = {'seconds': elapsed, 'median_seconds': statistics.median(elapsed),
                           'providers': session.get_providers()}
        del session
    # Profile separately so instrumentation does not distort benchmark timing.
    session = make_session(model, 'cuda', args.directory/'cuda_profile')
    session.run(['embeddings'], pack(ids[:64]))
    profile = session.end_profiling()
    events = json.loads(Path(profile).read_text())
    counts = Counter((e['args']['provider'], e['args'].get('op_name', ''))
                     for e in events if e.get('args', {}).get('provider'))
    if not any(provider == 'CUDAExecutionProvider' for provider, _ in counts):
        raise RuntimeError('No CUDA kernels recorded')
    results['profile_kernel_counts'] = [{'provider': p, 'operation': op, 'count': n}
                                       for (p, op), n in sorted(counts.items())]
    results['cpu_cuda_max_abs_error'] = float(np.max(np.abs(outputs['cpu']-outputs['cuda'])))
    assert np.allclose(outputs['cpu'], outputs['cuda'], atol=.002, rtol=.002)
    results['speedup'] = results['cpu']['median_seconds']/results['cuda']['median_seconds']
    (args.directory/'benchmark.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
