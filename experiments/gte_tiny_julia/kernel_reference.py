"""Matched ONNX CPU/CUDA reference for the Julia kernel experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from collections import Counter

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from prepare import DEFAULT, get, encode, sha, QUANT_SHA, FLOAT_SHA, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, default=DEFAULT)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    out = args.artifacts
    tokenizer = Tokenizer.from_file(str(out/'model/tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    panel = json.loads((out/'panel.json').read_text())
    paths = [('int8', get('onnx/model_quantized.onnx'), QUANT_SHA),
             ('fp32', get('onnx/model.onnx'), FLOAT_SHA)]
    timings = []
    for device in (['cpu'] if args.cpu_only else ['cpu', 'cuda']):
        if device == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
            continue
        if device == 'cuda':
            ort.preload_dlls()
        for precision, path, checksum in paths:
            assert sha(path) == checksum
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = args.threads
            opts.inter_op_num_threads = 1
            providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess = ort.InferenceSession(str(path), opts, providers=providers)
            if device == 'cuda' and sess.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError('CUDA provider failed to initialize')
            for batch in [1, 8]:
                encode(sess, tokenizer, panel, batch)
                samples = []
                for _ in range(5):
                    start = time.perf_counter()
                    encode(sess, tokenizer, panel, batch)
                    samples.append(time.perf_counter()-start)
                timings.append(dict(backend=f'onnx_{precision}_{device}', batch=batch,
                    samples_seconds=samples, texts_per_second=len(panel)/float(np.median(samples))))
                print(timings[-1]['backend'], batch, timings[-1]['texts_per_second'], flush=True)
            del sess
            # Separate instrumented execution: profiling overhead never enters timing.
            opts.enable_profiling = True
            opts.profile_file_prefix = str(out / f'profile-{precision}-{device}')
            sess = ort.InferenceSession(str(path), opts, providers=providers)
            encode(sess, tokenizer, panel[:2], 2)
            trace = json.loads(Path(sess.end_profiling()).read_text())
            assignments = Counter(e.get('args', {}).get('provider') for e in trace
                                  if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
            for row in timings:
                if row['backend'] == f'onnx_{precision}_{device}':
                    row['profiled_node_provider_counts'] = dict(assignments)
            del sess
    source = Path(__file__).read_bytes()
    write_json(out/'kernel-reference.json', dict(threads=args.threads, ort=ort.__version__,
        panel_sha256=sha(out/'panel.json'), source_sha256=hashlib.sha256(source).hexdigest(), timings=timings))


if __name__ == '__main__':
    main()
