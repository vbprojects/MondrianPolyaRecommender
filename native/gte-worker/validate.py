"""Reference-only parity, protocol and matched CPU benchmark for the native worker."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3;onnx-int8-ceab9964bcdb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Worker:
    def __init__(self, binary, bundle, batch=8):
        start = time.perf_counter()
        self.process = subprocess.Popen([str(binary), '--bundle', str(bundle), '--batch-size', str(batch), '--threads', '2'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, 'PATH': '', 'PYTHONPATH': '', 'PYTHONHOME': ''})
        try:
            self.ready = json.loads(self.process.stdout.readline())
            assert self.ready['ready'] and self.ready['model'] == MODEL
        except BaseException:
            self.process.kill(); self.process.wait()
            raise
        self.startup = time.perf_counter()-start

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def encode(self, texts):
        result = self.request(dict(op='embed', texts=texts))
        assert result['model'] == MODEL and result['count'] == len(texts) and result['dimension'] == 384
        vectors = np.frombuffer(base64.b64decode(result['vectors'], validate=True), dtype='<f4').reshape(len(texts), 384)
        assert np.isfinite(vectors).all()
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)
        return vectors

    def close(self):
        try:
            assert self.request(dict(op='close')) == {'ok': True}
            assert self.process.wait(timeout=10) == 0
        finally:
            if self.process.poll() is None:
                self.process.kill(); self.process.wait()
            self.process.stdin.close(); self.process.stdout.close()


def reference(sess, tokenizer, texts, batch):
    chunks = []
    for start in range(0, len(texts), batch):
        enc = tokenizer.encode_batch(texts[start:start+batch])
        feed = {name: np.asarray([getattr(e, field) for e in enc], dtype=np.int64)
            for name, field in [('input_ids', 'ids'), ('attention_mask', 'attention_mask'), ('token_type_ids', 'type_ids')]}
        h = sess.run(['last_hidden_state'], feed)[0]
        mask = feed['attention_mask'][..., None]
        pooled = ((h*mask).sum(axis=1, dtype=np.float32)/mask.sum(axis=1)).astype(np.float32)
        chunks.append(pooled/np.linalg.norm(pooled, axis=1, keepdims=True))
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'native/gte-worker/target/release/mondrian-gte-worker')
    parser.add_argument('--bundle', type=Path, default=ROOT/'output/gte_tiny_rust')
    parser.add_argument('--fixtures', type=Path, default=ROOT/'output/gte_tiny_julia')
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.bundle/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.bundle/'model_quantized.onnx'), options, providers=['CPUExecutionProvider'])
    # Use the same runtime binary as Python, rather than silently compare ORT versions.
    ort_dir = Path(ort.__file__).parent/'capi'
    runtime = ort_dir/f'libonnxruntime.so.{ort.__version__}'
    assert sha(runtime) == sha(args.bundle/'libonnxruntime.so'), 'Python/native runtime binaries differ'
    w = Worker(args.binary.resolve(), args.bundle.resolve())
    report = dict(model=MODEL, ort=ort.__version__, worker=w.ready, startup_seconds=w.startup,
        binary_bytes=args.binary.stat().st_size, runtime_bytes=runtime.stat().st_size,
        binary_sha256=sha(args.binary), model_sha256=sha(args.bundle/'model_quantized.onnx'),
        tokenizer_sha256=sha(args.bundle/'tokenizer.json'), runtime_sha256=sha(runtime),
        source_sha256=sha(Path(__file__).parent/'src/main.rs'), cargo_lock_sha256=sha(Path(__file__).parent/'Cargo.lock'),
        validation_source_sha256=sha(__file__), threads=2, parity=[], timings=[], retrieval=[],
        fixture_sha256={name: sha(args.fixtures/name) for name in ['tokens.json', 'parity.json', 'retrieval.json', 'panel.json']})
    try:
        fixtures = json.loads((args.fixtures/'tokens.json').read_text())
        for start in range(0, len(fixtures), 32):
            rows = fixtures[start:start+32]
            tokens = w.request(dict(op='tokenize', texts=[r['text'] for r in rows]))
            for row, ids, mask in zip(rows, tokens['ids'], tokens['mask'], strict=True):
                assert [x for x, m in zip(ids, mask, strict=True) if m] == row['ids']
        report['tokenizer_cases'] = len(fixtures)
        cases = json.loads((args.fixtures/'parity.json').read_text())
        # Also exercise multiple chunks and unequal padding in one request.
        cases.append(dict(texts=['Short.', 'Unicode café 日本語 🦋', 'long '*400]*12))
        for case in cases:
            actual = w.encode(case['texts'])
            expected = reference(session, tokenizer, case['texts'], 8)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
            report['parity'].append(dict(texts=len(actual), max_abs=float(np.abs(actual-expected).max()),
                min_cosine=float(np.sum(actual*expected, axis=1).min())))
        repeated = w.encode(['Repeat.', 'Repeat.'])
        np.testing.assert_array_equal(repeated[0], repeated[1])
        np.testing.assert_array_equal(repeated, w.encode(['Repeat.', 'Repeat.']))
        for screen in json.loads((args.fixtures/'retrieval.json').read_text()):
            assert screen['batch'] == 8
            queries = w.encode(screen['queries']); docs = w.encode(screen['docs'])
            expected_q = np.asarray(screen['onnx_int8']['queries'], dtype=np.float32)
            expected_d = np.asarray(screen['onnx_int8']['docs'], dtype=np.float32)
            np.testing.assert_allclose(queries, expected_q, atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(docs, expected_d, atol=2e-6, rtol=2e-5)
            def ndcg(q, d):
                values = []
                for i, qid in enumerate(screen['qids']):
                    relevant = set(screen['relevant'][qid]); scores = (d*q[i]).sum(axis=1)
                    order = sorted(range(len(d)), key=lambda j: (-float(scores[j]), screen['cids'][j]))[:10]
                    dcg = sum((screen['cids'][j] in relevant)/np.log2(r+2) for r, j in enumerate(order))
                    ideal = sum(1/np.log2(r+2) for r in range(min(10, len(relevant))))
                    values.append(dcg/ideal)
                return values
            actual_scores = ndcg(queries, docs); expected_scores = ndcg(expected_q, expected_d)
            assert actual_scores == expected_scores
            report['retrieval'].append(dict(dataset=screen['name'], queries=len(queries), documents=len(docs),
                ndcg10=float(np.mean(actual_scores)), per_query=actual_scores,
                max_abs=float(max(np.abs(queries-expected_q).max(), np.abs(docs-expected_d).max()))))
        for req in [dict(op='unknown'), dict(op='embed', texts=[]), dict(op='embed', texts=[1]),
                    dict(op='embed', texts=['x']*1025), dict(op='embed', texts=['🦋'*16385]),
                    dict(op='verify_nostr', events=[{}]*1025)]:
            assert 'error' in w.request(req)
        w.process.stdin.write('{broken json\n'); w.process.stdin.flush()
        assert 'error' in json.loads(w.process.stdout.readline())
        w.encode(['Still alive after invalid requests.'])
        # Cross-language BIP340/JSON serialization checks, not only invalid signatures.
        from coincurve import PrivateKey
        key = PrivateKey(bytes.fromhex('01'*32))
        events = []
        for content in ['hello', '日本語 café 🦋\n\u0000\u2028', 'quote " slash / backslash \\']:
            event = dict(pubkey=key.public_key_xonly.format().hex(), created_at=1735689600, kind=1,
                         tags=[['t', 'julia']], content=content)
            payload = [0, event['pubkey'], event['created_at'], event['kind'], event['tags'], event['content']]
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).digest()
            event.update(id=digest.hex(), sig=key.sign_schnorr(digest, aux_randomness=bytes(32)).hex())
            events.append(event)
        assert w.request(dict(op='verify_nostr', events=events))['valid'] == [True]*3
        bad = [dict(events[0], content='tampered'), dict(events[0], sig='00'*64),
               dict(events[0], created_at=True), dict(events[0], kind=65536), dict(events[0], tags=[['x', 1]]), None]
        assert w.request(dict(op='verify_nostr', events=bad))['valid'] == [False]*len(bad)
        assert w.request(dict(op='verify_nostr', events=[]))['valid'] == []
        report['protocol_passed'] = True
    finally:
        w.close()
    # Startup must fail before ready for a corrupt pinned asset.
    with tempfile.TemporaryDirectory() as directory:
        bad_bundle = Path(directory)
        (bad_bundle/'model_quantized.onnx').symlink_to((args.bundle/'model_quantized.onnx').resolve())
        (bad_bundle/'tokenizer.json').write_text('{}')
        failed = subprocess.run([str(args.binary.resolve()), '--bundle', str(bad_bundle)],
            capture_output=True, text=True, timeout=30)
        assert failed.returncode != 0 and not failed.stdout and 'checksum mismatch' in failed.stderr
    # Oversized frames terminate with bounded input buffering, rather than desynchronizing the protocol.
    failed = subprocess.run([str(args.binary.resolve()), '--bundle', str(args.bundle.resolve())],
        input=b'x'*(8*1024*1024+1)+b'\n', capture_output=True, timeout=30)
    assert failed.returncode != 0 and b'request exceeds 8 MiB' in failed.stderr
    panel = json.loads((args.fixtures/'panel.json').read_text())
    report['panel_sha256'] = sha(args.fixtures/'panel.json')
    for batch in [1, 8]:
        w = Worker(args.binary.resolve(), args.bundle.resolve(), batch)
        try:
            w.encode(panel); reference(session, tokenizer, panel, batch)
            samples = dict(rust_worker=[], python_direct=[])
            for repeat in range(5):
                order = ['rust_worker', 'python_direct'] if repeat % 2 == 0 else ['python_direct', 'rust_worker']
                for backend in order:
                    start = time.perf_counter()
                    if backend == 'rust_worker':
                        w.encode(panel)
                    else:
                        reference(session, tokenizer, panel, batch)
                    samples[backend].append(time.perf_counter()-start)
            for backend, values in samples.items():
                row = dict(backend=backend, batch=batch, samples_seconds=values,
                           texts_per_second=len(panel)/statistics.median(values))
                report['timings'].append(row)
                print(f'{backend} batch {batch}: {row["texts_per_second"]:.2f} texts/s', flush=True)
            status = Path(f'/proc/{w.process.pid}/status').read_text()
            report.setdefault('memory', []).append(dict(batch=batch,
                proc_status=[line for line in status.splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))]))
        finally:
            w.close()
    report['scope'] = 'Warm CPU; 16-text panel, median of 5 alternating passes. Rust includes pipe/JSON/base64 and output checks; Python is in-process. No CUDA. Memory is process RSS/HWM, not proportional memory.'
    report['passed'] = True
    (args.bundle/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Passed {len(fixtures)} tokenizer cases, {len(cases)} embedding cases and protocol checks; saved validation.json')


if __name__ == '__main__':
    main()
