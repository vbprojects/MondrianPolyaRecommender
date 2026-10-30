"""One-time export/reference tooling. Never imported by the Julia runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = 'TaylorAI/gte-tiny'
REV = '4cc5e73d86a67c601897257b467187234aa3bca3'
QUANT_SHA = 'ceab9964bcdb19ad013a2fe44bc1ddb3657ab0af1f48cddc349014d6085762ca'
FLOAT_SHA = '5fe59be58a9ad68621795a500e02b6b04a12f194b823af9c4c292854f340274d'
ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'output/gte_tiny_julia'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def get(name):
    return Path(hf_hub_download(REPO, name, revision=REV))


def export(out):
    out.mkdir(parents=True, exist_ok=True)
    source = get('onnx/model_quantized.onnx')
    assert sha(source) == QUANT_SHA, 'Unexpected pinned graph'
    graph = onnx.load(source).graph
    arrays = {x.name: numpy_helper.to_array(x) for x in graph.initializer}
    nodes = {x.name: x for x in graph.node}
    tensors = {}

    def save(name, array):
        a = np.asarray(array)
        assert a.dtype in (np.dtype('uint8'), np.dtype('int8'), np.dtype('float32'))
        filename = name + '.bin'
        (out / filename).write_bytes(a.astype(a.dtype.newbyteorder('<')).tobytes(order='F'))
        tensors[name] = dict(file=filename, shape=list(a.shape), dtype=a.dtype.name,
                             bytes=(out / filename).stat().st_size, sha256=sha(out / filename))

    for kind in ('word', 'position', 'token_type'):
        prefix = f'embeddings.{kind}_embeddings.weight'
        save(kind + '.weight', arrays[prefix + '_quantized'].T)
        save(kind + '.scale', arrays[prefix + '_scale'])
        save(kind + '.zero', arrays[prefix + '_zero_point'])
    save('embedding.norm.weight', arrays['embeddings.LayerNorm.weight'])
    save('embedding.norm.bias', arrays['embeddings.LayerNorm.bias'])
    for layer in range(6):
        for short, path in [('query', 'attention/self/query'), ('key', 'attention/self/key'),
                            ('value', 'attention/self/value'), ('attention', 'attention/output/dense'),
                            ('up', 'intermediate/dense'), ('down', 'output/dense')]:
            node = nodes[f'/encoder/layer.{layer}/{path}/MatMul_quant']
            weight = node.input[1]
            prefix = weight.removesuffix('_quantized')
            assert arrays[prefix + '_zero_point'].item() == 0
            save(f'{layer}.{short}.weight', arrays[weight])
            save(f'{layer}.{short}.scale', arrays[prefix + '_scale'])
            bias = f'encoder.layer.{layer}.' + path.replace('/', '.') + '.bias'
            save(f'{layer}.{short}.bias', arrays[bias])
        for short, path in [('attnorm', 'attention.output'), ('outnorm', 'output')]:
            for kind in ('weight', 'bias'):
                save(f'{layer}.{short}.{kind}', arrays[f'encoder.layer.{layer}.{path}.LayerNorm.{kind}'])
    shutil.copyfile(get('tokenizer.json'), out / 'tokenizer.json')
    shutil.copyfile(get('config.json'), out / 'config.json')
    write_json(out / 'weights.json', dict(format=1, model=REPO, revision=REV,
        source_sha256=QUANT_SHA, tokenizer_sha256=sha(out / 'tokenizer.json'),
        tensors=tensors, layout='little-endian, Julia column-major', max_tokens=256))
    return source


def session(path, threads, debug=False):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    if debug:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])


def inputs(tokenizer, texts):
    enc = tokenizer.encode_batch(texts)
    return {k: np.asarray([getattr(e, attr) for e in enc], dtype=np.int64)
            for k, attr in [('input_ids', 'ids'), ('attention_mask', 'attention_mask'),
                            ('token_type_ids', 'type_ids')]}


def encode(sess, tokenizer, texts, batch):
    outputs = []
    for start in range(0, len(texts), batch):
        feed = inputs(tokenizer, texts[start:start + batch])
        h = sess.run(['last_hidden_state'], feed)[0]
        mask = feed['attention_mask'][..., None]
        v = (h * mask).sum(1, dtype=np.float32) / mask.sum(1)
        v = v.astype(np.float32)
        outputs.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--data', type=Path, default=ROOT / 'output/gte_tiny_screen/bced8b63d83cdd78/data')
    parser.add_argument('--queries', type=int, default=10)
    parser.add_argument('--distractors', type=int, default=128)
    args = parser.parse_args()
    if args.threads < 1 or args.queries < 1 or args.distractors < 0:
        parser.error('invalid screen size or threads')
    out = args.output
    source = export(out / 'model')
    tokenizer = Tokenizer.from_file(str(out / 'model/tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    quant = session(source, args.threads)
    float_path = get('onnx/model.onnx')
    assert sha(float_path) == FLOAT_SHA
    fp32 = session(float_path, args.threads)
    texts = ['', 'Short.', 'Julia trees and probability.', '日本語 café #Astronomy 🦋',
             'Unicode\u2003whitespace\nnew line', '[MASK] literal [CLS]token[SEP]',
             'Café CAFÉ cafe\u0301 İSTANBUL Straße ΣΟΣ', 'a\x00b\ufffdc\u200bd\t e',
             "can't won't @user #tag https://example.com/a?b=3", '中文简体，繁體。한국어',
             '[PAD] [UNK] [MASK][MASK]', '[mask] [Cls]', 'x' * 101,
             '👩\u200d💻🦋🚀', 'galaxy ' * 400, '\r\n\t', 'a\u00a0b\u2028c\u0085d']
    rng = np.random.default_rng(42)
    chars = list('aAéİßΣς中文日本🙂-!?.,\t\n') + ['\u200b', '\x00', '\u0301']
    token_texts = texts + [''.join(rng.choice(chars, size=40)) for _ in range(100)]
    # Disable fixed padding for single token IDs: encode_batch uses dynamic batch padding.
    tokenizer.no_padding()
    write_json(out / 'tokens.json', [dict(text=t, ids=tokenizer.encode(t).ids) for t in token_texts])
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    cases = [texts[:1], texts[1:4], texts[4:10], [texts[14]], [texts[1], texts[14]]]
    fixture = []
    for group in cases:
        feed = inputs(tokenizer, group)
        fixture.append(dict(texts=group, ids=feed['input_ids'].tolist(), mask=feed['attention_mask'].tolist(),
            quant=encode(quant, tokenizer, group, len(group)).tolist(),
            fp32=encode(fp32, tokenizer, group, len(group)).tolist()))
    write_json(out / 'parity.json', fixture)
    # Layer boundaries from the unoptimized graph help localize numerical drift.
    debug = onnx.load(source)
    names = ['/embeddings/LayerNorm/Add_1_output_0'] + [
        f'/encoder/layer.{i}/output/LayerNorm/Add_1_output_0' for i in range(5)] + ['last_hidden_state']
    for name in names:
        if name not in {o.name for o in debug.graph.output}:
            debug.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    onnx.save(debug, out / 'debug.onnx')
    dbg = session(out / 'debug.onnx', args.threads, debug=True)
    group = texts[1:4]
    layers = dbg.run(names, inputs(tokenizer, group))
    write_json(out / 'layers.json', dict(texts=group, names=names,
                                       hidden=[h.tolist() for h in layers]))
    (out / 'debug.onnx').unlink()
    panel_path = args.data / 'latency-panel.json'
    panel = json.loads(panel_path.read_text())[:16] if panel_path.exists() else texts[1:9]
    write_json(out / 'panel.json', panel)
    timings = []
    for batch in (1, 8):
        for label, sess in [('onnx_int8', quant), ('onnx_fp32', fp32)]:
            encode(sess, tokenizer, panel, batch)
            samples = []
            for _ in range(3):
                start = time.perf_counter(); encode(sess, tokenizer, panel, batch)
                samples.append(time.perf_counter() - start)
            timings.append(dict(backend=label, batch=batch, samples_seconds=samples,
                                texts_per_second=len(panel) / np.median(samples)))
    screens = []
    for name in ('NanoNQ', 'NanoHotpotQA', 'NanoMSMARCO'):
        path = args.data / (name + '.json')
        if not path.exists():
            continue
        task = json.loads(path.read_text())
        qids = task['qids'][:args.queries]
        relevant = {q: task['relevant'][q] for q in qids}
        positives = {d for rel in relevant.values() for d in rel}
        candidates = sorted(positives | set(task['cids'][:args.distractors]))
        docs = dict(zip(task['cids'], task['docs']))
        queries = task['queries'][:len(qids)]
        doc_texts = [docs[d] for d in candidates]
        screen = dict(name=name, source_sha256=sha(path), qids=qids, cids=candidates,
                      queries=queries, docs=doc_texts, relevant=relevant, batch=8)
        for label, sess in [('onnx_int8', quant), ('onnx_fp32', fp32)]:
            screen[label] = dict(queries=encode(sess, tokenizer, queries, 8).tolist(),
                                 docs=encode(sess, tokenizer, doc_texts, 8).tolist())
        screens.append(screen)
        token_texts.extend(queries + doc_texts)
        print('Prepared', name, len(qids), len(candidates), flush=True)
    write_json(out / 'retrieval.json', screens)
    tokenizer.no_padding()
    write_json(out / 'tokens.json', [dict(text=t, ids=tokenizer.encode(t).ids)
                                    for t in dict.fromkeys(token_texts + panel)])
    write_json(out / 'reference.json', dict(model=REPO, revision=REV, quant_sha256=QUANT_SHA,
        float_sha256=FLOAT_SHA, threads=args.threads, ort=ort.__version__, numpy=np.__version__,
        python=platform.python_version(), timings=timings,
        scope='Reduced candidate retrieval screen; not comparable to full NanoBEIR scores'))
    print('Prepared reference artifacts:', out, flush=True)


if __name__ == '__main__':
    main()
