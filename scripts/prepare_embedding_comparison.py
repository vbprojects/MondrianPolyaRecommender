"""Matched, pinned ONNX inputs for the Julia embedding/clustering experiment.

No live index, existing embeddings, or model settings are changed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from embed_bluesky import MODEL, MODEL_REV, ROOT, make_session, pack, read_jsonl, sha

ARCTIC = 'Snowflake/snowflake-arctic-embed-xs'
ARCTIC_REV = 'd8c86521100d3556476a063fc2342036d45c106f'
TOPICS = {k: set(v.split()) for k, v in {
    'Art': 'art digitalart fanart illustration sketch drawing pixelart',
    'Books': 'booksky books bookchallenge reading writingcommunity',
    'Music': 'music nowplaying musicsky musicchallenge',
    'Nature': 'nature wildflowerhour birds birding flowers',
    'Gaming': 'gaming gamedev pokemon ffxiv ff14 ttrpg twitch videogames',
    'Politics': 'cdnpoli auspol politics uspoli ukpolitics uspol',
}.items()}
TAG = re.compile(r'(?<!\w)#([A-Za-z][A-Za-z0-9_]{2,30})\b')
ALL_TAGS = re.compile(r'(?<!\w)#[\w]+')


def normalized_text(text):
    return ' '.join(text.casefold().split())


def training_author(author, seed):
    digest = hashlib.sha256(f'{seed}:{author}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % 5 != 0


def select_posts(posts, count, seed):
    rng = random.Random(seed)
    order = list(range(len(posts)))
    rng.shuffle(order)
    natural, probe, seen, masked_seen = [], [], set(), set()
    topic_counts, author_counts = Counter(), Counter()
    for i in order:
        p = posts[i]
        key = normalized_text(p['text'])
        common = dict(source_row=i, uri=p['uri'], author=p['author'], text=p['text'],
                      train=training_author(p['author'], seed))
        if key not in seen and len(natural) < count:
            natural.append(dict(common, cohort='natural', input_text=p['text'], topic=''))
            seen.add(key)
        tags = {tag.lower() for tag in TAG.findall(p['text'])}
        labels = [name for name, aliases in TOPICS.items() if tags & aliases]
        masked = ' '.join(ALL_TAGS.sub(' ', p['text']).split())
        masked_key = normalized_text(masked)
        if len(labels) != 1 or len(masked_key) < 30 or masked_key in masked_seen:
            continue
        label = labels[0]
        if topic_counts[label] >= 160 or author_counts[label, p['author']] >= 4:
            continue
        probe.append(dict(common, cohort='topics', input_text=masked, topic=label))
        topic_counts[label] += 1
        author_counts[label, p['author']] += 1
        masked_seen.add(masked_key)
    if len(natural) != count:
        raise ValueError('Not enough unique texts in corpus')
    # Stratify the small probe by author, including at least five test authors
    # per topic. Union across topics prevents any author's train/test leakage.
    test_authors = set()
    for name in TOPICS:
        authors = sorted({p['author'] for p in probe if p['topic'] == name})
        rng.shuffle(authors)
        test_authors.update(authors[:max(5, round(.2*len(authors)))])
    for p in probe:
        p['train'] = p['author'] not in test_authors
    for name in TOPICS:
        assert sum(p['topic'] == name and not p['train'] for p in probe) >= 5, name
        assert sum(p['topic'] == name and p['train'] for p in probe) >= 5, name
    for cohort in (natural, probe):
        assert {p['author'] for p in cohort if p['train']}.isdisjoint(
            p['author'] for p in cohort if not p['train'])
        assert len({normalized_text(p['input_text']) for p in cohort}) == len(cohort)
    return natural + probe


class Encoder:
    def __init__(self, name, device):
        self.name = name
        self.repo, self.revision = (MODEL, MODEL_REV) if name == 'current' else (ARCTIC, ARCTIC_REV)
        files = ['onnx/model.onnx', 'tokenizer.json']
        if name == 'arctic':
            files += ['1_Pooling/config.json', 'sentence_bert_config.json']
        self.paths = {f: hf_hub_download(self.repo, f, revision=self.revision) for f in files}
        self.tokenizer = Tokenizer.from_file(self.paths['tokenizer.json'])
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        if name == 'arctic':
            cfg = json.loads(Path(self.paths['1_Pooling/config.json']).read_text())
            assert cfg['pooling_mode_cls_token'] and not cfg['pooling_mode_mean_tokens']
        self.session = make_session(self.paths['onnx/model.onnx'], device)
        self.inputs = [x.name for x in self.session.get_inputs()]
        self.outputs = [x.name for x in self.session.get_outputs()]
        if name == 'arctic':
            assert 'last_hidden_state' in self.outputs, self.outputs
        else:
            assert self.inputs == ['input_ids', 'offsets']
        self.d = 256 if name == 'current' else 384

    def tokenize(self, texts):
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=self.name == 'arctic')
        truncated = sum(len(e.ids) > 512 for e in encoded) if self.name == 'arctic' else 0
        # Truncate before SEP, preserving the official special-token contract.
        ids = [e.ids if self.name == 'current' or len(e.ids) <= 512 else e.ids[:511] + [102]
               for e in encoded]
        return ids, truncated

    def batch(self, ids, session=None):
        session = session or self.session
        if self.name == 'current':
            out = session.run(['embeddings'], pack(ids))[0].astype(np.float32)
        else:
            width = max(map(len, ids))
            tokens = np.zeros((len(ids), width), dtype=np.int64)
            mask = np.zeros_like(tokens)
            for i, row in enumerate(ids):
                tokens[i, :len(row)] = row
                mask[i, :len(row)] = 1
            feeds = dict(input_ids=tokens, attention_mask=mask, token_type_ids=np.zeros_like(tokens))
            out = session.run(['last_hidden_state'], {k: feeds[k] for k in self.inputs})[0][:, 0].copy()
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        assert np.isfinite(out).all() and np.all(norms > 0)
        return out / norms

    def encode_ids(self, ids, batch_size):
        result = np.empty((len(ids), self.d), np.float32)
        for start in range(0, len(ids), batch_size):
            result[start:start+batch_size] = self.batch(ids[start:start+batch_size])
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    ap.add_argument('--output', type=Path, default=ROOT/'data/embedding_comparison')
    args = ap.parse_args()
    assert args.count >= 1000 and args.batch_size > 0
    source = ROOT/'data/bluesky_100000'
    selected = select_posts(read_jsonl(source/'posts.jsonl'), args.count, args.seed)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out/'posts.jsonl').open('w') as f:
        for p in selected:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print('Cohorts:', Counter(p['cohort'] for p in selected), flush=True)
    print('Topic counts:', Counter(p['topic'] for p in selected if p['topic']), flush=True)
    manifest = dict(seed=args.seed, natural_count=args.count, rows=len(selected),
                    topic_aliases={k: sorted(v) for k, v in TOPICS.items()},
                    source_manifest=json.loads((source/'manifest.json').read_text()),
                    source_posts_sha256=sha(source/'posts.jsonl'), posts_sha256=sha(out/'posts.jsonl'),
                    selection='Shuffled unique normalized text; author-hash 80/20 natural split. '
                              'Independent masked-hashtag cohort, max 160/topic and 4/author/topic; '
                              '20% of authors/topic held out (minimum five), union across topics.',
                    versions=dict(onnxruntime=ort.__version__, **{k: importlib.metadata.version(k) for k in
                              ['numpy', 'tokenizers', 'huggingface_hub']}), models={})
    for name in ['current', 'arctic']:
        enc = Encoder(name, args.device)
        texts = [p['input_text'] for p in selected]
        started = time.perf_counter()
        ids, truncated = enc.tokenize(texts)
        tokenize_s = time.perf_counter() - started
        enc.batch(ids[:args.batch_size])
        started = time.perf_counter()
        X = enc.encode_ids(ids, args.batch_size)
        encode_s = time.perf_counter() - started
        assert X.shape == (len(selected), enc.d)
        assert np.max(abs(np.linalg.norm(X, axis=1)-1)) < 2e-6
        # Verify padding/batching does not change the embedding of the same text.
        singleton = np.vstack([enc.batch([x]) for x in ids[:8]])
        singleton_error = float(np.max(abs(X[:8]-singleton)))
        assert singleton_error < 5e-4, singleton_error
        cpu = make_session(enc.paths['onnx/model.onnx'], 'cpu')
        cpu_error = float(np.max(abs(X[:8]-enc.batch(ids[:8], cpu))))
        assert cpu_error < 5e-4, cpu_error
        del cpu
        archive_error = None
        if name == 'current':
            old = np.load(source/'embeddings.npy', mmap_mode='r')
            archive_error = float(np.max(abs(X[:32]-old[[p['source_row'] for p in selected[:32]]])))
            assert archive_error < 5e-4, archive_error
        bench = []
        for repeat in range(3):
            started = time.perf_counter()
            bi, _ = enc.tokenize(texts[:1000])
            token_s = time.perf_counter()-started
            started = time.perf_counter()
            enc.encode_ids(bi, args.batch_size)
            inference_s = time.perf_counter()-started
            bench.append(dict(tokenization_seconds=token_s, inference_seconds=inference_s,
                              total_seconds=token_s+inference_s))
        X.astype('<f4').tofile(out/f'{name}.f32')
        info = dict(repo=enc.repo, revision=enc.revision, dimensions=enc.d,
                    providers=enc.session.get_providers(), batch_size=args.batch_size,
                    pooling='static exported mean' if name == 'current' else 'CLS + L2 normalization; no query prefix',
                    tokenization_seconds=tokenize_s, inference_seconds=encode_s,
                    benchmark_1000=bench, truncated_posts=truncated,
                    singleton_max_error=singleton_error, cpu_max_error=cpu_error,
                    archive_max_error=archive_error, embedding_sha256=sha(out/f'{name}.f32'),
                    files={f: dict(bytes=Path(p).stat().st_size, sha256=sha(p)) for f, p in enc.paths.items()})
        manifest['models'][name] = info
        print(name, json.dumps({k: info[k] for k in ['dimensions', 'inference_seconds', 'benchmark_1000', 'cpu_max_error']}), flush=True)
        del enc
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Validated embeddings written to', out, flush=True)


if __name__ == '__main__':
    main()
