"""Bounded JSON-lines bridge for the existing pinned ONNX model and Nostr verification.
The index, adapters, kernels and Arrow persistence remain Julia. No raw vectors
are written to disk by this worker. stdout is reserved for the framed protocol.
"""
import argparse
import base64
import hashlib
import json
import re
import sys

MODEL = 'futur/Qwen3-Embedding-0.6B-model2vec-onnx'
REVISION = 'a3f60a85a2570e7976ea7f2d6505ad3e5152a96f'
QWEN_MODEL = MODEL + '@' + REVISION
GTE_MODEL = 'TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3'


class GteEncoder:
    """Pinned float ONNX, benchmark pooling; bound transformer batches separately."""
    def __init__(self, device):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        from embed_bluesky import make_session
        repo, revision = GTE_MODEL.split('@')
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, 'tokenizer.json', revision=revision))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('[PAD]'), pad_token='[PAD]')
        self.session = make_session(hf_hub_download(repo, 'onnx/model.onnx', revision=revision), device)

    def encode(self, texts):
        import numpy as np
        chunks = []
        for start in range(0, len(texts), 32):
            tokens = self.tokenizer.encode_batch(texts[start:start + 32], add_special_tokens=True)
            inputs = {'input_ids': np.asarray([t.ids for t in tokens], dtype=np.int64),
                      'attention_mask': np.asarray([t.attention_mask for t in tokens], dtype=np.int64),
                      'token_type_ids': np.asarray([t.type_ids for t in tokens], dtype=np.int64)}
            hidden = self.session.run(['last_hidden_state'], inputs)[0]
            mask = inputs['attention_mask'][..., None].astype(np.float32)
            chunks.append((hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1))
        return np.concatenate(chunks), self.session.get_providers()[0]


def verify_event(event):
    from coincurve import PublicKeyXOnly
    try:
        for key, size in [('id', 64), ('pubkey', 64), ('sig', 128)]:
            if not isinstance(event[key], str) or not re.fullmatch('[0-9a-f]{' + str(size) + '}', event[key]):
                return False
        if type(event['kind']) is not int or not 0 <= event['kind'] <= 65535:
            return False
        if type(event['created_at']) is not int or event['created_at'] < 0:
            return False
        if not isinstance(event['content'], str) or not isinstance(event['tags'], list):
            return False
        if not all(isinstance(t, list) and all(isinstance(v, str) for v in t) for t in event['tags']):
            return False
        payload = [0, event['pubkey'], event['created_at'], event['kind'], event['tags'], event['content']]
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).digest()
        return digest.hex() == event['id'] and PublicKeyXOnly(bytes.fromhex(event['pubkey'])).verify(bytes.fromhex(event['sig']), digest)
    except (KeyError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--model', choices=[QWEN_MODEL, GTE_MODEL], default=QWEN_MODEL)
    args = parser.parse_args()
    model = None
    dimension = 384 if args.model == GTE_MODEL else 256
    print(json.dumps({'ready': True, 'model': args.model, 'dimension': dimension}), flush=True)
    for line in sys.stdin:
        try:
            if len(line) > 8 * 1024 * 1024:
                raise ValueError('worker request exceeds 8 MiB')
            req = json.loads(line)
            if req['op'] == 'close':
                print('{"ok":true}', flush=True)
                break
            if req['op'] == 'verify_nostr':
                if len(req['events']) > 1024:
                    raise ValueError('too many events')
                out = {'valid': [verify_event(e) for e in req['events']]}
            elif req['op'] == 'embed':
                texts = req['texts']
                if not texts or len(texts) > 1024 or not all(isinstance(t, str) and t.strip() and len(t.encode()) <= 65536 for t in texts):
                    raise ValueError('expected 1..1024 nonempty texts, each at most 64 KiB')
                import numpy as np
                from embed_bluesky import make_session, pack
                if args.model == GTE_MODEL:
                    if model is None:
                        model = GteEncoder(args.device)
                    raw, provider = model.encode(texts)
                else:
                    if model is None:
                        from huggingface_hub import hf_hub_download
                        from tokenizers import Tokenizer
                        tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, 'tokenizer.json', revision=REVISION))
                        tokenizer.no_padding()
                        tokenizer.no_truncation()
                        session = make_session(hf_hub_download(MODEL, 'onnx/model.onnx', revision=REVISION), args.device)
                        model = tokenizer, session
                    tokenizer, session = model
                    ids = [e.ids for e in tokenizer.encode_batch(texts, add_special_tokens=False)]
                    raw = session.run(['embeddings'], pack(ids))[0].astype(np.float32)
                    provider = session.get_providers()[0]
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                if raw.shape != (len(texts), dimension) or not np.isfinite(raw).all() or (norms <= 0).any():
                    raise ValueError('invalid embedding output')
                vectors = np.asarray(raw / norms, dtype='<f4')
                out = {'dimension': dimension, 'count': len(texts), 'vectors': base64.b64encode(vectors.tobytes()).decode(),
                       'model': args.model, 'provider': provider}
            else:
                raise ValueError('unknown worker operation')
            print(json.dumps(out, separators=(',', ':')), flush=True)
        except Exception as exc:
            # Do not print input text, tokens, or secrets into error diagnostics.
            print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}), flush=True)


if __name__ == '__main__':
    main()
