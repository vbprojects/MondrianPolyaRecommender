"""Actual ONNX/PyTorch parity and live JSON-lines protocol (cached pinned model)."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from live_embedding_worker import GteEncoder, GTE_MODEL


class LiveGteTest(unittest.TestCase):
    def test_reference_and_protocol(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        repo, revision = GTE_MODEL.split('@')
        texts = ['Short.', 'Julia trees and probability.', '日本語 café #Astronomy 🦋',
                 'galaxy ' * 400, 'Unicode\u2003whitespace\nnew line', '[MASK] literal token'] * 6
        encoder = GteEncoder('cpu')
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)
        model = AutoModel.from_pretrained(repo, revision=revision, local_files_only=True, dtype=torch.float32).eval()
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            reference = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()
        raw, provider = encoder.encode(texts)
        vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        error = float(np.max(np.abs(vectors - reference)))
        self.assertLess(error, 2e-5)
        single, _ = encoder.encode([texts[0]])
        np.testing.assert_allclose(single[0]/np.linalg.norm(single[0]), vectors[0], atol=2e-5)
        self.assertEqual(provider, 'CPUExecutionProvider')

        proc = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/live_embedding_worker.py'),
                                 '--model', GTE_MODEL, '--device', 'cpu'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        def request(data):
            proc.stdin.write(json.dumps(data)+'\n');proc.stdin.flush()
            return json.loads(proc.stdout.readline())
        try:
            ready = json.loads(proc.stdout.readline())
            self.assertEqual((ready['model'], ready['dimension']), (GTE_MODEL,384))
            self.assertIn('error',request({'op':'embed','texts':['']}))
            start = time.perf_counter()
            result = request({'op':'embed','texts':texts})
            elapsed = time.perf_counter()-start
            self.assertEqual(result['count'],len(texts))
            actual = np.frombuffer(base64.b64decode(result['vectors']),dtype='<f4').reshape(-1,384)
            np.testing.assert_allclose(actual,reference,atol=2e-5)
            self.assertEqual(request({'op':'verify_nostr','events':[]})['valid'],[])
            self.assertTrue(request({'op':'close'})['ok'])
            self.assertEqual(proc.wait(timeout=30),0)
            out=ROOT/'output/live_gte_validation';out.mkdir(exist_ok=True)
            (out/'parity.json').write_text(json.dumps(dict(model=GTE_MODEL,dimension=384,max_tokens=256,
                pooling='masked mean, then L2 normalization',provider=provider,examples=len(texts),
                max_abs_onnx_pytorch_error=error,first_request_seconds=elapsed),indent=2)+'\n')
        finally:
            if proc.poll() is None: proc.kill();proc.wait()
            proc.stdin.close();proc.stdout.close()


if __name__ == '__main__':
    unittest.main()
