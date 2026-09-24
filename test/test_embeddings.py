"""Bridge contract tests; run with the isolated embedding Python environment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from embed_bluesky import pack, read_jsonl, sample_posts

class EmbeddingBridgeTests(unittest.TestCase):
    def test_jsonl_unicode_separators(self):
        rows = [{'text': 'a\u2028b\u2029c\n日本語'}, {'text': '</script><script>test</script>'}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'rows.jsonl'
            p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
            self.assertEqual(read_jsonl(p), rows)

    def test_scaled_sampling_keeps_unique_uris_and_alignment(self):
        def prefix(number, size):
            self.assertEqual(size, 10000)
            rows = [{'uri': f'{number}-{i}', 'text': f'post {number} {i}'} for i in range(3)]
            rows += [dict(rows[0]), {'uri': 'blank', 'text': '  '}]
            return f'shard-{number}', rows, {'records': len(rows)}
        with patch('embed_bluesky.fetch_prefix', side_effect=prefix):
            posts, sampling = sample_posts(40, 42, 10000)
            repeated, _ = sample_posts(40, 42, 10000)
        self.assertEqual(posts, repeated)
        self.assertEqual(len({p['uri'] for p in posts}), 40)
        self.assertEqual([p['embedding_row'] for p in posts], list(range(40)))
        self.assertEqual(sampling['eligible'], 60)
        self.assertEqual(sampling['skipped']['duplicate_uri'], 20)
        self.assertEqual(sampling['skipped']['empty_text'], 20)
        self.assertTrue(all(1 <= p['source_record'] <= 3 for p in posts))

    def test_ragged_batch_contract(self):
        result = pack([[11, 12], [3], [4, 5, 6]])
        np.testing.assert_array_equal(result['input_ids'], [11, 12, 3, 4, 5, 6])
        np.testing.assert_array_equal(result['offsets'], [0, 2, 3])
        self.assertEqual(result['offsets'].dtype, np.int64)
        with self.assertRaises(ValueError):
            pack([[1], []])

if __name__ == '__main__':
    unittest.main()
