import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import embedding_codec as codec

class CodecTests(unittest.TestCase):
    def test_packing_and_error_bound(self):
        rng=np.random.default_rng(12)
        values=rng.normal(size=(13,7)).astype(np.float32)
        groups=np.array([i%3 for i in range(13)],dtype=np.uint16)
        previous=codec.OUT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                codec.OUT=Path(tmp)
                for bits in (1,2,4,8):
                    decoded,files=codec.quantize(values,groups,bits,str(bits))
                    scales=np.fromfile(Path(tmp)/(str(bits)+'.scales.f32'),np.float32).reshape(3,7)
                    self.assertEqual(files['codes'],(values.size*bits+7)//8)
                    self.assertTrue(np.all(np.abs(decoded-values)<=scales[groups]/((1<<bits)-1)+1e-6))
                zero,_=codec.quantize(np.zeros((13,7),np.float32),groups,8,'zero')
                self.assertTrue(np.isfinite(zero).all())
        finally:codec.OUT=previous
    def test_saved_random_access(self):
        if not (codec.OUT/'results.json').exists(): self.skipTest('run benchmark first')
        import json
        results=json.loads((codec.OUT/'results.json').read_text())['results']
        for result in results:
            rows=[4,199,4,99999]
            small=codec.decode(result['name'],rows)
            self.assertEqual(small.shape,(4,256))
            np.testing.assert_array_equal(small[0],small[2])
            self.assertTrue(np.isfinite(small).all())
        with self.assertRaises(ValueError):codec.decode(results[0]['name'],[-1])

if __name__=='__main__':unittest.main()
