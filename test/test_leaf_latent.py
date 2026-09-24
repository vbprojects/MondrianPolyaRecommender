import unittest,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class LeafLatentTest(unittest.TestCase):
    def test_saved_experiment(self):
        raw=json.loads((ROOT/'output/leaf_latent/experiment.json').read_text())
        results=json.loads((ROOT/'output/leaf_latent/results.json').read_text())
        train=set(raw['train_rows']);test=set(raw['test_rows'])
        self.assertEqual(len(train),80000);self.assertEqual(len(test),20000)
        self.assertFalse(train&test);self.assertTrue(set(raw['target_rows'])<=test)
        self.assertEqual(len(raw['metrics']),5)
        for m in raw['metrics']:
            self.assertEqual(m['leaves'],64)
            self.assertAlmostEqual(m['mass'],80000,places=6)
            self.assertLess(m['max_split_mass_error'],1e-8)
            self.assertLess(m['max_split_mean_error'],1e-10)
            self.assertLess(m['max_conditional_diagonal_error'],1e-10)
            self.assertTrue(0<m['acceptance']<=1)
            self.assertTrue(np.isfinite(m['test_rmse']))
        for method in raw['proposals']:
            for bound,points in zip(raw['target_bounds'],method['points']):
                if not points:continue
                x=np.asarray(points)
                self.assertTrue(np.all(x>=np.asarray(bound['lo'])-1e-12))
                self.assertTrue(np.all(x<np.asarray(bound['hi'])+1e-12))
        for m in results['proposal_ig'].values():self.assertGreaterEqual(m['information_bits']['mean'],0)
if __name__=='__main__':unittest.main()
