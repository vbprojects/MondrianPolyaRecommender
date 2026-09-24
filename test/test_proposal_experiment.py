import unittest,sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from proposal_experiment import ig,ent,OUT
class ProposalTests(unittest.TestCase):
    def test_information_gain_identity(self):
        p=np.array([.2,.3,.5]);L=np.array([[.4,.4,.4],[.95,.1,.6]])
        score=ig(p,L)
        self.assertAlmostEqual(score[0],0.)
        for row,got in zip(L,score):
            q=row@p;yes=p*row/q;no=p*(1-row)/(1-q)
            entropy=lambda x: -sum(x*np.log(x))
            expected=entropy(p)-q*entropy(yes)-(1-q)*entropy(no)
            self.assertAlmostEqual(got,expected)
        self.assertGreater(score[1],0.)
    def test_artifacts(self):
        r=json.loads((OUT/'results.json').read_text());b=np.load(OUT/'banks.npz')
        self.assertFalse(np.isin(b['post_rows'],b['target_rows']).any())
        for m in ['tree_uniform','micro_diagonal','mppca','local_real_posts','random_posts']:
            self.assertEqual(b[m].shape,(64,64));self.assertTrue(np.all(b[m]<len(b['post_rows'])))
        with (OUT/'trials.csv').open() as stream: rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),600)
        for row in rows:
            self.assertTrue(0<=float(row['final_true_mass'])<=1)
            self.assertTrue(0<=int(row['feedback_count'])<=60)
        for row in r['sequential']:
            self.assertTrue(row['success_rate']['ci95'][0]<=row['success_rate']['mean']<=row['success_rate']['ci95'][1])
        for methods in r['frozen'].values():
            for result in methods.values():self.assertGreaterEqual(result['oracle_regret_bits']['mean'],-1e-10)
if __name__=='__main__':unittest.main()
