import unittest,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from proposal_knn import latent_coordinates,OUT,PREVIOUS
class KNNTests(unittest.TestCase):
    def test_posterior_coordinates(self):
        rng=np.random.default_rng(3);U,_=np.linalg.qr(rng.normal(size=(9,3)))
        basis=U.T;variance=np.array([5.,3.,2.]);noise=.2;mean=rng.normal(size=9);X=rng.normal(size=(7,9))
        W=U*np.sqrt(variance-noise)
        expected=(X-mean)@W@np.linalg.inv(W.T@W+noise*np.eye(3))
        np.testing.assert_allclose(latent_coordinates(X,mean,basis,variance,noise),expected,atol=1e-12)
    def test_first_latent_neighbor(self):
        model=json.loads((PREVIOUS/'models.json').read_text())
        root=PREVIOUS.parents[1]
        X=np.load(root/'data/bluesky_100000/embeddings.npy').astype(np.float64)
        mu=np.array(model['means'],np.float32);basis=np.array(model['bases'],np.float32)
        variance=np.array(model['variances'],np.float32);noise=np.array(model['noise'],np.float32)
        mass=np.array(model['mass']);rng=np.random.default_rng(819)
        ids=np.array(model['components'][0]);ids=rng.choice(ids,64,p=mass[ids]/mass[ids].sum())
        eps=rng.normal(size=(64,256));z=rng.normal(size=(64,4))
        raw=mu[ids]+np.sqrt(noise[ids,None])*eps+np.einsum('nrd,nr->nd',basis[ids],z*np.sqrt(np.maximum(0,variance[ids]-noise[ids,None])))
        bound=np.array(model['bounds'][0],np.float32)
        point=np.clip(raw.astype(np.float32)[0],bound[:256],np.nextafter(bound[256:],bound[:256])).astype(np.float64)
        k=ids[0];U=basis[k].astype(np.float64);mean=mu[k].astype(np.float64);var=variance[k].astype(np.float64)
        Z=latent_coordinates(X,mean,U,var,float(noise[k]));q=latent_coordinates(point,mean,U,var,float(noise[k]))
        distance=np.sum((Z-q)**2,axis=1);distance[model['rows']]=np.inf
        bank=np.load(OUT/'banks.npz')
        self.assertEqual(int(bank['post_rows'][bank['latent_1nn'][0,0]]),int(np.argmin(distance)))

    def test_saved_run(self):
        r=json.loads((OUT/'results.json').read_text());old=json.loads((PREVIOUS/'results.json').read_text())
        self.assertEqual(r['likelihood_scales'],old['likelihood_scales'])
        banks=np.load(OUT/'banks.npz');previous=np.load(PREVIOUS/'banks.npz')
        self.assertFalse(np.isin(banks['post_rows'],banks['target_rows']).any())
        for method in ['mppca','local_real_posts']:
            np.testing.assert_array_equal(banks['post_rows'][banks[method]],previous['post_rows'][previous[method]])
            for world in ['chord','fermat_graph']:
                a=next(x for x in r['sequential'] if x['method']==method and x['world']==world)
                b=next(x for x in old['sequential'] if x['method']==method and x['world']==world)
                self.assertAlmostEqual(a['success_rate']['mean'],b['success_rate']['mean'])
                self.assertAlmostEqual(a['final_true_mass']['mean'],b['final_true_mass']['mean'])
        for method in ['local_latent_5nn','local_original_1nn']:
            self.assertEqual(r['diagnostics'][method]['retrieved_outside_owner'],0.)
        self.assertEqual(len(r['sequential']),14)
if __name__=='__main__':unittest.main()
