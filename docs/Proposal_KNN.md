# kNN in local MPPCA space

[Interactive follow-up report](../output/proposal_knn/demo.html) · [results](../output/proposal_knn/results.json) · [trial records](../output/proposal_knn/trials.csv)

We reused the previous experiment's 64 target hypotheses, frozen rank-four local MPPCA components, exact Gaussian proposals and clipping, all 100,000 original vectors, 60 paired trials and 60 decision steps per method/world. Feedback likelihood scales are frozen to the previous run rather than recalibrated on the expanded candidate union. The original-space and local-real-post baselines reproduce their previous outcomes exactly. Only proposal-to-post retrieval changes.

## Retrieval definitions

MPPCA has multiple local coordinate systems, not one global latent space. For each proposal, we use its generating component to transform both the proposal and candidate posts. We never compare coordinate vectors produced by different bases directly.

For component covariance `C = U diag(lambda) U' + sigma² (I-UU')`, its posterior latent coordinates are `sqrt(lambda-sigma²)/lambda * U'(x-mean)`. We use Euclidean distance between these posterior coordinates for latent kNN. This choice includes PPCA shrinkage; unwhitened PCA coordinates would be another possible experiment.

| Variant | Retrieval |
|---|---|
| Original-space 1NN | Previous baseline: nearest original 256D vector to the MPPCA proposal |
| Latent 1NN | Nearest posterior latent vector among the full eligible corpus |
| Latent top-5 sample | Uniformly choose one of the five nearest latent neighbors per proposal |
| Full PPCA metric 1NN | Mahalanobis distance under the generating component, retaining isotropic off-subspace penalty |
| Within-region latent top-5 | Same top-5 latent sampling, restricted to real posts inside the proposal's owning box |
| Within-region original 1NN | Original-space nearest neighbor with the identical region restriction |
| Local actual posts | Previous random-within-region post bank control; no synthetic retrieval |

Top-5 sampling still emits **one candidate per proposal**, preserving the 32-proposal budget. It does not give the information-gain scorer 160 candidates. After candidate generation, every variant uses the same original-vector likelihood and information-gain rule. The feedback graph is not rebuilt in latent space.

All searches are exact brute-force kNN in this experiment, not an approximate neighbor index. Global latent searches project all candidate vectors into each needed local component chart. The full-metric calculation needs original vectors for off-subspace distances. Consequently this measures retrieval policy, not embedding-free storage or production lookup latency. Targets remain excluded from all candidate sets.

## Results

Fraction of trials reaching at least 50% posterior mass on the true simulated location within 60 decision steps:

| Variant | Chord feedback | Density-weighted graph feedback |
|---|---:|---:|
| Original-space 1NN | 43.3% | 53.3% |
| Latent 1NN | 30.0% | 50.0% |
| Latent top-5 sample | 23.3% | 53.3% |
| Full PPCA metric 1NN | 31.7% | 38.3% |
| Within-region latent top-5 | 33.3% | **63.3%** |
| Within-region original 1NN | 43.3% | 56.7% |
| Local actual posts | 36.7% | 70.0% |

Unrestricted latent retrieval often matches a post whose four latent coordinates are close but whose remaining coordinates differ substantially. About 87% of those retrieved posts fall outside the proposal's owning region, compared with about 46% for unrestricted original-space retrieval. The within-region variants enforce the original coarse partition and avoid that leakage. The full PPCA metric retrieves fewer distinct posts, limiting proposal diversity in this configuration.

Within-region latent top-5 improves mean final true-location mass over the unrestricted original-space baseline by 0.0906 in the graph world, with a paired 95% bootstrap interval [0.0013, 0.1775]. Against the **within-region original-space control**, however, the difference is 0.0824 with interval [-0.0184, 0.1845]. Against local actual-post sampling, it is 0.0122 [-0.0777, 0.1024]. These do not establish a clear advantage specifically from the latent metric. In the chord world, the corresponding local-latent comparisons all have intervals spanning zero and lower point estimates.

This supports retaining tree locality when using low-rank neighbors. It does not establish that MPPCA kNN is generally better. Locality controls were added after observing out-of-region retrieval, making this follow-up exploratory. There are multiple comparisons, one target/model/bank seed and only 60 trials per configuration; no multiplicity correction is applied.

## Scope and reproducibility

The simulator, transductive model training, finite proposal-bank reuse, heavy Gaussian clipping, perfect feedback-model specification and fixed user location have the same limitations as the [previous protocol](Proposal_Experiment.md). Capped failure steps are 61; these are not uncensored stopping-time estimates. The finite-union information-gain oracle now covers all seven methods, so oracle regret is not directly comparable to the previous five-method union. Absolute baseline information gain and localization outcomes remain comparable because likelihood scales are unchanged.

```sh
julia --project=. examples/proposal_knn.jl
.venv-embeddings/bin/python -m unittest discover -s test -p test_proposal_knn.py
```

The Julia entry point invokes the existing isolated NumPy bridge. No new package dependency is required. Inputs are the existing `output/proposal_experiment/` artifacts and original vector archive. Outputs are separate in `output/proposal_knn/`; prior results are preserved.

Validation checks posterior-coordinate algebra against the explicit PPCA loading-matrix formula, recomputes the first saved latent neighbor independently, excludes target IDs, checks within-region containment, and verifies exact baseline bank/outcome reproduction. The self-contained HTML report includes both feedback worlds, curves, uncertainty intervals and retrieval diagnostics.
