# Does local MPPCA improve information-gain proposals?

[Open the interactive report](../output/proposal_experiment/demo.html), [raw results](../output/proposal_experiment/results.json), or [trial records](../output/proposal_experiment/trials.csv).

This experiment tests proposal geometry, not vector reconstruction or L1 density matching. It uses the 100,000 original Bluesky vectors and the existing frozen local MPPCA model. It compares actual posts selected after proposals are mapped to the archive.

## Main findings

Under identical fixed user beliefs, MPPCA generated better best-in-batch information gain than the diagonal microcluster-style ablation:

| Common feedback geometry | Diagonal proposal best IG | MPPCA best IG | Paired improvement, 95% interval |
|---|---:|---:|---:|
| Embedding chord distance | 0.2226 bits | 0.2385 bits | +0.0159 [0.0099, 0.0224] bits |
| Density-weighted graph | 0.3460 bits | 0.3564 bits | +0.0104 [0.0061, 0.0145] bits |

Mean distance from synthetic proposal to retrieved real post was 0.878 for tree-uniform proposals, 0.240 for diagonal proposals and 0.203 for MPPCA. This supports the hypothesis that local covariance helps propose plausible positions, despite the earlier poor rank-four vector-codec result.

That advantage did not establish a clear sequential-localization win over diagonal proposals:

| Method | Chord: localized within budget | Graph: localized within budget |
|---|---:|---:|
| Tree uniform | 3.3% | 18.3% |
| Diagonal microcluster-style | 40.0% | 43.3% |
| Local MPPCA | 43.3% | 53.3% |
| Local actual-post sampling | 36.7% | 70.0% |
| Random actual-post sampling | 13.3% | 45.0% |

There are 60 trials per method per world, with 60 decision steps per trial. Success means true-hypothesis posterior first reaches 0.5. Paired mean final true-probability differences between MPPCA and diagonal proposals have intervals containing zero in both worlds: chord +0.0373 [-0.0551, 0.1294], graph +0.0740 [-0.0247, 0.1787]. Do not interpret the raw success percentages as definitive ordering.

The local actual-post baseline performs strongly and beats MPPCA on mean final true-location probability in the graph world in this run. There is no demonstrated reason yet to replace it with synthetic proposals. The common model matters: the best method changes across feedback geometries.

## Protocol

1. Draw 64 possible stationary user locations from original post vectors using seed 804. These target post IDs are excluded from candidate retrieval. The existing trained tree and PPCA parameters still saw this corpus, so this is a transductive experiment, not held-out model training.
2. For each hypothesis, find the deepest owning node with a retained MPPCA mixture on its original tree path. All synthetic methods use the same owner box.
3. Generate a bank of 64 proposals per hypothesis region. Tree-uniform uses uniform box samples. MPPCA selects components by retained mass and draws low-rank-plus-isotropic-noise Gaussian samples. The diagonal ablation uses the same component means, weights and covariance diagonals, with off-diagonal correlations removed. It is not the earlier independently trained microcluster implementation. This isolates the effect of learned directions.
4. Clip synthetic proposals to the same owner box, respecting the upper cut convention, and retrieve the nearest available actual post against all 100,000 original vectors, excluding the 64 target IDs. Retrieval is global; the nearest actual post can be outside the originating region. The fraction is recorded as `retrieved_outside_owner`. No embedding codec is involved.
5. Construct equal-sized banks of random actual posts from each owner box and from the global corpus as controls. These controls require regional/global post-ID access; the synthetic methods also require a retrieval mechanism. No storage-cost equivalence is claimed.
6. At each decision, sample 32 hypothesis indices from current belief and draw one proposal-bank entry for each. Deduplicate and exclude previously shown posts. Choose the remaining actual post with greatest expected binary information gain, under a shared likelihood model. There is no resampling to fill duplicates. If no unseen candidates remain, the decision step gives no feedback; counts are reported. The finite banks are reused, not new Gaussian draws every round.

Two evaluation worlds are held fixed across all policies:

- **Chord:** original-vector Euclidean distance on normalized embeddings.
- **Density-weighted graph:** symmetric 8-neighbor graph over the 64 hypotheses, with Euclidean minimum-spanning-tree bridges. Edge cost is chord distance divided by the quarter power of geometric-mean density. Density is a bounded inverse-squared neighbor-radius proxy. Candidate posts attach to their three closest anchors. Shortest paths supply distances. This is a coarse Fermat-like proxy, not exact integration of the tree density; it is built independently of each proposal method.

The shared probability of a like is `0.05 + 0.9/(1 + (distance/scale)^2)`, where scale is 0.25 times median candidate-to-hypothesis distance in that world. This is a stipulated simulator, not a model learned from people. The posterior uses exact Bayes updates without the interactive demo's tempering or prior refresh. The simulated user and inference use the same response model.

## Two evaluations

**Frozen beliefs:** 100 seeded Dirichlet belief distributions, shared across methods, isolate one-step candidate quality. Each gets the same hypothesis draws, bank-index draws and 32-proposal budget. Regret is relative to the best actual post in the finite union of all five candidate banks, not an oracle over the entire corpus. Bootstrap intervals resample the paired belief cases.

**Sequential trials:** 60 paired truth/feedback seeds per method per world. A uniform initial belief over 64 locations is updated for up to 60 decisions. Feedback thresholds use common random numbers across methods, although their chosen posts and subsequent beliefs differ. We record first passage to 0.5 true-hypothesis mass, final true mass, final MAP accuracy, feedback count and probability trajectories. First passage uses the evaluator's known simulated truth; a deployed agent would not know it. Reaching the threshold once need not imply stable final localization.

The capped-step metric assigns 61 to trials that do not reach the threshold. It is a bounded comparison score, not a mean uncensored stopping time. Final MAP and probability show whether an early success persisted. Bootstrap intervals and paired differences use 2,000 resamples of trials, conditional on this one dataset/model/proposal bank. They do not capture codebook training, target-set or bank-seed variability. No multiple-comparison correction is applied.

## Limitations and next decisions

About 93–94% of both Gaussian proposal types leave at least one coordinate of the owning box before clipping. Clipping is substantial and changes their distributions. This experiment tests the implemented clipped proposal mechanism, not an ideal truncated PPCA sampler. Low-rank proposals also need not lie on the unit sphere.

There are only 64 hypothetical interest locations. They are not semantic labels or validated topics. Local real-post sampling can work well because it starts directly on the empirical data distribution. The perfect feedback-model specification, fixed interest point, same-corpus codebook, finite proposal banks and small number of trials limit deployment conclusions.

The measured sequential computation is inexpensive because proposals, retrieval and response matrices are precomputed. `mapping_seconds` records offline projection-to-post retrieval; per-policy simulation time excludes it. These numbers are not end-to-end online latency benchmarks.

The evidence supports keeping local MPPCA as a proposal option, not declaring it the winner. The next stronger test would repeat codebook/target/bank seeds and introduce mismatched feedback or multiple user interests. A practical sampler should retain local actual-post sampling as a baseline or mixture component.

## Reproduce and inspect

```sh
julia --project=. examples/proposal_experiment.jl
.venv-embeddings/bin/python -m unittest discover -s test -p test_proposal_experiment.py
```

Julia loads the saved local MPPCA model and exports the frozen experiment geometry. The existing NumPy environment performs vectorized full-corpus retrieval and paired simulations, then generates the offline report. No new package dependency is required.

`results.json` includes uncertainty intervals, scores, trajectories, timings, seeds and model/input-manifest hashes. `trials.csv` records all 600 simulated sessions. `banks.npz` stores actual candidate row IDs, target IDs and each proposal bank. `models.json` contains frozen geometry for this run.

Tests verify the mutual-information/expected-entropy identity, uninformative likelihoods, target exclusion, bank dimensions, session counts, posterior bounds and nonnegative oracle regret. Browser checks cover both feedback worlds, tables, plots and mobile layout. Original archives and trained models remain unchanged.
