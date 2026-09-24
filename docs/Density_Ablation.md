# Does within-leaf PPCA help active localization?

[Interactive experiment](../output/density_ablation/demo.html) · [Summary](../output/density_ablation/summary.txt) · [Trials](../output/density_ablation/trials.csv)

The post codec is fixed to **centroid + 8-bit residuals**. This experiment asks whether a uniform leaf, a diagonal Gaussian, or a rank-four PPCA density gives better query candidates and inference geometry. Direct sampling of actual posts provides a control that needs no Gaussian proposal generator.

All inference and simulation code is Julia: `src/DensityAblation.jl` and `examples/density_ablation.jl`. Python standard-library code aggregates outcomes and builds the standalone viewer. Existing tree, codec, kernel and recommender implementations are unchanged.

## Measured results

The completed run has **33,600 simulated localization trials**, covering four workloads, three seeds, two tolerances and three inference settings. It took **150.6 seconds** after warmup on this workstation, using CPU and one BLAS thread. This includes experiment preparation and audit work; it is not a production serving benchmark.

The clearest architecture comparison keeps **actual-post candidate sampling fixed** and changes only the within-leaf density used for Fermat geometry. On the 100,000-post Bluesky archive, at the strict tolerance and an 80-response budget:

| Within-leaf density | Correct confident stop | Incorrect confident stop | Budget exhausted | Failure-penalized responses |
|---|---:|---:|---:|---:|
| Uniform | 12.7% | 19.7% | 67.7% | 74.4 |
| Diagonal Gaussian | 20.0% | 5.7% | 74.3% | 73.0 |
| PPCA rank 4 | 22.0% | 3.0% | 75.0% | 73.5 |

Each row has 300 paired trials. The tolerance is 5% of the reference median hypothesis distance, and a stop requires 90% posterior confidence. A failure costs 81 in the last column. Percentages can differ slightly from 100% after rounding.

**Learned within-leaf density helped this Bluesky simulation, but PPCA did not clearly beat the cheaper diagonal alternative.** PPCA gained 2.0 percentage points in correct stops over diagonal density; the paired 95% bootstrap interval is **−3.3 to +7.0 points**. Its incorrect-stop difference is −2.7 points, with an interval of −5.7 to 0.0. Its failure-penalized response count is slightly worse. The intervals describe resampled trials within these three fixed splits, not generalization to new users or datasets.

Relative to uniform density with the same actual-post sampler, PPCA increased correct stops by 9.3 points and reduced incorrect confident stops by 16.7 points. Uniform density's lower raw response count came partly from confidently stopping in the wrong location. Nevertheless, **75% of PPCA trials still exhausted the budget**: the high-dimensional localization problem remains largely unsolved at this tolerance.

Generating a Gaussian position and snapping it to a post did not show a compelling advantage. With the common reference Fermat geometry, PPCA proposals located 19.3% of Bluesky targets versus 19.0% for direct actual-post proposals. With PPCA density used for inference, direct actual-post proposals reached 22.0%, versus 16.7% for PPCA-generated proposals. These results apply to the finite proposal banks and tree-guided selection procedure tested here.

The other workloads prevent declaring a universal winner. With actual-post sampling, diagonal/PPCA correct-stop rates were 83.7%/82.7% on axis-aligned 2D topics, 90.0%/90.7% on correlated 2D topics, and 24.0%/21.7% on correlated 32D topics. Uniform density reached 25.3% on the 32D workload, and PPCA had more incorrect confident stops there. The chosen rank, tree cuts, covariance assumptions and density-to-distance approximation all matter.

For Bluesky's 32 leaves and 80,000 training posts per seed, average costs were:

| Density | Additional density state | Online fitting time | PDF cache / box-normalizer preparation |
|---|---:|---:|---:|
| Uniform | 0.5 KiB | <0.001 s | 0 s |
| Diagonal Gaussian | 288.8 KiB | 0.097 s | 0.001 s |
| PPCA rank 4 | 1,064.3 KiB | 1.023 s | 0.578 s |

The shared centroid + 8-bit post store is **22.00 MiB**, plus about 260 KiB of tree geometry and 2.00 MiB of training reservoirs. Density-state figures include fitting moments and Gaussian query caches, but exclude the experiment's graph and likelihood arrays. PPCA's extra state over diagonal is about 0.76 MiB; fitting is about 10.5 times slower in this CPU implementation. The direct-post controls reuse these same density models rather than allocating additional copies.

The evidence supports **centroid + 8-bit residuals for post identity and reconstruction, direct actual-post candidate sampling, and an independently replaceable density model**. Diagonal density is a reasonable inexpensive baseline; retain PPCA as an option to evaluate further rather than requiring it for every leaf or deleting it on the strength of a codec comparison. This experiment does not justify changing the existing streaming split/merge implementation, and synthetic likes do not establish real recommender quality.

## Experimental separation

Every case/seed uses one static tree scaffold, one set of Pólya region probabilities, one compressed post store and the same user hypotheses. Model-dependent codecs cannot account for a difference in this experiment.

The scaffold is the existing observed-range adaptive tree, initially fitted with a diagonal sketch. After its final cuts, all training points are routed again. Exact observed counts determine the shared Pólya branch probabilities, with concentration two and geometric branch priors. Diagonal and PPCA states are fitted with the existing online moment updates in those **same final leaves**. Uniform density needs only the counts/probabilities and cell volumes.

This deliberate refit isolates the value of within-region density. It **does not test the error of transferring historical moments or allocating historical mass during future splits and merges**, and it does not implement a replacement production streaming tree. Trees and users are static during feedback; time decay is disabled for this ablation.

The three settings separate candidate generation from density-dependent inference:

| Setting | Simulated feedback | Inference geometry | What changes between methods |
|---|---|---|---|
| Euclidean: proposals only | Original-vector Euclidean distance | Compressed-post Euclidean distance | Proposal banks |
| Fermat: proposals only | Independent reference density graph | Same reference density construction, with compressed post coordinates | Proposal banks |
| Fermat: density + proposals | Same independent reference graph | Uniform, diagonal or PPCA density graph | Density geometry and proposal banks |

Four strategies run in every setting: uniform locations snapped to posts, diagonal-Gaussian locations snapped to posts, PPCA locations snapped to posts, and directly sampled actual posts. The direct-post baseline uses uniform tree density in the complete setting. Two additional complete-setting controls use **actual-post sampling with diagonal density** and **actual-post sampling with PPCA density**. These controls determine whether a gain requires generating synthetic Gaussian positions or only a different distance model.

The extra controls were added after the first four-strategy run showed a large difference in confidence errors. Original strategies were rerun unchanged; validation compares their deterministic outcome hash with the original run. No model settings were tuned to make PPCA win.

## Data, post identities and hypotheses

Four workloads, each with three seeds:

- Axis-aligned 2D Gaussian topics.
- Rotated, correlated 2D Gaussian topics.
- Correlated 32D topics, with four dominant directions per topic plus small ambient noise.
- All 100,000 archived, 256-dimensional Bluesky embeddings.

Each synthetic workload has 12,000 points: 10,000 training and 2,000 held out. Bluesky uses a seeded 80,000/20,000 split. Seeds vary the split, tree and stochastic audit. This is a static shuffled-data experiment, not chronological replay.

Half of the synthetic held-out set, or 4,096 Bluesky held-out points, defines an independent nearest-neighbor density reference. From another held-out pool, select 64 distinct positions in its denser 70% according to that reference. These are the possible user locations. Their rows are not queryable training posts, although duplicated content or embeddings across the shuffled train/held-out split are not explicitly removed. Targets are sampled uniformly over the 64 hypotheses; inference has the same uniform prior. This finite, known hypothesis set is an easier problem than continuous unknown-location inference.

Each training leaf's posts are encoded in groups of at most 256, using the existing centroid + 8-bit implementation. All methods share the exact same reconstruction matrix and codec hash. Integer UUIDs identify training rows; displayed post rows map back to the original archive. Stored leaf memberships are original-data assignments at encoding time. No later tree changes occur in these trials, so the previous experiment's repeated-cut routing drift is intentionally excluded.

Original training and held-out vectors remain in an audit/simulator workspace. Candidate retrieval uses reconstructed training vectors. The original 64 hypothesis positions are shared known inference anchors. No method knows which anchor is the hidden target.

## Within-leaf density and proposals

Uniform density is the leaf's Pólya probability divided by its box volume. Gaussian densities multiply the same leaf probability by a within-box normalized Gaussian PDF.

Diagonal covariance uses exact online means and marginal scatter. PPCA uses the existing online rank-four sketch and isotropic residual variance, with the existing variance floor of `1e-7`. In 2D, rank is capped at two, giving a full-rank correlated control. The Gaussian log PDF uses the determinant lemma and Woodbury identity, avoiding a dense ambient covariance matrix. Tests compare it with an independent dense-covariance calculation.

Diagonal box normalization is a product of one-dimensional Gaussian interval probabilities. PPCA box normalization is Monte Carlo, with 4,096 independent draws per leaf and recorded minimum acceptance. Rejection sampling generates actual within-box proposals; points are not clipped to cell boundaries. Finite integration error remains, especially for low acceptance. The density-to-distance transform reduces its impact in high dimension but does not remove it.

Each method builds a bank of 64 proposal entries per leaf. A synthetic location is mapped to its nearest real training post **within that leaf**, using the common compressed vectors and Euclidean distance. The direct-post control samples existing post IDs from the same leaf. Banks may contain duplicate post IDs; their unique sizes, rejection counts, snapping distances and generation times are reported. Banks are fixed during each set of trials rather than regenerated after every response.

At each response:

1. Sum the current user posterior in every tree node.
2. Sample four internal splits with weight `left_posterior_mass × right_posterior_mass`.
3. Sample one hypothesis from each side, and four entries from that hypothesis's leaf bank: 32 proposal draws total.
4. Deduplicate and remove already displayed posts; score the remaining posts by expected information gain.
5. If the complete draw is exhausted, use a common actual-post fallback pool. Record the fallback rate.

When all posterior support lies in one leaf, hypotheses are sampled directly from the posterior. This is tree-guided information sampling; it is **not** the earlier shortest-path fractional-probe strategy. The geometry still determines the response probabilities used in its information-gain score.

## Independent distance and response model

The Fermat approximation has 512 vertices: 64 user hypotheses and 448 training-post landmarks. Connectivity is fixed for all methods using a symmetric 12-nearest-neighbor graph plus an MST, built from compressed coordinates. Query posts attach through their three nearest compressed-coordinate vertices. The simulator uses the same edges/attachment indices, but original coordinates and independent density weights. This avoids testing each fitted density against feedback generated by itself.

The reference density proxy is the distance to the 16th nearest independent held-out reference point. Higher reference density means a smaller radius. Its travel multiplier is radius divided by the median landmark radius, clamped to `[0.25, 4]`.

For a fitted density, the multiplier is `density^(-1/dimension)`, normalized by its own landmark median and bounded identically. This dimension correction avoids applying enormous raw high-dimensional PDF magnitudes directly as edge costs. It makes the reference kNN-radius proxy and fitted-PDF costs comparable in form. It is a chosen density-weighted metric, not a validated semantic distance.

An edge costs its Euclidean length times the average endpoint multiplier. Dijkstra distances are normalized so the median pairwise hypothesis distance is one for each geometry. Posts use the minimum weighted attachment-plus-path distance to each hypothesis. These are finite-graph, endpoint-quadrature approximations; they do not compute continuous Fermat geodesics. Global scale is calibrated internally, without observed user responses.

The simulator returns independent Bernoulli feedback:

```text
P(like | post, user) = 0.05 + 0.90 / (1 + (distance / 0.25)^2)
```

Inference uses the same functional form with its own estimated distances. In the complete setting it can therefore be misspecified. Normalized Bluesky embeddings and regularized Gaussian PDFs should be understood as geometric models; their ambient log scores are not calibrated semantic probabilities or actual measured human interests.

## Stopping and evaluation

There are 100 paired targets/feedback streams per seed, an 80-response budget, and two tolerances: 5% and 10% of the reference median hypothesis distance. The strict 5% result is the main comparison; 10% shows sensitivity to coarser localization.

For stopping, inference finds the hypothesis-centered tolerance ball with the greatest posterior mass, using **its own** distances. At the first response with at least 90% mass, it stops. The simulator then checks whether the true target lies within that estimated center's tolerance ball under the **reference** distances. Correct stops, incorrect confident stops and exhausted budgets are reported separately.

The metrics include:

- Correct-stop probability by response count, and false-stop curves.
- Mean responses consumed, including false stops and budget exhaustion.
- Mean among correct stops only, explicitly subject to selection bias.
- A failure-penalized score: first correct stop time, otherwise 81. This is a bounded comparison score, **not** an estimate of the uncensored mean time to locate all users.
- Predictive Brier score, candidate counts, fallback frequency and selection time.
- Density-state memory, fitting/query preparation time, held-out log-density score, proposal uniqueness and density/reference geometry disagreement.

Methods share each trial's target and Bernoulli uniforms. Proposal randomness is separately seeded by case seed, trial and step, never by hidden target. For each seed-one first trial, saved histories continue to 80 responses for visualization; reported first-stop outcomes and consumed-work metrics remain frozen at the first stop.

Paired PPCA-minus-alternative intervals use 1,000 bootstrap resamples of matched trials **within each fixed seed**. Separate comparisons hold actual-post proposals fixed while changing density. They describe Monte Carlo variation conditional on these landscapes/splits; three seeds cannot establish generalization to other corpora or actual users. Seed ranges are also displayed. The complete experiment has **33,600 trials**.

## Memory and performance scope

The density-state memory includes online moment states and cached Gaussian log-PDF parameters, plus the count vector. Uniform density requires just the count vector beyond shared geometry/probabilities. Shared post-codec bytes, tree geometry, and training reservoir memory are reported separately. The actual-post hybrid controls reuse a density model and add no second model instance.

Original audit arrays, landmark graphs, dense likelihood caches, trial records and viewer history are experiment infrastructure, excluded from the displayed density-state figures. They can dominate process RAM. The prototype is not a claim that all those caches scale to millions of posts or users.

All code runs on CPU with BLAS restricted to one thread. A disposable landscape warms the main code paths. Density fitting, proposal preparation and query selection have separate timings. Selection timing excludes precomputed likelihood construction and posterior updates; density fitting excludes common scaffold construction. No kernel/GPU speed comparison is made here.

## Reproduce

```sh
julia --project=. test/density_ablation.jl
julia --project=. examples/density_ablation.jl
python3 scripts/density_ablation_report.py
```

`--smoke` uses eight trials per method/setting and a 20-response budget, writing `smoke.json` rather than replacing full results. The main tests check Gaussian PDFs against dense covariance, normalization, sampler support, Dijkstra against Floyd–Warshall, response bounds, posterior normalization, hidden-target independence, unique query IDs, and correct treatment of confident false stops. The report includes all outcomes, including unsuccessful trials.

The Julia suite passed 57 assertions. Browser checks cover desktop/mobile layout, all experiment selectors and trace playback. The [output audit](../output/density_ablation/validation.json) checks pairing, unchanged codec identities, posterior normalization, stopping outcomes and agreement with the original 28,800-trial run before the two additional controls were added.
