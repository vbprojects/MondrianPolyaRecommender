# Leaf-only latent Gaussian experiment

[Open the report](../output/leaf_latent/demo.html) · [Results](../output/leaf_latent/results.json) · [Summary](../output/leaf_latent/summary.txt)

This implements the proposed change in ownership: **only active leaves retain Gaussian models**. On a split, the parent transfers its distribution to two newly fitted child models and releases its parameters and reservoir. There is no frozen root mixture or historical ancestor mixture. Internal nodes retain routing geometry only.

The first comparison uses **one component per leaf**. Collectively, the leaves form a tree-organized mixture. Multiple components within each leaf are deliberately deferred so that this experiment isolates covariance representation and split transfer.

## Representations

Five variants share exactly the same tree topology and input order:

- Diagonal Gaussian: mean plus coordinate variances.
- Rank-4 PPCA: learned directions plus isotropic residual noise.
- Rank-8 PPCA.
- Rank-4 factor plus diagonal noise.
- Rank-8 factor plus diagonal noise.

The low-rank-plus-diagonal variants are covariance sketches, not maximum-likelihood factor-analysis solvers. Factor magnitudes are globally shrunk if necessary to leave a positive diagonal remainder. PPCA noise uses discarded scatter trace. All covariance evaluations and updates use diagonal vectors, low-rank factors and small matrices; no dense 256×256 covariance is stored or factorized.

## Train/test protocol

The existing 100,000 normalized 256D Bluesky vectors are split using seed 192 into 80,000 training and 20,000 held-out observations. Unlike prior scaffold-based experiments, **the tree is learned online from training observations only**, using bounded leaf reservoirs. No test point is used to choose cuts, estimate moments or fit latent directions.

Root support is fixed a priori at `[-1,1]^256`, appropriate for these normalized vectors. The original vectors are not rescaled. Each variant sees the same observations; the experiment maintains their states side by side for controlled comparison.

A leaf becomes eligible for splitting after 512 new observations since its creation. There is a 64-leaf budget and depth cap 16. Axis selection is proportional to observed coordinate ranges in its reservoir. The threshold is uniform between the empirical 25th and 75th percentiles of that coordinate, avoiding tiny children. These are data-adaptive Mondrian-style cuts, **not an exact Mondrian process**. The same topology for every variant isolates leaf-model differences, but does not test model-specific cut optimization.

Each leaf retains at most 64 actual Float32 examples. Parent retained examples are routed into children. Child reservoirs then use standard reservoir replacement over inherited examples plus subsequent arrivals. An inherited example stands for one retained seed; historical observations are not all represented uniformly after repeated splits. This is a bounded, recent-data-biased reservoir, not an exact uniform sample of all historical members. It guides cuts and subspace initialization only; it does not supply a second copy of moment mass.

## What happens at a cut

For the parent's current Gaussian approximation, calculate the probability, mean and covariance diagonal for each side of the new halfspace. The Gaussian covariance is represented by a low-rank factor and diagonal noise. Its column along the split axis suffices for analytic conditional-moment calculations.

Child subspaces are initialized using their own retained real examples, refined by eight covariance-operator subspace iterations, and fitted to the conditional covariance without materializing it. Child means and masses come from the conditional target. Subsequent actual arrivals update weighted moments and the incremental scatter sketch. Parent states and samples are emptied after child creation.

Mass and weighted mean are conserved at transfer. Before recompression, conditional covariance plus between-child mean variance reproduces the parent Gaussian covariance diagonal up to roundoff. Recompression can change those diagonal values: PPCA forces isotropic residual noise, whereas diagonal and low-rank-plus-diagonal variants preserve the target diagonal up to the variance floor. The report records the maximum transfer and compression discrepancies separately.

**Crucial approximation:** conditioning formulas apply to the untruncated parent Gaussian and the new cut, not the Gaussian already conditioned on every inherited tree boundary. Consequently transferred child masses and moments approximate historical membership. Their sum remains correct, but their allocation can be wrong. Reported training mass total variation compares final transferred leaf weights with an exact rerouting of the archived training vectors, used only for evaluation.

## Sampling and Pólya probabilities

Sampling draws from a leaf's Gaussian and rejects every draw outside that leaf's box. No clipping, boundary projection, centroid replacement or fallback sample is used. For each of 64 held-out target regions, each model attempts to produce eight samples, with a maximum of 4,096 draws per region. All 512 requested samples were produced for each variant in this run. Rejection therefore samples the fitted Gaussian conditioned on its box, but does not retroactively correct the approximate moment-transfer masses.

Tree branch probabilities use the same Pólya mean formula as the earlier prototype: prior mass 2 centered on the split-axis volume ratio, combined with transferred subtree masses. Leaf probabilities sum to one. We report average test-region log probability, not continuous Gaussian log density: the latter would require the box normalization constants for valid comparisons.

## First run

| Model | Training state incl. tree/reservoir | Float32 decoder payload* | Held-out reconstruction cosine | Rejection acceptance | Best proposal IG |
|---|---:|---:|---:|---:|---:|
| Diagonal | 4.78 MiB | 128.5 KiB | 0.7699 | 40.1% | 0.1270 bits |
| PPCA rank 4 | 5.28 MiB | 320.8 KiB | 0.9703 | 33.0% | 0.1631 bits |
| PPCA rank 8 | 5.78 MiB | 576.8 KiB | 0.9893 | 31.0% | 0.1391 bits |
| Low-rank + diagonal, rank 4 | 5.28 MiB | 384.5 KiB | 0.9699 | 43.0% | 0.1334 bits |
| Low-rank + diagonal, rank 8 | 5.78 MiB | 640.5 KiB | 0.9891 | 45.0% | 0.1285 bits |

*Calculated inference parameter payload only: Float32 means, factors and noise plus one Float64 mass per leaf. It is not a saved per-post codec size. It excludes tree geometry, reservoirs, latent codes, post IDs and object overhead. The actual training state is Float64 for model statistics and Float32 for retained observations. Reservoirs account for approximately 4 MiB of the training footprint; shared geometry is about 0.51 MiB. The side-by-side experiment process holds all five variants and the evaluation archive, so process RSS exceeds these per-variant figures.

The combined five-model fit took 4.91 seconds on this machine after warmup. Per-model update and split times are recorded in results.json; they exclude reservoir/routing overhead and evaluation. These are local measurements, not production benchmarks.

Reconstruction uses the unquantized orthogonal projection into each leaf's retained basis. The diagonal model has no reduced-rank code and is reconstructed as the centroid. This measures the usefulness of retained low-rank coordinates, **not a storage-matched codec comparison**; a diagonal model could retain all 256 coordinates at extra cost. Rank 8 reconstructs better but does not win on proposal information.

## Proposal evaluation

Each accepted synthetic sample is mapped to its nearest actual **training** post among 80,000 candidates using original vectors. Held-out targets are never eligible for retrieval. The nearest post may leave the generating region; the fraction is reported. The Gaussian samples themselves always remain inside the owner box.

We evaluate 100 fixed synthetic beliefs over 64 held-out locations. Each method gets 32 shared hypothesis draws, then one of its eight region proposals. Unsuccessful regions get no replacement; duplicate retrieved posts are deduplicated. All actual candidates use a common chord-distance binary likelihood: `0.05 + 0.9/(1 + (distance/scale)^2)`. Scale is fixed from held-out hypothesis distances, not selected separately per method. This is an assumed simulator, not human feedback.

Rank-4 PPCA gains 0.0360 bits over diagonal proposals, with paired bootstrap interval [0.0281, 0.0438]. Rank-8 PPCA gains 0.0121 [0.0052, 0.0186]. Both low-rank-plus-diagonal differences have intervals spanning zero. Sampling actual local training posts gives 0.1695 bits, versus 0.1631 for rank-4 PPCA. The baseline is included in the detailed report.

These are one-step information-gain measurements, not sequential localization success. Intervals resample belief cases and are conditional on one training split, one topology and one sample bank. They do not establish broad superiority. The changed train/test protocol and topology mean these values should not be compared numerically with the previous 64-hypothesis experiments.

## Reproduce

```sh
julia --project=. examples/leaf_latent_experiment.jl
julia --project=. test/leaf_latent.jl
.venv-embeddings/bin/python -m unittest discover -s test -p test_leaf_latent.py
```

Core implementation: `src/LeafLatentTree.jl`, a standalone Julia module using standard-library linear algebra. The Julia experiment trains models, measures transfer/reconstruction, and samples. The existing isolated NumPy bridge performs full-archive proposal retrieval and generates the report. No extra dependencies were added.

Outputs in `output/leaf_latent/`: `models.jls`, `experiment.json` with exact train/test row IDs and samples, `results.json`, `summary.txt` and `demo.html`. Load trusted serialized models after including `src/LeafLatentTree.jl`; source and Julia version must be compatible.

Tests cover analytic standard-normal split moments, historical mass and global first-moment conservation, parent-state removal, bounded rank, positive covariance, sampling support and rejection budgets. Artifact tests check disjoint train/test data, finite metrics, split errors and all saved sample bounds. Original datasets and earlier experiments are unchanged.
