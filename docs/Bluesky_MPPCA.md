# Local online MPPCA on 100,000 Bluesky posts

The experiment has run on the same 100,000 saved 256-dimensional embeddings used for the original Bluesky comparison. Outputs are in `output/bluesky_mppca/`. Text and original embeddings remain intact. All observations have weight one; no time decay or timestamp filtering is applied.

| Method | Reachable model MiB | L1 difference from reference ↓ |
|---|---:|---:|
| Reference tree | 7.76 | 0 |
| Ancestor residual | 7.84 | 0.6912 |
| Microclusters | 19.68 | 0.1338 |
| Local online MPPCA | 16.01 | 0.4012 |

Memory was remeasured after deserializing the baseline objects, so it differs slightly from earlier measurements of freshly built objects. All include the shared scaffold as part of each model's footprint; exclude input vectors/text and process working memory. MPPCA's published counts/probability cache adds roughly 30 KiB beyond its reported geometry/summary footprint. Budgets and ownership policies differ, so this is not a controlled proof of superiority.

MPPCA fit took 1.53 seconds after warmup on this machine. Generating approximate tree counts took another 1.06 seconds. It retained 1,318 components at 752 owning nodes, preserving total mass 100,000. It improves over ancestor residuals on this reference-density metric but is less accurate than microclusters. The 2D local-MPPCA result does not carry over directly to this configuration.

## Implementation

The shared 1,911-node scaffold is loaded from `output/bluesky_tree/models.jls`, originally learned from this same dataset. Admission begins at the root and activates one predefined split when a leaf accumulates 128 observations. Historical mixtures remain at their ancestor owner; later arrivals go to newly active children. A local mixture permits two components, each with up to four principal directions. A second component may be seeded after 16 effective observations when the next vector is farther than 0.35 from every local mean. Thereafter responsibilities are soft, using component masses and Gaussian likelihoods.

Each weighted observation updates the component mean, total scatter trace and a rank-truncated scatter eigensystem. The update diagonalizes a matrix of at most 5×5 and projects back into 256 dimensions. Discarded trace supplies an isotropic noise estimate; noise has a floor of 1e-6. This is an approximate online PPCA-mixture estimator with truncated incremental PCA, not an exact implementation of a published online EM algorithm. Old assignments are never revisited; seeding, truncation and order influence results.

Global first moments and total second moment are conserved, but the complete covariance is not. The rank-four approximation cannot recover discarded directions. Some small components have fewer than four directions. This implementation stores summary parameters, not an individual component code and latent coordinate vector for every post. It is not yet the versioned embedding codec discussed earlier.

## How density is compared

Each component generates 256 deterministic-seed Gaussian samples. Coordinates outside its owning box are clipped into that box, respecting the tree's half-open cut convention. Samples supply fractional counts on the finest reference partition, followed by the same Pólya branch posterior calculation. This is a **censored Gaussian approximation**, not exact sampling from a Gaussian conditioned on a high-dimensional box. It can distort shape and add boundary mass. Density remains piecewise uniform on the tree leaves, not the direct MPPCA likelihood.

L1 is exact between the resulting published rectangular densities, but includes both PPCA modeling error and approximate allocation error. A second allocation seed changed the published density by L1 0.0339. Increasing samples from 64 to 256 changed reference error from 0.4042 to 0.4012; the difference from microclusters remains much larger than that change. Two seeds are a sensitivity check, not a confidence interval.

Because the input vectors lie on a unit sphere and local PPCA is an ambient Gaussian model, generated vectors need not lie on that sphere. Rank, noise, component count and allocation assumptions warrant further experiments. No recommendation quality, held-out likelihood or codec reconstruction advantage is established here. A Cartesian histogram is not included: 16 bins per coordinate would require `16^256` cells.

## Run and load

```sh
julia --project=. examples/bluesky_mppca.jl
julia --project=. test/bluesky_mppca.jl
```

```julia
# Defines the experiment's component type without executing main().
include("examples/bluesky_mppca.jl")
saved = deserialize("output/bluesky_mppca/models.jls")
```

Use trusted Julia serialization artifacts with matching source/Julia versions. `models.jls` includes the input manifest and configuration. `components.csv` lists component masses, ranks, noise and retained-scatter fractions. `leaves.csv` lists approximate counts, probabilities, reference probabilities and log densities. `summary.txt` records the run.

Tests validate weighted moments, orthonormal bases, rank bounds, agreement with batch covariance when no rank is discarded, local mixture growth, normalized density, tree mass conservation and owner preservation after boundary clipping. The full 100,000-post run separately checks count, first moments, total second moment, finite parameters and component budgets.
