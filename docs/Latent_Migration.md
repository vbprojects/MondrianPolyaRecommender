# Per-post codes through repeated tree splits and merges

[Interactive experiment](../output/latent_migration/demo.html) · [Summary](../output/latent_migration/summary.txt) · [Raw measurements](../output/latent_migration/results.json)

`src/LatentPostCodes.jl` adds an experimental pure-data representation: each `PostCodes` object holds exact UUIDs, a matrix of per-post coordinates, and a saved `Decoder` containing a copied mean and orthonormal basis. Decoder IDs describe versions independently of mutable tree indices. The module supplies encoding, decoding, coordinate mapping, split routing, and sibling-code merging. It is standalone Julia and does not change the production tree's storage or kernel paths.

For saved orthonormal bases and means, transport is:

```text
x_old_hat = mean_old + U_old * z_old
A = U_new' * U_old
b = U_new' * (mean_old - mean_new)
z_new = A * z_old + b
```

This exactly projects the **old reconstructed vector** into the new basis, up to floating-point error. It does not project the unavailable original vector. It needs no inverse, whitening, or division by small eigenvalues. Within a fixed rank, each post costs a small matrix-vector product after computing the shared transform. The shared setup costs roughly `dimension × rank²`; each migration costs roughly `rank²` per post.

A split routes each old reconstruction against the actual axis/cut, then converts its code to the chosen child decoder. A merge converts both sibling sets to the parent decoder and concatenates their UUIDs and codes. Routing an approximation can disagree with routing the original. Reprojection can also move a reconstruction across a cut; these codes are approximate retrieval coordinates, not a guarantee of geometric support inside every inherited leaf bound. Duplicate UUIDs are rejected. Decoder snapshots must never be mutated after publishing them.

## Experiment design

The experiment calls the existing `fit_leaf_tree`, `conditional_children`, `merge_sketches`, and `merge_tree!` implementations through their operation hooks. It uses their actual random cuts, conditional-moment transfers, exponential decay, and sibling collapse. It does not replace these operations with a fitted Gaussian-mixture library. The existing model covariance floor (`1e-7`) and reservoir-seed perturbation remain enabled in every policy; the comparisons vary per-post code transport rather than removing those existing safeguards.

Each run starts with 512 independent warmup observations and a separate fixed cohort of 384 post UUIDs. Every cycle trains on 256 fresh observations, permits up to eight leaves with a split threshold of 32 and a reservoir cap of 32, and then advances 12 half-lives without new arrivals. The merge threshold is 3. Every idle maintenance call must return the tree to one leaf. There are 80 cycles and three independent seeds per geometry.

All policies share the same statistical tree. The cohort's compressed representations never train the tree, so random perturbations cannot change cuts or contaminate another policy's model. Decoder snapshots change at structural events; the live density model continues to learn between events. This isolates code migration through structural changes rather than migrating all codes on every ordinary model update.

| Geometry | Ambient dimension | Code rank | Purpose |
|---|---:|---:|---|
| Full-rank control | 2 | 2 | Change coordinates without discarding dimensions |
| Stable manifold | 32 | 4 | Rank-matched stable directions, very small ambient noise |
| Nearly equal spectrum | 32 | 4 | Eight similarly strong directions compete for four retained directions |
| Very thin direction | 32 | 4 | One genuine direction has tiny variance |
| Rotating manifold | 256 | 4 | Four dominant directions rotate into four other directions and back |

The rotating case completes one full revolution in 80 cycles; the retained subspace already coincides with the original at the half-turn (cycle 40). Endpoint results therefore show how much information survived a return to the original subspace; intermediate checkpoints also show what happens while the basis points elsewhere. The persistent cohort deliberately survives much longer than a realistic post-expiration window. This is a stress experiment, not a claim about the expected lifetime of feed candidates.

## Policies

- **Sequential Float64/32/16:** repeatedly project the most recently stored codes. All arithmetic and decoder parameters remain Float64; only coordinate storage precision changes.
- **Jitter 0.1% / 1%:** add independent Gaussian coordinate noise at each migration, scaled by the new model's typical latent standard deviation. No original cohort values are read.
- **Restore variance with noise:** add Gaussian noise for the positive covariance deficit between migrated codes and the target model's latent covariance. It can inject variance but cannot identify each post's missing residual. Small groups with no more observations than latent dimensions are left unperturbed.
- **Remap from initial code:** retain the initial compressed code and decoder and always project that original reconstruction directly into the target decoder. This avoids repeated projections of already projected values.
- **Refresh originals every 10 cycles:** use ordinary sequential migration between refreshes, then re-encode the original audit vectors after every tenth merged state. Checkpoints immediately before refresh periods expose intervening drift. A real service would need stored originals or re-embedding from text to do this.
- **Original-vector oracle:** re-encode the true original vector at each decoder transition. Split decisions still use the current reconstruction, like other policies. After merging to the root, the representation is a direct projection of the original into the current root decoder.
- **Frozen initial decoder:** preserve the initial codes and decoder. Membership can move between tree regions, but the decoded coordinates stay fixed. It is a reference for keeping post storage independent of the evolving density model.
- **Map singular-value floor 0.1:** replace transform singular values below 0.1 with 0.1. This improves conditioning by deliberately deviating from projection.
- **Orthogonal transport:** replace all transform singular values with one (the orthogonal Procrustes factor). It preserves latent distances for posts undergoing the same transform, but can move their reconstructed ambient coordinates away from the originals. Different sibling transforms and offsets need not preserve distances across regions.

The perturbations are specific heuristics. Their measured results do not prove that every noise distribution or regularization strategy behaves the same way.

## Measurements

Reconstruction NRMSE is `sqrt(sum(abs2, reconstructed-original) / sum(abs2, original-mean(original)))`. Zero is exact; approximately one means error on the scale of the cohort's original spread. A variance ratio of one matches the original cloud's total centered variance, but does not establish that individual points are correct.

Recall@10 uses 24 fixed original-vector queries per run against all 384 reconstructed candidates, excludes each query's own UUID, and compares neighbors with exact original-vector Euclidean neighbors. It does not measure recommendation relevance or human feedback. Near-complete collapse makes distances nearly tied, so tiny recall differences at that point are not robust. Latent covariance condition numbers and entropy-based effective rank measure the code distribution within eligible buckets; neither substitutes for reconstruction or retrieval fidelity.

Each transition checks the unmodified affine formula against explicitly decoding and reprojecting. Each checkpoint verifies that all UUIDs appear exactly once and every reconstructed coordinate is finite. Recorded transform conditioning uses the modified transform for the singular-value-floor and orthogonal policies. The recorded equivalence error concerns the original projection formula before deliberate perturbation. `max_transport_projection_departure` records subsequent departures, including oracle refresh and jitter. Frozen storage performs no transformations; its transform metrics are placeholders.

An independent analytic control repeatedly maps a rank-one code to a basis rotated by an angle and back. Every individual transform has condition number one, yet code amplitude is multiplied by `cos(angle)^2` per round trip. This isolates information contraction from unstable inversion.

## Results and interpretation

The full run executed **6,378 actual splits and 6,378 merges**, with all UUIDs preserved. Maximum affine-map versus explicit decode/project discrepancy was `6.7e-16`. Forward-map singular values remained at or below one to numerical precision. The model's existing floors were left enabled. **The main failure was geometric drift and repeated information loss, rather than an unstable matrix inverse.**

For the stable rank-four manifold, sequential Float64 codes retained 100% of exact neighbors after 80 cycles, with NRMSE about `0.00025`. Float32 storage behaved almost identically. Float16 storage reduced recall to about 98.9%, while adding 1% jitter reduced it to about 71.2%. Random perturbation was unnecessary for this stable geometry.

The rotating 256D case shows why the complete trajectory matters:

| Policy | Recall at cycle 40 | Recall at cycle 80 | NRMSE at cycle 40 | NRMSE at cycle 80 |
|---|---:|---:|---:|---:|
| Sequential Float64 | 0.14% | 62.4% | 1.848 | 0.294 |
| Orthogonal transport | 0.28% | 99.0% | 1.999 | 0.0753 |
| Remap from initial code | 99.4% | 99.4% | 0.0752 | 0.0751 |

Values are means across three seeds. At cycle 40 the subspace has returned to the same plane, but sequential transport has carried posts around the rotating directions to approximately the opposite side of that plane. Orthogonal transport retains spread while preserving this wrong movement. Its impressive final score at cycle 80 alone would conceal the intermediate failure. Initial-code remapping refers back to the saved representation and recovers the original geometry whenever the target space can represent it. While the current space points into orthogonal directions, even this local view loses useful geometry; the frozen original decoder remains available for retrieval.

When eight nearly equal directions compete for rank four, sequential codes collapsed to essentially one point after repeated basis changes (variance ratio around `1e-30`). Final recall was around chance. Tiny jitter did not restore post-specific information. Covariance-restoring noise produced a wider cloud but about 2.5% neighbor recall. Flooring transform singular values at 0.1 also failed to prevent eventual collapse. Orthogonal transport avoided collapse but had NRMSE about 1.71 and only 3.5% recall. Better conditioning or more variance did not establish accurate identity-linked geometry.

The initial-code control cannot restore detail already lost in the first encoding: on the nearly equal-spectrum case it reached only 12.6% recall in the new local view. Keeping the frozen original decoder yielded 29.2%, and refreshing from actual originals yielded 30.3% at the final checkpoint. The rank-four bottleneck itself is substantial for that eight-direction distribution.

These results favor **keeping an immutable compressed representation as the source of truth**, with current-region codes as derived views. On splits/merges, transform from that retained source rather than overwriting it with another projection. Original-vector refresh or an additional residual code is needed when the first compression is insufficient. Per-post random jitter should not be added by default on the evidence of these tests. An orthogonal transport may suit deliberately moving latent objects, but it is not a general way to preserve a fixed post's embedding.

## Reproduce and audit

```bash
julia --project=. test/latent_post_codes.jl
julia --project=. examples/latent_migration.jl
julia --project=. test/latent_migration_audit.jl
python3 scripts/latent_migration_report.py
```

The saved `output/latent_migration/transition_audit.jls` contains the actual original UUIDs, initial codes, and every sequential Float64 split/merge transition for the rotating 256D seed-1 run. Its verifier replays the journal using the independent public split/merge code API, compares codes and decoder IDs, and checks complete, duplicate-free UUID inventories after every event. It also verifies the frozen decoder control. Only deserialize this locally generated, trusted Julia artifact; it is a research artifact, not a durable interoperable storage format.

The original vector cohort is included in that artifact solely for audit. Fixed-cohort frozen/anchor controls retain one initial decoder. A continuous deployment must also measure how many decoder versions remain referenced by newly arriving posts, expire unused versions, and choose a refresh policy. This experiment does not claim bounded decoder storage for an endless stream.

Final verification: **11,876 assertions passed**, including replay of all 844 saved structural events. The complete experiment contains 4,860 measured policy checkpoints; UUID inventories and finite reconstructions passed throughout. The viewer was checked at desktop/mobile sizes with no JavaScript errors. [Validation record](../output/latent_migration/validation.json).
