# Static trees on 100,000 Bluesky posts

The models are saved in `output/bluesky_tree/models.jls`. All **100,000 posts and all 256 embedding dimensions** are used, without dimensionality reduction, coordinate rescaling, or time decay. Original text and vectors remain in `data/bluesky_100000/`; compression affects the model's retained statistics, not the archive.

## Construction

The root is the empirical bounding box, padded slightly in every coordinate. Splitting picks a coordinate with probability proportional to its observed range and a uniform cut inside that range. Exponential geometric waiting times use the sum of observed ranges. The most populated pending cell is refined first. Splitting stops below 128 posts, at depth 24, or at the 2,047-node budget. Seed 7 produces **1,911 nodes / 956 leaves**. Some leaves can exceed 128 posts because of the depth or node limits.

This extends the prototype with a **data-adaptive Mondrian-style heuristic**, not an exact Mondrian process or projective online Mondrian–Pólya Forest. Observed-range cuts, occupancy stopping and mass-priority scheduling alter the process. Cut times are geometric metadata, not post timestamps. The complete rectangular cells cover the padded root box; they are not fitted topic boundaries.

Reference branch probabilities use the Beta posterior mean `(left_count + 2q)/(left_count + right_count + 2)`, where `q` is the fraction of the parent's split-axis interval assigned to the left child. Leaf densities are uniform inside their boxes. Volumes and densities are handled in log space to avoid underflow in 256 dimensions; branch probabilities use one-dimensional interval ratios.

The two compressors activate branches prospectively on this shared scaffold, starting at the root. They see the posts in their saved, seeded random order. A leaf activates its predefined split when its local summary count reaches 128; at most one new split is activated per arrival. This experiment learns the scaffold from the whole corpus first. It is not a one-pass topology learner, held-out evaluation or temporal replay.

## Ancestor residuals

An incoming post contributes one count to the active leaf containing it. When that leaf splits, historical count stays at that ancestor and both children start with zero local evidence. Future posts reach the new children. Each post contributes at exactly one node; subtree counts include local residual plus descendants.

For prediction, each node's residual defines a uniform component over its entire box. A root-uniform prior of mass 2 is added, and the mixture is normalized by `100000 + 2`. Its mass passes down to child cells according to geometric volume ratios. This does not invent child assignments for discarded observations. It is a normalized mixture model rather than the reference's Beta branch-count posterior.

Early evidence therefore remains coarse even when later observations create deeper branches. This run retains **752 nonzero residual counts across 395 active leaves**. The sum of local residuals is exactly 100,000. No pooling/merging or expiry is triggered in this static fit.

## Microclusters

Each active leaf holds up to four microclusters. An incoming vector joins the nearest Euclidean centroid, unless its distance exceeds 0.2 and a cluster slot is available. Clusters store count, a 256D centroid, and a 256D diagonal central sum of squares, updated with Welford's method. These are ordinary moment summaries, **not mixture PPCA**; no local subspace or off-diagonal covariance is fitted.

When a leaf splits, each entire cluster is routed to one child using its centroid. A cluster's true members might straddle the cut: the retained moments cannot reconstruct their individual destinations. Consequently branch counts are approximate even though global count, coordinate sums and diagonal second moments are conserved. Later arrivals update clusters within the active leaf they actually reach.

The final clusters supply subtree counts for the same Pólya branch posterior as the reference. Density remains piecewise uniform over tree leaves; this implementation does not evaluate a Gaussian mixture from cluster variances. The run retains **2,946 clusters across 934 active leaves**. On the original unit vectors, Euclidean distance is related to cosine distance, but cluster centroids need not have unit norm; nearest-centroid assignment retains that geometric bias.

## Recorded comparison

| Model | Active leaves | Summary records | Reachable memory | Fit time | L1 difference from reference |
|---|---:|---:|---:|---:|---:|
| Reference | 956 | 956 leaf counts | 7.77 MiB | 0.572 s | 0 |
| Ancestor residual | 395 | 752 nonzero residuals | 7.85 MiB | 0.077 s | 0.6912 |
| Microclusters | 934 | 2,946 clusters | 19.79 MiB | 0.185 s | 0.1338 |

Timings exclude compilation warmup, input loading and export. Compressor times exclude the shared scaffold fit. Memory counts each model with its scaffold and allocated statistics, excluding the external embeddings/text. These budgets are not byte-matched. The reference already stores only geometry and counts: neither compressor is smaller than that static count-only baseline. Both avoid owning the 97.66 MiB raw vector matrix; the microcluster model retains additional moments for possible future adaptations.

All models represent 100,000 posts and normalize to one within `2.3e-16`. L1 is integrated over the common finest rectangular partition; its range is 0 to 2. It measures approximation to this reference, not semantic quality or recommendation performance. High-dimensional rectangular volume is an ambient-space model for unit-normalized vectors, which occupy a spherical manifold. Good normalization does not establish a useful density on that manifold.

## Reproduce and inspect

```sh
julia --project=. examples/bluesky_tree.jl
julia --project=. test/runtests.jl
```

```julia
using MondrianPolyaToy, Serialization
include("examples/load_bluesky.jl")
X = load_bluesky_embeddings("data/bluesky_100000")
saved = deserialize("output/bluesky_tree/models.jls")
tree = saved.partition
micro = saved.models["microclusters"]
residual = saved.models["residual"]
p = embedding_probabilities(tree, micro)
leaf = embedding_leaf(tree, X[:, 1], micro.active)
log_density = embedding_logdensity(tree, X[:, 1], micro; probabilities=p)
```

Load only trusted Julia Serialization files with compatible source and Julia versions. The artifact embeds the input manifest, including hashes, and configuration. Portable CSV exports are also provided:

- `nodes.csv`, `bounds.csv`: split topology, counts, depth, geometric times, log-volumes and all coordinate bounds. Node and axis indices are one-based; axis zero denotes a leaf.
- `reference_leaves.csv`, `residual_leaves.csv`, `microclusters_leaves.csv`: active leaves, probabilities, log-densities and subtree counts.
- `assignments.csv`: zero-based `embedding_row` plus the three leaf assignments. Join to `posts.jsonl` using `embedding_row` to inspect the text in any cell.
- `residuals.csv`: local ancestor masses and active split flags.
- `microclusters.csv`: count, mean and central sum of squares per cluster/coordinate. Counts repeat across coordinate rows; count each cluster once. Divide central sum of squares by count for population variance.
- `summary.txt`: measured statistics and caveats.

Tests check branch count conservation, finite log-volumes including constant coordinates, reference assignments, normalized predictions, prospective ancestor ownership, cluster budgets and global moment conservation. The saved 100,000-post models were reloaded and their total residual mass and microcluster first/second moments checked against the original vectors.

The [local online MPPCA experiment](Bluesky_MPPCA.md) now compares rank-four summaries on the same 100,000 embeddings.
