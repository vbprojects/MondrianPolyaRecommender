# Exponential decay for leaf-local PPCA

`src/LeafLatentTree.jl` now accepts chronological timestamps and a positive half-life. Time units are caller-defined and must match. Omitting the half-life preserves the static behavior.

```julia
fit = fit_leaf_tree(X;
    timestamps=seconds_since_start,
    half_life=6*3600,
    configs=[(:ppca, 4)],
    split_after=128)

# Nonmutating query; all leaf/subtree masses evaluated at the same time.
q = leaf_probabilities(fit; time=fit.time + 3600, kappa=2.)
# q.probabilities contains leaf probabilities, zero at internal nodes.
# q.masses includes subtree masses at internal nodes.
```

Before an arrival, the destination leaf's mass, diagonal central scatter, and low-rank scatter eigenvalues are scaled by `exp2(-elapsed/half_life)`. Mean and basis are unchanged. The ordinary online moment update then adds a unit-weight arrival. Between observations the covariance is unchanged except at numerical floors/extinction. At extreme elapsed times weight underflows to zero; subsequent observations restart the statistics.

A split transfers the already-decayed parent moments and mass. Both children start at the split time with zero new-arrival evidence. Their inherited mass still contributes to density. The split threshold uses decayed **new-arrival weight since birth**, preventing ancient arrivals from eventually triggering cuts despite low current traffic.

Decay is lazy during fitting; inactive leaves may have different materialization times internally. At return, all active leaves are advanced to the final timestamp. Existing observer callbacks receive raw nodes and must not interpret intermediate raw masses as synchronized. `leaf_probabilities` evaluates returned models at any current/future time without mutation and rebuilds subtree masses. Its geometric Pólya prior does not decay, so sufficiently faded evidence tends toward the geometric prior. Query cost is linear in node count.

For finite half-lives the split reservoir becomes a bounded FIFO window of the most recent retained examples, including inherited examples at birth. It is not an exponentially weighted reservoir or a time-based expiration guarantee. Static runs retain their previous reservoir policy. Decay alone does not merge nodes. Optional sibling merging now reclaims weak subtrees: set `merge_below` to a positive effective-mass threshold below `split_after`, and `merge_interval` to a positive interval in timestamp units. See [Sibling merging](Sibling_Merging.md) for moment conservation, idle maintenance, and the CPU/GPU comparison. Daily storage expiration is separate; archived data is not deleted. Inactive leaves retain bounded reservoirs until replacement or merging.

The existing one-Gaussian halfspace transfer still ignores inherited box truncation. Low-rank covariance and historical child assignment remain approximate. This API fits a supplied stream in one call; it does not yet expose a resumable tree-insertion service.

## Validation and real-data check

```bash
julia --project=. test/leaf_latent_decay.jl
julia --project=. test/leaf_latent.jl
julia --project=. test/leaf_latent_tracking.jl
python3 scripts/prepare_leaf_decay.py
julia --project=. examples/leaf_decay.jl
```

The decay tests compare against direct exponentially weighted moments, verify split conservation, future-time queries, invalid timestamps, and numerical extinction. The real-data experiment sorts the 89,798 posts with valid 2024 timestamps from the existing 100k archive. It compares disabled decay with a six-hour half-life, using rank 4, a 128 effective-arrival split threshold, and a 2,048-leaf ceiling. This bounded historical sample is not the full platform stream. Results are in [summary](../output/leaf_decay/summary.txt) and [JSON](../output/leaf_decay/results.json). The reported mass TV compares estimated final leaf masses with exact exponentially weighted routing of those same inputs, not recommendation quality.
