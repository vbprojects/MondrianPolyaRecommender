# Static user localization from binary feedback

Open [the experiment](../output/localization/demo.html). The left panel shows the static content density, the hidden user and queried post; the right shows the posterior distribution over user locations. Playback compares six query strategies on the same first preselected target. Aggregate curves and outcome tables use all trials.

```sh
julia --project=. examples/localization_demo.jl         # 200 trials/policy, 80 responses
julia --project=. examples/localization_demo.jl 500 120 # more trials and larger budget
julia --project=. test/runtests.jl
```

## Mechanistic assumptions

The user does not move. The content tree does not change during questioning. Time decay is disabled with `half_life=Inf`.

A synthetic set of 4,000 posts forms three dense topic regions, a connecting population and background. Out-of-domain Gaussian proposals are replaced by uniform background points. A seeded Mondrian partition (up to 255 nodes, lifetime 20) with Pólya branch probabilities supplies the density. This experiment uses the existing tree, not kernel smoothing.

A 21×21 grid supplies 441 possible hidden locations. Retain grid states at or above the 65th percentile of estimated grid density, and sample the user proportional to estimated density within those states. Inference receives the same dense-region prior. This is deliberately an easy, matched-model experiment: no niche users, unexpected priors, encoder error or unknown response parameters.

Each observed post is assigned to its grid cell center for the querying experiment. There are 330 occupied candidate locations. Thus this is finite, quantized localization, not continuous-coordinate recovery. Query locations can be repeated; observations are independent given target and query, without fatigue or item-specific preferences. Random queries sample locations proportional to their post counts; other policies select locations explicitly.

## Fermat geometry and likelihood

Connect the grid through horizontal, vertical and diagonal neighbors. The local scalar cost is `clamp(density^(-0.5), 0.2, 5.0)`. An edge costs its Euclidean length times the average endpoint scalar cost. All-pairs shortest graph distances are calculated once and shared by all strategies and trials.

These are grid-approximated Fermat distances. Endpoint quadrature does not integrate every tree-cell crossing, and graph paths do not equal continuous geodesics. The bounds avoid arbitrarily cheap or expensive routes. Graph resolution and cost caps are experimental choices.

Let `s` be 12% of the median all-pairs graph distance. For distance `d` from queried post to user:

```
P(like | query, user) = 0.05 + 0.90 / (1 + d/s)
P(dislike | query, user) = 1 - P(like | query, user)
```

The probability is 95% at zero distance and approaches 5% at large distance. This is a regularized inverse-distance response, not a singular `1/d`. A negative is therefore evidence against a location, not proof that the user cannot be there.

After every response, each state's prior probability is multiplied by its likelihood for that response, then normalized. No target position is accessed during query selection or posterior updating.

## What counts as finding the user?

Exact point identification is not the success criterion. Tolerance is a Fermat radius equal to 10% of the median all-pairs distance, approximately 0.062744 in this experiment.

At every step, find the grid-centered radius ball containing the most posterior mass. That center is the estimate. Stop at the first response where its posterior mass reaches 90%.

Ground truth is used only afterward to classify that first stop as correct or incorrect: is the actual user within the same radius of the estimate? A coincidentally close estimate with low confidence is not counted as a completed localization. An incorrect confident stop stays a failure even if continued diagnostic sampling later fixes it.

The timeframe is an interaction budget, default 80 responses. There is no assumed human response latency. Mapping samples to minutes requires a response-time model; execution time of this Julia simulation is not user interaction time.

## Query policies

- Random posts: draw according to empirical post counts, independent of feedback.
- Highest predicted like: select the post location with greatest posterior expected like probability.
- Information gain: select the post expected to reveal the most information about hidden location. Its score is binary entropy of the predictive response minus posterior-average conditional response entropy. A response probability near 50% is not sufficient if every hidden state predicts the same noisy response.
- Tree representatives + information: select one fixed candidate per occupied tree leaf, using the location with the largest observed count. Maximize information over those 68 representatives. Representative choice does not observe the user. This tests a simple candidate-compression optimization, not the best possible tree-guided active search.

The same density and Fermat geometry are used by every policy. The last strategy changes the query candidate set, not the posterior resolution. Restricting to representatives can prevent informative probing within a region. It should not be interpreted as evidence that the tree itself is unsuitable for active learning; the new path strategies test adaptive proposals as one alternative; a full hierarchical optimization of information gain remains unimplemented.

## Tree-guided Fermat path proposals

Two additional strategies now generate candidates from competing user hypotheses instead of retaining one fixed representative per leaf.

At each response, sum the user posterior within every tree node. Sample four internal splits with replacement, weighting each by `left_posterior_mass × right_posterior_mass`. For each sampled split, draw one hidden-location hypothesis from each child, conditional on its posterior. This favors branches with meaningful competing support; it uses user belief rather than posting volume. Branches are a proposal mechanism, not a semantic-distance assertion.

Recover the shortest graph path between the two hypotheses using Dijkstra predecessor caches. Choose the graph vertex nearest each requested fraction of accumulated Fermat cost, then map that vertex to its nearest occupied post location under the same Fermat distance. Deduplicate the resulting posts and score their information gain against the entire posterior.

- Midpoint only: use the 50% point from each of four paths, at most four candidates.
- Asymmetric probes: use 0%, 20%, 35%, 50%, 65%, 80%, and 100%, at most 28 distinct candidates.

If there is no split with posterior mass on both sides, draw two hypotheses directly from the posterior and propose along their path. This supports refinement within one leaf. If both draws coincide, their nearest post remains a valid fallback.

The weighted midpoint is discretized to a graph vertex, then snapped to an available post; it is not an exact continuous geodesic midpoint. Equal-distance probes alone cannot distinguish a symmetric two-location hypothesis, which motivates the asymmetric fractions and endpoints. Midpoint-only is included as an ablation, not expected to have zero information for every posterior.

Gold lines in the animation show sampled paths, blue dots show candidate posts, and the orange ring shows the selected query. Paths are those proposed before the displayed response; the posterior panel shows belief after that response.

The implementation adds shared predecessor, tree-membership and nearest-post caches. These caches are computed once for the static landscape. Per-response metrics record distinct candidates actually scored, recovered path edges, and query-selection time including proposal generation, through the first stop or budget. They exclude posterior updates, localization estimation, graph construction and all shared precomputation. Timing paths receive a small warmup; reported latency is a local diagnostic, not a production benchmark. Full candidate scoring still visits every hidden state for each proposal.

## Initial measurements

200 independent target/feedback trials per strategy on one shared landscape; the same target list is used across strategies. Success intervals are individual 95% Wilson intervals, conditional on this landscape and simulator, not paired significance tests.

| Policy | Correct within 80 responses | Incorrect confident stop | Exhausted budget | Mean responses consumed | Mean responses among correct stops |
|---|---:|---:|---:|---:|---:|
| Random posts | 23.0% (17.7–29.3) | 1.0% | 76.0% | 71.65 | 45.09 |
| Highest predicted like | 82.5% (76.6–87.1) | 4.0% | 13.5% | 36.45 | 29.88 |
| Information gain | 86.5% (81.1–90.6) | 6.5% | 7.0% | 32.02 | 27.72 |
| Tree representatives + information | 53.0% (46.1–59.8) | 2.0% | 45.0% | 49.32 | 24.09 |
| Tree path midpoint only | 67.5% (60.7–73.6) | 5.5% | 27.0% | 42.94 | 28.85 |
| Tree path asymmetric probes | 90.0% (85.1–93.4) | 5.5% | 4.5% | 32.49 | 29.75 |

Exhaustive information gain reduces consumed responses compared with random sampling. Asymmetric path probes have the highest observed success, but their interval overlaps exhaustive information gain: this does not establish superiority. Its success interval overlaps the greedy policy's; this experiment does not establish a statistically significant difference between those two policies. Its additional incorrect stops also matter.

Asymmetric path probes evaluate an average of 18.26 posts per response, versus 330 for exhaustive information gain (about 18× fewer). Midpoint-only evaluates 3.64 and fixed representatives evaluate 68. Path-proposal overhead means this ratio is not an end-to-end speedup. The extra all-pairs predecessor and nearest-post precomputation is excluded from per-query timing. Those quadratic caches are appropriate to this 441-state benchmark, not an unmodified large-scale deployment.

Exhaustive information gain has at least as much immediate expected information as any subset when evaluated on the same posterior. A sampled strategy's slightly higher realized success in these trials is not a contradiction: random outcomes and different query histories change subsequent beliefs, and one-step entropy reduction does not optimize every stopping objective.

The tree-representative policy's 24.09 success-only average is **not** evidence of better overall efficiency: 47% of its trials do not end in correct localization. The success-only mean excludes those failures. Mean responses consumed includes incorrect stops and counts non-stopping trials at 80. Neither quantity is an uncensored mean number of responses required for correct localization. We cannot estimate that unconditional mean from this finite experiment without additional assumptions or a larger budget.

See [trial-level CSV](../output/localization/metrics.csv) and [summary](../output/localization/summary.txt). `stopping_sample=0` means no stop within budget, not a zero-query success. The `final_error_after_budget` field uses the continued diagnostic trajectory; first-stop correctness is reported separately.

## Implementation and verification

[src/Localization.jl](../src/Localization.jl) contains landscape construction, graph distances, response likelihoods, Bayesian updates, query selection and trial simulation. The likelihood and its binary entropy are precomputed for all query/state pairs. This trades quadratic memory for cheap repeated trial evaluation at 441 states; it is not the intended implementation for millions of posts or 324-dimensional state grids.

The current implementation runs each trial to the full budget for diagnostic histories even after the first stop. Reported response consumption still uses the first stop; policy evaluation timings include that continued simulation.

Additional tests verify shortest-path recovery, accumulated-cost fractions, snapping to available posts, within-leaf fallback, information maximization over proposals, hidden-truth independence and work accounting through the first stop. Tests also verify graph symmetry/connectivity/triangle inequalities, probability bounds, Bayes updates, zero information for uninformative queries, candidate membership, posterior normalization, first-stop semantics and first-query independence from hidden truth. The full suite has 2,383 passing assertions. Browser checks cover playback, strategy changes, scrubbing, reveal controls and mobile layout.

The next useful sensitivity tests are multiple landscape seeds, finer state grids, different response noise/radii, and likelihood mismatch. Those are not included in the reported results. This static benchmark intentionally precedes moving-user tracking and real text embeddings.
