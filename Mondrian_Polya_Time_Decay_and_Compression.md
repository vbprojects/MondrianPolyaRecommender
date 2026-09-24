CONCEPTUAL MODEL AND TECHNICAL DESIGN

Mondrian–Pólya  
Density Estimation

With Time Decay and Compression

A bounded-memory probability landscape for local and shared-feed recommendation

20 September 2026 \| Version 1.0

<img src="Mondrian_Polya_assets/media/image1.png" style="width:6.75in;height:2.475in" alt="An adaptive map: geometry determines resolution; probability allocation determines concentration. Schematic, not empirical results." />

An adaptive map: geometry determines resolution; probability allocation determines concentration. Schematic, not empirical results.

| **Central principle** Decay evidence continuously, revise geometry deliberately, and retain only the information needed for the revisions you intend to support. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|

This document consolidates the discussion into an implementable research design. It covers the published Mondrian–Pólya foundation, exponential forgetting, exact and approximate compression, structural inference, computational budgets, density-derived distance, and the surrounding recommendation pipeline.

Published constructions, mathematical consequences derived here, and proposed engineering extensions are distinguished throughout. No implementation benchmarks or recommendation-quality gains are claimed.

DECISION BRIEF

# Executive summary

The desired object is an **adaptive probability map**, not a collection of Gaussian topics. A recursive partition describes where the model can distinguish regions; mass allocations and a within-region density describe what it believes occupies those regions. More detail can be allocated where useful without retaining every historical post.

Mondrian–Pólya trees provide a relevant starting point, but they do not automatically supply all the behavior discussed. Exponential forgetting, hard memory budgets, demand-driven refinement, compressed historical evidence, and preference-aware traversal are separate choices. Combining them changes the statistical model and must be tested rather than inherited as a guarantee.

## Recommended starting point

Build a small reference implementation with a **bounded retained embedding set, explicit wall-clock decay, and fixed installed cuts between scheduled structural updates**. Use it to verify mass accounting, density normalization, deletion semantics, and measured memory. Then compare it with a summary-only design that leaves uncertain historical evidence at ancestors.

| **Decision**       | **Recommended initial choice**                              | **Reason**                                       |
|--------------------|-------------------------------------------------------------|--------------------------------------------------|
| Representation     | Fixed encoder and coordinate transform                      | Old cuts and summaries remain interpretable.     |
| Probability update | Lazy discounted counts; baseline priors kept separate       | Constant extra work per visited branch.          |
| Geometry           | Bounded rebuilds or supported insertion/deletion            | Counts alone cannot repair historical extrema.   |
| Compression        | Counts first; moments and a small split sketch where useful | Pay for future questions, not unused statistics. |
| Uncertainty        | One tree or a small forest before topology particles        | Measure sensitivity before multiplying state.    |
| Retrieval          | Separate bounded post-ID cache/index                        | A density cannot reconstruct actual posts.       |

## The most important tradeoff

You can have exact statistics for a fixed representation, or approximate flexibility to construct new representations from compressed history. You cannot generally retain a constant-size summary that answers every future spatial cut exactly. A sound implementation exposes this limitation rather than disguising it as a posterior calculation.

The longer-term research candidate is a **multiresolution residual-evidence density**: old observations remain represented at the resolution at which they were compressed, while new observations populate finer descendants. This is a proposed extension, not the unchanged published streaming Mondrian–Pólya algorithm.

SCOPE AND CONVENTIONS

# Reading guide and notation

Sections 1–16 develop the density estimator itself. Sections 17–20 connect it to candidate acquisition, two-stage embeddings, ONNX deployment, and shared infrastructure. Sections 21–22 provide evaluation and implementation decisions. Appendices give worked checks and pseudocode.

| **Part**                  | **Read for**                                                                                      |
|---------------------------|---------------------------------------------------------------------------------------------------|
| 1–3 Foundations           | What is stored, how a tree becomes a density, and what the published method supplies.             |
| 4–5 Forgetting            | Evidence decay versus structural change; absolute activity versus normalized probability.         |
| 6–9 Compression           | Sufficient statistics, sketches, residual evidence, and the information that cannot be recovered. |
| 10–14 Implementations     | Retention policies, Bayesian inference, data structures, and quantified resource accounting.      |
| 15–16 Geometry            | A density-weighted distance and its numerical and semantic limitations.                           |
| 17–22 System design       | Sampling, encoders, hosting, calibration, and staged implementation.                              |
| Appendices and references | Numerical examples, algorithm contracts, and primary sources.                                     |

## Notation

| **Symbol**          | **Meaning**                                                                     |
|---------------------|---------------------------------------------------------------------------------|
| xᵢ, tᵢ              | A fixed-version embedding in ℝᵈ and its chosen event time.                      |
| T, Rᵥ, h, F         | Partition tree, node region, actual traversal depth, and forest size.           |
| β, H                | Wall-clock decay rate and half-life H = ln(2)/β.                                |
| Λ, τᵥ               | Mondrian refinement lifetime and cut time; not wall-clock age.                  |
| mᵥ, sᵥ, Qᵥ          | Discounted mass, first moment, and second moment for a defined observation set. |
| Pₜ(R), pₜ(x), Aₜ(x) | Regional probability, normalized density, and activity-weighted density.        |
| B, N, K             | Retained point budget, allocated tree nodes, and auxiliary summary budget.      |
| rᵥ, Sᵥ              | Residual evidence stored directly at a node, and its cached subtree subtotal.   |

**Status labels:** “Published” denotes a cited method. “Derived” denotes equations following from stated assumptions. “Proposed” denotes an engineering or modeling choice requiring validation. All numerical budgets are illustrative calculations, not measured throughput.

CONCEPTUAL FOUNDATION

# 1. The map is the compressed representation

A standard axis-aligned split asks whether one coordinate is below a threshold. Successive answers define a region. The path through the tree stores those constraints compactly; no sphere, ellipsoid, centroid, or covariance is required to represent a basic partition cell.

Compression occurs when many embeddings are replaced by **a region and its evidence**. A leaf may remember that a large amount of recent activity lies somewhere inside its bounds, without remembering each location. Complicated density shapes arise from unions of cells, not from forcing one component to have a complicated parametric shape.

<img src="Mondrian_Polya_assets/media/image1.png" style="width:6.75in;height:2.475in" alt="Figure 1. A new cut can increase available resolution without changing density. Later evidence can establish a genuine contrast between the children." />

Figure 1. A new cut can increase available resolution without changing density. Later evidence can establish a genuine contrast between the children.

## Three independent roles

**Geometry** defines which spatial questions are distinguishable. **Probability** estimates how mass is allocated among the regions. **Retention** determines which future geometry changes remain supportable. Combining these in one data structure does not make them the same quantity.

Optional moments serve extra purposes: proposing cuts, detecting local heterogeneity, or defining directional travel costs. They do not turn the cell into an ellipse. An explicit Gaussian leaf is a different modeling choice and must be normalized over its intended support.

## A topic is not necessarily a leaf

A conversation may occupy several neighboring cells, one subtree, or overlapping neighborhoods under different trees. Calling every leaf a category is a useful interface shortcut, not a discovered ontology. A region with more posts deserves consideration for refinement, but repeated identical posts do not necessarily reveal additional semantic structure.

| **Design constraint** Keep the coordinate system fixed while its tree is active. Changing the encoder, tokenization, projection, or normalization changes the meaning of old boundaries. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

PUBLISHED FOUNDATION VERSUS EXTENSIONS

# 2. What the published construction supplies

**Published foundation.** Dickens et al. combine Mondrian partitions with Pólya probability allocation and give a streaming variant with insertion and deletion procedures. Its density representation includes an extra distinction between observed bounding boxes and the complementary space around them. This matters because observed child boxes alone need not cover their parent. \[1\]

A Mondrian tree uses randomized, geometry-dependent cuts. Its online extension relies on a consistency property called projectivity: the appropriate extension has the same partition distribution as constructing the model on the enlarged observation set. This is not equivalent to selecting whichever new cut maximizes a user-defined objective. \[2\]

## A minimal cut mechanism

For a rectangular region with coordinate spans Δ₁,…,Δ_d, a schematic Mondrian construction draws an exponential cut increment with rate equal to their sum, chooses an axis in proportion to its span, and selects a location on that axis. The refinement budget Λ determines how far the construction proceeds. \[2\]

$$L_{R} = \sum_{j = 1}^{d}\Delta_{j},\quad\quad E \sim Exp\left( L_{R} \right),\quad\quad\Pr(J = j) = \Delta_{j}/L_{R}$$

The model is sensitive to coordinate scaling: rescaling one feature changes its contribution to the cut rate. Fix feature scaling with the encoder version. Axis alignment is inexpensive and easy to route through, but an oblique semantic structure may require many cells.

## Why the streaming density needs special treatment

The streaming paper separates an ordinary left/right cut from restriction to the observed child box. The remaining space is represented implicitly rather than silently discarded. Complement regions need not themselves be single rectangles, and degenerate observed boxes require special handling. \[1\]

## What the proposed system adds

| **Feature**                                    | **Status in this design**                                                             |
|------------------------------------------------|---------------------------------------------------------------------------------------|
| Exponential aging of branch evidence           | Explicit extension with a stated weighted-likelihood interpretation.                  |
| Hard node and memory budgets                   | Engineering constraints; not supplied by the word “nonparametric.”                    |
| Traffic- or utility-driven refinement          | Alternative split policy; original Mondrian guarantees do not automatically transfer. |
| Historical residual mass at internal nodes     | Different multiresolution density model.                                              |
| Semantic shortest paths and private user state | Downstream applications, not consequences of density fitting.                         |

DEFINITIONS AND NORMALIZATION

# 3. From partition mass to density

For the following derivation, assume a finite partition of a bounded domain Ω into disjoint cells Rℓ. Let θᵥ be the conditional probability of taking the left branch. The right branch has probability 1 − θᵥ. Starting with root mass one yields consistent child probabilities:

$$P\left( R_{L} \right) = P\left( R_{v} \right)\theta_{v},\quad\quad P\left( R_{R} \right) = P\left( R_{v} \right)\left( 1 - \theta_{v} \right)$$

A leaf probability is a path product. Choose a normalized within-cell density qℓ supported in its cell; then the whole density is:

$$p_{t}(x) = \sum_{\ell}^{}P_{t}\left( R_{\ell} \right)q_{\ell}(x),\quad\quad\int_{R_{\ell}}^{}q_{\ell}(x)\, dx = 1$$

For uniform cells, qℓ(x) = 1/Vol(Rℓ) inside the cell. The global density there is P(Rℓ)/Vol(Rℓ), not the conditional density itself. Mass is dimensionless; density depends on the reference measure and coordinate units.

## A baseline-adjusted formulation

Instead of uniform volume, choose a fixed positive reference density b with regional mass B(R). Use its normalized restriction inside each cell:

$$q_{\ell}(x) = \frac{b(x)\mathbf{1}_{R_{\ell}}(x)}{B\left( R_{\ell} \right)},\quad\quad p_{t}(x) = b(x)\frac{P_{t}\left( R_{\ell} \right)}{B\left( R_{\ell} \right)}\quad\left( x \in R_{\ell} \right)$$

This turns P(R)/B(R) into a concentration ratio relative to the baseline. It avoids interpreting equal Euclidean volume as equal semantic opportunity, but requires computing or approximating B(R). A learned baseline introduces its own uncertainty and must not be treated as cost-free.

## Density-preserving refinement

When splitting R, assign child mass in proportion to the chosen reference: P(RL) = P(R)B(RL)/B(R), and similarly right. Both children initially retain the same density multiplier as their parent. A boundary appears without creating evidence. Subsequent observations can justify different child concentrations.

## Support must be explicit

Unit-normalized embeddings lie on a sphere, so their ideal distribution need not have an ordinary full-dimensional Lebesgue density. A rectangular density is then a chosen finite-resolution approximation. Alternatives include a reduced fixed projection, a directional reference measure, or a deliberately smoothed model. Handle out-of-domain points explicitly; do not silently clamp them into a boundary cell.

EXACT UPDATE FOR A SPECIFIED TARGET

# 4. Exponential forgetting without tree sweeps

Define the stream being modeled first: all admitted public posts, a random sample, or a user-directed candidate stream. Choose event time consistently—publication time or ingestion time—and record that choice. With a fixed rate β, the effective mass in a fixed region is:

$$m_{v}(t) = \sum_{i:x_{i} \in R_{v}}^{}e^{- \beta\left( t - t_{i} \right)},\quad\quad H = \frac{\log 2}{\beta}$$

If the last materialization was at t₀, all old evidence receives the same factor a = exp\[−β(t − t₀)\]. Updating only when a node is accessed is therefore exact for these additive summaries. A late-arriving post contributes its current weight rather than necessarily contributing one.

$$c_{L} \leftarrow ac_{L} + w\mathbf{1}_{L}(x),\quad\quad c_{R} \leftarrow ac_{R} + w\mathbf{1}_{R}(x)$$

## Keep baseline priors separate

For a fixed split with a Beta(a₀,b₀) prior, the convenient discounted update is Beta(a₀+cL,b₀+cR). Its predictive left probability is:

$${\widehat{\theta}}_{v}(t) = \frac{a_{0} + c_{L}(t)}{a_{0} + b_{0} + c_{L}(t) + c_{R}(t)}$$

Decay the accumulated evidence, not the baseline prior. Without new observations the smoothed prediction returns toward that prior. By contrast, the ratio of two raw counts decayed at the same rate remains unchanged. These are different, intentional behaviors.

## Statistical interpretation

The update is conjugate for the chosen power-likelihood target on fixed regions:

$${\widetilde{p}}_{t}(\theta) \propto p_{0}(\theta)\prod_{i}^{}p\left( x_{i} \mid \theta \right)^{\exp\left\lbrack - \beta\left( t - t_{i} \right) \right\rbrack}$$

Generalized Bayesian updating supplies a framework for such evidence weighting. It is not automatically the ordinary posterior of a latent process with changing topology; predictive calibration must be checked. \[5\] Choosing cuts using the same observations also requires care before interpreting plug-in split scores as fully Bayesian model evidence. \[12\]

## Parent–child aging is not a dependency chain

For disjoint child observation sets using the same clock and kernel, mP = mL + mR at any common time. All three quantities age together. A parent does not wait for descendants to be removed. With multiple half-lives, keep separate internally consistent statistic sets rather than giving parent and child unrelated decay rates.

STRUCTURAL FORGETTING

# 5. Aging evidence is not erasing geometry

An old outlier can have negligible weight while still defining a bounding-box extreme. Multiplying its weight does not move that coordinate. A minimum, maximum, or arbitrary historical rank cannot be recovered from counts and moments after the supporting observations have been discarded.

| **Operation**                | **What can remain exact?**                          | **What must be decided?**                             |
|------------------------------|-----------------------------------------------------|-------------------------------------------------------|
| Decay evidence               | Timestamped additive counts and moments             | Half-life, event time, numerical precision.           |
| Collapse a subtree           | Its pooled counts and moments                       | How much density shape and retrieval detail to lose.  |
| Delete one retained point    | Its contribution and, with suitable state, geometry | Remaining extrema, rerouting, and cached quantities.  |
| Move or rotate a cut         | Future routing under the new cut                    | How to reassign old evidence; often approximate.      |
| Discard a negligible summary | Known removed modeled mass                          | Whether that loss is acceptable for future questions. |

## Safe structural policies

Keep installed cuts fixed between explicit maintenance operations. A changing local PCA direction may propose a replacement, but must not silently rotate the active boundary while retaining old child counts. Rebuild, replay retained observations, use a bounded approximation, or leave unresolved history at the parent.

Deleting an internal cut while preserving its descendants is not merely deleting a pointer: descendant regions inherit the old ancestor constraint. Either preserve that constraint in the new representation or reconstruct the affected subtree. Bottom-up coarsening is convenient because the accounting is straightforward.

## Absolute activity is a separate field

$$A_{t}(x) = M_{t}p_{t}(x),\quad\quad\int p_{t}(x)\, dx = 1,\quad\quad\int A_{t}(x)\, dx = M_{t}$$

Here Mₜ is the total discounted admitted mass. During a quiet period, Aₜ can fade everywhere even though a normalized pₜ must still integrate to one. For a visualization, show this activity separately from relative probability and from posterior support for a boundary.

## Pruning and maintenance

Exponential weights do not become exactly zero. Use a tolerance or a known approximation budget. For an inactive summary with mass m at t₀, the threshold ε is reached at t₀ + log(m/ε)/β. A deadline queue can schedule checks, but bursts of expired items still create maintenance work. Hysteresis—stricter conditions to grow than to remain split—reduces oscillation.

WHAT CAN SURVIVE DISCARDED HISTORY

# 6. Exact compressed statistics

For a fixed region or summary, store a mass, a first moment, and optionally a full second moment. The same update works for any fixed additive feature map φ. These are algebraic consequences of the chosen exponential kernel, not special properties of a Gaussian partition.

$$m = \sum_{i}^{}w_{i},\quad\quad s = \sum_{i}^{}w_{i}x_{i},\quad\quad Q = \sum_{i}^{}w_{i}x_{i}x_{i}^{\top}$$

$$m \leftarrow am + w,\quad\quad s \leftarrow as + wx,\quad\quad Q \leftarrow aQ + wxx^{\top}$$

The mean is s/m; the population-form weighted covariance is Q/m − (s/m)(s/m)ᵀ. Under a Gaussian weighted likelihood, all observation dependence is through these statistics. They are sufficient for that fixed likelihood, not for arbitrary future cuts. Aging count and moment summaries also has a concrete precedent in the ClusTree implementation. \[7\]

## Optional statistics and their costs

| **State**                    | **Space per summary** | **What it supports**                                                   |
|------------------------------|-----------------------|------------------------------------------------------------------------|
| Mass + timestamp             | O(1)                  | Activity, mixture weights, fixed-branch evidence.                      |
| First moment                 | O(d)                  | Centroid; simple local dispersion summaries with extra scalars.        |
| Diagonal second moments      | O(d)                  | Coordinate variances, cheap proposals; no correlations.                |
| Full symmetric second moment | O(d²)                 | Covariance and Gaussian/PPCA likelihood calculations.                  |
| Low-rank sketch              | O(dr)                 | Approximate local directions; merging/truncation can lose information. |
| Squared-weight sum           | O(1)                  | Weight-concentration diagnostic and effective sample size.             |

$$W_{2} = \sum_{i}^{}w_{i}^{2},\quad\quad W_{2} \leftarrow a^{2}W_{2} + w^{2},\quad\quad n_{eff} = m^{2}/W_{2}$$

Effective sample size here is a weight-concentration diagnostic, not a replacement for Bayesian evidence strength. Under uniform aging without arrivals, n_eff stays unchanged while m shrinks. Both may be useful for different decisions.

## Mergeability and numerical care

At a common timestamp, sums and moments add exactly for disjoint observation sets. Pooling them does not necessarily preserve the full density. Use float64 accumulators or stable centered-moment updates; Q/m − μμᵀ can suffer cancellation. Symmetrize estimated covariance and regularize small eigenvalues before inversion.

A single statistic set cannot retrospectively change its half-life, encoder, or feature map. Retain replay data or maintain a small bank of rates when those changes are important.

RETROSPECTIVE RESOLUTION

# 7. The limit of compression—and useful sketches

Two one-dimensional datasets can share identical count, mean, and variance yet disagree on a future split:

$$A = \{ - 1, - 1,1,1\},\quad\quad B = \{ - \sqrt{2},0,0,\sqrt{2}\}$$

Both have count 4, sum 0, and squared sum 4. A cut at −0.5 sends two observations from A left, but only one from B. No function of those shared moments can recover both correct answers. Answering every threshold exactly would encode the empirical distribution, not a fixed-size general summary.

## Microclusters: keep a little structure beneath a leaf

Maintain a bounded collection of small moment summaries rather than one summary for a whole region. Their retained moments merge additively. This preserves some multimodality, although selecting which summaries to merge is itself a lossy policy. A summary budget bounds storage; it does not guarantee every future cut is accurate.

With conservative bounding boxes, a candidate half-space places some summaries entirely left, some right, and crosses others. Without assuming a shape inside each box, the left mass is bounded by:

$$m_{certain} \leq m_{L} \leq m_{certain} + m_{crossed}$$

This derived bound makes ambiguity visible. A Gaussian approximation within crossed summaries gives a point estimate, but that estimate is not the discarded empirical distribution. Conservative boxes can also remain large because of old extrema.

## Histograms, random projections, and coresets

A small histogram along a fixed axis or projection retains approximate threshold counts; bins can additionally hold moments. Streaming histogram-based split selection has a published classification precedent. \[8\] Histograms cannot generally be rotated after discarding observations, and separate marginals do not reconstruct the full joint distribution.

Time-decay coreset research provides compact summaries for specified clustering objectives. \[9\] A guarantee for k-means cost is not automatically a guarantee for every half-space probability, density value, or geodesic. Choose a sketch for the downstream question it must answer.

## Identities and deletion remain different requirements

A compressed summary does not remember which post contributed which location. Exact removal of a named historical post needs its embedding and weight, a contribution ledger, or replay. A posting-list tombstone can prevent future serving without exactly undoing that post’s influence on an already compressed model. State this retention contract explicitly.

A SUMMARY-ONLY RESEARCH CANDIDATE

# 8. Proposed design: residual evidence at ancestors

When a leaf is split, do not pretend its discarded observations have known child assignments. Keep their coarse summary at the parent. Route new observations into children. This preserves the information actually available and allows finer structure to emerge prospectively. Dynamic-tree retirement is a related published motivation, but this residual-mixture construction is a proposed alternative. \[4\]

<img src="Mondrian_Polya_assets/media/image2.png" style="width:6.75in;height:2.295in" alt="Figure 2. Residual summaries contain disjoint evidence cohorts. Cached subtree totals are bookkeeping quantities; they are not automatically the integrated probability of a spatial region." />

Figure 2. Residual summaries contain disjoint evidence cohorts. Cached subtree totals are bookkeeping quantities; they are not automatically the integrated probability of a spatial region.

Let rᵥ be discounted evidence stored directly at node v and let Sᵥ be the subtotal of residual evidence in its subtree. Each observation contributes to exactly one residual summary:

$$S_{v} = r_{v} + S_{L} + S_{R},\quad\quad M = \sum_{v}^{}r_{v}$$

Give each residual summary a density qᵥ normalized over its region. A uniform density is the simplest choice; moments may guide future structure without changing that choice. Define:

$$p_{t}(x) = \frac{1}{M_{t}}\sum_{v}^{}r_{v}(t)q_{v}(x)$$

Ancestor and descendant supports overlap, but the evidence is not double-counted. The model is a mixture of coarse and fine explanations. On a fixed nested tree, evaluating uniform residual densities at x only requires the nodes on its path.

| **Not the same model** This is not ordinary Pólya branch-count inference with fractional historical child counts. Its normalization is explicit, but its uncertainty model and split policy require separate design and validation. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

ACCOUNTING, SAMPLING, AND APPROXIMATION

# 9. Keeping the residual model coherent

## Subtree evidence is not spatial probability

An ancestor’s residual density extends into descendant regions. Therefore Sᵥ/M counts evidence stored below v, but generally omits the ancestor contribution to the spatial region Rᵥ. The actual probability must integrate every overlapping component:

$$P_{t}\left( R_{v} \right) = \frac{1}{M_{t}}\sum_{u}^{}r_{u}(t)\int_{R_{v}}^{}q_{u}(x)\, dx$$

For nested uniform rectangular components, the relevant overlap fractions are volume ratios. This distinction is essential when deriving branch probabilities, displaying regional mass, sampling targets, or defining a density-based metric. Do not substitute a bookkeeping subtotal for a probability integral.

## Sampling is straightforward at the model level

To sample a synthetic location, first choose a residual component with probability rᵥ/M, then sample from qᵥ. Cached subtree subtotals support a recursive choice among “stop here,” “go left,” and “go right.” In a plain leaf-based Pólya density, sample a root-to-leaf branch sequence and then a point from that leaf. Neither procedure produces an actual post without an independent content index.

## Splitting and merging

Installing an empty child structure leaves the residual density unchanged. New observations can make the fine components matter. A bounded finer sketch may transfer confidently located historical subgroups into children, but transferred evidence must be removed from the parent summary to avoid duplication.

Coarsening can pool masses and moments exactly. Replacing several component densities by one uniform or moment-matched density generally changes the density function. Keeping the old mixture preserves more shape but does not reclaim all its storage. That is a deliberate memory–fidelity tradeoff.

## A rigorous pruning bound for the represented model

If removed summaries account for fraction δ of the current mixture mass, write P = (1 − δ)P_keep + δP_drop. After renormalizing the retained components:

$$TV\left( P,P_{keep} \right) = \delta TV\left( P_{drop},P_{keep} \right) \leq \delta$$

This bounds the change to the current compressed model, not its error against the true corpus. It also does not bound every downstream geodesic: a low-mass corridor can matter greatly to connectivity. Evaluate density, retrieval, and path error separately.

## Geometry policy

Retain the parent support while its residual component is active. Aggressively shrinking that support would move historical probability without evidence. Prune negligible components, rebuild from available detail, or explicitly approximate their redistribution. All summary families still require a hard node budget if indefinite refinement is allowed.

TRADEOFF MATRIX

# 10. Implementation families at a glance

The following alternatives are implementations to compare, not interchangeable parameter settings. “Exact” means exact for the explicitly retained observations or fixed weighted likelihood—not exact recovery of an unobserved platform-wide distribution.

| **Family**                               | **Stored detail**                                           | **Main benefit**                                             | **Main limitation**                                                          |
|------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------|
| A. Fixed partition + discounted evidence | Cuts, branch counts, timestamps; optional moments           | Smallest update path; no historical observations needed      | Old geometry cannot adapt itself; arbitrary deletions cannot be recovered.   |
| B. Exponential random expiration         | Retained points, deletion support, expiry queue             | Geometry forgets through ordinary point removal              | No hard memory cap; additional sampling noise.                               |
| C. Bounded time-biased reservoir         | At most B points plus tree                                  | Controls memory and recency-biased sampling                  | Density is fitted to a sample; turnover and estimation variance.             |
| D. Weighted bounded cache + rebuilds     | Recent embeddings, timestamps, weighted summaries           | Easy to inspect, replay, benchmark, and correct              | Tail is truncated; fixed-count retention changes its time horizon.           |
| E. Residual summaries + split sketches   | Bounded coarse/fine summaries; little or no raw history     | History-independent memory and honest unresolved assignments | Modified model; delayed refinement, coarsening loss, limited exact deletion. |
| F. Bayesian structural averaging         | Candidate subtrees or particles plus conditional statistics | Represents uncertainty about cuts and topology               | Higher memory and maintenance cost; retained detail still matters.           |

## Decision rules

Choose **A** to establish the numerical baseline. Choose **D** to establish a practical adaptive baseline whose errors can be diagnosed by replay. Choose **C** when a strict sample cap and principled age bias are central. Investigate **E** when avoiding retained observations is an explicit requirement. Add **F** only after demonstrating that uncertainty about topology materially affects predictions.

A small forest averages several partition-based density estimates and costs roughly a linear factor in tree count. It is not automatically a posterior sample over all possible trees, and disagreement between its members is not automatically a calibrated uncertainty interval.

| **Recommendation** Compare a bounded-cache reference against the residual-summary candidate using exactly the same admitted stream, encoder, event-time policy, and memory budget. Otherwise architecture differences become confounded with different data exposure. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

FORGETTING THE OBSERVATIONS THAT DEFINE GEOMETRY

# 11. Retention policies in more detail

## Random expiration

Assign each admitted observation an independent lifetime Lᵢ ~ Exp(β), then delete it at tᵢ + Lᵢ. For a fixed region R, expected surviving counts match exponentially weighted counts:

$$\mathbb{E}\left\lbrack C_{R}(t) \right\rbrack = \sum_{i:x_{i} \in R}^{}e^{- \beta\left( t - t_{i} \right)}$$

This is an exponential-survival construction, related to Bernoulli time-biased sampling. \[6\] The equality concerns counts in a fixed region. Fitting a normalized, adaptive density is nonlinear, so it does not imply that the expected fitted density equals a deterministic weighted-data density.

A min-heap schedules expirations in O(log B) per event, on top of tree deletion work. If the admitted arrival rate is stationary at r posts per second, expected retained population is r/β = rH/ln 2. At r = 1 and H = 3,600 seconds, that is about 5,194 points. This is an expectation, not a hard maximum.

## Bounded time-biased sampling

R-TBS is a published scheme designed to bound sample size while controlling relative age-dependent inclusion probabilities, including exponential decay under varying arrival rates. \[6\] It is a principled retention layer rather than a second density estimator. Unweighted sample counts reflect the retained sample; they should not be treated as full-stream counts or full-stream posterior precision.

Avoid double forgetting: if inclusion already falls as exp(−β·age), weighting retained points by that same factor generally produces exp(−2β·age) in expected unnormalized contributions. Choose the estimator and sampling correction together.

## Truncated weighted history

An easier reference model retains observations for horizon W and weights them exponentially while present:

$$w_{i}(t) = e^{- \beta\left( t - t_{i} \right)}\mathbf{1}\{ 0 \leq t - t_{i} \leq W\}$$

At deletion, remove the observation’s remaining weight, not one. Under a stationary arrival rate, the expected fraction of exponential mass beyond W is exp(−βW). About 6.64 half-lives leaves a 1% tail. A time horizon still does not bound memory during bursts; a hard item cap changes the effective time horizon.

## Rebuild rather than repair

Rebuilding a bounded sample is often the simplest way to obtain current geometry. It may be amortized over many arrivals or limited to one subtree per maintenance cycle. Keep the rebuild’s statistical target explicit: a reservoir, a weighted recent cache, and a summary-only history are different datasets, even when their memory sizes match.

WHERE BAYESIAN MACHINERY HELPS

# 12. Split priors and structural inference

Two questions must be separated: “What traffic allocation is plausible?” and “Is a finer representation worth its complexity?” A branch prior Beta(κq,κ(1−q)) describes an allocation baseline q and prior strength κ. It is not a hard cap on category occupancy. A node budget is an engineering constraint, not a probability prior.

## Optional Pólya trees

Optional Pólya trees add a stopping decision and a distribution over available refinements. They can average coarse and fine explanations instead of committing to one topology. \[3\] For one candidate split, a schematic recursive evidence calculation is:

$$Z_{v} = \rho_{v}Z_{v}^{stop} + \left( 1 - \rho_{v} \right)Z_{v}^{split}$$

The posterior stopping weight is ρᵥZ_stop/Zᵥ. With more possible axes, thresholds, and descendants, candidate search expands quickly. Limited-lookahead OPT addresses computational growth by restricting exploration; that is a useful precedent for a bounded local search. \[10\]

A topology probability can control the contribution or displayed opacity of a boundary. It is not the same as a hand-designed “split score.” Furthermore, log-beta evidence terms are nonlinear in decayed counts. Lazy counts alone do not make every cached subtree evidence value current. Refresh structural scores under a maintenance budget and report their age.

## Particle filters and unscented estimation

Dynamic Trees for Learning and Design uses particle learning for evolving regression/classification tree states and simple conditional leaf models. \[11\] For a density model, particles could represent alternative topologies while conjugate parameters are integrated out. Resampling can lose structural diversity; local subtree proposals or rejuvenation may be needed.

A Gaussian or unscented estimator is not a representation of discrete alternatives such as “this cut exists” versus “this cut does not exist.” It could operate on continuous parameters conditional on a fixed structure, but does not eliminate topology inference. Deterministic count aging needs neither method.

## Tooling and runtime choice

Stan requires discrete latent choices to be marginalized rather than sampled directly. Pyro provides SMC machinery but not a ready-made mutable forest. Gen supports involutive MCMC for structural moves. \[20–22\] These are useful experimental tools; a small custom array-based tree may be more appropriate for the deployed runtime.

For a non-Bayesian budget policy, require minimum independent evidence, measurable split benefit, and a node budget. Keep duplicate-weighted activity separate from raw counts so a flood does not automatically buy more representational capacity.

IMPLEMENTATION BLUEPRINT

# 13. Data structure and update contract

Use indexed arrays or compact records with stable node identifiers. Store installed geometry, probability evidence, timestamps, and optional sketch state separately. Published streaming sMPT updates involve both cut and observed/complement probability parameters; a simplified complete-partition prototype should not claim to reproduce that model unless it implements those distinctions. \[1\]

ModelHeader  
schema_version, encoder_hash, coordinate_transform_hash  
time_basis, decay_rates, geometry_version, snapshot_time  
Node  
parent, left, right, split_axis, split_value  
region_bounds, geometry_metadata  
evidence_counts, evidence_time, prior_parameters  
residual_summary_ref, sketch_ref, cache_epoch  
RecordIndex \# optional, bounded separately  
post_id, revision_hash, event_time, embedding_ref, tombstone

## Fast path

Validate the record and model version; deduplicate the revision; compute the embedding; advance a monotone materialization clock; calculate the arrival’s age-adjusted weight; route through the current cuts; materialize touched evidence; add the contribution; commit the state update and consumed job identifier together.

For a residual model, the arrival is assigned to one residual cohort and cached subtree subtotals are adjusted. Do not add the same point to both an ancestor residual and a child residual. For a branch model, ancestor counts are legitimate cumulative summaries, but are not separate mixture components.

## Maintenance path

Propose cuts from available sketches or retained points; evaluate them on held-out or prequential evidence; split or coarsen under the budget; invalidate derived densities and path caches; publish a new geometry version. Keep old version identifiers usable long enough for in-flight retrieval requests, or return a redirect/refinement response.

## Persistence and late events

Use snapshots plus a bounded write-ahead log or durable checkpoint mechanism. On resume, age from stored timestamps rather than executing every missed timer. A late event is evaluated at the current clock with weight exp\[−β(now − event_time)\]. Define clock-skew handling, future timestamps, post edits, and deletions explicitly.

A durable queue provides retryable jobs, not automatic exactly-once statistical updates. The logical record identity should include network, canonical ID, revision/content hash, and encoder version. Reject results for stale revisions and use an idempotent transaction at the state writer.

CALCULATED BUDGETS, NOT BENCHMARKS

# 14. Computational and memory burden

Let d be dimension, h actual visited depth, F forest size, N allocated nodes per tree, and B retained embeddings. The streaming paper gives O(dh) insertion/deletion work under its data-structure assumptions. \[1\] Finding replacement extrema, rebuilding subtrees, refreshing dependent priors, and global visualization work must still be included in implementation measurements.

| **Operation**                        | **Approximate accounting**                      | **Important qualification**                              |
|--------------------------------------|-------------------------------------------------|----------------------------------------------------------|
| Known-path count aging               | O(Fh)                                           | No full-tree timer sweep is required.                    |
| Box-aware insertion or density query | O(Fdh)                                          | Do not replace h by log B without a balance assumption.  |
| Full covariance along each path      | O(Fhd²)                                         | Optional; diagonal moments cost O(Fhd).                  |
| All-node probability refresh         | O(FN)                                           | Local changes can affect many global leaf masses.        |
| Rebuild B retained points            | O(FdBh), straightforward construction           | Actual construction and batching can differ.             |
| Particle topology model              | Roughly multiplies state/work by particle count | Sharing helps, but rejuvenation and divergence add cost. |

## Illustrative retained-point forest

For B = 3,000, d = 32, and F = 8, a full binary tree with at most one retained item per leaf has at most 5,999 ordinary nodes. Storing two float32 box endpoints per coordinate costs about **11.7 MiB across eight trees**. The embeddings stored once cost about **0.37 MiB**. This excludes links, priors, counters, indexing, text, and temporary workspaces.

## Illustrative summary-only state

| **512 summaries at d = 32**                          | **Float64 payload** |
|------------------------------------------------------|---------------------|
| Mass + first moments + diagonal second moments       | 260 KiB             |
| Mass + first moments + full symmetric second moments | About 2.19 MiB      |

Bounds, timestamps, sketches, and model weights are additional. An upper bound on summary count makes this independent of the number of historical posts, but not independent of dimension or chosen forest size.

## Measure the full pipeline

Report input posts per second, token-length distribution, embedding throughput, tree-update latency, maintenance p95/p99, memory peaks, battery/thermal behavior, retrieval bandwidth, and queue backlog separately. A small tree does not make an encoder cheap. A fast encoder does not make whole-network fan-out economical. No hardware throughput from the earlier exploratory discussion is assumed here.

PROPOSED DOWNSTREAM GEOMETRY

# 15. Density-derived semantic distance

The desired interpretation is travel through a medium: high-density regions are easier to traverse and low-density gaps are costly. Density-based Riemannian and Fermat distances provide a published mathematical precedent; adapting one to a time-decaying partition is a separate design. \[14\]

At a fixed snapshot t, choose a positive local travel cost cₜ(x) and define the length distance:

$$d_{t}(x,y) = \inf_{\gamma(0) = x,\,\gamma(1) = y}\int_{0}^{1}c_{t}\left( \gamma(u) \right) \parallel \dot{\gamma}(u) \parallel \, du$$

For example, form a robust density ratio ρₜ(x) = pₜ(x)/b(x) against a positive baseline and choose a decreasing capped cost:

$$c_{t}(x) = clip\left( \left( \rho_{t}(x) + \varepsilon \right)^{- \alpha},c_{\min},c_{\max} \right),\quad\quad\alpha > 0$$

Positive finite bounds prevent zero-cost collapse and infinite barriers. On a suitable connected domain, the construction is symmetric and satisfies the triangle inequality. The result is a modeling definition, not proof that text meaning or user preference obeys this geometry.

## Use density, not leaf mass

A density-preserving split changes cell mass and volume together. A cost based on mass alone would change merely because the representation became finer. Using the same density field preserves the continuous metric under that refinement. This is an important regression test.

## Directional travel is optional

$$G_{t}(x) = c_{t}(x)^{2}A_{v},\quad\quad L(\gamma) = \int\sqrt{{\dot{\gamma}}^{\top}G_{t}(\gamma)\dot{\gamma}}\, du$$

A regularized local inverse covariance is one possible positive-definite Aᵥ. It makes movement along high-variance directions cheaper, but covariance is also affected by cell shape, mixed topics, and sampling. Normalize scale and cap eigenvalues. Adding a covariance-based tensor is not required for the density model.

## What the metric means—and does not mean

It estimates connectivity through the modeled, sampled discourse. A dense bridge can connect otherwise distant posts, but popularity is not truth, usefulness, or personal preference. Spam can create artificial shortcuts; uncertainty, duplicate weighting, baseline normalization, and bounded speed are safeguards to evaluate, not guarantees of semantic correctness.

APPROXIMATION CHOICES

# 16. Computing paths and visualizing change

For piecewise-constant c, a path crossing cells has cost Σℓ cℓ·length(γ∩Rℓ). Within a cell the optimal segment is straight, but the best boundary crossing location depends on neighboring costs. A continuous optimum need not pass through cell centroids.

| **Path method**             | **Benefit**                                 | **Tradeoff**                                                                         |
|-----------------------------|---------------------------------------------|--------------------------------------------------------------------------------------|
| Centroid graph              | Small and easy to update                    | Can miss corridors and overestimate paths; k-nearest edges may cross unmodeled gaps. |
| Boundary portals            | Better approximates crossing locations      | Portal count and adjacency grow rapidly with dimension.                              |
| Sample/representative graph | Reuses retained embeddings or microclusters | Connectivity depends on sample coverage; misses unseen bridges.                      |
| Continuous optimization     | Can refine a candidate path                 | Piecewise discontinuities and nonconvex geometry complicate optimization.            |

For a graph edge, integrate the cost along its actual segment through every crossed region rather than averaging endpoint densities. That avoids treating a long low-density gap as cheap merely because its endpoints are dense. Dijkstra or A\* then finds the graph’s optimum, not necessarily the continuous optimum.

## A tree path is not a spatial geodesic

Parent/child ancestry measures hierarchy, not physical adjacency. Two leaves on opposite sides of a root cut may be spatially adjacent. Conversely, nearby centroids may require traversing other cells. Build and invalidate the spatial graph explicitly; do not use lowest-common-ancestor depth as a substitute for the intended metric.

## Time and visualization

Freeze the density at a snapshot for each distance query. If the medium changes while travel occurs, the problem becomes time-dependent routing and may no longer define a symmetric distance. Keep snapshot version and density time in the returned result.

Use boundary presence or opacity for resolution/support, fill for relative density, brightness or a separate field for absolute activity, and a distinct overlay for user-state uncertainty. A two-dimensional projection can overlap disjoint high-dimensional cells. Its visible areas are not the volumes used in the actual density calculation.

## Compare alternatives before committing

Diffusion-style and resistance-style graph quantities aggregate multiple paths, unlike a shortest path that can depend on one fragile bridge. They are separate possible recommendation features, not equivalent versions of the same metric. Test ranking stability and niche discovery against ordinary cosine distance before adding substantial path machinery.

THE DENSITY DOES NOT SOLVE DISCOVERY

# 17. Sampling and candidate acquisition

Sampling a location from the modeled density is cheap: choose a probability-weighted leaf or residual component, then sample within it. Finding a real, recent post near that location is a different retrieval problem. A density tree does not retain text, author IDs, or candidate availability unless a separate index is maintained.

## The observed distribution depends on the sampler

A feed assembled from follows, keyword searches, semantic queries, and broad discovery is not a uniform sample of the whole network. Record provenance and selection policy. A tree built from those candidates estimates the admitted distribution unless a defensible sampling correction is applied.

$${\widehat{m}}_{R}(t) = \sum_{i\ included}^{}\frac{e^{- \beta\left( t - t_{i} \right)}}{\pi_{i}}\mathbf{1}\{ x_{i} \in R\}$$

For known positive inclusion probabilities πᵢ, this inverse-probability total is design-unbiased for a fixed region and target population. A normalized ratio and an adaptively fitted tree need not be unbiased. Unknown search rankings cannot be repaired by pretending their result positions are inclusion probabilities. Small πᵢ also produce high variance.

## Separate mass fields, one geometry

A common partition can carry distinct statistics for broad discovery, targeted retrieval, duplicate-adjusted content, and user feedback. Do not label targeted exposure “global popularity.” Shared geometry saves storage, but a structure chosen primarily from one stream may under-resolve another.

## Cheap candidate gates

Attach a small lexical or static-embedding routing representation to relevant regions. Use it to propose keyword/tag queries or filter locally before expensive encoding. Keep a randomized audit channel outside the gate, because a rejected post otherwise supplies no evidence that the rejection was wrong. Selective-label correction is a recognized concern in importance-weighted active learning. \[23\]

AT Protocol exposes streaming interfaces with filters; it does not provide an arbitrary private latent-region sampler. \[19\] Cross-network adapters can normalize content records, but ActivityPub instances and Nostr relays do not necessarily expose the same population, history, or completeness. Do not assume one universal public stream.

## Network economics and privacy

If each of U devices receives a stream at byte rate S, aggregate last-mile traffic is approximately U·S before overhead. Local filtering saves encoder calls, not upstream bandwidth. A shared public index can publish cacheable coarse-region chunks and broad deterministic samples. Querying those regions still leaks interests to the service or network observer; local personalization is not anonymous retrieval.

LEARNING WHEN FURTHER COMPUTATION IS USEFUL

# 18. Cheap and expensive representations

Two encoders can feed two trees: a cheap representation covers the admitted stream, while an expensive representation handles selected posts. Paired encodings of the same posts train their correspondence without requiring other users’ reactions. The mapping concerns content; the private model still decides preference.

## Coordinate mapping

$$\widehat{e}(x) = As(x),\quad\quad A = BS_{\lambda}^{- 1},\quad\quad S_{\lambda} = S + \lambda I$$

Here S = Σwᵢsᵢsᵢᵀ and B = Σwᵢeᵢsᵢᵀ are discounted paired sufficient statistics. They can be updated without retaining all pairs when both feature spaces are fixed. For sparse high-dimensional TF–IDF, a dense covariance is inappropriate; use a smaller fixed projection or sparse online optimization.

A linear TF–IDF mapper is effectively a learned table of token or n-gram vectors followed by weighted pooling. It cannot recover distinctions absent from its input, such as word order in identical unigram bags. Under squared error, ambiguity can produce a mean teacher coordinate that lies between genuine modes.

## Region correspondence

$$M_{ij} = \Pr\left( e(x) \in B_{j} \mid s(x) \in A_{i} \right),\quad\quad{\widehat{M}}_{ij} = \frac{C_{ij} + a_{j}}{\sum_{k}^{}\left( C_{ik} + a_{k} \right)}$$

Discounted paired cross-counts Cᵢⱼ represent many-to-many relationships between cheap regions Aᵢ and expensive regions Bⱼ. The routing policy can combine them with the user model’s expected utility in each expensive region. This preserves uncertainty instead of forcing every cheap observation to one predicted coordinate.

When either tree splits, old parent cross-counts do not reveal child assignments. Retain the coarse association, learn finer correspondence from new pairs, or replay a bounded cache. Never insert every predicted embedding into the expensive density as though it were an independent teacher observation; that reinforces the mapper’s errors.

## Useful cheap models

Model2Vec is a concrete example of static token embeddings distilled from a sentence encoder. \[24\] It provides a starting point for a lookup-and-pool gate, not a guarantee of niche-topic recall. A cheap routing tree may be supervised to reduce uncertainty about expensive destinations rather than optimized solely as another density tree.

Evaluate recall of useful candidates at a fixed expensive-encoding budget, including rare topics and out-of-gate audits. Mean coordinate error alone does not measure the operational objective.

ELASTIC INFERENCE WITHOUT CHANGING THE MODEL SILENTLY

# 19. 2D Matryoshka and ONNX deployment

2D Matryoshka sentence embeddings are trained to support variation in both transformer depth and output dimensionality. “2D” means these two computational axes, not a two-coordinate semantic map. Standard dimensional truncation mainly saves storage and downstream work; early exit removes transformer blocks. \[15\]

## Coordinate-family consistency is not identity

A prefix of a full embedding at one depth is directly defined. A vector from a shallower layer is generally different from the same-sized vector at a deeper layer. Joint training can make shallow representations useful, but does not establish identical partitions across depth. Version trees by encoder, depth, dimension, pooling, and normalization; learn or validate any cross-depth mapping.

## Three ONNX deployment patterns

| **Pattern**                              | **Advantage**                               | **Cost or limitation**                                                                |
|------------------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------|
| Separate truncated graphs                | Simplest fixed-depth exports and benchmarks | Escalating a post repeats the earlier layers.                                         |
| One full graph with intermediate outputs | Easy access to diagnostic embeddings        | Requesting an early output is not a portable guarantee that later blocks are skipped. |
| Composable encoder stages                | Can reuse token hidden states on escalation | Extra graph boundaries, activation storage, batching and provider-transfer overhead.  |

ONNX Runtime exposes inference sessions, execution providers, and explicit tensor I/O. \[18\] Stage boundaries are an architecture-specific export task. Carry token hidden states, masks, positions, and any other required tensors—not only the pooled sentence embedding. Validate model-specific intermediate normalization and pooling.

Sentence Transformers notes that its ONNX transformer export may require pooling and normalization outside the graph. \[16\] Preserve the model’s actual sentence-embedding computation. After taking a Matryoshka prefix, renormalize when required by its training and similarity convention.

## ARM, quantization, and FlashAttention

Benchmark the exported graph on the intended runtime and execution provider. Transformer fusion and quantization support depend on graph structure and hardware. \[17\] Native ARM inference, browser WebAssembly/WebGPU, and a device NPU are different deployment paths; provider availability and operator coverage must be checked separately.

FlashAttention reduces attention memory traffic; exporting ONNX does not carry a particular PyTorch CUDA kernel into every provider. \[25\] For short posts, feed-forward layers, batching, tokenization, and memory transfers can dominate. Record tokens/second and end-to-end posts/second, not an unqualified throughput claim.

ONE ALGORITHM, OPTIONAL PUBLIC INFRASTRUCTURE

# 20. Standalone and shared deployment

<img src="Mondrian_Polya_assets/media/image3.png" style="width:6.75in;height:2.475in" alt="Figure 3. Shared infrastructure performs public-content work. Personal state remains local, but retrieval requests can still disclose interests." />

Figure 3. Shared infrastructure performs public-content work. Personal state remains local, but retrieval requests can still disclose interests.

The same logical components can run on one machine: stream adapters, a bounded job queue, an encoder, one serialized density writer, a post cache, and a local policy. A hosted service can amortize only the non-private corpus work, while clients keep their own preference state and choose candidates locally.

## Worker and state-writer contract

Embedding workers may be interchangeable local or rented machines. Jobs carry a content reference or bounded payload plus canonical ID, revision, event time, encoder hash, and job identity. Results must be finite, correctly shaped, tied to the intended model, and committed idempotently. A signature authenticates a worker; it does not prove the embedding is correct. Audit untrusted workers and reject stale results.

Use one canonical tree writer initially. Independently trained tree topologies cannot be merged by simply adding node counts. Additive statistics are mergeable only when their feature definitions, regions, clocks, decay kernels, and observation ownership agree. Workers can parallelize encoding without distributing topology mutation.

## Retrieval and evolving cells

Store posting lists outside node payloads, keyed by model/geometry version, region, and time bucket. A split does not magically reassign old IDs to children. Retain parent buckets, re-embed or reroute retained vectors, or publish a versioned refinement map. Keep deletion tombstones distinct from whether historical density influence can be exactly reversed.

## Costs to measure

Use measured input rate and batched encoder throughput to estimate compute. Add idle time, model loading, queue operations, storage, replicas, egress, and fan-out separately. Public-content indexing can be shared; no specific provider pricing or cloud throughput is assumed. A standalone deployment has no compulsory hosted bill, but still pays in hardware, power, network use, and attention to maintenance.

EVALUATION SHOULD TEST THE ENTIRE FEEDBACK LOOP

# 21. Spam resistance and online calibration

High density can reflect a legitimate event, duplication, coordinated activity, or an overly coarse encoder. Low density can reflect valuable novelty or noise. Neither is a spam label by itself. Robust Random Cut Forest is a useful streaming-anomaly comparator, not proof that density alone solves abuse detection. \[13\]

## Separate observation and refinement evidence

$$m_{v}^{raw} = \sum_{i}^{}w_{i},\quad\quad m_{v}^{refine} = \sum_{i}^{}w_{i}a_{i},\quad 0 \leq a_{i} \leq 1$$

The proposed factor aᵢ can cap duplicate or source-concentrated influence on capacity allocation. Preserve raw counts to make the intervention inspectable. Apply each observation’s chosen weight consistently along its path. Semantic embeddings and compressed statistics are not automatically private; rare-content summaries can still disclose information.

## Evaluate before incorporating each observation

| **Test family**               | **What to measure**                                                                                    |
|-------------------------------|--------------------------------------------------------------------------------------------------------|
| Normalization and bookkeeping | Integrated density; parent/child consistency; residual overlap accounting; no negative masses.         |
| Predictive calibration        | Pre-update log score, branch probabilities, uncertainty coverage, behavior after long silence.         |
| Drift and deletion            | Moving clusters, new islands, old outliers, repeated coordinates, edits, late arrivals and removal.    |
| Compression fidelity          | Held-out regional probabilities, replay discrepancies, moment error, split ambiguity, coarsening loss. |
| Geometric stability           | Density-preserving split invariance, path-cost changes, corridor removal, spam shortcut sensitivity.   |
| Recommendation utility        | Niche candidate recall, user-feedback prediction, novelty/diversity, expensive calls per useful post.  |
| Systems behavior              | Steady-state and peak memory, latency tails, restart parity, queue backlog, network and energy cost.   |

## Prevent self-confirming measurements

Use an independently sampled audit stream where possible. Training and testing only on posts the current router admits hides its false negatives. Maintain comparable exposure across alternatives. Report bootstrap or repeated-stream uncertainty appropriate to temporal dependence; random shuffling can erase the drift problem being tested.

The content model and the private preference model need separate calibration. A well-normalized density can still be a poor predictor of likes. Likewise, attractive geometry is not evidence that the recommender improves satisfaction. Treat the visualization as a diagnostic interface, not a performance metric.

RECOMMENDED EXPERIMENT SEQUENCE

# 22. Implementation roadmap and decisions

## Stage 1 — make the probability accounting undeniable

Use one fixed complete partition, a frozen low-dimensional coordinate system, lazy decayed branch counts, and a separate total-activity scalar. Validate against direct weighted replay. Include out-of-domain behavior and a genuine positive baseline. This establishes a trustworthy numerical core before topology changes obscure failures.

## Stage 2 — establish an adaptive retained-data reference

Retain a bounded embedding cache and compare scheduled rebuilds with the published streaming machinery where implemented faithfully. Fix the source sample, memory cap, and half-life. Measure replacement-extrema work, structural latency, and prediction shifts. A working reference is more valuable than prematurely optimizing an unverified approximation.

## Stage 3 — compare compression policies

Add microcluster or histogram support, then implement residual historical evidence at ancestors as a separate model family. Check exact conservation of retained statistics, prospective learning after splits, and explicit density changes after merges. Compare at equal peak memory, including sketches and temporary rebuilding buffers.

## Stage 4 — add geometry and acquisition

Evaluate a capped density-distance feature against cosine distance. Then test a cheap-to-expensive mapper or a 2D-Matryoshka cascade. Preserve an out-of-gate audit budget. Only introduce a public semantic index when measured stream bandwidth or repeated encoding justifies shared infrastructure.

## Stage 5 — add structural Bayesian inference selectively

Measure instability across reasonable tree seeds and small ensembles. Use bounded candidate averaging or particles only if that instability causes measurable prediction errors. Keep uncertainty from partition choice, sample selection, encoder ambiguity, and preference inference conceptually separate.

| **Decision to record before implementation** | **Why it changes correctness**                              |
|----------------------------------------------|-------------------------------------------------------------|
| Target population and sampling policy        | Determines what the estimated probability means.            |
| Event time, half-life, and late-event rule   | Determines which evidence is currently influential.         |
| Deletion and replay contract                 | Determines what can be removed exactly after compression.   |
| Support, normalization, and encoder version  | Determines density units and validity of old cuts.          |
| Memory and structural-update budget          | Determines how much refinement can be retained.             |
| Primary success metric                       | Distinguishes density fidelity from recommendation quality. |

| **Bottom line** The strongest design is not “a tree that remembers everything cheaply.” It is a tree that states precisely which evidence it remembers, at what resolution, and which later operations must be approximate. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

DERIVED EXAMPLES

# Appendix A. Worked numerical checks

## A1. Parent and children decay consistently

Suppose a region has mass 900, split into 600 and 300. After one half-life without arrivals, the masses are 450, 300, and 150. The 2:1 branch ratio is unchanged. The absolute activity is halved. A smoothed Beta prediction can nevertheless move toward its prior because evidence has weakened relative to fixed prior strength.

## A2. Fixed-prior versus count decay

With a Beta(1,1) baseline and counts 80 and 20, the predictive left probability is 81/102 ≈ 0.7941. After ten half-lives, the counts are 0.078125 and 0.01953125, giving approximately 0.5140. The raw ratio remains 0.8. This distinguishes diminishing confidence from changing relative empirical allocation.

## A3. Refinement can preserve density exactly

| **Region**                        | **Probability mass** | **Volume** | **Density** |
|-----------------------------------|----------------------|------------|-------------|
| Parent                            | 0.30                 | 0.10       | 3           |
| Left, initialized proportionally  | 0.21                 | 0.07       | 3           |
| Right, initialized proportionally | 0.09                 | 0.03       | 3           |
| Left, after new evidence          | 0.26                 | 0.07       | ≈ 3.714     |
| Right, after new evidence         | 0.04                 | 0.03       | ≈ 1.333     |

The first split changes resolution only. The later allocation introduces a density contrast while preserving total regional probability. A travel cost based on density should not change at the first split; one based only on leaf mass would.

## A4. Residual bookkeeping is not spatial probability

Assume a parent covers a unit-volume region and retains uniform residual mass 900. Its equal-volume left and right children contain new residual masses 60 and 40. Total evidence is 1,000. The left subtree subtotal is 60, but the left region also contains half the parent component:

$$P\left( R_{L} \right) = \frac{900(1/2) + 60}{1000} = 0.51,\quad\quad P\left( R_{R} \right) = 0.49$$

Using 60/1,000 as the left-region probability would be wrong. The discrepancy is the inherited coarse component, not a normalization bug.

## A5. Expected cache size under exponential survival

With stationary admission r = 1 post/second and half-life H = 1 hour, rH/ln 2 ≈ 5,194 surviving posts are expected. A 3,000-item hard cap therefore requires an admission policy, a different half-life, or a bounded sampling construction—not just independent expiration.

REFERENCE CONTRACTS, NOT A FULL LIBRARY

# Appendix B. Update pseudocode and invariants

These routines illustrate fixed-region evidence and proposed residual bookkeeping. They are not substitutes for the published sMPT geometry insertion/deletion algorithms. Use monotone materialization time, validated event timestamps, and one consistent observation weight.

def touch(summary, now, beta):  
assert now \>= summary.time  
a = exp(-beta \* (now - summary.time))  
summary.mass \*= a  
summary.first \*= a  
summary.second \*= a  
summary.weight_square_sum \*= a \* a  
summary.time = now  
  
def add(summary, x, event_time, now, beta, base_weight=1):  
touch(summary, now, beta)  
w = base_weight \* exp(-beta \* (now - event_time))  
summary.mass += w  
summary.first += w \* x  
summary.second += w \* outer(x, x)  
summary.weight_square_sum += w \* w

def split_with_residual_history(leaf, cut):  
\# Historical summary stays attached to this exact region.  
\# Children start with zero residual evidence.  
assert cut_is_valid_for(leaf.region, cut)  
leaf.install_cut(cut)  
leaf.left = empty_child(left_region(leaf.region, cut))  
leaf.right = empty_child(right_region(leaf.region, cut))  
invalidate_density_and_path_caches()  
  
def residual_density(x, tree, now):  
total = current_total_residual_mass(tree, now)  
if total \<= 0:  
return configured_baseline_density(x)  
value = 0  
for node in route_containing_x(tree, x):  
value += current_mass(node.residual, now) \* node.q(x)  
return value / total

## Must-pass invariants

Counts remain nonnegative within numerical tolerance. Every residual observation has one owner. At a shared timestamp, cached subtotals equal local residual plus child subtotals. Regional probabilities include ancestor residual overlap. Densities integrate to one on the declared domain. Resuming from a snapshot matches uninterrupted aging. Replayed job IDs do not add evidence twice.

Deleting a retained weighted observation subtracts its current weight, current weighted moments, and squared current weight. Exact removal of a retired named observation is unsupported unless the contribution can be reconstructed. A changed cut or encoder invalidates any cached assignment that depended on the old geometry.

PRIMARY SOURCES AND OFFICIAL DOCUMENTATION

# References • foundations

Citations in the body refer to these numbered sources. Titles are clickable. Original derivations and proposed extensions are identified in the text; citing a precursor does not transfer its guarantees to the proposed system.

**\[1\] Dickens, C., Meissner, E., Moreno, P. G., and Diethe, T. (2020).**

[<u>Interpretable Anomaly Detection with Mondrian Pólya Forests on Data Streams.</u>](https://arxiv.org/abs/2008.01505)

Primary source for the batch/streaming construction, complementary regions, and update procedures.

**\[2\] Lakshminarayanan, B., Roy, D. M., and Teh, Y. W. (2014).**

[<u>Mondrian Forests: Efficient Online Random Forests.</u>](https://arxiv.org/abs/1406.2673)

Foundation for online Mondrian partition extension and projectivity.

**\[3\] Wong, W. H., and Ma, L. (2010).**

[<u>Optional Pólya tree and Bayesian inference.</u>](https://arxiv.org/abs/1010.0490)

Bayesian stopping, partition choice, and density estimation.

**\[4\] Anagnostopoulos, C., and Gramacy, R. B. (2012 preprint).**

[<u>Dynamic trees for streaming and massive data contexts.</u>](https://arxiv.org/abs/1201.5568)

Data retirement and the limits of repartitioning discarded covariates.

**\[5\] Bissiri, P. G., Holmes, C. C., and Walker, S. G. (2013 preprint; published 2016).**

[<u>A General Framework for Updating Belief Distributions.</u>](https://arxiv.org/abs/1306.6430)

Framework for explicitly chosen loss- or likelihood-weighted updates.

**\[6\] Hentschel, B., Haas, P. J., and Tian, Y. (2019).**

[<u>Temporally-Biased Sampling Schemes for Online Model Management.</u>](https://arxiv.org/abs/1906.05677)

Time-biased inclusion probabilities and bounded reservoir sampling.

**\[7\] MOA project. Source code, accessed 20 September 2026.**

[<u>ClusTree: ClusKernel.java.</u>](https://raw.githubusercontent.com/Waikato/moa/master/moa/src/main/java/moa/clusterers/clustree/ClusKernel.java)

Implementation precedent for aging microcluster counts and moments.

**\[8\] Ben-Haim, Y., and Tom-Tov, E. (2010).**

[<u>A Streaming Parallel Decision Tree Algorithm.</u>](https://jmlr.org/beta/papers/v11/ben-haim10a.html)

Fixed-memory histograms for streaming split selection; classification setting.

**\[9\] APPROX/RANDOM proceedings (2019).**

[<u>Improved Algorithms for Time Decay Streams.</u>](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.APPROX-RANDOM.2019.27)

Compact approximation algorithms for specified time-decay clustering objectives.

**\[10\] Jiang, H., Mu, J. C., Yang, K., Du, C., Lu, L., and Wong, W. H. (2013).**

[<u>Computational Aspects of Optional Pólya Tree.</u>](https://arxiv.org/abs/1309.5489)

Limited-lookahead inference and computation–accuracy tradeoffs.

**\[11\] Taddy, M. A., Gramacy, R. B., and Polson, N. G. (2009 preprint; published 2011).**

[<u>Dynamic Trees for Learning and Design.</u>](https://arxiv.org/abs/0912.1586)

Particle learning over local tree changes in regression and classification.

**\[12\] Ma, L., and Bruni, J. (2024 preprint).**

[<u>A partial likelihood approach to tree-based density modeling and its application in Bayesian inference.</u>](https://arxiv.org/abs/2412.11692)

Statistical treatment of data-dependent partition selection.

PRIMARY SOURCES AND OFFICIAL DOCUMENTATION

# References • geometry and systems

**\[13\] Guha, S., Mishra, N., Roy, G., and Schrijvers, O. (2016).**

[<u>Robust Random Cut Forest Based Anomaly Detection on Streams.</u>](https://proceedings.mlr.press/v48/guha16.html)

Dynamic random-cut summaries and a useful anomaly-detection comparator.

**\[14\] Sorrenson, P., Behrend-Uriarte, D., Schnoerr, C., and Koethe, U. (2025).**

[<u>Learning Distances from Data with Normalizing Flows and Score Matching.</u>](https://proceedings.mlr.press/v267/sorrenson25a.html)

Density-based metrics, Fermat distances, and geodesic estimation.

**\[15\] Li, X., Li, Z., Li, J., Xie, H., and Li, Q. (2024).**

[<u>2D Matryoshka Sentence Embeddings.</u>](https://arxiv.org/abs/2402.14776)

Elastic sentence representations across embedding dimensions and encoder depths.

**\[16\] Sentence Transformers. Official documentation, accessed 20 September 2026.**

[<u>Speeding up Inference.</u>](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html)

ONNX export, pooling, normalization, and inference backends.

**\[17\] ONNX Runtime. Official documentation, accessed 20 September 2026.**

[<u>Transformers optimizer.</u>](https://onnxruntime.ai/docs/performance/transformers-optimization.html)

Graph optimization and model-specific benchmarking considerations.

**\[18\] ONNX Runtime. Official documentation, accessed 20 September 2026.**

[<u>Python API overview.</u>](https://onnxruntime.ai/docs/api/python/api_summary.html)

Inference sessions, execution providers, and tensor I/O.

**\[19\] AT Protocol. Official documentation, accessed 20 September 2026.**

[<u>Streaming Data.</u>](https://atproto.com/guides/streaming-data)

Network event streams and their filtering interfaces.

**\[20\] Stan. Official documentation, accessed 20 September 2026.**

[<u>Latent Discrete Parameters.</u>](https://mc-stan.org/docs/stan-users-guide/latent-discrete.html)

Marginalizing discrete latent choices rather than sampling them directly.

**\[21\] Pyro. Official documentation, accessed 20 September 2026.**

[<u>pyro.infer.smcfilter.</u>](https://docs.pyro.ai/en/stable/_modules/pyro/infer/smcfilter.html)

Sequential Monte Carlo interface and state-management constraints.

**\[22\] Gen.jl. Official documentation, accessed 20 September 2026.**

[<u>Markov Chain Monte Carlo.</u>](https://www.gen.dev/docs/stable/ref/inference/mcmc/)

Involutive MCMC and structural proposal machinery.

**\[23\] Beygelzimer, A., Dasgupta, S., and Langford, J. (2008 preprint).**

[<u>Importance Weighted Active Learning.</u>](https://arxiv.org/abs/0812.4952)

Selection probabilities and selective-observation bias.

**\[24\] Minish. Official documentation, accessed 20 September 2026.**

[<u>Model2Vec introduction.</u>](https://minish.ai/packages/model2vec/introduction/)

Static token embeddings distilled from sentence encoders.

**\[25\] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022).**

[<u>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.</u>](https://arxiv.org/abs/2205.14135)

Attention-specific memory and kernel optimization; not a universal encoder-throughput claim.

Scope note: This document makes no claim to a completed implementation, measured hardware throughput, cloud-price estimate, or empirical recommendation improvement. These remain validation tasks in the proposed roadmap.
