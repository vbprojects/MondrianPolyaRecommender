# Read-only API v1

Implemented by `src/MondrianServer.jl`. All three query endpoints are read-only; internal ingestion and maintenance still write to the store.

## Shared identity and coordinate contract

Every response identifies `api_version`, persistent `store_id`, `embedding_space_id`,
`generation`, and `effective_at` (Unix seconds). Persist a random store UUID once;
post/node IDs remain store-local. Serialize 64-bit post, node and generation values
as decimal strings to avoid JavaScript integer precision loss.

The embedding-space descriptor contains model ID and pinned revision, dimension,
pooling/preparation identifier, output normalization and coordinate dtype.
Hash the persisted descriptor JSON to make `embedding_space_id`. The current live model
produces normalized 256D vectors; the GTE-tiny experiment is a different 384D space.

Vector requests supply this ID and exactly `dimension` finite coordinates. Reject
a wrong space/dimension or coordinates outside the advertised root box. Never
silently switch encoders, clamp, project or normalize the query. Non-unit interior
points are allowed: a client's interest location need not be a stored post.
For cosine scoring only, use norms in the distance formula; a zero-norm query is
invalid for nearby ranking, but is valid for routing to a term-summary leaf.

Clients should cache the tree pack and retain response generations with feedback.
A leaf identifier is meaningful only with the store and tree revision: splits and
merges can retire it. Requests are routed against the current committed tree.
Optionally send `client_tree_revision`; responses report whether it differs.
Old packs are not historical read transactions and v1 does not promise reads from
arbitrary old generations.

Queries do not submit feedback, write client profiles, call social-platform APIs
or trigger embedding inference. Return native platform IDs so the client can
hydrate posts through the relevant service.

## GET /v1/tree?view=partitions|ppca

Default: `partitions`. Return a complete immutable pack at one committed generation
and one common decay timestamp, including:

- Schema/version, store/embedding descriptors, generation, tree revision,
  effective time, creation time, freshness deadline and maintenance lag (not source/firehose lag).
- Root ID and root coordinate bounds (currently [-1, 1] on every axis).
- Nodes: persistent ID, parent, node type; split dimension, cut and child IDs
  for internal nodes. JSON dimensions are **zero-based**; Julia exports axis - 1.
  Routing is left for `x[axis] < cut`, right otherwise.
- Leaf entries: ID, retained post count, decayed evidence mass, Pólya probability,
  log box-volume and log piecewise-constant density.
- Density half-life, nondecaying branch-prior parameter `kappa`, estimator name
  and reference measure. Keep retention separate from half-life.
- `view=ppca`: additionally, each leaf's mean, effective rank, low-rank covariance
  factor W and diagonal noise D; covariance is `W Wᵀ + diag(D)`. Declare matrix
  shape and row-major wire order, variance floor, and a mass-based reliability flag.

Root bounds plus the cuts reconstruct each leaf box, avoiding two repeated
dimension-length arrays per leaf. No reservoirs, post vectors, text or term
dictionaries are included. The topology hash changes only when topology changes;
the data generation can also change for metadata/maintenance.

For the normalized partition model:

```text
log_volume(leaf) = sum(log(upper[j] - lower[j]))
log_density(leaf) = log(polya_probability(leaf)) - log_volume(leaf)
```

The probability comes from the existing volume-weighted branch priors and decayed
child evidence, not just a leaf's fraction of total mass. Probabilities sum to one.
These densities use ambient coordinate volume, not surface area on the embedding
unit sphere, and are not calibrated probabilities of semantic relevance.
For zero probability use a JSON null log density plus `zero_density=true`, never
non-JSON Infinity/NaN. PPCA parameters come from the existing sketch-to-parameters
conversion, not raw scatter mistaken for covariance.

**PPCA caveat:** current Gaussian moments are not normalized over their leaf boxes.
Export `support_normalization="untruncated_gaussian"`; do not label
`leaf_probability * GaussianPDF` as a normalized within-leaf density.
Clients can use moments for local geometry or an explicitly approximate Gaussian
potential. A normalized truncated model requires a separate normalization method.

This pack supports a client's chosen density-weighted path calculation, for example
integrating `rho(x)^(-alpha)` along a curve. The client chooses alpha, density
regularization, admissible domain (box versus sphere), graph/path discretization,
and whether to use moments. The pack supplies a density model, **not** a precomputed
geodesic graph or a guarantee of useful Fermat paths. Low-density regions and
high-dimensional box volume need numerical care; log density is the primary value.

Cache complete serialized packs by generation/time/view with ETags and compressed
HTTP transfer. Before returning 304, check freshness as well as ETag: time decay
can change probabilities despite unchanged topology. Bound refresh to five seconds
initially, shorten at the next hard expiry, and invalidate after a processed
deletion. If a fresh pack cannot be produced before its deadline, return a
retryable error rather than representing stale summaries as current.

Clients may extrapolate raw masses with the half-life between refreshes. They
must recompute branch probabilities with the unchanged prior; normalized leaf
probabilities do not all decay by one common multiplier.

## POST /v1/nearby

Request fields:

| Field | Default / rule |
| --- | --- |
| `embedding_space_id` | Required. |
| `vector` | Required coordinate array in that space. |
| `limit` | 25; integer 1–25. |
| `leaf_budget` | 4; capped at 16 and current leaf count. |
| `candidate_budget` | 256; capped at 1,024; at least limit. |
| `client_tree_revision` | Optional hint for detecting a changed tree. |

Algorithm:

1. Select candidate leaves with the existing box-distance routing heuristic.
   Include the containing leaf explicitly, with stable tie breaking for other boxes.
2. Collect unique eligible post IDs using addressable posting slots and stored
   leaf counts. Allocate some probes to every nonempty selected leaf, then allocate
   the remainder proportionally to remaining posting slots. A sparse Fisher–Yates
   draw prevents duplicate slots; expired rows are discarded without decoding.
3. Decode at most the configured candidate budget from Float16 storage, in batches
   no larger than existing 1,024-row fetch limits.
4. Score candidates with cosine distance. Sort ascending, break ties by post ID,
   return at most 25. Zero-norm reconstructed posts are skipped and counted.
5. Return identity, current leaf and offsets for each returned post.

**Confirmed choice:** approximate regional top-k, not global exact kNN. The box heuristic is
Euclidean even when reranking uses cosine; report both stages. Sampling is uniform
within the chosen allocation, not weighted by likes or personalized server state.
It is candidate generation for the client. If budgets, eligibility, deadlines or
available posts yield fewer than 25, return fewer with an explicit reason.

Response item:

```text
post_id: decimal string, store-local
identity: {kind: "bluesky", native_id: "at://did:.../app.bsky.feed.post/..."}
platform: "bluesky"
leaf_id: decimal string
distance: cosine distance to the submitted query
delta: length-d Float32-valued JSON array
```

Define `delta = decoded_stored_embedding - submitted_query`.
Do not substitute a leaf center, PPCA projection or regenerated sample for the
stored post's vector. The client reconstructs approximately as `query + delta`;
error includes original Float16 quantization and delta rounding. Compute distance
using the decoded vector, not a silently normalized delta. Include the query origin
or an unambiguous query digest plus the coordinate policy in the response.

For 25 posts, delta payload alone is about 25 KiB in binary Float32 at 256D, or
37.5 KiB at 384D; JSON is larger. Deltas do not reduce dimensionality or themselves
guarantee wire compression. Tree packs support gzip; nearby and term responses currently use plain JSON. Benchmark
binary/quantized offsets later.

Response diagnostics: snapshot/generation, tree revision, routed leaf, probed leaf
IDs, candidates scored, posting draws, metric, storage dtype, `approximate=true`,
returned count, and underfill reason. Use a request-local random generator, not the
ingestion RNG; queries must never change later tree cuts. Mark responses no-store by default.

## POST /v1/terms

**Confirmed:** YAKE phrases and hashtags represent the same kind of interest cue
for this interface. Reuse the current canonicalized, deduplicated contributions;
a separate hashtag backfill is not a prerequisite.

Inputs: `embedding_space_id`, `vector`, optional `limit` (default 25, maximum 100),
`ranking` (`common` by default, optionally `distinctive`),
`client_tree_revision`, and opaque `page_cursor`.

Route to the **one current leaf containing the vector**. Return combined terms:

- Canonical readable term, without a synthetic hashtag prefix.
- Raw document count: once per post, including a term appearing both as a YAKE
  phrase and a hashtag.
- Exponentially decayed document support.
- Raw rate = term count / covered post count; decayed rate = term support /
  decayed covered post support. Return null rates for a zero denominator.
- Ranking score: support for common terms; the existing positive log-lift score
  for distinctive terms (at least three supporting posts).
- Leaf post total, covered count, English-eligible count, coverage fraction,
  raw/decayed denominators, keyword half-life, extractor/configuration fingerprint,
  generation and effective timestamp.

Default sort: decayed support descending, canonical term ascending. Distinctive
ranking uses the existing global background and add-one smoothing. A fully covered
leaf with no retained terms returns an empty table; incomplete backfill is reported
as coverage, not disguised as absence of terms.

The denominator includes all covered posts, including hashtag-only non-English
posts and posts with empty keyword results. The English-eligible count is useful
metadata but is not the common rate denominator.

Paginate the full leaf frequency table rather than requiring an unbounded response.
Bind the cursor to store, leaf, generation, ranking and effective timestamp. If the
generation is no longer the current snapshot, return `409 snapshot_changed`;
clients restart. Cursors also expire on server restart. No long-lived transaction is held between HTTP calls. Rapid
ingestion can invalidate pagination; retained paginated snapshots are a later
extension if needed. A clipped first page must say `truncated=true` and include
a continuation cursor. A ranking's minimum-support filter is separate from paging.

These are frequencies of **retained extracted terms**, not exhaustive word counts.
Preserve the existing policy: up to eight phrases plus four supplemental distinct
hashtags per post. Return the policy/fingerprint so clients can interpret coverage
and extraction limits. Increasing the hashtag cap or changing the extractor is
not part of this server plan.

Clients may send terms to a provider's keyword or hashtag search as appropriate.
Do not assume every phrase was originally a hashtag; the existing summaries do not
preserve that provenance. Actual provider search capabilities remain a client concern.

**Optional later extension:** `kind=hashtag|phrase|all` or separate endpoints if useful.
This requires phrase/hashtag provenance per contribution and an explicit versioned
backfill. A term may have both kinds on one post; the combined view must still count
it once. Current aggregates cannot reconstruct exact source-kind or per-platform
frequencies, and v1 should not pretend to return those breakdowns.

## Errors and read-only semantics

Use JSON error codes: 400 malformed JSON/options, 413 body too large, 422 invalid
vector/embedding space, 409 changed paginated snapshot, 429 bounded queue/rate
limit, 503 unavailable/freshness deadline. Provide Retry-After when appropriate.
Never return a successful empty result for unavailable storage or failed extraction.

GET/POST queries do not advance source cursors, change statistics, alter RNG state,
or run maintenance. Internal maintenance operates on its own schedule. Read caches
and operational counters may change. The public surface exposes no write,
rebuild, ingestion-control or feedback endpoint.

## Runtime bounds

The default accepts 64 KiB bodies, at most eight active HTTP requests and 64 owner
jobs, with a five-second owner deadline. Queued jobs that expire never execute.
An already-running durable mutation finishes safely even if its caller times out.
Term ranking scans aggregate rows in pages of 256 with a deadline and bounded
top-k memory; pagination limits response size, **not total scan work**.

Packs and each serialized reply are capped at server_budget/32 (1 MiB by default).
A larger PPCA pack returns `503 pack_too_large`; increase the configured server
budget or use the default partition pack. This is conservative buffer sizing,
not a total-process RSS cap. Julia, HTTP transport/TLS and the embedding worker
have separate runtime memory costs.

`GET /healthz` reports HTTP liveness. `GET /readyz` also checks the owner, aggregate
freshness and background worker health. CORS is disabled by default; configure
explicit origins. Requests are not logged with bodies or query coordinates.
