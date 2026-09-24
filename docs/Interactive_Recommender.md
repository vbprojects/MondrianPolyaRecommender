# Try the ancestor-residual recommender

Open [the local demo](../output/interactive/demo.html) in a browser. It is a self-contained file: it requires no server, account or network access. Click **Like**, **Dislike**, or **Skip** to get another post. Undo reverses the last action; Start over resets the session. Ratings persist in browser local storage when supported; Download ratings exports a JSON history containing embedding row IDs, actions, timestamps and prediction diagnostics. No feedback is sent to Bluesky or any other service. Browser storage can be unavailable or cleared, so download a session you want to keep.

Build it again with Julia:

```sh
julia --project=. examples/interactive_recommender.jl
```

The build reads the saved ancestor-residual model and original embedding archive. The resulting HTML bundles the candidate post text and precomputed response probabilities. Browser JavaScript evaluates information gain and updates the session, so there is no running Julia backend to manage. No microclusters are used.

## Policy and assumptions

The 395 occupied active residual leaves are possible user-interest regions. Each has a representative actual post nearest its mean embedding. Initial belief is the residual model's leaf probability, renormalized over occupied leaves. Each region supplies its representative and up to three seeded random posts, yielding 1,436 candidates before exact-text deduplication in the browser. This is a small fixed candidate pool, not a search across all 100,000 posts after every answer.

A symmetric 12-neighbor graph connects region representatives by embedding chord distance, with minimum-spanning-tree edges guaranteeing connectivity. Edge distances are discounted by a bounded local occupancy density proxy. This is a heuristic Fermat-like graph metric, not the exact continuous geodesic of the residual box density. The density proxy uses neighborhood mass divided by squared neighborhood radius; exponent 2 is a heuristic, not an inferred intrinsic dimension. A candidate connects to its three nearest representatives. Shortest paths approximate its distance to each hypothesized interest location.

The assumed probability of a like is `0.05 + 0.9/(1 + (distance/scale)^2)`, with scale 0.4 times the median representative graph distance. These parameters are not fitted to human feedback. A single stationary interest location is assumed; multiple interests, wording, sentiment, language and author preferences can violate that model.

For each unseen candidate, expected information gain is binary entropy of its predicted response minus posterior-weighted binary response entropy under individual hypotheses. The candidate with the greatest score is shown. The response is not forced to bisect the tree: a like can support a localized region while a dislike rules against it, giving asymmetric changes in belief. This is a graph-distance adaptation of the information-gain strategy; it does not regenerate shortest-path midpoint proposals after every action.

Updates multiply each hypothesis weight by the square root of the observed response likelihood, normalize, and mix in 1% of the initial prior. This softens noisy feedback and allows recovery. It is a heuristic tempered update, so the displayed information gain is the untempered model's criterion, not an exact prediction of the implemented update's entropy reduction. Skip excludes that post without updating belief. Undo recomputes belief from the remaining history.

This is an exploration demo: information gain can favor a post the model expects you to dislike. The concentration bar is not measured accuracy or calibrated confidence. No claim of recommendation quality follows from completing a session.

## Validation

Julia validates finite graph distances, normalized prior and bounded likelihoods during export. The browser smoke test checks Like/Dislike, posterior normalization, Skip, Undo, persistence, reset, no repeated text, expected-information scoring, rating download, safe literal text rendering, mobile layout and absence of external requests:

```sh
# Requires Playwright and its Chromium browser. Point PLAYWRIGHT_MODULE to an
# existing installation if it is not in Node's normal module search path.
node test/interactive_recommender.cjs
```

The original text is unfiltered historical content. The inspector renders it as text, never HTML. The session export contains row IDs rather than copying post text.
