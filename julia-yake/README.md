# YAKE in Julia

A standalone, dependency-free Julia implementation of YAKE's default statistical keyword extractor. No Python process, embedding model, GPU or training is needed. It includes the English stopword list.

From the repository root:

```sh
julia --project=julia-yake julia-yake/examples/extract.jl "Space telescopes observe distant galaxies."
```

Or start Julia with `--project=julia-yake`:

```julia
using Yake
extractor = Extractor(n=3, window_size=1, top=8, dedup_lim=0.9)
keywords = extract_keywords(extractor, "Space telescopes observe distant galaxies.")
[(k.phrase, k.score) for k in keywords]
```

Lower scores mean greater document-local importance. Scores are not probabilities and are not comparable across documents. Configuration supports maximum phrase length, co-occurrence window, output limit, deduplication threshold and custom stopwords. A deduplication threshold of 1 disables fuzzy deduplication. Reuse an extractor across documents or threads without mutating its stopword set; each call owns its document statistics.

The algorithm computes capitalization, frequency, sentence position, sentence spread and co-occurrence features, then combines term scores into phrases with YAKE's internal-stopword weighting. It deduplicates ranked candidates by normalized Levenshtein similarity. The upstream terminal-`s` normalization is retained for scoring compatibility; it is not a general stemmer.

## Compatibility

The scoring target is Python `yake==0.6.0`, with its default features and `seqm` deduplication. This is not an implementation of every optional Python API or multilingual stopword package.

On 1,557 pinned cases (519 texts across three configurations), scoring on **identical tokens** passes phrase/score parity within `rtol=1e-12, atol=1e-14`. One near-tied pair exchanges rank due to floating-point rounding. The largest score difference is 2.22e-16.

The native sentence tokenizer is deliberately smaller than segtok. Token sequences match on 1,482/1,557 cases; fully ranked phrase/score outputs match on 1,481/1,557 cases. These figures include raw, uncleaned text and repeated configurations. Abbreviations, punctuation, URLs, unusual Unicode boundaries and HTML entities can differ. This is algorithmic scoring parity, not a claim of complete tokenizer parity. Common and numeric HTML entities are supported; the full HTML5 entity table is not included.

See [measured performance](benchmark/RESULTS.md) for the separate 100,000-post test using the production input-cleaning policy.

## Existing tree interface

[JuliaYakeExtractor](../src/JuliaYakeExtractor.jl) implements the existing `AbstractKeywordExtractor` contract. It uses three-word phrases, top 24 candidates and an 8,192-character limit. The existing shared Julia layer retains up to eight phrases plus four hashtags and handles canonicalization, identities and fingerprints.

```julia
# Run with --project=environments/storage
include("src/KeywordExtraction.jl")
using .KeywordExtraction
extractor = JuliaYakeExtractor()
inputs = [KeywordInput("post:1", "Space telescopes #NASA", ["NASA"], "en")]
prepared = prepare_keywords(extractor, inputs)
```

URLs and mentions are removed before extraction without modifying stored text. Explicitly non-English posts contribute only hashtags; unknown language uses English and is flagged. Extraction uses original text independently of embedding truncation.

The new extractor has its own version/fingerprint. Existing Python-prepared stores are not silently mixed or migrated. To change an existing store, use the existing explicit `enable_keywords!(store, extractor; rebuild=true)` and resumable `backfill_keywords!(store, extractor)` flow. This rebuilds keyword contributions, not embeddings. Existing demos continue to use their original adapter until explicitly changed.

## Tests and reproduction

From the repository root:

```sh
# Core package tests: Julia only, no external dependencies.
julia --project=julia-yake julia-yake/test/runtests.jl
# Scoring parity against the checked-in fixture (JSON3 in existing storage environment).
julia --project=environments/storage julia-yake/test/parity.jl
# Adapter, multithreaded extraction, disk restart and resumable backfill.
julia --threads=4 --project=environments/storage test/julia_yake.jl
# Existing keyword lifecycle regression suite (includes original Python adapter).
julia --project=environments/storage test/disk_keywords.jl
# Optional: regenerate reference fixtures, requiring pinned Python YAKE and local data.
.venv-embeddings/bin/python julia-yake/benchmark/reference.py
# 100k real-post comparison + paired 10k preparation speed, requiring local data and Python.
julia --threads=1 --project=environments/storage julia-yake/benchmark/run.jl
```

The benchmark checks post identity and text digest against the existing Python sidecar. Preparation timings exclude warmup/model inference/storage; Python process startup is excluded, but its normal request/response serialization is included.

## License

This implementation adapts YAKE's source formulas and includes its stopwords; the upstream distribution's actual license is **AGPL-3.0-or-later**. Adapted segtok word-boundary rules retain their MIT notice. See [NOTICE.md](NOTICE.md), [LICENSE](LICENSE), and [segtok attribution](notices/segtok-LICENSE.txt). The notices apply to this folder and do not relicense unrelated repository files.
