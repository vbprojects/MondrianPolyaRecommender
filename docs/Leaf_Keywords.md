# Pluggable keyword extraction and leaf summaries

Keyword extraction is independent of embeddings and tree geometry. `KeywordExtraction.AbstractKeywordExtractor` produces ranked candidate phrases; shared Julia code normalizes, deduplicates and caps them. Storage accepts `PreparedKeywords`, not YAKE objects or Python state. A Julia extractor can implement `extractor_id` and `extract_keywords` without changing ingestion or persistence. The test suite includes such an implementation.

The initial adapter uses CPU YAKE 0.6.0. Its dedicated worker imports no embedding model, Torch or ONNX. YAKE is a single-document statistical extractor; its scores rank candidates within a document and are not probabilities. Scores remain in the preparation sidecar for inspection and do not weight document counts. See the [upstream implementation](https://github.com/INESCTEC/yake).

## Using the interface

```julia
include("examples/disk_ablation_common.jl")
using .DiskTreeStore.KeywordExtraction

extractor = PythonYakeExtractor()
store = DS.open_store("data/my_keyword_tree"; create_config=DS.config(mode=:f16))
DS.enable_keywords!(store, extractor)
# records: Vector{PostRecord}; X: 384 × length(records) embedding matrix
keywords = prepare_keywords(extractor, keyword_input.(records))
DS.append_batch!(store, records, X; keywords, time=now)
summary = DS.leaf_keywords(store, leaf_id; ranking=:distinctive)
close(extractor)
DS.close_store(store)
```

Methods for another extractor:

```julia
struct MyExtractor <: AbstractKeywordExtractor end
KeywordExtraction.extractor_id(::MyExtractor) = "my-extractor-v1;configuration=..."
function KeywordExtraction.extract_keywords(e::MyExtractor, inputs)
    # Return one KeywordResult per input, in the same order.
    # Each Candidate contains phrase::String, rank::Int, and optional Float64 score.
    # Return KeywordResult(candidates, extractor_id(e), truncated::Bool).
end
```

Version all changes that could change candidates, including preprocessing and length limits. The shared policy has its own fingerprint. `prepare_keywords` checks output count and extractor identity. Disk ingestion validates each prepared identity, relevant-content digest and fingerprint before entering a transaction. Returning an empty successful result is different from a worker failure; failures raise and prevent keyword-enabled ingestion.

Default policy:

- English YAKE; one-to-three-word phrases; up to eight distinct phrases and four additional hashtags. A phrase and matching hashtag count once.
- Original text is retained unchanged. The YAKE adapter removes HTTP(S)/www URLs and account mentions, limits its input to 8,192 Unicode characters, and reports truncation. This is independent of the encoder's 256-token limit.
- Julia applies Unicode compatibility normalization, case folding and whitespace normalization. No stemming, synonym merging or named-entity resolution. Stored terms are at most 256 UTF-8 bytes.
- Available `language` or `langs` metadata is respected. Explicitly non-English posts contribute hashtags only; unknown-language posts are processed as English and flagged. The existing 100,000-post fixture lacks language metadata.

## On-disk state and lifecycle

`kw_terms` holds a disk dictionary; `kw_posts` stores each post's bounded list of term IDs and preparation metadata; `kw_counts` holds exact raw document counts and lazily decayed masses keyed by leaf and term. There is no inverted term-to-post retrieval index. Scores and unused raw candidates need not occupy the production database.

Counts are exact for the retained extractor output, not every word appearing in the original text.

Each prepared post contributes once to coverage; English-eligible posts also contribute to an eligibility counter. Term rates use **covered posts**, including hashtag-only non-English posts, so their numerator and denominator describe the same population. Keyword mass is independent of PPCA mass. The keyword half-life defaults to the density half-life but is separately configurable when enabling keywords.

Counts age from the indexed content version's `density_at`. Replay, backfill and hashtag-only updates preserve that age. Text edits retire old contributions and give the new indexed version a new origin time. `keyword_update!` accepts metadata-only changes without an embedding matrix and rejects changed text. Older revisions do not overwrite newer versions.

Splits and merges reuse the existing posting-movement paths: moving each post subtracts its stored contribution from the old leaf and adds it to the new leaf. Global counts remain unchanged. Deleted/expired posts lose their contributions; garbage collection removes dictionary entries that no active post uses. SQLite pages become reusable; this does not shrink the database file automatically.

Counts, postings, keyword configuration and generation commit in the same SQLite transaction. On any commit failure, discard the poisoned store and reopen. Abrupt-exit tests cover failures before payload publication, before the SQL commit and after the commit.

`leaf_keywords(store, leaf; limit=10, ranking=:distinctive, time=store.clock)` first processes due logical expiry. It returns terms, counts, decayed support, score, coverage, eligible count, version, generation and timestamp. It can therefore perform writes and trigger the existing density rebuild when expiry is due. Physical payload reclamation remains separate. Common labels sort by decayed support. Distinctive labels use support × positive log(local/global rate), with add-one/Beta(1,1) smoothing and at least three raw supporting posts. Small or uninformative leaves may have no distinctive labels.

Leaf summaries stream 256 aggregate rows at a time and retain at most 100 results. The last summary is cached, keyed by generation/time/query, inside the existing 16 MiB work-buffer reservation; writes and cache clearing invalidate it. Dictionaries and per-post contributions are not loaded wholesale. SQLite uses the existing bounded database cache. Exact per-leaf term tables can nevertheless occupy substantial disk space: bounded RAM is not bounded vocabulary size.

## Backfill and compatibility

Existing stores open with keywords disabled and incur no per-post keyword SQL work. Explicitly enable, then run:

```julia
DS.enable_keywords!(store, extractor)
DS.backfill_keywords!(store, extractor; batch_size=128, max_batches=10)
DS.keyword_stats(store)  # partial coverage is reported
```

Each backfill page commits independently. Restart and repeat to continue missing active posts without loading the catalog. Backfill reads stored text, never embeddings. Changing an extractor or keyword half-life requires `enable_keywords!(...; rebuild=true)` and a backfill; this explicitly clears old keyword tables while preserving the tree and vectors. New keyword-enabled arrivals must supply prepared results. The current live Arrow pipeline remains unchanged.

## Reproduce the experiment

```bash
.venv-embeddings/bin/pip install -r scripts/requirements-keywords.txt
.venv-embeddings/bin/python scripts/prepare_keywords.py --workers 4
julia --project=environments/storage test/disk_keywords.jl
julia --project=environments/storage test/keyword_crash_process.jl
julia --project=environments/storage examples/keyword_experiment.jl disabled
julia --project=environments/storage examples/keyword_experiment.jl hashtags
julia --project=environments/storage examples/keyword_experiment.jl yake
julia --project=environments/storage examples/keyword_experiment.jl online
julia --project=environments/storage examples/validate_keywords.jl
python3 scripts/keyword_report.py
node test/keyword_viewer.cjs
```

Experiment stores must be fresh; the script refuses to overwrite them. `prepare_disk_ablation.py --keywords` also invokes the same extraction module alongside embedding preparation, including when embeddings already exist. This opt-in operation preserves embedding values and the encoder configuration.

Open [the interactive comparison](../output/keywords/demo.html), [raw results](../output/keywords/results.json), [pinned configuration](../output/keywords/config.json), and [measurement summary](../output/keywords/summary.txt). Preparation runs separately from storage workers. Storage variants use identical inputs, seed, Float16 format and PPCA configuration, with a 32 MiB managed budget. Process RSS is reported separately. Cache-cleared summary measurements clear application/SQLite caches, not the operating system's file cache. One-seed timing is exploratory, not a production capacity claim.

The viewer shows early cuts, the full corpus, and decay-induced merges. Example posts are the first eight in a leaf's sequence order, with their retained terms. They are inspection aids, not an unbiased sample or a semantic quality score. Unknown-language extraction, fragmented phrases and generic vocabulary remain visible limitations. No recommendation-quality improvement is claimed.

The `online` run measures 10,000 posts through the real Julia-to-Python extractor interface and durable disk ingestion together, using one keyword worker. It excludes embedding inference, consumes the same fixed embeddings, and reports preparation-inclusive throughput separately from storage-only timing.

## Initial measurements

The 100,000-post comparison used a 32 MiB managed budget, not a 32 MiB process-memory limit. Process RSS was approximately 793–819 MiB across methods, including Julia runtime/compiler and allocator state.

| Method | Durable storage posts/s | Added database pages | Summary p95, application caches cleared | 128 → 5 leaf merge |
|---|---:|---:|---:|---:|
| Disabled | 1,589 | — | — | 43.3 s |
| Hashtags only | 1,236 | 10.0 MiB | 2.1 ms | 97.7 s |
| YAKE + hashtags | 809 | 53.1 MiB | 46.1 ms | 158.9 s |

YAKE preparation processed the corpus in 164.1 seconds (609 posts/s) with four CPU workers; none hit the 8,192-character cap. These preparation timings were collected on the shared workstation while correctness checks also ran. The three storage runs were sequential. The disabled and hashtag merge probes were measured in separately reopened processes; YAKE's merge followed its application-cache-cleared summary measurements. All used the same target time, topology, and 32 MiB budget.

YAKE's measured file footprint including WAL increased by 67.9 MiB over the disabled run. Database-page and physical-file numbers differ because WAL and free pages are not included in the page totals. The keyword dictionary contained 346,145 distinct retained terms. Exact per-post contributions make edits and splits verifiable but carry nontrivial storage and write costs.

The major performance follow-up is bounded batching of counter deltas during structural changes. The current implementation updates counters per moved post, so a large merge is expensive, even though regular ingestion exceeds 100 posts/s. Labels include useful topics such as art and Thanksgiving, alongside generic words and awkward fragments. Bare domains without a scheme or `www` can still leak into extracted phrases; the current cleanup is intentionally small and should be improved in a versioned policy update. Keep this optional and review the labels before presenting them as client interests; neither those examples nor throughput establish recommender quality.

The direct 10,000-post, single-worker YAKE + durable-ingestion run achieved **159 posts/s**, excluding embedding inference. This is measured pipeline throughput; the separate four-worker preparation plus storage timings yield a serial-composition estimate of 342 posts/s for 100,000 posts. These are different execution configurations, not competing measurements of the same pipeline.

An independent in-memory audit, outside measured storage workers, checked all 100,000 posts after merging in each keyword store: 25,051 hashtag counters and 732,333 YAKE counters agreed exactly in raw counts, with maximum absolute decay-mass error about **7.1 × 10⁻¹²**. All three real-data ingestion runs produced identical tree cuts.

The retained YAKE output averages 5.58 terms per post. Of 346,145 distinct terms, 312,463 (90.3%) occur in only one post. The three-post minimum keeps these out of distinctive labels, but their contributions remain on disk for correct updates and possible later repetition. This is a concrete storage/label-quality tradeoff worth revisiting before larger-scale use.

Replay assigns simulated creation/ingestion times at 100 posts/s, retaining historical source times as metadata. The 72-hour jump is a deliberate bulk-maintenance stress case, not a continuously ingested three-day feed. See [validation results](../output/keywords/validation.json) for the 2,680 passing assertions, full-corpus counter audit and browser checks.

The optional [native Julia YAKE implementation](../julia-yake/README.md) now implements
the same extraction contract through `JuliaYakeExtractor()`. It has no Python runtime
dependency and has a separate extractor fingerprint. Switching existing keyword
stores requires an explicit keyword rebuild/backfill; embeddings remain reusable.
See its README for scoring parity, tokenizer differences and throughput measurements.
