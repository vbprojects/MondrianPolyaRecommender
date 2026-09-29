# Native YAKE measurements

Measured on the existing 100,000 historical Bluesky posts, AMD Ryzen 7 7800X3D,
Julia 1.13.0, one Julia execution thread. Python reference: YAKE 0.6.0.
Pinned settings and raw measurements are in [results.json](results.json).

| Measurement | Result |
|---|---:|
| Native extraction, 100,000 posts | 4.21 s |
| Native extraction throughput | 23,740 posts/s |
| 256-post batch p50 / p95 | 9.66 / 14.78 ms |
| Mean top-eight set Jaccard vs Python | 99.45% |
| Identical ordered top eight | 98.06% |
| Top-one agreement, at least one nonempty output | 99.23% |
| Julia preparation, matched first 10,000 posts | 0.574 s; 17,434 posts/s |
| Python adapter preparation, same 10,000 posts | 51.90 s; 193 posts/s |

These are warmed single-run workstation measurements, not sustained end-to-end
ingestion benchmarks. The paired preparation comparison includes shared Julia
canonicalization, digest generation and hashtag supplementation. The Python adapter
includes its normal IPC/JSON costs but excludes process startup. This observed
ratio is about 90× for the complete adapter path; it is not a claim about YAKE
arithmetic alone. Both paths use one worker, with no embedding inference or storage
writes. Other short correctness tests ran concurrently on the workstation.

The full-data extraction timing sums calls on 256-post batches and excludes input
parsing, reference comparison and final canonicalization. Inputs are held in memory
by this benchmark harness; the extractor itself uses document-local state.
Inputs lack language metadata, so both paths use the existing unknown-language
English assumption. URLs and mentions are removed and original text is capped at
8,192 characters. Lowercased/canonicalized top-eight phrases are compared before
hashtag supplementation. Empty-vs-empty sets count as Jaccard 1.
The script verifies identity and text SHA-256 against every Python reference row.

Scoring parity is tested independently on 1,557 cases (519 source texts and three
configurations). All cases pass within numerical tolerance when given identical
tokens, allowing ordering changes only between numerically tied scores.
There is one such tied pair; maximum score difference is 2.22e-16.
Strict native end-to-end parity is lower on the raw tokenizer fixture:
1,481/1,557 cases. The native tokenizer intentionally does not duplicate all segtok
boundary rules. Agreement with Python is a compatibility measure, not a measure
of recommendation or keyword-label quality.

Validation: **3,531 assertions passed** across standalone tests (44), fixed-token
parity (1,557), native adapter/disk integration (18), and existing keyword lifecycle
regressions (1,912). Tests cover Unicode, empty results, configuration validation,
language handling, truncation, concurrent reuse, version rejection, disk restart,
resumable backfill, edits, expiry, splits/merges, rollback and cache eviction.

Recommendation: the native adapter is ready for an explicit experimental keyword
rebuild using the existing interface. It comfortably exceeds the extraction-only
100 posts/s target in this run. Keep the separate extractor fingerprint because
tokenization differences can change labels. No existing store, embedding model,
tree routing policy or live ingestion default was switched.
