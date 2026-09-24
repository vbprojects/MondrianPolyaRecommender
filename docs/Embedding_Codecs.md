# Frozen embedding codec experiment

The experiment encoded all 100,000 Bluesky vectors into actual packed files. Original embeddings and text remain intact. [Full results](../output/embedding_codec/results.json) and [summary](../output/embedding_codec/summary.txt).

Both families share 1,318 frozen local component centroids and identical assignments. Each post chooses the nearest centroid among owners along its original tree path. The centroid-residual family is a microcluster-style codec using those centroids, not the independently fitted 2,946-microcluster model from the earlier density comparison. This paired design isolates representation coding from different clustering.

| Codec | Total MiB | Bytes/post | Recall@10 | Mean cosine | IG choice agreement | Mean IG regret (bits) |
|---|---:|---:|---:|---:|---:|---:|
| centroid_residual_1bit | 6.58 | 69.05 | 2.0% | 0.822928 | 2% | 0.035614 |
| centroid_residual_2bit | 9.64 | 101.05 | 17.6% | 0.979060 | 4% | 0.017433 |
| centroid_residual_4bit | 15.74 | 165.05 | 69.3% | 0.999265 | 56% | 0.001355 |
| centroid_residual_8bit | 27.95 | 293.05 | 95.5% | 0.999997 | 94% | 0.000006 |
| ppca_rank4_latent_4bit | 7.61 | 79.75 | 10.0% | 0.962138 | 2% | 0.023292 |
| ppca_rank4_latent_8bit | 7.80 | 81.75 | 10.0% | 0.962261 | 4% | 0.022282 |
| ppca_rank4_latent8_tail1 | 12.13 | 127.24 | 3.6% | 0.836672 | 0% | 0.033320 |
| ppca_rank4_latent8_tail2 | 15.19 | 159.24 | 23.9% | 0.981704 | 4% | 0.015085 |
| ppca_rank4_latent8_tail4 | 21.29 | 223.24 | 70.6% | 0.999374 | 52% | 0.001345 |

## Interpretation

At a common 16 MiB ceiling, the centroid 4-bit configuration uses 15.74 MiB and preserves 69.3% of exact top-10 neighbors. PPCA latent8+tail2 uses 15.19 MiB and preserves 23.9%. These are near-matched actual sizes, not exactly equal bytes. At an 8 MiB ceiling, PPCA latent-only fits and achieves 10% recall versus 2% for the feasible centroid 1-bit configuration. PPCA therefore has a benefit at the tightest tested budget, but absolute retrieval preservation is poor there. All budget-feasible variants are listed in results.json; no selection algorithm was tuned on test scores.

The rank-four latent-only code loses individual variation outside the subspace. Increasing latent precision from 4 to 8 bits changes little. Adding a 1-bit residual can actually worsen reconstruction: the symmetric two-level quantizer has no zero code and can introduce large errors near zero. These are simple uniform quantizers, not optimized product quantization or entropy coding. Per-component maximum scales are sensitive to outliers.

8-bit centroid residuals preserve 95.5% of exact neighbors and 94% of IG choices while using 27.95 MiB, versus 98.42 MiB for Float32 vectors plus UInt64 post IDs. This is about 3.52× smaller. Exact choice agreement can penalize near-ties; IG regret reports the information lost by choosing the compressed-model winner, evaluated under the original vectors. These are codec-preservation results, not a human recommendation-quality result.

## Encoding and storage

Centroid codes store quantized offsets in all 256 dimensions at 1, 2, 4 or 8 bits per coordinate. PPCA codes store four latent coordinates at 4 or 8 bits, optionally followed by 1/2/4-bit residuals in 256 dimensions. The latter residual is computed after reconstructing the quantized latent vector. Scale arrays and decoder parameters use Float32. Packed bit payloads are actually written, unpacked and verified. Random-access decoding supports selected post rows.

Totals include the shared means, per-component quantizer scales, PPCA bases when required, UInt16 component IDs, UInt64 post row IDs, and Int32 component owner IDs. Original full URI/text lookup is external: the 8-byte ID is a foreign key, not the whole post identifier string. Shared tree storage, text, filesystem block overhead and JSON metadata are excluded equally. Noise/scatter parameters train PPCA but are not needed to decode latent vectors, so they are excluded. Output directories contain multiple variants and redundant shared files; the sum of the whole directory is not a single-codec storage footprint.

This is a frozen offline codec trained/calibrated on the corpus being encoded. Existing online MPPCA parameters are frozen before assignment and encoding; quantizer ranges use that same corpus. It is not a test of online codebook migration or generalization to unseen future posts. The original vector archive was not deleted.

## Evaluation

100 seeded query vectors are compared against all 100,000 decoded post vectors; each query excludes itself. Original query vectors are retained for evaluation only. Recall@10 measures overlap with original-vector neighbors. Exact duplicate texts/vectors can produce ties; no tie-aware adjustment is applied. Reconstructed vectors are normalized for cosine ranking; raw coordinate RMSE is measured before normalization.

Information-gain evaluation uses 50 fixed synthetic belief distributions over 128 original-vector hypotheses and 256 candidate posts each. Only candidates are decoded; hypotheses are unchanged. The response function is 0.05 + 0.9/(1 + squared_chord_distance/0.25). This uses cosine/chord geometry, not the interactive demo's density-weighted Fermat graph. No actual user feedback, localization trials or confidence intervals are included. This checks whether codec distortion changes a fixed scoring procedure.

## Reproduce and decode

```sh
julia --project=. examples/embedding_codec.jl
.venv-embeddings/bin/python -m unittest discover -s test -p test_embedding_codec.py
```

Julia exports the frozen tree-local codebook; the existing isolated NumPy environment supplies bit packing and batched metric calculations. No new Python dependency is required. Binary parameters are exported in native little-endian layout on this machine.

```python
import sys
sys.path.insert(0, "scripts")
from embedding_codec import decode
vectors = decode("centroid_residual_4bit", rows=[0, 100, 99999])
# shape (3, 256); raw reconstructed vectors
```

Saved payload hashes are in results.json. Tests cover packing including partial final bytes, quantization error bounds, zero ranges, finite random-access reconstruction and invalid indices. Every benchmark variant also verifies decoded saved bytes against its in-memory reconstruction on over 1,000 sampled rows.

Follow-up: [per-post coordinate migration under actual tree splits and merges](Latent_Migration.md) tests UUID preservation, numerical conditioning, cumulative geometric drift, and perturbation policies.

The [streaming codec comparison](Streaming_Codecs.md) adds continuing arrivals, decoder-version storage, inherited bases, adaptive rank, and daily post expiry on synthetic streams and the 100,000 Bluesky embeddings.
