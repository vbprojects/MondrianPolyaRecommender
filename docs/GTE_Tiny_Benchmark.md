# GTE-tiny benchmark — 2026-09-28

GTE-tiny is a credible candidate for the ONNX INT8 stage. Its unquantized model substantially improves on the current static embedding baseline in this screen, performs similarly to MiniLM, and matches both small transformers on Twitter paraphrase AP. BGE-small remains clearly stronger on retrieval. These results support testing INT8; they do not validate the quantized model or change the application's encoder.

The [Colab L4 notebook](../encoder_experiments_colab/09_bert_hash_static_comparison.ipynb) now includes `TaylorAI/gte-tiny`, pinned to `4cc5e73d86a67c601897257b467187234aa3bca3`. It uses masked mean pooling, no query instruction, and normalized native 384D vectors, following the [model card](https://huggingface.co/TaylorAI/gte-tiny). Parameters are explicitly converted to float32 because the checkpoint configuration advertises float16. The notebook includes a four-model selection example and paired GTE-tiny retrieval comparisons against all three baselines below.

## Full quality screen

| Model | Mean NDCG@10 | Twitter AP | Twitter ROC-AUC |
| --- | ---: | ---: | ---: |
| Current Qwen Model2Vec ONNX, 256D | 0.0183 | 0.4615 | 0.6385 |
| MiniLM-L6-v2, 384D | 0.5801 | 0.8470 | 0.9381 |
| **GTE-tiny, 384D** | **0.6026** | **0.8484** | **0.9396** |
| BGE-small-en-v1.5, 384D | 0.6886 | 0.8484 | 0.9383 |

Retrieval averages NanoNQ, NanoHotpotQA and NanoMSMARCO equally: 50 queries each, with complete candidate corpora of 5,035, 5,090 and 5,043 documents, respectively. Twitter uses all 51,534 pairs (45,371 unique texts). These are full benchmark subsets, not the notebook's smoke fixtures. No classification probes or new encoder training were run.

Every model uses the same data revisions, input texts, candidate IDs and qrels as notebook 09, a 256-token cap, float32 vectors, and exact cosine ranking. Tokenizers retain different text at a common token cap; per-model truncation counts are saved. BGE receives its recommended retrieval-query instruction. Static encoding uses the existing project's ONNX adapter and original pinned model. Its poor results here characterize this artifact/protocol, not all static embedding models.

GTE-tiny's individual retrieval scores are 0.5648 on NanoNQ, 0.7114 on NanoHotpotQA, and 0.5315 on NanoMSMARCO.

## Paired retrieval uncertainty

10,000 bootstrap replicates, seed 42, with query IDs resampled within each dataset and dataset means weighted equally. The adjusted intervals cover the three GTE-tiny contrasts as a separate Bonferroni family.

| GTE-tiny minus baseline | Mean difference | Ordinary 95% interval | Family-adjusted interval |
| --- | ---: | --- | --- |
| Current static | +0.5843 | [+0.5243, +0.6451] | [+0.5115, +0.6578] |
| MiniLM | +0.0225 | [−0.0217, +0.0670] | [−0.0303, +0.0777] |
| BGE-small | −0.0860 | [−0.1269, −0.0467] | [−0.1343, −0.0378] |

The small point advantage over MiniLM is inconclusive; the gap behind BGE-small is clear in this screen. Twitter AP point estimates for these three contextual models are nearly identical. A connected-text bootstrap comparison of GTE-tiny against current static is also saved; Twitter significance against MiniLM or BGE was not tested. Pretraining overlap is unknown, and the small Nano query sets do not establish performance on every deployment corpus.

## Measured encoding cost

Local hardware: AMD Ryzen 7 7800X3D, two CPU threads, and NVIDIA GeForce RTX 4070 SUPER. **These are not L4 timing measurements.** Quality inference used GPU for the contextual encoders and CPU for current static. CPU and GPU timing workers loaded models separately.

| Model | CPU batch-1 median, ms | CPU batch-32 texts/s | GPU batch-1 median, ms | GPU batch-32 texts/s |
| --- | ---: | ---: | ---: | ---: |
| Current static | 6.06 | 171.2 | Not measured | Not measured |
| MiniLM | 6.43 | 54.7 | 2.99 | 2,106.5 |
| GTE-tiny | 6.50 | 55.4 | 2.89 | 2,182.6 |
| BGE-small | 14.62 | 27.6 | 4.85 | 1,295.6 |

Timings include tokenization, packing, transfers, inference, pooling and output normalization to CPU float32 vectors. They exclude loading, network access, search and disk caches. The fixed panel contains 128 real retrieval/social texts, with ten warmup batches and three measured passes. Batch-32 padding makes its per-text cost depend strongly on the mixed length distribution. Throughput is total texts divided by total elapsed time; batch-1 values are medians, so they are not reciprocal throughput figures. Raw samples and p95 values are retained. Timing-ratio significance and rotated-order trials were not performed.

## ONNX INT8 next stage

The same pinned repository contains both graphs. Inspection found:

| Artifact | Bytes | Relevant graph evidence |
| --- | ---: | --- |
| `onnx/model.onnx` | 90,387,629 | Float initializers and ordinary MatMul |
| `onnx/model_quantized.onnx` | 22,887,653 | INT8/UINT8 initializers, 48 MatMulInteger and 48 DynamicQuantizeLinear nodes |

The published quantized graph is about 3.95× smaller on disk than the float32 ONNX graph. This comparison is between ONNX files, not the released PyTorch checkpoint or resident memory. File inspection confirms integer quantization; **the ONNX graphs have not been benchmarked for inference quality or speed in this run**.

Next, check the float32 ONNX graph against the PyTorch reference, then run the INT8 graph through the identical retrieval and Twitter inputs. Keep pooling, token limits and normalization unchanged. Report paired quality deltas against unquantized GTE-tiny and measured CPU latency on the deployment hardware. A reasonable screening target to fix before that run is at most 0.01 absolute loss in mean NDCG@10 and Twitter AP; inspect uncertainty and per-dataset losses before adopting it. This is a proposed INT8 tolerance, not a measured result or a user-specified requirement.

Quantizing model weights does not shrink the default index vectors: 384D float32 still requires 1,536 bytes/document, compared with the current 256D model's 1,024 bytes/document, before index metadata. The current runtime and index configuration remain unchanged.

## Reproducibility and artifacts

Run identity: `bced8b63d83cdd78`. The run directory retains the executed worker, model/data revisions, environment, immutable input manifests, task predictions, numerical checks, timing samples, output checksums, report source and completion marker.

- [Configuration](../output/gte_tiny_screen/bced8b63d83cdd78/config.json)
- [Quality table](../output/gte_tiny_screen/bced8b63d83cdd78/report/quality.csv)
- [Paired retrieval differences](../output/gte_tiny_screen/bced8b63d83cdd78/report/paired-retrieval-differences.csv)
- [Twitter grouped interval](../output/gte_tiny_screen/bced8b63d83cdd78/report/paired-twitter-differences.json)
- [Latency table](../output/gte_tiny_screen/bced8b63d83cdd78/report/latency.csv)
- [CPU quality/cost plot](../output/gte_tiny_screen/bced8b63d83cdd78/report/quality-cost-cpu.png)
- [ONNX graph hashes and inspection](../output/gte_tiny_screen/bced8b63d83cdd78/gte-onnx-artifacts.json)

Validation covered notebook schema and Python syntax, synchronized model registries, real GTE-tiny numerical checks, complete four-model quality/timing execution, paired statistics, plots, and verified result bundles. Colab Drive behavior and L4 performance remain unmeasured locally.
