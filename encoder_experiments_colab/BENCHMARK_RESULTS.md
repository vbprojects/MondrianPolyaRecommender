# Encoder benchmark results

Results collected through **2026-09-28** cover BERT Hash Femto/Pico/Nano, static embeddings, MiniLM, BGE-small, trained Ettin exits, and GTE-tiny. GTE-tiny offers the quality and encoding-cost balance we want: strong retrieval, Twitter paraphrase scores comparable to BGE-small, and CPU latency close to MiniLM.

The comparison uses 150 retrieval queries across NanoNQ, NanoHotpotQA and NanoMSMARCO, each with its complete candidate corpus, plus all 51,534 Twitter URL paraphrase pairs. Retrieval is the equally weighted mean NDCG@10 across the three datasets; AP is average precision on Twitter. Higher is better. Quality runs share pinned data, a 256-token cap and normalized float32 vectors; BGE uses its retrieval-query instruction.

| Encoder / configuration | Vector dimensions | Mean NDCG@10 | Twitter AP |
| --- | ---: | ---: | ---: |
| BGE-small-en-v1.5 | 384 | **0.6886** | 0.8484 |
| **GTE-tiny** | **384** | **0.6026** | **0.8484** |
| MiniLM-L6-v2 | 384 | 0.5801 | 0.8470 |
| Trained Ettin, depth 8, same-exit retrieval | 256 | 0.5195 | 0.7207 |
| Trained Ettin, full depth 10 | 384 | 0.5052 | 0.7338 |
| Static Retrieval MRL | 1024 | 0.5045 | 0.8175 |
| Static Retrieval MRL | 256 | 0.4794 | 0.8111 |
| BERT Hash Nano embeddings | 128 | 0.4709 | 0.8232 |
| Static Retrieval MRL | 128 | 0.4559 | 0.8035 |
| BERT Hash Pico embeddings | 80 | —* | 0.8040 |
| BERT Hash Femto embeddings | 50 | 0.2991 | 0.7734 |
| POTION-base-8M | 256 | —* | 0.8195 |
| BERT Hash Nano base | 128 | 0.1230 | 0.7277 |
| BERT Hash Pico base | 80 | 0.0599 | 0.7066 |
| BERT Hash Femto base | 50 | 0.0344 | 0.6248 |
| Current Qwen Model2Vec ONNX | 256 | 0.0183 | 0.4615 |

\* These two retrieval values are omitted from the saved notebook's truncated output; the original Drive report is needed to recover them. Their Twitter results are retained. BERT Hash names refer to the NeuML checkpoints, with `-embeddings` distinguishing embedding-tuned models from base controls. Current-static results describe the particular Qwen Model2Vec artifact and adapter, not static embeddings generally.

Ettin was evaluated at all ten depths and 128/256/384 dimensions, including same-exit and full-depth-corpus retrieval. Depth 8 / 256D was its best observed retrieval setting; its best Twitter AP was 0.7498 at depth 1 / 128D. These are different configurations selected retrospectively from the sweep, not an independently validated combined winner.

GTE-tiny scored 0.5648 on NanoNQ, 0.7114 on NanoHotpotQA and 0.5315 on NanoMSMARCO. Its retrieval advantage over MiniLM is inconclusive: the paired difference is +0.0225, with a 95% bootstrap interval of [−0.0217, +0.0670]. BGE-small's retrieval advantage is clear: GTE minus BGE is −0.0860 [−0.1269, −0.0467]. Similar Twitter point estimates do not establish statistical equivalence.

The earlier frozen-embedding classification comparison also favored BGE-small:

| Encoder, 384D | GoEmotions micro-F1 | TweetTopic macro-F1 |
| --- | ---: | ---: |
| BGE-small | 0.3960 | 0.7042 |
| MiniLM | 0.3355 | 0.6724 |
| Trained Ettin, full depth | 0.3294 | 0.6165 |

These probes used 1,000 emotion and 999 topic test examples, with validation-tuned emotion thresholds. Ettin's best observed topic exit reached 0.6574 at depth 5. GTE-tiny has not been evaluated on these classification probes; its selection is based on retrieval, paraphrase ranking and encoding cost.

The four-model GTE run measured the following end-to-end encoding costs, including tokenization and normalization:

| Encoder | CPU batch-1 median, ms | CPU batch-32 texts/s | GPU batch-32 texts/s |
| --- | ---: | ---: | ---: |
| Current static | 6.06 | 171.2 | Not measured |
| MiniLM | 6.43 | 54.7 | 2,106.5 |
| **GTE-tiny** | **6.50** | **55.4** | **2,182.6** |
| BGE-small | 14.62 | 27.6 | 1,295.6 |

Timing hardware was a Ryzen 7 7800X3D with two CPU threads and an RTX 4070 SUPER, using a fixed 128-text panel, ten warmup batches and three measured passes. These are local timings; the broader BERT Hash quality run used a Colab L4. GTE-tiny delivered about twice BGE-small's CPU batch throughput in this panel, while the current static encoder remained faster at batch encoding.

Sources: saved outputs in [notebook 09](09_bert_hash_static_comparison.ipynb), L4 run `cfdd396ef971b0e0`; [notebook 07](07_local_baseline_comparison.ipynb), local Ettin/probe run `432fa34e2e42fe47`; and the [GTE-tiny report with raw result links](../docs/GTE_Tiny_Benchmark.md), run `bced8b63d83cdd78`. The L4 artifacts live under `MyDrive/encoder_experiments/bert-hash-l4-01/cfdd396ef971b0e0/`.

The intended deployment format is **ONNX INT8**. Its published graph is 22.9 MB versus 90.4 MB for FP32 ONNX, approximately 3.95× smaller. Graph inspection is complete; inference parity, quantized quality and deployment-CPU speed still need validation. The scores above are for unquantized GTE-tiny. Its default 384D float32 index vectors remain 1,536 bytes each, versus 1,024 bytes for the current 256D baseline.

**We are going forward with GTE-tiny.**
