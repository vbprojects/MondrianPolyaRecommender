# BERT Hash versus static embeddings and small transformers

Status: broader experiment specification, with an executable [Colab L4 screening notebook](09_bert_hash_static_comparison.ipynb). The notebook documents its reduced timing suite and optional probes/bootstrap; it does not implement every follow-up measurement below. Reference: [notebook 07](07_local_baseline_comparison.ipynb). Model metadata checked on 2026-09-26; exact model and dataset revisions are in [the experiment manifest](09_bert_hash_benchmark_manifest.json).

The experiment asks whether tiny contextual BERT Hash encoders improve retrieval and social-text quality enough over static embeddings to justify their inference cost, and how much quality they retain relative to MiniLM, BGE-small, and the trained Ettin export. Compare frozen released checkpoints first. This measures useful deployed systems; differences in training data prevent attributing the results solely to architecture.

## Model matrix

| Group | Model | Output dimensions | Role |
| --- | --- | --- | --- |
| Hash embeddings | `NeuML/bert-hash-femto-embeddings` | 50 | Main tiny contextual candidate |
| Hash embeddings | `NeuML/bert-hash-pico-embeddings` | 80 | Intermediate size |
| Hash embeddings | `NeuML/bert-hash-nano-embeddings` | 128 | Largest hash candidate |
| Hash base controls | `NeuML/bert-hash-{femto,pico,nano}` | 50 / 80 / 128 | Masked mean pooling of pretrained encoders |
| Current static | `futur/Qwen3-Embedding-0.6B-model2vec-onnx` | 256 | Existing project deployment baseline |
| General static | `minishlab/potion-base-8M` | 256 | Additional Model2Vec baseline |
| Retrieval static | `sentence-transformers/static-retrieval-mrl-en-v1` | 1024 native; 128 / 256 secondary | Retrieval-trained static baseline |
| Small transformer | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Notebook 07 baseline |
| Small transformer | `BAAI/bge-small-en-v1.5` | 384 | Notebook 07 baseline |
| Small transformer | `TaylorAI/gte-tiny` | 384 | Unquantized candidate before ONNX INT8 evaluation |
| Trained Ettin | Local `social-encoder/` export | 128 / 256 / 384 | All exported exits and full depth |

The requested `NeuML/bert-hash-femto` is explicitly included. Its base-model results must be labeled separately from embedding-tuned results. The base family compresses the token embedding table; this does **not** imply binary sentence vectors or Hamming-distance search. Use ordinary floating-point embeddings and cosine similarity. [Base-model card](https://huggingface.co/NeuML/bert-hash-femto).

The embedding variants use masked mean pooling and produce 50D, 80D, and 128D vectors respectively. Use their single-vector outputs without a reranker. Some model-card tables include a second-stage ColBERT reranker; those numbers are not the protocol here. [Femto](https://huggingface.co/NeuML/bert-hash-femto-embeddings), [Pico](https://huggingface.co/NeuML/bert-hash-pico-embeddings), [Nano](https://huggingface.co/NeuML/bert-hash-nano-embeddings).

Use the current static model through the project's existing ONNX tokenization/packing path in [embed_bluesky.py](../scripts/embed_bluesky.py). It is a static export, not a 0.6B transformer. Use the published adapters for [POTION](https://huggingface.co/minishlab/potion-base-8M) and [Static Retrieval MRL](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1); preserve their token weighting and pooling. The latter supports trained dimension prefixes.

The local Ettin export currently specifies exit depths 1–9, a 10-layer backbone, and trained dimensions 128/256/384. Read these from the verified artifact at execution time. An early exit runs its prefix plus one independent transformer head; full depth runs the backbone without an exit head. Treat it as the **trained social Ettin** model. Frozen original Ettin or notebook 08 exports, if added later, require separate artifact identities and labels.

GTE-tiny was added on 2026-09-28 at revision `4cc5e73d86a67c601897257b467187234aa3bca3`. Use masked mean pooling, no query instruction, 384D output, and L2 normalization. Explicitly cast parameters to float32 for the reference run, since its configuration advertises float16. Compare it against current static, MiniLM and BGE-small with paired query differences; keep these three macro contrasts in their own interval family. [Model and pooling instructions](https://huggingface.co/TaylorAI/gte-tiny).

If that quality is suitable, evaluate the same revision's published `onnx/model_quantized.onnx` before adopting INT8. First establish parity between an ONNX float32 graph and the PyTorch reference, then measure INT8 retrieval/AP changes, CPU batch-1/batch-throughput latency, actual execution provider, and graph bytes on identical inputs. Inspect the graph's quantization operators and integer weights rather than inferring INT8 from its filename. Quantized weights do not imply quantized output vectors or reduced 384D index storage; CPU INT8 acceleration does not imply an L4 speedup. Keep the existing deployment unchanged until the quantized candidate has been measured.

## Evaluation tasks

| Task | Fixed evaluation data | Primary metric | Additional metrics |
| --- | --- | --- | --- |
| Retrieval | NanoNQ, NanoHotpotQA, NanoMSMARCO from notebook 07 | Equal-dataset mean NDCG@10 | Per-dataset NDCG@10, MRR@10, Recall@10, Hit@1 |
| Social paraphrase | Entire pinned Twitter URL test mirror | Average precision | ROC-AUC, positive prevalence |
| Emotion probe | GoEmotions simplified, official splits | Micro-F1 | Macro-F1, per-label AP, labels predicted per example |
| Topic probe | TweetTopic temporal train_2020 / validation_2020 / test_2021 | Macro-F1 | Accuracy, per-class F1 |

Retain notebook 07's pinned data revisions, corpus text construction, query IDs, document IDs, and complete candidate corpora. Save ordered text/ID/qrel hashes. Do not subsample candidates for the scored run. Check the qrel schema: reproduce notebook 07's binary relevance for the compatibility run, and preserve graded relevance in separately labeled evaluations if a new dataset supplies it. Never silently coerce grades or change title concatenation between models.

Score exact cosine search using L2-normalized float32 vectors. Use blocked matrix multiplication if needed; record deterministic document-ID tie breaking. Each model encodes its own queries and documents. For Ettin, evaluate both same-exit retrieval and early queries against full-depth documents at the same trained dimension. Never compare hash or static queries directly with an Ettin/BGE document index. Emit only one full-depth row per dimension: the two Ettin modes coincide at full depth.

For Twitter URL, save a score per pair and fit no threshold on test. Cache repeated texts once but reconstruct the original pair order and multiplicity for scoring. Keep this test evaluation-only.

For probes, reproduce notebook 07's initial caps of 4,000 train / 512 validation / 1,000 test with seed 42 and its text deduplication priority (test, then validation, then train). Save the resulting IDs and actual counts. Train the same fresh one-vs-rest logistic regression protocol on each frozen embedding: `C=10`, balanced class weights, `liblinear`, `max_iter=1000`, seed 42. Select the GoEmotions global threshold from 0.05–0.95 in 37 steps on validation only, with a fixed tie rule choosing the smallest threshold. Also report the fixed-0.5 result. Log convergence warnings and labels absent from training/evaluation.

Use actual output dimensions in every probe row. Report all trained Ettin dimensions, unlike notebook 07's full-width-only probe. The current export has already used GoEmotions and TweetTopic during encoder training; these probes measure transfer from the available checkpoints, not equal prior supervision. A new probe split does not undo earlier encoder exposure. Audit any available notebook 06 training IDs against evaluation texts, report overlap, and keep any filtered sensitivity analysis separate. General pretraining overlap remains unknown.

## Input and dimension controls

Use float32 as the common quality reference, with a 256-token cap per model tokenizer, including special tokens and any query instruction. This matches the trained Ettin export's supported evaluation cap. MiniLM uses mean pooling; BGE uses CLS pooling and notebook 07's query instruction for retrieval queries only. Hash models use masked mean pooling. Static models use their native pooling and no special-token injection unless their published adapter requires it. L2-normalize after any supported dimension slice.

A common token count does not guarantee identical visible text. Save original/retained token counts and truncation rates per model and task. Also report a sensitivity subset consisting of examples that fit all tokenizers without truncation; for retrieval, keep the complete candidate corpus and select queries only when both the query and every relevant document fit. Clearly label this as a subset, since candidate truncation still varies.

Keep all models at native dimensions in the main table. Add only supported prefixes for Static Retrieval MRL and Ettin. Do not slice BGE, MiniLM, POTION, or the current static export merely to match the tiny models. Dimension reduction with train-fitted PCA can be a later, explicitly labeled compression experiment. No padding to 384D and no arbitrary slicing to 50D/80D.

For static models whose adapter ignores `max_seq_length`, implement and verify the 256-token limit at the token-ID layer while preserving pooling semantics. Include full-text static inference only as a separately labeled deployment sensitivity run. Do not silently give static models all of a long document while truncating every transformer.

## Latency, throughput, and memory

CPU batch-1 latency is the primary deployment efficiency measure. Test one thread and `min(8, available CPUs)` threads, controlling PyTorch, ONNX Runtime, BLAS, and tokenizer concurrency. Record actual CPU model, affinity, thread settings, OS, package versions, and power settings when available. Run each model in an isolated process so resident Ettin weights cannot distort a hash model's memory result.

Use a fixed, seeded panel of 512 distinct real texts, balanced across short social posts and retrieval documents. Save its IDs and input lengths before running any model. Time query and document encoding separately where prompts differ. Preserve the same text order and batch boundaries for all models; distinguish document-length strata using a fixed reference tokenizer.

For each hardware/thread configuration, measure batch sizes 1, 8, 32, and 128 (record OOM explicitly). Warm up at least 10 batches, then run five timed passes over the panel with model order rotated between passes. Report batch-1 median/p95 latency, full-batch median/p95 latency, and total texts divided by total elapsed time. Retain every timing sample. Bootstrap latency ratios by pass or batch block rather than treating repeated encodings as independent users. For GPU runs, synchronize before and after each timed batch and record its actual execution provider.

The main timing boundary starts with raw strings and ends with normalized float32 vectors on CPU. Include tokenization, packing, host/device transfer, inference, pooling, and normalization. Exclude downloads, loading, embedding-cache reads, nearest-neighbor search, and probe fitting. Separately measure cold model load time from an already-downloaded local snapshot and exact-search time on the actual benchmark corpus. Do not reuse [benchmark_embeddings.py](../scripts/benchmark_embeddings.py) timings as directly comparable: its documented boundary excludes tokenization and normalization.

Use each model's supported backend and disclose it: current static ONNX, POTION Model2Vec, Static Retrieval Sentence Transformers, contextual models PyTorch. This is a deployment comparison, not an isolated kernel/architecture experiment. CPU is required for all models; GPU is secondary where supported. Report unsupported GPU paths explicitly rather than labeling CPU fallback as GPU performance. Keep BF16, quantization, compilation, and optimized ONNX transformer exports in separate follow-up tables with quality rechecked.

Record model artifact bytes (including tokenizer), loaded parameter counts, peak process RSS, and peak GPU allocation/reservation. For Ettin, distinguish full export size from an optionally materialized single-exit deployment and count prefix plus head parameters. For static models, count the whole stored table separately from token lookup activity; a large table does not imply transformer-like compute.

Report index-vector storage independently of model storage. Raw float32 vectors require `4 × N × dimension` bytes:

| Dimension | Bytes/vector | Raw vectors for 1 million documents, decimal GB |
| --- | --- | --- |
| 50 | 200 | 0.200 |
| 80 | 320 | 0.320 |
| 128 | 512 | 0.512 |
| 256 | 1024 | 1.024 |
| 384 | 1536 | 1.536 |
| 1024 | 4096 | 4.096 |

These are arithmetic projections, excluding IDs, metadata, ANN graph overhead, and temporary search buffers. A measured million-document search benchmark requires a real corpus and is a separate stage.

## Uncertainty and decision rules

Save per-query and per-pair predictions. For retrieval differences, use 10,000 paired bootstrap replicates with seed 42, resampling query IDs within each dataset and averaging dataset means equally. Report 95% intervals for each dataset and the macro difference. NanoBEIR's small query sets make this a screening experiment; inconclusive intervals remain inconclusive.

For Twitter, resample connected groups of repeated normalized tweet text (and shared URLs if provided), retaining paired model predictions and recomputing AP/AUC. Record component sizes and discard bootstrap draws lacking both classes. If grouping information is unavailable, label pair-level intervals as potentially optimistic. For probes, paired resampling of held-out examples estimates uncertainty conditional on the fitted classifier; it does not estimate training-set variability.

Predeclare primary comparisons: each of the three embedding-tuned hash models versus the current static export, separately for retrieval macro NDCG and Twitter AP. Use simultaneous 95% intervals via a Bonferroni-adjusted bootstrap interval across the three model contrasts within each endpoint. Report ordinary 95% intervals too, labeled exploratory. Comparisons with POTION, Static Retrieval, BGE, MiniLM, base hash controls, and the Ettin grid are descriptive unless independently confirmed.

Use the quality/latency/storage Pareto frontier to identify candidates. Proposed practical tolerances, to freeze before execution: at most 0.02 absolute loss in retrieval macro NDCG and 0.01 in Twitter AP versus the current static model; report tolerances independently for comparisons to MiniLM and BGE. A lower confidence bound above the negative tolerance supports noninferiority; a bound crossing it is inconclusive. A quality win requires a lower bound above zero. Report a speed win only when the warmed latency-ratio interval supports it. Parameter count alone is not a speed result.

Ettin's many profiles and the static dimension variants remain a full descriptive grid. Do not pick a best test-scoring exit and claim an unbiased winner. Freeze the chosen deployment profile using development data or operational budgets, then confirm it once on a larger untouched evaluation set with full candidate corpora. Additional full-size BEIR datasets and the repository's author-held-out masked-hashtag probe can provide follow-up evidence; the latter is a weak-label application check, not a replacement for semantic relevance judgments.

## Execution stages and notebook changes

1. **Preflight.** Verify the local Ettin manifest; download pinned model snapshots; inspect custom BERT Hash code and load it with the pinned revision. Smoke-test its compatibility with notebook 07's Transformers 5.1.0 and ModernBERT path. The hash checkpoint advertises Transformers 4.57.3, so do not assume compatibility. If separate environments are needed, persist vectors through the same adapter contract and record both locks. Select attention backends per architecture and record them. Preserve the user's working environment while establishing the experiment environment.
2. **Correctness smoke run.** Check output width, finite/nonzero norms, repeatability, batch/singleton agreement, padding invariance, and pooling parity with each published adapter. Retain Ettin's canonical full-forward and actual skipped-layer checks. Exercise empty/whitespace inputs and define one common data exclusion policy before scoring; report any model-specific failures instead of silently dropping rows. Use tiny fixture data solely to verify execution and output files.
3. **Frozen quality run.** Run all hash variants, all three static baselines, MiniLM, BGE, and all verified Ettin profiles on the shared tasks. Cache raw full-width features once per model/role/text set and derive supported normalized prefixes. Run fresh probes from those cached vectors.
4. **Efficiency run.** Run the isolated-process CPU suite, then supported GPU paths. Time only the selected Ettin depth/head for each profile, not a single all-exit forward attributed to every exit. Add measured storage/memory and uncertainty plots.
5. **Confirmation.** Freeze a deployment candidate before inspecting a new test set. Expand only if the Nano screen leaves a useful quality/cost tradeoff. Equal-budget domain adaptation, if desired, is a new experiment with shared training data, teacher, validation selection, and fixed evaluation sets.

The new notebook `09_bert_hash_static_comparison.ipynb` uses notebook 07 as reference and implements the L4 screening subset. It keeps Transformers 5.1.0 for Ettin and uses isolated Transformers 5.17.0/tokenizers 0.23.1 workers for the pinned hash code, which fails to import under 5.1.0. It defaults to frozen float32 inference, batch 32, a 256-token cap, all Ettin quality profiles, and optional fresh classifier probes. The selected Drive export determines Ettin's identity. The following adaptations guide both the screen and the expanded protocol:

- Replace the two-item `BASELINES` switch with adapters exposing model identity, role-specific encoding, native dimension, supported dimensions, backend, and measured resource metadata.
- Replace the probe's `normalized(raw, 384)` and `dim=384`, and plot labels hard-coded to `384D`, with each profile's declared and asserted output width. Slicing fewer columns must not silently mislabel a 50D vector as 384D.
- Extend paired comparisons beyond Ettin-versus-baseline; deduplicate full-depth Ettin retrieval rows and save Twitter/probe predictions for uncertainty estimates.
- Extend cache identity with model/code/tokenizer revisions, pooling, prompt, length policy, dimension policy, ordered text hashes, implementation hash, environment lock, and artifact identity. Never reuse notebook 07 cache files under a different protocol.
- Move timing and memory collection into one-model worker processes; turn loading, encoding, vector conversion, and search into explicit measurement boundaries.

Save under `output/bert_hash_benchmark/<configuration-hash>/`: resolved configuration and environment, input manifests, correctness checks, retrieval aggregates and per-query scores, Twitter pair scores, probe IDs/predictions, paired differences, raw latency samples, resource metrics, and plots. Main figures: retrieval quality versus CPU batch-1 latency, Twitter AP versus latency, retrieval quality versus bytes/vector, and the complete Ettin depth/dimension curves. Show CPU and GPU separately and annotate backend and training status. Write a completion marker only when every required profile/task has succeeded; retain explicit failure records otherwise.
