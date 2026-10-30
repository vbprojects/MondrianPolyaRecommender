# Notebook-first encoder experiments

[Benchmark results and model decision](BENCHMARK_RESULTS.md): consolidated quality, classification and encoding-cost results, with GTE-tiny selected for the next deployment stage.

[Quantized GTE-tiny in Julia](../experiments/gte_tiny_julia/README.md): a dedicated Julia tokenizer and INT8 inference experiment, with ONNX parity checks, CPU timings and a reduced retrieval screen. Inference does not use Python or ONNX Runtime.

Open a notebook in VS Code and connect it to a **Colab A100** runtime. Each notebook is independent and contains all of its model, data, training, checkpointing, evaluation and export code. No local package installation or repository checkout is needed in Colab.

- [01_neobert.ipynb](01_neobert.ipynb): adapt the MIT-licensed 28-layer NeoBERT into an embedding teacher using GTE supervision and retrieval triplets, then copy eight complete blocks into a student and distill from that adapted NeoBERT teacher. Width, attention heads, tokenizer and pooling are preserved. The unlicensed GooAQ checkpoint is not used.
- [02_ettin.ipynb](02_ettin.ipynb): longer training of the entire Ettin-32M backbone on Natural Questions, SQuAD, and AllNLI. Defaults continue the original pilot for research, with live GTE supervision, Matryoshka dimensions, lexical negatives, mixed input lengths, Drive recovery and final sanity checks. The completed original pilot is preserved in [archive/02_ettin_pilot.ipynb](archive/02_ettin_pilot.ipynb).

- [03_ettin_prefix_continuation.ipynb](03_ettin_prefix_continuation.ipynb): continue the successful Ettin pilot. Load its validation-best checkpoint, compare 128/256/384-dimensional retrieval, and optionally refine these prefixes with Matryoshka training. All continuation code is inline; notebook 02 does not need to be running.

- [04_ettin_exit_ablations.ipynb](04_ettin_exit_ablations.ipynb): compare frozen-backbone early exits using raw pooling, pooled MLPs, token-wise MLPs, and copied transformer blocks. Includes every intermediate layer, dimension prefixes, shared-index retrieval, latency/parameter costs, Drive resume, exports and bottom-cell sanity checks.

- [05_ettin_transformer_long.ipynb](05_ettin_transformer_long.ipynb): continue the trained transformer exit heads for a longer mixed-data run, with the Ettin backbone frozen. Includes restartable mixed sampling, validation safeguards, before/after retrieval and bottom-cell sanity checks.

- [06_ettin_social_end_to_end.ipynb](06_ettin_social_end_to_end.ipynb): the consolidated research pipeline: full-width backbone adaptation, Matryoshka prefix refinement, then independent transformer exits with the selected backbone frozen. Adds MS MARCO, GoEmotions, and TweetTopic; includes Twitter URL evaluation and an original-training-file adapter. All code is inline. The final cell calls `runtime.unassign()` after saving and verifying outputs.

Notebook 01 and the archived original Ettin pilot use pinned `Alibaba-NLP/gte-modernbert-base` as the initial embedding supervisor, pinned MS MARCO training triplets, and pinned NanoMSMARCO/NanoHotpotQA benchmarks. These are English pilot experiments, not reproductions of published model scores or replacements for multilingual/social-media evaluation.

## Run

1. Run installation in a fresh runtime. If you already imported different library versions, restart the kernel after installation.
2. Mount Drive using the notebook's `google.colab.drive.mount` cell.
3. Edit `RUN_NAME` and `CFG`. Each notebook documents its defaults and a small smoke configuration; notebook 02 now uses the longer mixed-data recipe below.
4. Run the remaining cells in order. Inspect the teacher baseline and each student's before/after retrieval results.

Persistent output is `/content/drive/MyDrive/encoder_experiments/<RUN_NAME>/`. Hugging Face downloads and upload staging use the runtime's local disk. You can upload this folder to Drive for convenience, but notebook execution does not depend on its location.

## Longer Ettin backbone training (notebook 02)

The revised `02_ettin.ipynb` defaults to `INIT_MODE='research_continuation'`, as requested: load the checksum-verified validation-best `student` checkpoint from `ettin-pilot-01`, including its training-only alignment head. Write a separate `ettin-backbone-diverse-01` Drive run. The source needs `config.json`, `checkpoints/student/`, and `pairs.json`; its original training text is excluded from new holdouts. Set `INIT_MODE='pretrained'` under another run name to initialize from pinned pretrained Ettin instead.

Defaults train all backbone layers for **up to 10,000 updates**, with a fresh AdamW/warmup/cosine schedule peaking at `1e-5` for continuation (`3e-5` for pretrained initialization). Matryoshka ranking and teacher geometry train 128/256/384D throughout. The source cycle is 40% Natural Questions, 40% SQuAD, 20% AllNLI, with up to 60k/40k/60k unique-query training rows respectively. Teacher-screened lexical negatives augment the QA pairs; AllNLI retains labeled contradictions. GTE is frozen and runs online at the current input cap, so there is no old 256-token target cache to reuse incorrectly.

The first 6,000 updates use 512-token inputs and actual contrastive batch size 64. Later QA updates alternate with batch size 32 at 1,024 tokens, sampling genuinely longer positives where available. A saved length audit reports coverage and truncation. Validation stays fixed at 256/1,024 tokens and reports every dimension/source plus long-positive subsets. Selection averages those validation MRRs and requires mean full-width MRR to stay within 0.02 of the starting model. Early stopping begins after update 6,500, with six evaluations of patience and 0.001 meaningful-improvement threshold. Step zero remains eligible.

Checkpoints every 100 updates include the complete model, optimizer, schedule, all source/length sampler states, RNG and early-stopping state. Rerunning resumes the committed state. The final cells save learning curves, before/after NanoNQ/NanoHotpotQA results, a verified inference export, `handoff.json`, and independent bottom-cell sanity checks including a long-input position diagnostic. The selected model is restored before test reporting or export.

**Licensing:** the existing pilot retains its MS MARCO history. Microsoft's [upstream terms](https://microsoft.github.io/msmarco/Notice.html) restrict that dataset to non-commercial research; new data do not erase that lineage. The new direct training mix uses SQuAD (CC BY-SA 4.0), Natural Questions (CC BY-SA 3.0), and AllNLI (SNLI CC BY-SA 4.0 plus MultiNLI's mixed OANC/CC/public-domain terms). These are not blanket Apache/MIT data licenses. The notebook contains upstream links, research references, attribution/ShareAlike notes and exclusions, and saves pinned dataset cards plus complete model-license notices with the export. It does not label continued weights as commercially cleared. NanoMSMARCO is excluded from this notebook's benchmark downloads.

The recipe adapts ideas from Qwen3 embedding training, Nomic's Matryoshka training and GTE supervision; it does not reproduce their scale or scores. Notebooks 04/05 remain unchanged and still target the earlier pilot. Their data adapters and exit heads need updating for the new backbone before reuse; `handoff.json` describes the mixed-data schema.

## Continue the trained Ettin pilot

Open `03_ettin_prefix_continuation.ipynb` on an A100 and run its cells in order. Defaults read `/content/drive/MyDrive/encoder_experiments/ettin-pilot-01` and write a separate `ettin-prefix-eval-01` folder. `SOURCE_CHOICE='best'` reads the pilot checkpoint's saved best step (400 in the completed pilot), verifies its checksum, and locks that selection for subsequent reruns. An integer selects another retained step.

The default is **evaluation only**: it compares 128/256/384-dimensional prefixes on NanoMSMARCO and NanoHotpotQA, exports the original selected model, verifies reload parity for each prefix, and ends with the independent sanity check. It needs the source `config.json` and `checkpoints/student/` plus access to the pinned tokenizer and benchmarks. No teacher model is loaded.

To refine prefixes, set `RUN_MATRYOSHKA=True` and choose a **new `RUN_NAME`** before running from the top. This additionally needs the source `pairs.json` and complete `gte_cache/`. It reuses those cached targets, averages ranking loss over the prefixes, and selects checkpoints using mean validation MRR across dimensions. The unchanged source is saved as step zero and can remain the best checkpoint. Defaults are 300 updates at learning rate `1e-5`; optimizer, sampler, schedule, and RNG state resume from Drive. Test metrics are recorded after restoring the selected checkpoint.

Every prefix uses the same backbone: slice the output first, then normalize. Smaller vectors save storage and similarity-search work; they do not reduce encoder computation. Source artifacts remain unchanged. Keep the source checkpoint available for reruns; this continuation locks its identity rather than duplicating the entire pilot.

## Compare Ettin early-exit ablations

Open `04_ettin_exit_ablations.ipynb` on an A100. It reads the best retained checkpoint from `ettin-pilot-01` and writes to a new `ettin-exit-ablation-01` Drive folder. Keep the source `config.json`, `checkpoints/student/`, and `pairs.json`; GTE targets are not required. The original trained Ettin supplies final-vector supervision, and its backbone remains frozen.

Defaults train three independent head banks for 300 updates each, with batch size 16, seed 42, every intermediate depth (1–9 for Ettin-32M), and dimensions 128/256/384. Raw normalized pooling and the original full-depth output are controls. All banks see the same batch order; both MLP variants start with identical residual initialization. The transformer variant copies the final block, so this compares practical architectures rather than controlling for parameter count and initialization. Choose a new run name when changing settings. For a short smoke run, set `STEPS=20` and `EXIT_DEPTHS=[2,5,8]`.

Each stage checkpoints only its heads, optimizer, schedule, sampler, RNG state, and validation-best step to Drive. Rerunning resumes incomplete stages. Selection averages validation MRR over depth, dimension, and same-exit/full-depth-corpus retrieval; step zero can win. Test rows and NanoBEIR report results without choosing checkpoints. Repeated seeds are supported via `SEEDS`.

Outputs under the destination include:

- `evaluations/ablation-results.csv` and JSON: NanoMSMARCO/NanoHotpotQA metrics at each exit and dimension, for both retrieval modes.
- `evaluations/cost-latency.json` and `quality-cost-latency.csv`: actual prefix execution latency, active/stored parameter counts, and a joined quality/cost table. GPU timings exclude tokenization and host/device transfer; prefix dimensions share the same encoding cost.
- Depth/quality and latency/quality PNG plots, plus the independent sanity report generated by the bottom cell.
- `exports/exit-ablation/`: one shared backbone/tokenizer and separate safetensors files for every selected head bank, with provenance and reload checks. These are notebook-loadable PyTorch weights, not ONNX.

The notebook implements frozen-head ablations; it does not jointly retrain the backbone with 2D Matryoshka loss. Every head trains across dimension prefixes. A result at one depth does not establish quality at untested depths or compatibility with an existing full-depth corpus index; the notebook measures both explicitly.

## Continue the transformer exits

Open `05_ettin_transformer_long.ipynb` on an A100 and run all cells in order. All code is inline. Defaults read `ettin-exit-ablation-01`, select the validation-best `transformer-seed-42` checkpoint, and write a new `ettin-transformer-long-01` Drive folder. It needs the ablation's `config.json`, transformer checkpoints and `exports/exit-ablation/`, plus the original pilot's `pairs.json`. Source paths are editable near the top; no GTE model or GTE target cache is needed.

The run adds **3,000 updates at batch size 32** with a fresh warmup/cosine schedule peaking at `3e-5`. It retains the existing independent transformer heads, depths, vector dimensions and input-length cap. Every update trains all exit heads; the original backbone stays frozen. The batch-source cycle is 50% existing MS MARCO, 25% SQuAD question–passage pairs, and 25% AllNLI triplets. Up to 30,000 training rows per added source are cached on Drive, with separate validation/test rows and exact-text overlap checks. This gives 96,000 row presentations at default settings, not 96,000 unique rows.

Checkpoint selection uses weighted validation MRR and rejects candidates whose MS MARCO score drops more than 0.02 below the starting model. Step zero can remain best. The selected source's MS MARCO sampler order/cursor is preserved; all three sampler states, source-cycle position, optimizer, schedule and RNG states resume from Drive. Checkpoints are committed every 100 updates and on validation improvements. Settings changes require a new run name.

The final cells compare source and selected heads on NanoMSMARCO/NanoHotpotQA at every exit/dimension in both same-exit and full-depth-corpus modes, save plots and CSV/JSON results, export a verified PyTorch bundle, and run independent semantic/plumbing checks. ONNX is deferred. For a short Colab smoke run, use the small-run settings documented in the notebook's settings section under a separate run name.

## Resume and edit

Interrupt a training cell normally to save the last completed update, then edit and rerun the appropriate definitions and training cell. After a runtime disconnect, rerun the notebook: existing target shards are reused, and each stage resumes from its latest verified checkpoint. Completed training stages do not perform additional optimizer updates.

Checkpoints contain weights, optimizer, learning-rate schedule, sampler order/cursor, random states, and experiment/data identities. Incomplete uploads are ignored; a corrupted latest generation falls back to the previous valid generation. Keep several GB free on Drive, particularly for NeoBERT's full teacher. The latest two and validation-best checkpoints are retained per stage.

Architecture, data and loss configuration changes require a new `RUN_NAME`; edits to cell implementation remain possible for debugging. Do not run two sessions writing to the same run folder concurrently.

## Interpret results

Training combines contrastive ranking, teacher alignment and pairwise similarity matching. An accumulation setting does **not** enlarge the actual in-batch negative pool. The fixed validation subset selects checkpoints; test triplets and NanoBEIR only report results. Exact text overlaps with benchmarks and between splits are filtered, but this does not establish semantic deduplication or absence of upstream pretraining contamination.

Dimension Matryoshka training is enabled throughout revised notebook 02; it remains optional in the earlier pilot/continuation workflows. It trains output prefixes, not early exits. Export/reload parity is checked at the end of each notebook. Exports omit optimizer state and the training-only alignment head.

NeoBERT exports require the class definitions in its notebook to reload. Ettin exports contain a standard Transformers backbone plus tokenizer and pooling metadata. Neither experiment relies on the previous Qwen/KDA implementation.

## Validation performed locally

Validated both notebooks' schema and Python cells with the pinned dependencies on a local CUDA GPU. Loaded the actual pinned NeoBERT and Ettin-32M checkpoints; checked padding invariance, finite gradients, real GTE target extraction, cache reuse, and one optimizer step on real triplets for each student. Export/reload parity passed, including preservation of the saved maximum input length. Tiny-model checks covered upstream NeoBERT equation parity (using a portable SwiGLU substitute), corrupted-checkpoint fallback, and an interrupted accumulation window resuming to exactly the uninterrupted run's weights.

The continuation notebook also passed local checks using a real Ettin checkpoint and existing target cache: best-step selection, source locking, prefix scoring on a small labeled fixture, export/reload parity, a training update, completed-stage resume, and restoration of step zero when validation does not improve. Its final sanity cell ran successfully.

These are implementation checks, not trained-quality results. The full A100 pilot and Google's mounted Drive filesystem have not been exercised from this workspace. Loading Ettin as an encoder reports unused MLM-head tensors; those vocabulary-prediction tensors are intentionally excluded.

The exit-ablation notebook passed a local CUDA smoke run with real Ettin weights: pooled/token/transformer training at three depths, unchanged-backbone hashing, full-forward parity, skipped-layer instrumentation, completed-stage resume, interrupted-update recovery to exactly the uninterrupted weights, both retrieval modes on a small fixture, all-depth latency measurements and joined plots/tables, export/reload checks, and all final sanity plumbing checks. This does not substitute for the full A100 retrieval experiment.

The longer transformer continuation passed a local end-to-end CUDA smoke run using real Ettin weights and small samples of all three datasets. Checks covered exact source-output parity, mixed-data updates, retained source cursor with a changed batch size, completed-run resume, data-cache reuse, unchanged-backbone hashing, before/after retrieval reporting on a small fixture, export/reload parity, and the final sanity cell. A partial-update interruption resumed to exactly the uninterrupted weights, sampler states and schedule; corrupted-checkpoint fallback and step-zero restoration also passed. Separate updates exercised all nine exits at batch size 32 and 256 tokens for both pair and triplet inputs. These checks do not establish longer-run quality or validate Colab's mounted Drive behavior.

Revised notebook 02 passed a real-model CUDA smoke run in research-continuation mode using the pinned GTE supervisor and small samples of Natural Questions, SQuAD and AllNLI. Checks covered source checkpoint loading, training-only negative mining, split/benchmark exclusion, mixed-length sampling, frozen-teacher/backbone-update behavior, cache reuse, completed-run resume, export/reload parity and the final sanity diagnostics. A backward interruption resumed to exactly the uninterrupted model weights, sampler states and schedule. Corrupted-checkpoint fallback, persisted early stopping and restoring step zero also passed. Separate full-size updates passed at batch 64 × 512 tokens and batch 32 × 1,024 tokens (about 3.9 GiB peak allocated on the local GPU). A preliminary pretrained-mode smoke run also passed. The new full training run and Colab-mounted Drive remain untested here; these checks establish implementation behavior, not improved retrieval quality.

## Consolidated social-media run (06)

Start `06_ettin_social_end_to_end.ipynb` on an A100. Defaults load the selected checkpoint from `ettin-pilot-01` and write a new `ettin-social-unified-01` directory. The inspected Drive listing did not contain a completed diverse-backbone checkpoint or export. Set `INIT_MODE='pretrained'` if no existing project checkpoint is available. No other notebook needs to execute. Defaults are 10,000 full-width backbone updates, 5,000 Matryoshka updates, and 10,000 transformer-exit updates; edit `PHASES` to change these budgets. Exit branches are initialized from the newly selected backbone, not reused from the incompatible old backbone.

The training mix contains retrieval triplets/pairs from MS MARCO, NQ, SQuAD and AllNLI, plus **auxiliary classification** on GoEmotions and TweetTopic. Same emotion/topic does not imply paraphrase. Twitter URL's readily available MTEB mirror contains only test data and is kept exclusively for evaluation. To train on Twitter URL too, set `TWITTER_TRAIN_FILE` to the authors' original training TSV (or the documented equivalent JSONL) on Drive before the first run. Its file hash is part of the run identity; URL/text-connected groups stay in one split. Without that file, the notebook explicitly reports Twitter URL as evaluation-only. The data are used under the agreed research scope; TweetTopic's `other` license and underlying platform rights remain unresolved and are recorded, not relabeled permissively.

Every stage saves complete optimizer/scheduler/sampler/RNG state with checksums and restores its validation-best model before the next stage. Test data never select checkpoints. No small long-passage pool is oversampled. Reports cover retrieval at every dimension/depth, full-depth shared-index retrieval, social classification, Twitter URL paraphrase ranking, padding/repeatability sanity, and export reload parity. One inference export contains the backbone once plus the exit heads, tokenizer, metadata and notices. Training-only auxiliary heads stay in resumable checkpoints.

**The final cell disconnects and releases the Colab runtime with `runtime.unassign()`.** It runs only if execution reaches the end after exports and checks. Earlier failures remain visible for debugging; this is not a shutdown-on-error handler. Check `RUN_COMPLETE.json` in Drive when returning to the experiment.

Local validation of notebook 06: notebook schema/cell syntax; nine GPU updates in each stage on a small real-data mix covering all six automatic training sources; completed-stage resume without extra updates; export/reload parity; numerical/padding and actual prefix-execution checks; optional Twitter TSV grouping and paraphrase-loss gradients. The smoke test used two exit depths and short sequences on an RTX 4070 SUPER, not a full A100 training run.

## Local baseline comparison (07)

[07_local_baseline_comparison.ipynb](07_local_baseline_comparison.ipynb) runs in a local Python kernel on CUDA or CPU. It compares pinned `all-MiniLM-L6-v2` and `bge-small-en-v1.5` against every exported Ettin exit and its trained 128/256/384D prefixes. Baselines use their native 384D outputs; BGE receives its recommended instruction for retrieval queries only.

Manually download notebook 06's run folder: keep `config.json`, `checkpoints/prefix/` and `checkpoints/exits/` with all retained generations (`state.pt` and `COMMITTED.json`). Set notebook 07's `ARTIFACT_DIR` to that local run folder; Set `ARTIFACT_MODE='checkpoints'` for this layout. The current default is the manually copied repository-root `social-encoder/` export. It selects the best saved exit heads and their matching prefix backbone, verifying checksums and the backbone tensor hash. A completed `exports/social-encoder` remains supported with `ARTIFACT_MODE='export'`. No Google authentication or Drive download code is included.

The notebook evaluates matched NanoNQ/NanoHotpotQA/NanoMSMARCO retrieval, same-exit and full-depth-index modes, and Twitter URL paraphrase ranking. It saves per-query metrics, paired bootstrap differences, plots, independent prefix-only encoding latency, and parameter counts. Optional fresh frozen-embedding classifiers (`RUN_PROBES=True` by default) compare GoEmotions and TweetTopic at full 384D across all exits and baselines; emotion thresholds are tuned only on validation. These probe F1 scores use a different protocol from notebook 06's auxiliary heads.

Embedding caches and reports go under `~/encoder-benchmarks/results/<configuration-hash>/`. All inference code is inline, with no project-package dependency or Colab shutdown cell. Use an existing working PyTorch installation; CPU evaluation will take longer.

Notebook 07 validation: syntax/schema checks and a reduced real-model GPU run covering both baselines, an exported Ettin smoke model, all evaluation/report paths, classification probes and prefix-only timing. The full user-trained artifacts have not been downloaded; provide them manually for the full comparison.

Notebook 06 final shutdown: retries artifact checksum verification for up to 20 minutes, requires selected and final checkpoints for all three stages, then calls Colab `drive.flush_and_unmount` with a 20-minute timeout before releasing the runtime. Verification or flush failure leaves the runtime assigned for inspection. Mounted-file checks alone are not remote upload verification; successful DriveFS flush is the synchronization gate.

### 08 — full-width Ettin exit comparison

[08_ettin_fullwidth_exits.ipynb](08_ettin_fullwidth_exits.ipynb) compares frozen original Ettin-32M with notebook 06's selected pre-prefix backbone. Both variants train fresh independent transformer exits at depths 2/4/6/8 for 10,000 updates using only 384D vectors; no dimension-prefix loss or old exit weights are used. The same verified notebook 06 mixture/splits supply retrieval pairs and social texts. Social texts use teacher alignment without auxiliary classification heads. Original pretrained Ettin is an MLM model, so its full-depth retrieval baseline must be measured rather than assumed strong.

Use an A100 Colab runtime. Set `SOURCE_RUN` to the existing notebook 06 run containing `config.json`, `checkpoints/backbone/`, and `data/mixture/`. Edit the inline definition cells, then run the final orchestration cell to execute both variants sequentially. Each variant has isolated resumable checkpoints, a verified frozen-backbone snapshot, held-out cosine/MRR, NanoBEIR same-exit/shared-index results, a notebook-07-compatible export, and final semantic/plumbing sanity checks. `comparison.csv` and `comparison.json` summarize both variants. Runtime release is optional and requires a successful Drive flush.

### 09 — BERT Hash and static baselines on Colab L4

[09_bert_hash_static_comparison.ipynb](09_bert_hash_static_comparison.ipynb) is a self-contained frozen-encoder benchmark targeting an **L4**, with no training or A100 requirement. It compares NeuML Femto/Pico/Nano base and embedding-tuned checkpoints, the current Qwen Model2Vec ONNX export, POTION-8M, Static Retrieval MRL, MiniLM, BGE-small, GTE-tiny, and the trained Ettin export. Exact public-model/data revisions and all worker code are embedded; no repository clone or companion manifest is required.

Select an L4 runtime and set `ETTIN_EXPORT` to a completed notebook 06/08 `exports/social-encoder` directory on Drive, or an uploaded extracted `social-encoder/`. The manifest and all source files are verified before copying to the local SSD. Default settings use float32, batch 32, 256 tokens, one model process at a time, disk-backed vectors, and blocked exact search. Set `SMOKE=True` for an installation/path check, then `False` for scored results. Different settings create separate configuration directories.

The full default measures matched NanoNQ/NanoHotpotQA/NanoMSMARCO retrieval, Twitter URL AP/AUC, paired retrieval intervals, CPU latency, and supported GPU latency. All Ettin exits/dimensions receive quality evaluation; timing defaults to depths 1/3/5/9 plus full depth. `RUN_PROBES=True` adds fresh GoEmotions/TweetTopic classifiers; `TWITTER_GROUP_BOOTSTRAP=True` adds connected-text AP intervals. Static quality inference is explicitly CPU-based; timing tables record backend and device. POTION uses its published Sentence Transformers static adapter. The broader [experiment design](09_bert_hash_benchmark_design.md) describes additional confirmation and sensitivity experiments beyond this compact screen.

The pinned hash code needs Transformers 5.17.0/tokenizers 0.23.1, installed in a separate directory for hash workers. Ettin and other workers retain Transformers 5.1.0. Completed model/task bundles, logs, predictions and reports are checksummed under `MyDrive/encoder_experiments/<RUN_NAME>/<configuration-hash>/`; re-running unchanged settings skips verified tasks. Optional runtime release occurs only after verification and a Drive flush.

For a focused GTE-tiny screen, set `MODEL_IDS` to `['TaylorAI/gte-tiny', 'futur/Qwen3-Embedding-0.6B-model2vec-onnx', 'sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']`; that selection needs no Ettin export. GTE-tiny is pinned to `4cc5e73d86a67c601897257b467187234aa3bca3`, uses masked mean pooling and native 384D vectors, and is explicitly evaluated with float32 parameters. The report includes paired retrieval differences against all three selected baselines and optional grouped Twitter AP differences against current static. Evaluate the published ONNX INT8 artifact separately after establishing suitable unquantized quality; quantized weights do not reduce the default float32 index-vector size.

The [completed local GTE-tiny screen](../docs/GTE_Tiny_Benchmark.md) measured all 150 Nano queries with complete corpora and all 51,534 Twitter pairs for these four models. GTE-tiny reached mean NDCG@10 0.6026 and Twitter AP 0.8484; its retrieval difference versus MiniLM was inconclusive, while BGE-small scored clearly higher on retrieval. CPU/GPU timings, paired intervals and artifact links are included. The published INT8 graph was inspected but has not yet been evaluated for inference quality or speed. The Colab notebook also prints progress during long encoding tasks.

Local validation: notebook schema/cell/embedded-worker syntax; real-model adapter checks for all twelve model entries; all exported Ettin depths and dimensions; reduced real-data retrieval, Twitter, both probes, CPU/GPU timing, resume, report plots, paired intervals, corruption detection and retrieval tie handling. A batch-32/256-token check used about 268 MiB of allocated GPU tensors for BGE-small and 309 MiB for Ettin with all exits on the local GPU. These are allocated-tensor measurements, not total device residency. Colab Drive behavior, L4 performance and full-dataset quality have not been measured here.
