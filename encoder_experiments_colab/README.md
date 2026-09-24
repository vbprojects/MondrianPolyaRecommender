# Notebook-first encoder experiments

Open a notebook in VS Code and connect it to a **Colab A100** runtime. Each notebook is independent and contains all of its model, data, training, checkpointing, evaluation and export code. No local package installation or repository checkout is needed in Colab.

- [01_neobert.ipynb](01_neobert.ipynb): adapt the MIT-licensed 28-layer NeoBERT into an embedding teacher using GTE supervision and retrieval triplets, then copy eight complete blocks into a student and distill from that adapted NeoBERT teacher. Width, attention heads, tokenizer and pooling are preserved. The unlicensed GooAQ checkpoint is not used.
- [02_ettin.ipynb](02_ettin.ipynb): train the pretrained Ettin-32M encoder for embeddings, with selectable Ettin-68M. The pretrained backbone is retained intact. GTE supervises embedding alignment and pairwise relationships; retrieval ranking trains the native student vectors. The alignment head is training-only.

- [03_ettin_prefix_continuation.ipynb](03_ettin_prefix_continuation.ipynb): continue the successful Ettin pilot. Load its validation-best checkpoint, compare 128/256/384-dimensional retrieval, and optionally refine these prefixes with Matryoshka training. All continuation code is inline; notebook 02 does not need to be running.

The two initial experiments use pinned `Alibaba-NLP/gte-modernbert-base` as the initial embedding supervisor, pinned MS MARCO training triplets, and pinned NanoMSMARCO/NanoHotpotQA benchmarks. These are English pilot experiments, not reproductions of published model scores or replacements for multilingual/social-media evaluation.

## Run

1. Run installation in a fresh runtime. If you already imported different library versions, restart the kernel after installation.
2. Mount Drive using the notebook's `google.colab.drive.mount` cell.
3. Edit `RUN_NAME` and `CFG`. Defaults use 20,000 candidate training rows and 256-token inputs. For an initial plumbing check, use 2,000 rows and approximately 20 steps per stage under a separate run name.
4. Run the remaining cells in order. Inspect the teacher baseline and each student's before/after retrieval results.

Persistent output is `/content/drive/MyDrive/encoder_experiments/<RUN_NAME>/`. Hugging Face downloads and upload staging use the runtime's local disk. You can upload this folder to Drive for convenience, but notebook execution does not depend on its location.

## Continue the trained Ettin pilot

Open `03_ettin_prefix_continuation.ipynb` on an A100 and run its cells in order. Defaults read `/content/drive/MyDrive/encoder_experiments/ettin-pilot-01` and write a separate `ettin-prefix-eval-01` folder. `SOURCE_CHOICE='best'` reads the pilot checkpoint's saved best step (400 in the completed pilot), verifies its checksum, and locks that selection for subsequent reruns. An integer selects another retained step.

The default is **evaluation only**: it compares 128/256/384-dimensional prefixes on NanoMSMARCO and NanoHotpotQA, exports the original selected model, verifies reload parity for each prefix, and ends with the independent sanity check. It needs the source `config.json` and `checkpoints/student/` plus access to the pinned tokenizer and benchmarks. No teacher model is loaded.

To refine prefixes, set `RUN_MATRYOSHKA=True` and choose a **new `RUN_NAME`** before running from the top. This additionally needs the source `pairs.json` and complete `gte_cache/`. It reuses those cached targets, averages ranking loss over the prefixes, and selects checkpoints using mean validation MRR across dimensions. The unchanged source is saved as step zero and can remain the best checkpoint. Defaults are 300 updates at learning rate `1e-5`; optimizer, sampler, schedule, and RNG state resume from Drive. Test metrics are recorded after restoring the selected checkpoint.

Every prefix uses the same backbone: slice the output first, then normalize. Smaller vectors save storage and similarity-search work; they do not reduce encoder computation. Source artifacts remain unchanged. Keep the source checkpoint available for reruns; this continuation locks its identity rather than duplicating the entire pilot.

## Resume and edit

Interrupt a training cell normally to save the last completed update, then edit and rerun the appropriate definitions and training cell. After a runtime disconnect, rerun the notebook: existing target shards are reused, and each stage resumes from its latest verified checkpoint. Completed training stages do not perform additional optimizer updates.

Checkpoints contain weights, optimizer, learning-rate schedule, sampler order/cursor, random states, and experiment/data identities. Incomplete uploads are ignored; a corrupted latest generation falls back to the previous valid generation. Keep several GB free on Drive, particularly for NeoBERT's full teacher. The latest two and validation-best checkpoints are retained per stage.

Architecture, data and loss configuration changes require a new `RUN_NAME`; edits to cell implementation remain possible for debugging. Do not run two sessions writing to the same run folder concurrently.

## Interpret results

Training combines contrastive ranking, teacher alignment and pairwise similarity matching. An accumulation setting does **not** enlarge the actual in-batch negative pool. The fixed validation subset selects checkpoints; test triplets and NanoBEIR only report results. Exact text overlaps with benchmarks and between splits are filtered, but this does not establish semantic deduplication or absence of upstream pretraining contamination.

Dimension Matryoshka training is optional and disabled by default. Enable it before starting a new run, after you have a useful full-vector baseline from an earlier run. It trains output prefixes, not early exits. Export/reload parity is checked at the end of each notebook. Exports omit optimizer state and the training-only alignment head.

NeoBERT exports require the class definitions in its notebook to reload. Ettin exports contain a standard Transformers backbone plus tokenizer and pooling metadata. Neither experiment relies on the previous Qwen/KDA implementation.

## Validation performed locally

Validated both notebooks' schema and Python cells with the pinned dependencies on a local CUDA GPU. Loaded the actual pinned NeoBERT and Ettin-32M checkpoints; checked padding invariance, finite gradients, real GTE target extraction, cache reuse, and one optimizer step on real triplets for each student. Export/reload parity passed, including preservation of the saved maximum input length. Tiny-model checks covered upstream NeoBERT equation parity (using a portable SwiGLU substitute), corrupted-checkpoint fallback, and an interrupted accumulation window resuming to exactly the uninterrupted run's weights.

The continuation notebook also passed local checks using a real Ettin checkpoint and existing target cache: best-step selection, source locking, prefix scoring on a small labeled fixture, export/reload parity, a training update, completed-stage resume, and restoration of step zero when validation does not improve. Its final sanity cell ran successfully.

These are implementation checks, not trained-quality results. The full A100 pilot and Google's mounted Drive filesystem have not been exercised from this workspace. Loading Ettin as an encoder reports unused MLM-head tensors; those vocabulary-prediction tensors are intentionally excluded.
