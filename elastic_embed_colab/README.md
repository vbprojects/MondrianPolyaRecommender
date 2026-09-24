# Qwen → KDA: first-pass Colab runners

This is the experimental implementation of [the implementation plan](plans/qwen_kda_matryoshka_implementation_plan.md). Model/training logic lives in `src/elastic_embed`; notebooks only install dependencies, configure paths, and invoke CLI commands. It is ready for Colab debugging, not a trained model or a completed release gate.

Start with [01_teacher_cache.ipynb](notebooks/01_teacher_cache.ipynb), then [02_initialization_recovery.ipynb](notebooks/02_initialization_recovery.ipynb), [03_train.ipynb](notebooks/03_train.ipynb), and [04_export_eval.ipynb](notebooks/04_export_eval.ipynb). Upload the entire **`elastic_embed_colab` folder** to Google Drive → **My Drive**, then open the first notebook with Colab. The default mounted location is `/content/drive/MyDrive/elastic_embed_colab`. If you upload it elsewhere, change `BUNDLE` in each notebook’s setup cell. All package code, configs, notebooks, tests, this guide, and the implementation plan are included in this folder.

Select a GPU runtime first. The default is BF16 teacher/student training on a supporting GPU; an A100 is the intended first debugging target. The teacher is exactly `Qwen/Qwen3-Embedding-8B`, with no automatic quantization/substitution. Its two-byte weights alone need about 16 GB. A 16 GB T4 is not the default execution target. `teacher.max_memory` and `device_map` allow explicit CPU offload, subject to host RAM. Student-only stages use the cached targets and release the teacher process.

Run outputs go to Google Drive; downloaded HF model shards default to local Colab disk. A detached process is not guaranteed to survive Colab disconnection. Rerun cache/train commands with `--resume` after reconnecting. Writes use a temporary file plus atomic replacement, and cache manifests mark completed shards. This assumes the backing filesystem honors rename semantics; Drive synchronization can still lag, so verify saved artifacts before ending the session.

## First run

From the bundle root (`elastic_embed_colab/`; the notebooks switch here automatically):

```bash
bash scripts/bootstrap_colab.sh
python -m elastic_embed.cli runtime-check --config configs/smoke.yaml
python -m elastic_embed.cli prepare-data --config configs/smoke.yaml
python -m elastic_embed.cli cache-teacher --config configs/smoke.yaml --resume
python -m elastic_embed.cli initialize --config configs/smoke.yaml
python -m elastic_embed.cli cache-mohawk --config configs/smoke.yaml --resume
python -m elastic_embed.cli train --stage mohawk-matrix --config configs/smoke.yaml --resume
python -m elastic_embed.cli train --stage mohawk-block --config configs/smoke.yaml --resume
python -m elastic_embed.cli cache-recovery --config configs/smoke.yaml --resume
python -m elastic_embed.cli recover --config configs/smoke.yaml --resume
python -m elastic_embed.cli train --stage full-depth --config configs/smoke.yaml --resume
python -m elastic_embed.cli train --stage prefix --config configs/smoke.yaml --resume
python -m elastic_embed.cli train --stage matryoshka --config configs/smoke.yaml --resume
python -m elastic_embed.cli evaluate --config configs/smoke.yaml
python -m elastic_embed.cli export --config configs/smoke.yaml
python -m elastic_embed.cli benchmark --config configs/smoke.yaml
```

The notebooks create `runs/smoke-mohawk-config.yaml` on Drive once and reuse it across all four notebooks, with absolute source/output paths derived from `BUNDLE`. Edit that saved file to change settings after its first creation. Template configs use paths relative to the **bundle root**. Outputs stay under `runs/smoke-mohawk/` (or `runs/pilot-mohawk/` and `runs/main-mohawk/` for the larger templates). When continuing a notebook-created run from the CLI, use `--config runs/smoke-mohawk-config.yaml` instead of the template to preserve the exact config identity. Smoke/pilot/main cap preparation at 20k/200k/1M examples respectively. No job launches without an explicit `--config`.

For a cheap wiring check, edit the config before starting: use 1,800 examples, minimum 20/language, 5 optimizer steps per stage, effective batch 8, recovery examples 64, recovery steps 5, and freeze steps 1. `runtime-check --quick` reduces the shape matrix and records that limitation. For a random-init control, set `mohawk.enabled: false` and `initialization.mode: random`; skip `cache-mohawk`, both MOHAWK stages, `cache-recovery`, and `recover`. For a small MOHAWK wiring check, use 32 calibration training documents, 18 validation documents, matrix steps 3, block steps 5, block freeze steps 2, and MOHAWK effective batch 8. Keep teacher identity unchanged. A reference-only local debug run can set `model.backend: reference`, `training.device: cpu`, and `training.precision: float32`; it will be very slow for real training.

Microbatch defaults to 2; reduce it explicitly if Colab runs out of memory. Automatic microbatch sizing is not implemented yet. Exact resume requires the same config/cache identity; to change microbatch or schedules after a checkpoint, start a new run or deliberately implement a checkpoint migration. Steps are optimizer updates, so 300 steps at effective batch 256 processes 76,800 examples, with repetition when the corpus is smaller. Checkpoints are saved every 25 updates and at stage completion. Work since the last saved optimizer step is replayed after an interruption. No FP16 loss-scaler path is claimed; BF16 and FP32 do not need one.

### Debugging a failed runtime check

The notebook runner uses the notebook kernel’s Python interpreter, streams stdout/stderr into the cell, and saves each command’s output under `runs/logs/`. A subprocess exit status alone does not identify the failure. Check the printed traceback and the run directory’s `runtime_report.json`: it includes separate `reference_error`, `fla_error`, and `onnx_error` fields when those stages fail. Gradient checks match parameters by name and retain per-tensor errors in `gradient_checks`, including all failures rather than stopping at the first mismatch.

The BF16 gradient gate requires both relative L2 error and maximum absolute error divided by the reference tensor’s maximum absolute value to be at most **3%**, with denominator floors of `1e-5`. Nonfinite gradients always fail. This avoids treating cancellation around individual near-zero entries as a whole-tensor failure, while checking aggregate drift and individual spikes. The old elementwise mismatch counts remain in the report for inspection; the FP32 policy and forward tolerances are unchanged. These are engineering parity thresholds, not evidence of downstream model quality.

Notebooks 01, 02, and 03 include an **Update the Drive copy from this notebook** cell. When your local notebook is connected to a Colab kernel, running that cell writes the embedded initialization and runtime diagnostic patch to the mounted Drive bundle, with backups of replaced files. Then rerun `run('runtime-check')`. This works without re-uploading the entire folder. It leaves configs, training data, caches, and checkpoints intact. To refresh the embedded patch after further local code edits, run `python scripts/build_drive_patch.py` from the bundle root.

The runtime report records native BF16 support separately from support including emulation. A true emulation-inclusive result on a T4 does not establish native BF16 acceleration; the A100 remains the intended default training target. Passing forward checks alone does not pass the training gate: gradient checks must also succeed.

If an older runner fails inside `model.py` → `scan` with `failed to open libnvrtc-builtins.so.13.0`, apply the notebook’s Drive-update cell and rerun the check. The reference recurrence and MOHAWK matrix probes now use ordinary eager PyTorch operations, avoiding TorchScript’s optional CUDA fusion compiler. Their math, gradients, and device placement are unchanged; optimized FLA training still uses FLA. The same recurrence is scripted only during ONNX export to retain a dynamic `Loop`. This is a targeted workaround for that compilation path, not a repair of the runtime’s CUDA installation. NVRTC package versions are included in the runtime report for further diagnosis if another operation encounters the library error.

If initialization fails with `KeyError: 'model.embed_tokens.weight'`, apply the Drive-update cell in notebook 02 and rerun `run('initialize')`. The pinned Qwen embedding checkpoint uses bare backbone names (`embed_tokens.weight`, `layers.*.mlp.*_proj.weight`). The loader now resolves both bare and `model.`-prefixed names and records the exact keys in `initialization_report.json`. Existing sentence caches do not need rebuilding.

## Data inputs and provenance

The starter [source manifest](configs/sources.yaml) covers the plan's 18 languages. The starter dataset revisions are pinned. Dataset/config existence is checked; if you supply `main` for a custom source, preparation resolves it to an immutable SHA and saves the lock in `corpus_manifest.json`. The equal `available_count` values are **explicit placeholder estimates**, making initial language weights equal. Replace them with your source counts to use meaningful square-root sampling. Preparation enforces the English cap and per-language minimum, and fails on missing/insufficient sources instead of silently dropping languages.

Add permitted informal/conversational/parallel text as local JSONL sources. For example, a sources YAML entry:

```yaml
- path: /content/drive/MyDrive/corpus/spanish-posts.jsonl
  language: es
  script: Latn
  available_count: 10000
  license: YOUR_ACTUAL_LICENSE
  terms_reference: YOUR_ACTUAL_TERMS_URL
```

Each JSONL line needs `text`; optionally include `id`, `source_document_id`, `conversation_id`, `duplicate_group_id`, and `parallel_group_id`. Group IDs should be stable; duplicate/parallel IDs are treated as globally namespaced IDs. Provide source terms yourself; public accessibility is not a licensing determination. Sources with the same language share its quota in proportion to declared availability. The default web sources alone do **not** provide the curated short, code-switching, negation, or parallel evaluation sets requested in the plan.

Processing strips outer whitespace and selects a sentence-ending prefix within the 256/512 token budget where possible, otherwise a decoded token prefix. It re-tokenizes the exact visible text, stores those token IDs, and gives **identical IDs** to teacher and student. Roughly 15% of documents are assigned the 512 profile; actual 257–512 counts are recorded. Inspect these counts: short sources might not exercise that profile. Unicode, case, punctuation, and emojis are retained. Exact text deduplication and transitive document/conversation/duplicate/parallel groups precede splitting. Optional benchmark JSONL exclusion matches exact canonical spans only; no near-duplicate detector is implemented.

## Model, training, and artifacts

The default student has **61,587,912 parameters**: 12 distinct causal KDA blocks, width 384, six 64-dimensional heads, factorized rank-256 input embeddings, 1024-channel SwiGLU FFNs, and a shared readout. Runtime checks enforce the 80M ceiling. Inference stops at the requested block. Full multi-exit training collects all exits from one forward pass.

FLA is installed at commit `954438d1fcb5e1bb05c22f9908de9c5c2df74ae5`; Qwen/tokenizer revision is `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. Torch/CUDA remain the Colab installation subject to minimum dependency requirements; the runtime report records exact versions and VCS installation metadata. `bootstrap_colab.sh` writes a full pip freeze to `/tmp/elastic-embed-colab-pip-freeze.txt`; save it with a debugging report. `runtime-check` refuses an unverified FLA install commit.

The reference backend implements this specific pinned KDA configuration, including causal short convolutions, per-channel decay, update gates, and sigmoid-gated RMS readout. Recurrence/gate calculations use FP32. It is a correctness/debug backend, not a fast training implementation. The full preflight tests batches 1/4/16, lengths 1/7/32/63/64/65/128/256/512, variable padding, repeated calls, extreme gates, and FLA gradients. FLA failures are recorded and block teacher caching for the FLA configuration. Missing ONNX support is recorded separately; a working training backend can still run research training.

Useful outputs, relative to the selected run directory:

| Artifact | Contents |
| --- | --- |
| `runtime_report.json`, `one_block.onnx` | Versions, parameter count, backend checks, errors, CPU execution trace |
| `corpus.jsonl`, `corpus_manifest.json` | Exact spans/IDs, group splits, pinned sources, language and truncation counts |
| `teacher_cache/` | FP16 normalized 384-prefix shards, checksums, IDs, small FP32 full-width audit shards |
| `bridge.pt`, `initialization_report.json` | Internal SVD basis and explicit tensor/channel transfer record |
| `mohawk_cache/` | Contiguous projected teacher inputs, attention/block/norm outputs, short head-mean attention matrices, exact token views |
| `mohawk-matrix.pt`, `mohawk-block.pt`, `mohawk-*_report.json` | Resumable alignment stages and held-out per-layer/per-language loss histories |
| `recovery_cache/` | Up to 16 valid token positions, 12 projected post-block states per document |
| `initialized.pt`, `recovery.pt`, `full-depth.pt`, `prefix.pt`, `matryoshka.pt` | Model and staged resume checkpoints |
| `training.jsonl` | Losses, per-grid last-microbatch metrics, aggregate gradient norm, examples/tokens/memory |
| `evaluation.json` | Full grid, per-language alignment, neighborhood/parallel metrics, teacher/cache audits |
| `onnx/`, `benchmark.json` | CPU-checked optional depth profiles, wrapper metadata, end-to-end timings |

SVD uses an uncentered randomized fit on a deterministic uniform sample of embedding rows, then a rank-256 factorization over all projected rows. The bridge never changes teacher output coordinates. FFN transfer selects one shared channel set using joint gate/up/down weight RMS saliency; activation-based saliency is not implemented. KDA, norms, and readout start from valid fresh defaults. The MOHAWK stages then train KDA and block norms against Qwen behavior; no arbitrary projection of norm gain vectors or RoPE-dependent Q/K weights is used. `random`, `table`, and `table-ffn` are explicit modes; use identical corpus/cache and optimizer budgets in separate run directories when comparing them. Main-scale corpus metadata/token IDs are currently loaded into host memory: smoke/pilot are the intended debugging scales before optimizing dataset streaming.

Recovery captures raw post-block residuals through hooks, including the final block **before** teacher final RMSNorm. Only sampled positions are projected/cached. Recovery first freezes transferred tensors and then unfreezes at the configured step; that decision is a fixed pilot setting, not automatic validation-based selection. Its internal loss decays away before output training. The full-depth stage aligns the native 384 prefix, the prefix stage trains all five dimensions at depth 12, and the Matryoshka stage trains all 25 pairs with a shallow-exit ramp. Nearby examples are selected against a bounded random candidate pool of teacher vectors; relational geometry uses the actual microbatch. The baseline does not add contrastive/GOR losses.

## MOHAWK adaptation and Qwen knowledge retention

MOHAWK is enabled in the supplied configs and implemented in `src/elastic_embed/mohawk.py`. The original method aligns mixing matrices, independently aligns mixing-block outputs, and then performs end-to-end distillation. Our width/depth reduction requires the following explicit adaptations; this is not a reproduction of Phi-Mamba. [MOHAWK methods](https://arxiv.org/html/2408.10189v1#S4)

The default sequence is:

```text
Qwen embedding/FFN initialization
  → cache-mohawk
  → train --stage mohawk-matrix
  → train --stage mohawk-block
  → cache-recovery → recover
  → full-depth → prefix → matryoshka
```

1. **Keep compatible learned structure.** Qwen's token table and selected SwiGLU channels are projected/factorized into the student. The one-based `mohawk.teacher_layers` mapping defaults to `[3,6,...,36]` and is shared by FFN transfer, MOHAWK calibration, and whole-student recovery. KDA-specific parameters start from upstream defaults, then learn Qwen attention behavior. Qwen RoPE operations and incompatible Q/K weight tensors are not copied into KDA.
2. **Orient token mixing.** For each mapped block, Qwen runs with its real Q/K norms, RoPE, and causal softmax. We cache its head-mean attention matrix. Identity value probes materialize KDA's exact latent recurrent mixing coefficients for a fixed sequence of queries, keys, decay gates, and update gates. The loss compares head means after Frobenius normalization. This removes overall scale (KDA's normalized Q/K constrain its coefficient magnitude) and avoids inventing a correspondence between 32 Qwen heads and six student heads. It is a coarse orientation objective, not per-head equality. Only Q/K projections, Q/K convolutions, and decay/update parameters train here; value/output parameters are fitted next. Materialization uses FP32 reference math on short prefixes, with gradients checked against the reference KDA forward.
3. **Fit attention and complete blocks independently.** Each student block consumes the **projected pre-block residual from its mapped teacher block**, not the previous student block. We minimize relative squared error for the attention branch output (after Qwen's output projection, before residual addition), whole-block output, and projected normalization outputs. Norms learn their behavior from activations, not arbitrary projected gain vectors. Token embeddings and the shared readout stay frozen; transferred FFNs stay frozen for the configured initial steps and then may adapt at the lower learning rate. Blocks are processed and backpropagated one at a time to bound activation memory.
4. **Recover the composed student.** Independent block fitting does not account for the two omitted Qwen blocks between mappings. Whole-student recovery subsequently matches the mapped teacher residuals on real token inputs, with a Qwen sentence-embedding anchor. Final output stages preserve Qwen's native prefix coordinates. Internal losses do not remain active in the final Matryoshka stage.

The calibration cache is separate from the sentence and sparse recovery caches. It stores five contiguous projected state streams (`inputs`, attention outputs, block outputs, and two normalized outputs), short head-mean attention matrices, token IDs, and lengths. Every stream uses the same frozen bridge. Teacher and student see the same contiguous token prefix starting at position zero; shorter documents are right padded only when batching. The cache's limited prefix is **not** paired with a whole-document sentence target. The later sentence/recovery stages still use the corpus's 256/512 token profiles.

Default smoke calibration selects up to 256 train and 32 validation records with deterministic language round-robin coverage. Validation comes from the existing group-safe validation split and never participates in gradient updates. Its default arrays need about **1.7 GB**; pilot calibration needs about **6.5 GB**. `max_cache_gb` checks the estimated array storage before loading Qwen; these figures exclude teacher downloads, checkpoints and filesystem overhead. `max_length: 128` controls block calibration and `matrix_length: 64` limits quadratic matrix materialization. Increase calibration lengths deliberately when investigating longer-context transfer. Eager attention is used during this additional teacher pass; optimized FLA is used for block fitting when configured.

Reports include alignment loss before training and at saved checkpoints, by layer and language, plus a macro-language mean. They measure internal approximation, not downstream generalization. Compare MOHAWK on/off runs with matching corpus, semantic targets, initialization, and a documented total optimization budget. Evaluate held-out languages, informal/negation examples, and retrieval tasks before treating the warmup as an improvement. The deployed student remains the same **12-block causal KDA encoder**, with no added inference parameters; it is not a bidirectional BERT model.

### Existing Colab runs

Relocating this code bundle does not migrate old Drive checkpoints. Exact resume hashes include config paths: keep the original saved config and its referenced locations if continuing an existing run, or start a fresh run under this bundle’s `runs/` directory. Do not rewrite paths inside an existing run config and expect exact resume to succeed.

The notebooks now use `smoke-mohawk-config.yaml` and the `smoke-mohawk` output directory; pilot/main defaults also use new `*-mohawk` directories. Existing configs without a `mohawk` section retain the previous behavior. Do not silently add MOHAWK to an existing optimizer checkpoint: the stages enforce the new bridge, layer mapping, cache identity, and predecessor lineage. Re-run initialization in the new directory. Existing corpus files, corpus manifest, runtime report, and **completed sentence cache** can be copied into the new directory if their tokenizer/teacher/cache settings are unchanged, avoiding a repeat of the expensive sentence-teacher pass. Keep the old run intact for comparison. Older initialization files lack the provenance fields needed by MOHAWK and must be regenerated.

## Export and remaining research work

The public wrapper is:

```python
from elastic_embed.deployment import Encoder
encoder = Encoder('/content/drive/MyDrive/elastic_embed_colab/runs/smoke-mohawk/onnx')
vectors = encoder.encode(texts, depth=3, truncate_dim=128, normalize=True)
representation_id = encoder.representation_id(depth=3, truncate_dim=128)
```

It validates profiles, applies the same span preprocessing, and normalizes after slicing. Empty/all-padding input is rejected. Each depth export has dynamic batch/sequence axes and a standard ONNX `Loop` recurrence at opset 17, so no `LinearAttention` provider capability is assumed. Each graph emits one raw 384 readout; dimensions are handled in the wrapper. Initializers are currently duplicated between graphs and their real byte sizes are reported. FP32 parity is checked on export, including prefix normalization; no quantization or browser/ARM support is claimed.

Evaluation currently measures held-out alignment distributions, per-language/macro scores, neighbor recall, early-exit displacement, parallel matching when pairs exist, and teacher-prefix/FP16 cache distortion. It does not run external MTEB/STS tasks, define pilot quality thresholds, measure Mondrian cell crossings, establish quality/latency utility of shallow exits, or automate the ablation matrix. Microbatch search, dataset near-deduplication, automatic curated-source acquisition, shared ONNX external initializers, quantization, ARM/WASM/WebGPU execution, and end-to-end 8B/FLA training validation remain Colab/research follow-up work. Do not treat the generated model card status as a release decision.

For local correctness checks without model downloads:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest tests -q
```

Upstream references: [Qwen embedding model](https://huggingface.co/Qwen/Qwen3-Embedding-8B), [pinned FLA KDA](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/layers/kda.py), [FineWeb2 source/terms](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2). See [third-party notices](THIRD_PARTY_NOTICES.md).


## Independent checkpoint sanity check

In notebook 03, run the **Update the Drive copy from this notebook** cell, then the **Independent checkpoint sanity check** section at the bottom. You can run that section before or after any training stage; it does not trigger training. Existing checkpoints and caches are not modified.

From the bundle root, the equivalent command is:

```bash
python scripts/sanity_check.py --config runs/smoke-mohawk-config.yaml
```

By default it explicitly compares `mohawk-block.pt`, `full-depth.pt`, and `matryoshka.pt` on identical examples. Missing checkpoints are reported and skipped, and incomplete checkpoints are labeled. Checkpoint hashes, step counts, backend, precision, and the fixture identity are saved in a new timestamped `runs/smoke-mohawk/sanity-*.json` report. To include the intermediate stages:

```bash
python scripts/sanity_check.py --config runs/smoke-mohawk-config.yaml --stages initialized mohawk-block recovery full-depth prefix matryoshka
```

The check encodes 88 fixed short texts per checkpoint: 20 English query/positive/hard-negative/unrelated-negative cases and eight translated query probes. It tests whether relevant passages outrank negatives, retrieval over 60 fixed candidate passages (hit@1, hit@5, MRR), multilingual-to-English retrieval, nonfinite/zero vectors, and near-constant normalized outputs. Retrieval ties are counted conservatively so identical vectors cannot score well by their array order. All configured depth/dimension combinations are evaluated from a single all-exit encoding pass; six representative profiles are printed. Timing is not a measurement of early-exit speed.

These are authored debugging examples, **not an official benchmark**. Hard negatives deliberately include opposing intents and negation; low hard-negative accuracy alone does not prove the model is unusable. Each multilingual language has only one query. The pairwise similarity statistics include related examples too. Do not train on these probes or interpret repeated tuning against them as independent evaluation. Only tokenizer files are needed from Qwen; no teacher weights, teacher targets, benchmark download, or corpus loading is required.

Read `easy_accuracy`, retrieval hits, individual examples, and collapse flags together. Compare stages on the same profile. Post-MOHAWK sentence scores are only a baseline: that stage freezes the sentence readout and fits blocks independently, so poor sentence embeddings there do not by themselves indicate failed matrix/block alignment. The full-depth versus Matryoshka comparison is more informative about retention of trained sentence behavior. Use `--device cpu --backend reference` for CPU debugging, or `--batch-size 2` to reduce evaluation memory.
