# Implementation specification: multilingual Qwen → elastic KDA embedder

Prepared 23 September 2026. This is an implementation handoff, not a report of completed training or verified deployment performance.

## 1. Objective and fixed requirements

Build one small multilingual text encoder that approximates the native Matryoshka embeddings of `Qwen/Qwen3-Embedding-8B`, at multiple computation depths and output dimensions.

The deployed model must have 12 distinct Kimi Delta Attention (KDA) blocks, a residual/hidden width of 384, and a maximum sentence-embedding width of 384. It must support actual early termination of computation, rather than running all blocks and selecting an intermediate output afterward. Default input length is 256 tokens; explicitly train and test a 512-token profile. Keep the earlier model-size ceiling of 80 million trainable parameters, including the token table and all inference heads.

The training sequence is:

    runtime preflight
        → prepare and split multilingual corpus
        → cache Qwen sentence embeddings
        → teacher-derived partial weight initialization
        → MOHAWK-style mixer orientation and independent block alignment
        → short internal/function distillation
        → full-depth embedding distillation
        → multi-depth, multi-dimension embedding distillation
        → evaluate, export, and benchmark ONNX

There is no MLM or next-token language-model training phase in the initial implementation. Treat the advantage of weight initialization over output-only distillation as an empirical question, not an established result for this particular architecture.

Qwen3-Embedding-8B advertises native output dimensions from 32 through 4096 and multilingual support. Its published checkpoint declares Apache-2.0. Use that specific embedding checkpoint, not an ordinary Qwen chat model, as the semantic teacher. [S1]

## 2. Decisions that prevent implementation ambiguity

### 2.1 Pure KDA first; not an undisclosed hybrid

The primary model uses KDA in all 12 blocks. A 3:1 KDA/softmax hybrid may be implemented as an explicitly named comparison, but it is not a substitute for the requested all-KDA model. Kimi Linear's published large-model results involve a KDA/MLA hybrid; they do not establish the performance of this proposed small, pure-KDA embedding model. [S3]

For the first implementation, use **causal KDA with last-valid-token pooling**. This is compatible with sequential teacher/student hidden-state alignment and keeps the recurrent implementation simple. The final pooled state can depend on the entire preceding input; causal token processing does not prevent sentence embedding.

Do not describe this configuration as a bidirectional BERT encoder. A bidirectional KDA experiment must explicitly specify forward/backward scans, their parameter sharing, fusion, padding behavior, and additional compute. It is a later ablation, not an invisible configuration change.

### 2.2 One multilingual tokenizer, factorized input embeddings

Retain the exact teacher tokenizer revision and token IDs initially. This removes token-alignment ambiguity during internal-state distillation. Do not silently replace it with an English-only tokenizer.

The teacher configuration currently specifies a vocabulary of 151,665 tokens. A dense `V × 384` student table alone would contain 58,239,360 parameters. [S2] Therefore use a factorized input embedding:

    token ID → Embedding(V, 256) → Linear(256, 384, bias=False)

This input component has 38,924,544 parameters when V=151,665. Input factorization is independent of the 384-dimensional residual stream and final output. Factorized embedding parameterization has precedent in ALBERT; its quality for this student still needs an ablation. [S10]

A rough parameter estimate for this specification is 62M (approximately 61.6M under the component count used here), depending on the exact KDA gates and normalization implementation. Compute the true count programmatically and fail configuration validation above 80M. Compare input ranks 128/192/256 only after rank 256 is working; a smaller input rank is not automatically quality-neutral.

### 2.3 No PCA for the semantic targets

For teacher output T(x), define:

    t_d(x) = L2Normalize(T(x)[:d])

Use these native prefixes directly. Do not fit a PCA projection, learn a replacement teacher output space, or rotate these target coordinates.

SVD/PCA is allowed ONLY as an optional weight-initialization tool for input/internal representations. That is a different operation from changing the teacher's final Matryoshka targets.

## 3. Student architecture

Recommended starting specification, subject to operator preflight:

| Component | Initial setting |
|---|---|
| Blocks | 12 independently parameterized KDA + feed-forward blocks |
| Residual width | 384 |
| KDA heads | 6 |
| Key and value head dimensions | 64 and 64 |
| Value expansion | 1.0 |
| Short convolution | Kernel size 4, using the pinned KDA implementation's semantics |
| Feed-forward | SwiGLU; 1024 intermediate channels |
| Normalization | Pre-RMSNorm; separate learned block norms |
| Sequence direction | Causal |
| Input factorization | V × 256, then 256 × 384 |
| Pooling | Last valid token |
| Embedding readout | One shared 384 → 384 linear head across exits |
| Exit normalization | Shared readout normalization, then prefix selection and L2 normalization |
| Trained depths | 1, 3, 6, 9, 12 |
| Trained output dimensions | 64, 128, 192, 256, 384 |
| Default/max trained lengths | 256 / 512 tokens |

Block skeleton:

    h = h + KDA(RMSNorm(h))
    h = h + SwiGLU_FFN(RMSNorm(h))

Keep the KDA implementation faithful to a pinned upstream reference. The FLA implementation includes Q/K/V projections, short convolutions, decay/update gates, output gating/normalization, and an output projection; replacing just softmax with a recurrence is not a complete KDA implementation. [S4]

A conceptual per-head reference recurrence is:

    S_decay = diag(alpha_t) @ S_previous
    error   = v_t - S_decay.T @ k_t
    S       = S_decay + beta_t * outer(k_t, error)
    output  = scale * q_t.T @ S

The state has shape `[key_dim, value_dim]`. Match the upstream choices for query/key normalization, scaling, gate parameterization, and state orientation exactly. The equation above is a correctness guide, not a replacement for those details.

Do not import Qwen RoPE operations into KDA by default. Initialize KDA-specific gates, convolution weights, and mixer parameters using the chosen KDA implementation's initialization.

Implement two mixer backends with matching semantics: a slow plain-PyTorch reference for tests, and the pinned optimized FLA backend for training. A third backend is the ONNX lowering. Keep all three behind one interface.

## 4. Phase 0: runtime and correctness gate

Perform this before paying to generate the full teacher cache.

The current ONNX documentation defines a standard-domain `LinearAttention` function at opset 27. ONNX Runtime's current main-branch kernel list also contains `com.microsoft::LinearAttention` kernels. These are different operator domains/versions, and main-branch support does not establish availability in an installed release or a browser execution provider. [S5, S6]

Implement a one-block, randomly initialized 384-wide KDA test and record:

- Exact Python, Torch, FLA, ONNX, ONNX Runtime, CUDA/Triton versions and source commits.
- Operator domain, version, execution provider, input/state dtypes, and shape restrictions.
- Forward parity: PyTorch reference versus FLA versus ONNX.
- Gradient parity: reference versus FLA on small test inputs.
- Actual provider assignment and fallback behavior from an execution trace.

Test batch sizes 1/4/16 and lengths 1/7/32/63/64/65/128/256/512. Include variable-length batches, chunk boundaries, padding, repeated inputs, and extreme-but-valid gates. Test head dimension 64 and six heads rather than assuming a large-model kernel accepts them efficiently.

Masking requirements: each document begins with independent zero recurrent and convolution state; padding must not alter valid outputs. Use right padding initially. Packed training is allowed only after explicit document-boundary reset tests pass. Reject all-padding inputs through a documented error path. Never reuse state between unrelated posts.

Test native ONNX Runtime CPU first. Test ARM CPU, browser WASM, and browser WebGPU separately when those environments are available. Mark untested environments as untested; native CPU/CUDA success is not browser success. [S7]

A standard-operator Scan/Loop implementation may serve as a correctness fallback, but do not label it a fast deployment solution without measurement. Triton custom/autograd kernels do not become portable just because the enclosing module exports.

Deliver `runtime_report.json`, a one-block ONNX test model, and tests. If the required provider has no working path, report the missing capability and keep a reference model available; do not quietly change the main model to full attention or claim deployment is ready.

## 5. Phase 1: representative multilingual corpus

### 5.1 Dataset selection and size gates

Create a versioned corpus manifest rather than downloading an entire web corpus.

Use FineWeb2 language-script subsets as one possible multilingual source, and the original FineWeb for English, as directed by the FineWeb2 card. Add curated short text and permitted conversational/post-like data so training is not dominated by long formal web pages. The FineWeb2 card specifies ODC-By and additional Common Crawl terms; record source-specific terms instead of treating public availability as blanket permission. [S11]

Initial configurable scale gates:

    smoke:       20,000 examples
    pilot:      200,000 examples
    main:     1,000,000 examples

These are project starting budgets, not claims that one million examples will preserve the teacher's multilingual capabilities. Increase only after measuring scaling on held-out tasks and recording compute/storage use.

Default pilot language set: English, Spanish, Portuguese, French, German, Russian, Arabic, Hindi, Bengali, Mandarin Chinese, Japanese, Korean, Indonesian, Turkish, Vietnamese, Thai, Swahili, and Telugu. This is a proposed coverage set, not the user's declared language preference. Make it configurable and map every language to actual available source configurations. Do not silently omit unavailable languages.

For the multilingual component, sample roughly as `p(language) ∝ available_count**0.5`, apply an English cap of 30%, and enforce minimum per-language representation. Save actual counts and sampling weights. Keep language and script separate. Include code-switching, transliteration, negation, emoji, and short informal sentences in targeted subsets.

Reserve a permitted parallel/paraphrase subset where available. Keep entire translation/paraphrase groups in the same split. Translation pairs are useful evaluation/training structure, not an excuse to assume every pair with different text is a negative.

### 5.2 Data processing contract

Deduplicate before splitting. Split by parent document, conversation, and duplicate/translation group as appropriate. Exclude selected benchmark examples and near duplicates from training when detectable; do not claim to know the teacher's own benchmark exposure.

Select sentence-aware text spans that fit the student context. Teacher and student must see the same visible text/token span. A target computed from a whole article cannot supervise a student that sees only its first 256 tokens without introducing avoidable label noise.

Keep normal Unicode, case, punctuation, and emojis. Apply only documented preprocessing; avoid aggressive normalization that erases distinctions in non-Latin scripts. Save length and truncation statistics by language, including discarded/truncated rates.

Recommended record schema:

    id: content-derived stable ID
    text: exact canonical visible text
    language: normalized language code
    script: script label where known
    source_dataset/source_revision/source_id
    source_document_id
    duplicate_group_id
    parallel_group_id: optional
    source_license/terms_reference
    split: train|validation|test
    token_count
    tokenizer_revision
    preprocessing_version
    text_sha256

## 6. Phase 2: cache Qwen sentence targets

Use `Qwen/Qwen3-Embedding-8B` in eval/inference mode with an exact locked revision. Follow its documented last-valid-token pooling rather than using mean pooling accidentally. For the initial symmetric post/document embedding task, use an empty prompt consistently; query-instruction tasks require their own explicitly identified mode/cache. [S1]

Use BF16 where supported and sufficient memory exists; otherwise validate the selected inference precision. An 8B teacher has approximately 16 GB of two-byte weights before activations and runtime overhead, so do not assume a 16 GB GPU can hold it comfortably. Colab hardware and session availability vary, including for paid tiers. [S12]

Do not silently quantize the teacher or substitute its 4B/0.6B sibling. A quantized teacher or different checkpoint must have a different manifest/cache ID and a prefix-level quality comparison against the intended teacher. If the intended teacher cannot run, report that constraint rather than producing mislabeled targets.

For each input cache only:

    cached_target384 = L2Normalize(teacher_embedding[:384])

Store the vector as FP16, and derive every smaller target later with:

    target_d = L2Normalize(cached_target384.float()[:d])

This recovers the same normalized prefix up to cache rounding. It avoids five separate vectors and avoids caching the full 4096-dimensional output for every example. One million 384-dimensional FP16 vectors occupy 768,000,000 bytes, approximately 0.768 GB, before metadata.

Keep full 4096-dimensional vectors on a small validation subset to measure the teacher's own truncation loss. Retain a small FP32 target audit subset to test FP16 cache error, especially at 64 dimensions.

Cache manifest must include teacher checkpoint and tokenizer revisions, pooling, prompts, normalization order, precision/quantization, preprocessing, visible input span/hash, and sample ID. Write sharded arrays plus metadata with atomic completion markers and checksums. Resume by completed shard; never mix incompatible runs.

Final embedding caches cannot support internal block distillation. That needs a separate, smaller hidden-state calibration cache or a live teacher pass.

## 7. Phase 3: teacher-derived weight initialization and recovery

### 7.1 Scope and honesty

Qwen and the student differ in width, depth, attention mechanism, heads, and normalization details. There is no assumed lossless conversion of Qwen softmax attention weights into KDA weights. Call this stage **partial weight warm-start plus functional recovery**, not an already validated Qwen-to-KDA converter.

GUIDE provides precedent for embedding-derived initialization, and MOHAWK provides precedent for staged Transformer-to-SSM functional distillation. Neither establishes this exact 36-layer, 4096-wide Qwen to 12-layer, 384-wide KDA transfer. [S8, S9]

### 7.2 Concrete initialization experiment

Implement a reproducible `init_from_teacher` experiment:

1. Read required teacher tensors shard by shard. Avoid loading several full-size teacher copies.
2. Fit a frozen internal-space projection `P ∈ R^(4096×384)` from the teacher input embedding table, using a documented SVD/uncentered-PCA convention. Use a deterministic seed and optionally frequency weights from the training split only. This bridge is NOT the teacher's final Matryoshka embedding space.
3. Form `E0 = E_teacher @ P`. Factorize E0 into student table `A ∈ R^(V×256)` and projection `B ∈ R^(256×384)` with a rank-256 approximation. Record reconstruction error and activation scale.
4. Map student block j to teacher block `m(j)=3j`, for j=1..12, when the pinned teacher has the verified 36-block configuration. Validate from configuration; never silently assume the count. [S2]
5. Initialize compatible feed-forward components as a controlled experiment. Select the same 1024 teacher intermediate channels for gate/up/down using a documented deterministic saliency rule. A calibration activation-based score is preferable to unrelated selections for the three matrices.
6. Initialize all KDA-specific attention, decay, update, convolution, and output-gating components using the upstream KDA defaults. Do not densely rotate Qwen attention heads or copy RoPE-dependent Q/K tensors as though they were already KDA.
7. Initialize student RMSNorm scales to valid defaults rather than applying an arbitrary vector projection to the teacher's norm gains. Use a separately initialized shared embedding readout.

For row-vector notation, candidate FFN transfer with teacher gate/up matrices of shape D×F, down matrix F×D, hidden bridge P, and selected channels J is:

    W_gate_student = P.T @ W_gate_teacher[:, J]
    W_up_student   = P.T @ W_up_teacher[:, J]
    W_down_student = W_down_teacher[J, :] @ P

Transpose appropriately for PyTorch's stored linear-weight convention. This is only an initialization heuristic: RMSNorm, gated nonlinearities, discarded layers/channels, and the new mixer prevent exact function preservation. Log which tensors were transferred, factorized, selected, or freshly initialized.

Required ablation: identically configured random-init student with the same training data, optimizer budget, and output losses. Also support an embedding-table-only warm start to determine whether FFN transfer helps or hurts.

### 7.3 MOHAWK-style calibration before whole-student recovery

The Colab implementation now provides `cache-mohawk` and `train --stage mohawk-matrix|mohawk-block`. Preserve projected token-table/FFN initialization, then train KDA behavior against the mapped Qwen blocks before composing the student. This is a reduced-width/depth adaptation of MOHAWK, not a claim of reproducing its architecture or quality. [S9]

Cache contiguous pre-block residuals, attention branch outputs, post-block residuals, and both norm outputs projected through the same frozen P. Cache short head-mean teacher softmax matrices using its actual Q/K normalization, RoPE, and causal mask. Teacher and student calibration views use identical token prefixes; do not reuse whole-document sentence targets for truncated calibration views.

The first warmup materializes KDA recurrent mixing coefficients with identity value probes and aligns Frobenius-normalized head means. The 32-to-6 head mismatch is handled by averaging; no one-to-one head correspondence is claimed. Matrix materialization is restricted to short calibration prefixes and uses reference FP32 math. The second warmup fits complete attention and block functions independently on teacher-provided inputs, including functional norm alignment. Preserve transferred FFNs during the first part, then optionally unfreeze at the lower transfer learning rate. Embeddings and the shared readout are frozen throughout these independent-block stages.

Use disjoint training/validation calibration records from the corpus's existing splits, report per-language/per-layer validation before and during training, and save full resume state. Store the bridge hash and shared teacher-layer mapping in checkpoints and cache identities. Then perform whole-student recovery to handle the omitted teacher layers and the changed input distribution. Defaults, storage estimates, limitations, and migration instructions are in [the Colab guide](../README.md#mohawk-adaptation-and-qwen-knowledge-retention).

### 7.4 Small internal-state recovery cache

Use a separate 2K–8K-example multilingual calibration split, drawn only from training. Capture the teacher's selected post-block states and project them through frozen P. Retain up to 16 deterministic valid token positions per document, including the last valid token, rather than every hidden state for the entire corpus.

For 8,192 examples × 16 positions × 12 layers × 384 channels × 2 bytes, the selected target states occupy about 1.21 GB before metadata. Do not describe caching all 4096-wide teacher states as cheap.

Because tokenizer IDs are shared, the sampled token positions align. Specify whether targets are before/after each block's normalization. Any later tokenizer change requires a new alignment method and a new recovery cache.

Train the student against normalized projected teacher states at matched depths, with a full-depth sentence-embedding anchor:

    L_recovery = mean_j,positions MSE(normalize_hidden(Hs_j),
                                     normalize_hidden(Ht_m(j) @ P))
                 + lambda_anchor * L_sentence_384

Freeze P and teacher targets. Start by training new KDA parameters/readout, then unfreeze transferred components if validation supports it. Treat the exact freeze duration and loss weight as pilot hyperparameters. Decay the internal-state loss to zero before the final multi-exit stage so the semantic task is not permanently constrained by an arbitrary hidden-state basis.

After targets are cached, unload the teacher. Student optimization should not require keeping an 8B teacher in GPU memory.

## 8. Phase 4: full-depth embedding distillation

First train only the canonical full-depth output against Qwen's native 384-dimensional target. Once this learns reliably, introduce the other dimension prefixes at depth 12. Only then add early-depth supervision.

For pooled hidden state h_l(x), shared readout R, and readout normalization N:

    z_l(x)   = R(N(h_l(x)))
    s_l,d(x) = L2Normalize(z_l(x)[:d])

The primary output loss is:

    L_direct(l,d) = mean_i [1 - dot(s_l,d(x_i), t_d(x_i))]

Normalized vector MSE is an equivalent alternative up to a constant. Do not sum both as though they were independent objectives.

Direct coordinate alignment is important here: pairwise-only geometry losses permit rotations/reflections, which are undesirable when mixed-depth vectors must share a coordinate system and feed an axis-aligned tree.

## 9. Phase 5: two-dimensional teacher-prefix distillation

Train depths `L={1,3,6,9,12}` and dimensions `D={64,128,192,256,384}`. The initial implementation supervises the full 5×5 grid, not only a diagonal of five operating points. All requested hidden states come from ONE student forward pass. Additional prefix slicing and loss calculations must not trigger separate encoder passes.

Suggested deployment profiles are `(1,64), (3,128), (6,192), (9,256), (12,384)`, but the complete grid enables controlled depth/width ablations. The depth/dimension distinction is the central feature of 2D Matryoshka: reducing depth can skip encoding work, while truncating the output mainly changes downstream vector cost. [S13, S14]

### 9.1 Loss components

Let normalized student/teacher batches at dimension d have similarity matrices `G_S=S S.T` and `G_T=T T.T`. Mask the diagonal and duplicate/self entries.

    L_rel(l,d) = mean_{i != j} (G_S[i,j] - G_T[i,j])**2

Use batches containing a mix of random examples and semantically nearby teacher-selected examples, including cross-language neighbors. Uniformly random batches alone can make local geometry poorly tested. Pairwise losses operate on the actual microbatch or explicit gathered tensors; gradient accumulation does not magically enlarge their similarity matrix.

Optional SAME-DIMENSION cross-depth alignment:

    L_depth(l,d) = mean_i [1 - dot(s_l,d(x_i), stopgrad(s_12,d(x_i)))]

Use this only for l<12. It targets the full student's prefix at the same dimension, not an unrelated full-width similarity matrix.

Proposed initial objective:

    L = sum_{l,d} normalized_weight(l,d) *
            [L_direct(l,d) + 0.1 * L_rel(l,d)]
        + 0.05 * mean_{l<12,d} L_depth(l,d)

The numerical weights are tunable starting values, not literature-established optima. Suggested relative depth weights are `[0.25,0.5,1,1,2]`; suggested dimension weights are `[0.5,0.5,0.75,1,2]`. Normalize weights and log loss/gradient contribution per exit. Ramp shallow-exit weights over the first 10% of this stage.

Keep the external teacher fixed and the full-depth direct loss strong. Stop-gradient on a self-distillation target does NOT freeze the full student's representation: shared backbone/readout parameters can still change from other losses.

GOR and supervised contrastive losses are optional ablations, disabled in the baseline. Do not impose global isotropy/orthogonality without checking compatibility with the teacher geometry, and do not treat all distinct multilingual examples as negatives.

### 9.2 No unsupported geometric guarantees

Matryoshka prefixes are nested coordinates, but their separately normalized pairwise geometries need not be identical. Do not force `teacher64` and `teacher384` to have equal similarities as an unconditional objective. First measure this intrinsic teacher mismatch; any extra cross-dimension geometry penalty is an explicit, low-weight experiment.

Training does not guarantee monotonic error reduction for every document or invariant tree membership after refinement. Report those properties as measured statistics, not architectural guarantees.

## 10. Training engineering and resource controls

Provide a package-first implementation and thin Colab notebooks that call it. Do not hide model/training logic in notebook cells.

Proposed optimizer starting settings: AdamW, learning rate 1e-4 on transferred parameters and 3e-4 on newly initialized parameters, weight decay 0.01 on ordinary matrix weights, warmup 5%, gradient clipping 1.0. Preserve upstream exclusions for decay-rate/gate parameters and norms; sweep rather than assume these values are optimal.

Use BF16 training only on supporting hardware; otherwise use a validated precision configuration. Keep recurrent accumulation/gate math in the numerically stable dtype required by the reference implementation. Record all casts.

Start with an automatically sized microbatch, effective batch target 256 via accumulation, and direct-loss training that does not need a large in-batch negative matrix. Include some 257–512-token examples explicitly; training only at 256 is not validation of a 512 profile.

Track processed examples, actual nonpadding tokens, wall-clock time, peak GPU/host memory, and storage, including teacher preprocessing. Do not promise that Colab Pro is enough for a specific language/quality target before the pilot. [S12]

Checkpoint model/optimizer/scheduler/scaler states, RNG states, sampler position, source manifests, teacher cache identity, current loss schedule, and progress counters. Shard caches, use atomic checkpoint completion, and verify interrupted/resumed runs. Require explicit scale settings so a smoke test cannot accidentally launch a million-example teacher job.

## 11. Evaluation and release gates

### 11.1 Functional correctness

Required tests cover prefix slicing and renormalization; student/teacher sample-ID integrity; padding/last-token pooling; recurrent-state isolation; cached versus freshly generated teacher prefixes; one-forward multi-exit training; actual early exit; export parity; invalid depth/dimension requests; deterministic cache creation; and interruption recovery.

For an early-exit inference call, use hooks/profiling to prove no later block executes. Merely returning h_3 from a fully executed 12-block graph is a failed implementation.

### 11.2 Semantic and geometric evaluation

For every depth/dimension operating point and each evaluated language, report direct cosine alignment, cosine-error distribution, similarity-matrix error, teacher-neighbor recall@10/@50, and downstream retrieval/STS/bitext-mining or clustering measures on chosen held-out tasks.

Report separate compression gaps:

    teacher 4096 → teacher d: intrinsic prefix/truncation effect
    teacher d → student 12,d: student capacity/training effect
    student 12,d → student l,d: early-exit effect

Include cross-language retrieval and parallel-pair matching; a macro-language score alongside traffic-weighted averages; code-switching/negation/order-sensitive checks; and exact language/test coverage. Do not claim 100-language parity because the teacher supports 100+ languages.

Use identical-data and identical-budget controls: random initialization; table-only transfer; table+FFN transfer/recovery; full-depth-only KD; full 2D KD. A same-width/depth softmax model is a useful runtime/quality control, not a silent fallback. Model2Vec is a throughput reference, not a contextual-equivalence claim.

Set release thresholds from the pilot and document them. Do not fabricate target MTEB scores or declare the one-layer exit useful without measuring it. Keep experimental exits behind configuration until they offer a measured quality/latency benefit.

### 11.3 Deployment benchmarks

Measure end-to-end latency including tokenization and normalization, p50/p95 latency, texts/s, peak memory, model/package bytes, cold load, and warmed inference at actual short-text lengths. Compare native CPU, ARM, WASM, and WebGPU only when measured on those targets. Linear asymptotic scaling does not imply lower latency at 256 tokens.

Start ONNX parity in FP32. A reasonable initial investigation threshold is maximum absolute output error around 1e-4 and per-example cosine agreement near 0.9999 for FP32 reference comparisons, with backend-specific tolerances justified by recurrence/state error and downstream metrics. These are engineering targets, not observed results. Quantized exports need separate thresholds and quality evaluation.

## 12. ONNX packaging and public inference contract

Export one truncated graph per selected depth. Each graph may emit an unnormalized 384-dimensional readout, followed by host-side prefix selection and normalization. This avoids 25 separate model copies. Supply ready-to-use wrappers so callers cannot accidentally forget renormalization.

Where supported, share external initializer data across depth graphs; otherwise ship profiles as optional downloads and report actual duplicate storage. Do not claim weight sharing works in a particular browser without testing.

Quantize ordinary projections/input tables only when provider support and parity permit it. Keep numerically sensitive recurrent-state/gate calculations at an appropriate floating precision. Validate lower-dimension prefixes after quantization, not only the 384-dimensional output.

Public API to implement:

    encode(texts, depth=12, truncate_dim=384, normalize=True)

Validate the trained depth/dimension combinations. Include an embedding representation identifier covering model revision, tokenizer, preprocessing, prompt/task mode, depth, dimensionality, pooling, normalization, and quantization profile.

For refinement of a stored document, rerun its text to the requested depth by default. A pooled 64/384-dimensional vector is not sufficient to resume computation at a later layer; that requires the full token-level hidden state at the exit. Token-level caching is an optional bounded cache, not the default for an entire corpus.

## 13. Tree integration contract

The downstream tree must not assume that upgrading an embedding leaves its coordinates or cell membership unchanged. Record embedding version/depth/dimension and define a routing policy explicitly.

A safe initial policy is to retain each item's original coarse routing vector and use higher-resolution vectors only within explicitly refined regions. If refined embeddings replace routing coordinates, audit boundary crossings and perform the required reassignment/rebuild; a point can cross a higher-level boundary, not just a local split.

Compare vectors only at an agreed depth/dimension/normalization or through an explicitly evaluated mixed-depth policy at the same dimension. Never zero-pad a 64-dimensional normalized vector to compare it with a 384-dimensional normalized vector as though their cosine were equivalent.

Measure paired coarse/fine displacement and routing disagreement on held-out data. New model versions require cache/index migration; a shared model name is not evidence of unchanged geometry.

## 14. Repository and implementation milestones

The uploadable implementation lives in `elastic_embed_colab/`, alongside its [Colab guide](../README.md). Run the commands below from that bundle root. The research-oriented layout below remains a suggested organization; the first-pass implementation uses flat Python modules.

Suggested layout within the bundle:

    src/elastic_embed/
        config.py
        data/{prepare,manifests,sampling}.py
        teacher/{cache_embeddings,cache_hidden,pooling}.py
        models/{kda_reference,kda_fla,student,readout}.py
        initialization/{input_factorization,ffn_transfer}.py
        losses/{prefix_kd,relational,depth_alignment}.py
        training/{recovery,embedding,checkpointing}.py
        export/{onnx_lowering,profiles,quantization}.py
        evaluation/{alignment,neighbors,multilingual,runtime,tree}.py
        cli.py
    configs/{smoke,pilot,main,ablations}.yaml
    notebooks/{01_teacher_cache,02_initialization_recovery,03_train,04_export_eval}.ipynb
    tests/
    reports/
    MODEL_CARD.md
    DATA_PROVENANCE.md
    THIRD_PARTY_NOTICES.md

First-pass CLI commands (from the bundle root):

    python -m elastic_embed.cli runtime-check --config configs/smoke.yaml
    python -m elastic_embed.cli prepare-data --config configs/pilot.yaml
    python -m elastic_embed.cli cache-teacher --config configs/pilot.yaml --resume
    python -m elastic_embed.cli initialize --config configs/pilot.yaml
    python -m elastic_embed.cli cache-mohawk --config configs/pilot.yaml --resume
    python -m elastic_embed.cli train --stage mohawk-matrix --config configs/pilot.yaml --resume
    python -m elastic_embed.cli train --stage mohawk-block --config configs/pilot.yaml --resume
    python -m elastic_embed.cli cache-recovery --config configs/pilot.yaml --resume
    python -m elastic_embed.cli recover --config configs/pilot.yaml --resume
    python -m elastic_embed.cli train --stage full-depth --config configs/pilot.yaml
    python -m elastic_embed.cli train --stage matryoshka --config configs/pilot.yaml
    python -m elastic_embed.cli evaluate --config configs/pilot.yaml
    python -m elastic_embed.cli export --config configs/pilot.yaml
    python -m elastic_embed.cli benchmark --config configs/pilot.yaml

Milestone A: reference/optimized/ONNX one-block parity and a 12-block shape/parameter test.

Milestone B: deduplicated multilingual smoke corpus, correct teacher cache, and resume tests.

Milestone C: reproducible partial weight transfer, sparse hidden cache, and comparison with random/table-only initialization.

Milestone D: full-depth 384-dimensional student learns meaningful held-out alignment.

Milestone E: 2D teacher-prefix training, per-exit/per-language reports, and real early exit.

Milestone F: ONNX packages with measured provider support, quantization reports where applicable, and a model card with limitations.

Keep a functioning baseline after every milestone. A failed research ablation is a valid reported result; it must not be disguised by changing the teacher, context, data split, or architecture.

## 15. Source references and evidence boundaries

These references establish components and prior methods. The combination, hyperparameters, budgets, transfer formulas, architecture sizing, and release gates in this document are proposed engineering decisions, not published or measured results for the proposed student. URLs and revisions must be locked again by the implementation agent before use.

[S1] Qwen3-Embedding-8B model card: https://huggingface.co/Qwen/Qwen3-Embedding-8B

[S2] Qwen3-Embedding-8B configuration: https://huggingface.co/Qwen/Qwen3-Embedding-8B/blob/main/config.json

[S3] Kimi Linear paper: https://arxiv.org/abs/2510.26692 ; official project: https://github.com/MoonshotAI/Kimi-Linear

[S4] FLA KDA layer implementation: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py

[S5] ONNX standard LinearAttention specification: https://onnx.ai/onnx/operators/onnx__LinearAttention.html

[S6] ONNX Runtime current main-branch kernel list: https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/OperatorKernels.md ; release compatibility: https://onnxruntime.ai/docs/reference/compatibility.html

[S7] ONNX Runtime WebGPU documentation: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

[S8] GUIDE: Guided Initialization and Distillation of Embeddings: https://arxiv.org/abs/2510.06502

[S9] MOHAWK / Transformers to SSMs: https://arxiv.org/abs/2408.10189

[S10] ALBERT / factorized embedding parameterization: https://arxiv.org/abs/1909.11942

[S11] FineWeb2 dataset card, language subsets, and terms: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

[S12] Google Colab FAQ / resource availability: https://research.google.com/colaboratory/faq.html

[S13] 2D Matryoshka Sentence Embeddings: https://arxiv.org/abs/2402.14776

[S14] Sentence Transformers Matryoshka2dLoss documentation: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshka2dloss

[S15] Starbucks-v2: Improved Training for 2D Matryoshka Embeddings: https://arxiv.org/abs/2410.13230
