# Arctic Embed XS versus the current embedding model

The matched experiment finds **better topical neighborhoods with Arctic XS, but much less improvement from the current tree partitions**. Changing the encoder improves the underlying space; the existing random axis-aligned cuts do not automatically preserve those improvements.

Open the [interactive comparison](../output/embedding_comparison/demo.html), [machine-readable results](../output/embedding_comparison/results.json), or [short summary](../output/embedding_comparison/summary.txt). The viewer provides separate PCA maps, clickable held-out posts, training examples per group, and identical queries with neighbors from both models.

## Models and inputs

| Encoder | Pinned revision | Dimensions |
|---|---|---:|
| Current: `futur/Qwen3-Embedding-0.6B-model2vec-onnx` | `a3f60a85a2570e7976ea7f2d6505ad3e5152a96f` | 256 |
| `Snowflake/snowflake-arctic-embed-xs` | `d8c86521100d3556476a063fc2342036d45c106f` | 384 |

“Arctic XS” means Snowflake's original XS encoder. The current model is the **static Model2Vec export**, not the full Qwen transformer. Arctic uses the official unquantized ONNX export, special tokens, a maximum of 512 tokens, CLS pooling and L2 normalization. This follows its [pooling configuration](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/blob/d8c86521100d3556476a063fc2342036d45c106f/1_Pooling/config.json) and [model-card instructions](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs). Posts use document encoding without the retrieval-query prefix for symmetric post-to-post clustering. No input required truncation in this run.

Both models embed exactly the same texts in exactly the same row order. Clustering uses normalized vectors at native dimensions. PCA is fitted on training vectors **only for display**. Its axes are separate for each encoder; the plots are projections, not geometric slices of tree cells.

The experiment uses the existing [100,000-post historical archive](Bluesky_Embeddings.md), sampled from bounded prefixes of dataset shards. It is not a uniform sample of all Bluesky or a single-day feed. There are two independent cohorts:

1. **20,000 distinct normalized texts**, selected with seed `20260923`. Original text is unchanged. An author hash assigns 15,981 posts to training and 4,019 to evaluation. This cohort has no reliable semantic labels.
2. **623 weakly labeled topic posts**, drawn from the same archive using predefined hashtag families. Every hashtag is removed before either encoder sees an input. Multi-family posts and inputs shorter than 30 normalized characters are excluded, masked-text duplicates are removed, and selection is capped at 160 posts/topic and four posts/author/topic. Twenty percent of authors per topic, with a minimum of five, are selected for evaluation; their union is held out across all topics. This yields 497 training posts and 126 evaluation posts from 113 evaluation authors.

An author or normalized exact text never crosses training/evaluation **within a cohort**. The cohorts can overlap because they are separate fits. Ordinary topic words, URLs, languages and near-duplicate templates can still influence results. Topics and aliases are recorded in the input manifest; neither clustering algorithm sees them.

| Weak topic | Training / evaluation posts | Current exact same-topic P@10 | Arctic exact same-topic P@10 |
|---|---:|---:|---:|
| Art | 127 / 33 | 33.0% | 67.6% |
| Books | 129 / 31 | 64.8% | 73.9% |
| Gaming | 102 / 24 | 27.9% | 32.5% |
| Music | 81 / 20 | **37.5%** | 22.0% |
| Nature | 31 / 10 | 11.0% | 34.0% |
| Politics | 27 / 8 | 7.5% | 33.8% |
| **Equal-weight mean over topics** | **497 / 126** | **30.3%** | **43.9%** |

P@10 is the fraction of ten retrieved training posts sharing the query's hidden topic. Macro averaging gives each topic equal weight. Random training-post retrieval has expected macro P@10 of 16.7% here. Arctic improves the average by **13.7 percentage points**, but loses on Music. A paired bootstrap resampling evaluation authors gives an approximate 95% interval of **+8.3 to +18.1 points**. This interval conditions on the fixed training set and weak labels; it does not establish live recommendation quality or performance on other corpora.

Inspection revealed repeated book-challenge templates across authors. A follow-up sensitivity check excludes training candidates with at least six shared word trigrams and Jaccard overlap of at least 0.5 with the query. URLs, mentions and nonletters are removed for overlap detection only. Sixteen of 126 queries lose candidates; macro P@10 becomes **26.5% current versus 43.1% Arctic**. The retrieval advantage survives this wording-overlap control. This is an exploratory check, not a comprehensive near-duplicate detector; clustering fits still contain the templates.

## Clustering and tree protocol

Both encoders use identical training/evaluation splits and seeds `17, 29, 43`:

- **Cosine baseline:** spherical k-means, k-means++ initialization, normalized centroid updates, up to 200 iterations or unchanged assignments. All twelve fits converged before the iteration cap. This is a batch diagnostic, not a replacement streaming implementation.
- **Adaptive tree:** the existing `LeafLatentTrees.fit_leaf_tree`, including per-leaf rank-4 PPCA, reservoir capacity 64 and maximum depth 32. Seeded training order is shared between models. Split thresholds are 128 arrivals for the natural cohort and 16 for the smaller probe. Both models reach the same final budgets: 64 leaves and 16 leaves respectively, for every seed. Half-life is infinite and merging is disabled to isolate embedding and partition geometry.

The tree selects axes in proportion to observed reservoir ranges, then cuts between the first and third quartiles on that axis. PPCA estimates local density and transfers moments after splits; **it does not choose topic-optimized cut directions**. Pólya probabilities are checked to sum to one, and transferred mass is checked against the training count. Group membership is routing through the actual full-dimensional tree.

During setup the probe split threshold was reduced from 24 to 16 to make all fits reach the same 16-leaf budget, and the k-means iteration cap was increased from 40 to 200 to allow convergence on the natural cohort. The final comparison uses the settings above.

## Results

These are means over three clustering seeds on held-out authors:

| Topic-probe metric | Current Model2Vec | Arctic XS |
|---|---:|---:|
| Exact cosine same-topic P@10, before partitioning | 30.3% | **43.9%** |
| Spherical k-means topic ARI | 0.077 | **0.228** |
| Adaptive-tree topic ARI | 0.025 | **0.054** |
| K-means training-majority balanced accuracy | 33.1% | **46.5%** |
| Tree training-majority balanced accuracy | 25.0% | **27.9%** |
| Same-leaf cosine retrieval: topic P@10 | 26.3% | **32.0%** |
| Same-k-means-group cosine retrieval: topic P@10 | 29.1% | **50.5%** |

ARI measures agreement between partitions and weak topic labels after adjusting for chance: zero is roughly chance, one is identical grouping. It supports a clearer topical clustering advantage for Arctic under the cosine baseline than under the adaptive tree. Tree ARI ranges across seeds are 0.018–0.031 for current and 0.041–0.065 for Arctic; these are seed ranges, not confidence intervals.

For the **20,000-post natural cohort**, exact neighbors are measured for 512 common held-out queries against all 15,981 training posts:

| Partition method / encoder | Exact top-10 neighbors found in one partition | Mean fraction of training candidates searched |
|---|---:|---:|
| Adaptive tree / current | 27.0% | 1.76% |
| Adaptive tree / Arctic | 10.6% | 1.74% |
| Spherical k-means / current | 68.4% | 3.29% |
| Spherical k-means / Arctic | 44.5% | 2.85% |

This is a **one-partition diagnostic**, not the project's full asymmetric sampling policy. Neighbors are defined separately in each encoder's own space. A current-model neighbor may differ from an Arctic neighbor in meaning, so the table measures preservation of each model's geometry, not comparative human relevance. Candidate fractions differ across methods; this is not a matched-compute retrieval contest.

The tree reduces the candidate set substantially, but one-leaf retrieval loses many neighbors, especially in Arctic's space. Multi-leaf probing or a different cut geometry should be tested before assuming better embeddings alone improve the public retrieval service. Same-group topic retrieval can outperform unrestricted cosine retrieval because the restriction changes the ranking pool; it does not mean every direct neighbor was retained.

Held-out cosine silhouette is recorded on a fixed sample of at most 1,024 posts. On the natural cohort it is 0.103 for current k-means and −0.062 for Arctic k-means, despite Arctic's advantage on the topic probe. Silhouette depends on the encoder's own geometry, cluster spread and anisotropy; it is not interchangeable with semantic correctness. Negative tree silhouettes show that many geometric neighbors lie across cuts, not that the density model is mathematically invalid.

## Runtime and storage

Embedding inference used an RTX 4070 SUPER through ONNX Runtime's CUDA provider. Julia clustering and tree fits ran on the Ryzen 7 7800X3D CPU, with four BLAS threads. Warm embedding medians on the **same 1,000 posts**, batch size 64:

| Measure | Current | Arctic XS |
|---|---:|---:|
| Tokenization + inference + normalization / 1,000 | **0.255 s** | 0.303 s |
| Inference for all 20,623 inputs, excluding tokenization | **5.09 s** | 7.12 s |
| Float32 vector payload / million posts | **0.95 GiB** | 1.43 GiB |
| Natural-cohort tree state, mean over seeds | **5.31 MiB** | 7.90 MiB |

Timed batches exclude downloads, session initialization and warm-up. Outputs are materialized on the host before the timer ends. Current uses the existing mixed/FP16 export; Arctic uses the official unquantized export. This compares the deployed representations, not identical arithmetic precision or maximum achievable throughput. Fit times, state byte counts and provider details are in `results.json`; short fit timings are noisy workstation measurements.

Tree state includes bounds, PPCA, reservoirs and fit bookkeeping, but excludes input matrices, text, post codecs, IDs and runtime overhead. Raw vector storage is 50% larger with Arctic because it has 384 rather than 256 dimensions. No codec quality or end-to-end ingestion memory benchmark is included.

## Reproduce

The existing `data/bluesky_100000/posts.jsonl`, `embeddings.npy` and `manifest.json` are required. Reuse the project's Python embedding environment and Julia ingestion environment:

```sh
# Matched rows and fresh embeddings for both models; revision-pinned downloads.
.venv-embeddings/bin/python scripts/prepare_embedding_comparison.py --device cuda
# --device cpu is supported; timings will differ.

# Julia clustering/tree experiment and self-contained viewer.
julia --project=environments/ingestion examples/embedding_comparison.jl

# Focused metric/algorithm and browser checks.
julia --project=environments/ingestion test/embedding_clustering.jl
julia --project=environments/ingestion test/embedding_comparison_protocol.jl
node test/embedding_comparison_viewer.cjs
# Set PLAYWRIGHT_MODULE if Playwright is installed at a different path.
```

Preparation writes `data/embedding_comparison/{posts.jsonl,current.f32,arctic.f32,manifest.json}`. SHA256 hashes bind source data, row order, model files and vectors. Each `.f32` file stores consecutive posts in Python row-major order, equivalent to a Julia `dimensions × posts` matrix. Existing embeddings and the live index configuration are unchanged.

Validation includes CPU/GPU agreement, singleton/batched inference agreement, unit norms, shape and finiteness checks, and comparison of fresh current-model outputs to archived embeddings. The archived spot check was exact; maximum checked CPU/GPU absolute differences were `2.24e-4` for current and `5.03e-5` for Arctic. Julia metric and protocol tests and desktop/mobile viewer checks passed.

The small hashtag probe, absent language labels, native dimension difference, residual template duplication and absence of user feedback limit the conclusions. Arctic merits further testing, particularly together with broader tree traversal; these results do not establish a production encoder choice.
