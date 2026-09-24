# MNIST with flattening only

Open [the leaf explorer](../output/mnist/demo.html) to switch between the reference tree and ancestor residuals, inspect digit histograms and original images in each leaf, and view the held-out confusion matrix.

The run uses the standard 60,000 training and 10,000 test images, each 28×28 UInt8 pixels. Source: [MNIST dataset description](https://www.tensorflow.org/datasets/catalog/mnist); original IDX files are downloaded from the Google CVDF MNIST mirror, with URLs and SHA-256 hashes recorded in `output/mnist/manifest.json`.

The only input transformation is flattening each image's original row-major raster into a 784-element column. Values remain UInt8 integers from 0 to 255. There is no normalization, standardization, binarization, pixel removal, PCA, embedding model or data augmentation. MNIST itself already consists of prepared 28×28 digit images; we make no additional image adjustments. Canvas rendering preserves the original raster orientation and intensities.

## Models and results

This reuses the project's adaptive Mondrian-style partition and Pólya branch counts, with seed 7, a 2,047-node budget, minimum split occupancy 128, depth limit 24 and branch prior mass 2. The reference learns geometry exclusively from training pixels. Ancestor residuals use that shared scaffold and process the original training file order, retaining older evidence at ancestors after prospective splits. No microclusters or time decay are used.

The root support is fixed in advance at `[0,255]^784`. This is a support choice, not input preprocessing: it accommodates valid held-out pixels even in coordinates that were constant in training. The model uses continuous rectangular volumes as a density surrogate for discrete pixels. Density values are therefore not discrete image probabilities; logs avoid numerical underflow.

| Model | Active leaves | Training leaf purity | Test majority accuracy | Fit time |
|---|---:|---:|---:|---:|
| Reference | 975 | 55.28% | 55.32% | 0.452 s |
| Ancestor residual | 347 | 49.12% | 49.63% | 0.040 s |

The common scaffold has 1,949 nodes. Both models represent all 60,000 training images, and leaf probabilities sum to one within `2.3e-16`. Reachable model memory is approximately 23.62 MiB for the reference and 23.71 MiB for residuals, including their complete scaffold but excluding images and label diagnostics. Fit timings exclude compilation warmup, loading, diagnostics and export; residual time also excludes scaffold construction.

Labels are read only after fitting to construct per-leaf training histograms. Each test image is routed through the fitted partition and receives the majority training label of its leaf. An empty cell falls back to the global training majority. Test labels are used only to score those predictions. This is a post-hoc readout of unsupervised geometry, not a supervised classifier or class posterior derived from the Pólya density. For the residual model, histograms are computed by rerouting archived training images through the final active leaves; they are additional diagnostic information not recoverable from residual scalars alone.

As in the Bluesky experiment, this is an occupancy-adapted Mondrian-style heuristic, not an exact/projective Mondrian process. One seed and one preset budget were used; there was no test-driven hyperparameter tuning. These results show some digit grouping in raw pixel space, with substantial mixing remaining.

## Reproduce

```sh
julia --project=. examples/mnist_tree.jl
julia --project=. test/runtests.jl
```

The Julia loader uses standard-library Downloads and the system `gzip` executable. It validates IDX magic, dimensions, exact payload lengths, label ranges and split sizes. Downloads are cached in `data/mnist/`. No Python, GPU model or added Julia package is required.

```julia
using MondrianPolyaToy, Serialization
include("examples/load_mnist.jl")
data = load_mnist()
saved = deserialize("output/mnist/models.jls")
x = data.test[:, 1]  # 784 original pixel values
leaf = embedding_leaf(saved.partition, x, saved.residual.active)
```

`output/mnist/` contains the trusted local Julia serialization, source manifest, summary, viewer, and train/test assignment CSVs for each model. CSV image row indices are zero-based; node indices are one-based. Test files contain true label, training-majority prediction and log density. The original IDX data remain available.

Validation includes all existing project tests, new IDX flattening/fixed-support tests, normalization and count checks, and browser checks for exact displayed pixel values, model/leaf selection, confusion matrices and mobile layout. The browser test is `test/mnist_viewer.cjs` and requires Playwright.

See also the [PCA projection, subtree-collapse and exact-slice comparison](MNIST_PCA.md). PCA is used only for visualization.
