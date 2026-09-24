# PCA views of the MNIST tree

Open [the interactive comparison](../output/mnist_pca/demo.html). It contains four panels with identical coordinates:

1. **Projected true digits:** all 10,000 test images, colored by their actual labels.
2. **Projection-aware subtree collapse:** the same points, assigned using original pixels to retained ancestors of the original tree.
3. **Projected original assignments:** no collapse; full-dimensional leaf assignments shown in 2D.
4. **Exact affine slice:** polygons formed by intersecting the original cells with `mean + u PC1 + v PC2`, restricted to the original root support. No projected observations are overlaid on these polygons.

Choose reference or ancestor-residual tree, adjust the collapse threshold, and color scatterplot groups by their training-majority digit or ancestor ID. The digit legend applies to majority coloring and the slice, not the arbitrary ancestor colors.

PCA is fitted exclusively on centered training pixels for visualization. The trained tree, original image inputs and models are not altered. PC1 accounts for 9.70% and PC2 for 7.10% of training variance: together only 16.80%. Components for constant pixel coordinates are set to their mathematically zero values to remove eigensolver roundoff. There is no pixel rescaling or supervised PCA.

## Your method: retain original ancestry, collapse subtrees

For each original internal cut, compare its actual left/right assignments with those obtained from the two-PC reconstructions of the training points that reach that node. Score the agreement with balanced accuracy, averaging left and right recall. A constant one-sided prediction scores 0.5 even if the original split is strongly imbalanced.

A subtree is collapsed only when **none of its active internal cuts** meets the selected agreement threshold. Weak ancestors remain when required to reach a stronger descendant. Thus the retained partition is a pruning of the original tree, and every image remains assigned to a true ancestor of its original leaf. Scores are computed on training data and are a heuristic, not cross-validated estimates of 2D separability. A differently fitted separator in 2D could perform better than the projected original cut.

The root cut has balanced agreement 0.5. Immediately collapsing every weak cut would therefore reduce the whole tree to its root. The descendant-aware rule avoids losing all useful deeper distinctions at once.

| Model | No collapse | Threshold 0.60 | Threshold 0.80 |
|---|---|---|---|
| Reference | 975 groups; 55.32% test majority accuracy | 462 groups; 48.91% | 140 groups; 37.75% |
| Ancestor residual | 347 groups; 49.63% | 139 groups; 43.25% | 54 groups; 34.25% |

The labels and readout use training node histograms. Test labels only produce descriptive accuracy. Exploring thresholds while looking at test results makes this an exploratory visualization, not a final unbiased model-selection benchmark. Group overlap can remain: pruning preserves ancestry, not a unique 2D partition. This operation does not compact/rewrite saved models or measure memory savings.

## My method: separate projection from intersection

The projected-assignment view shows what the original tree actually did to the images. The exact slice instead shows what it would do to points that genuinely lie on the chosen plane. For cut `x[j] < cut`, clipping uses `PC1[j] u + PC2[j] v < cut - mean[j]`. Every root-support inequality is also applied. Projection can reconstruct negative pixel values, so large parts of the displayed plane may be outside `[0,255]^784`; we leave those areas blank rather than silently changing the tree support or clipping image values.

This geometric slice may intersect very few cells, even when the full tree has hundreds of leaves. That does not mean those other leaves are empty. They can lie away from the plane along discarded directions. Comparing these views makes the loss of information visible without treating slice polygons as boundaries for projected image memberships.

## Reproduce and validate

```sh
julia --project=. examples/mnist_pca.jl
julia --project=. test/mnist_pca.jl
# With Playwright installed:
node test/mnist_pca_viewer.cjs
```

The build needs the existing MNIST dataset and `output/mnist/models.jls`. It uses Julia standard-library LinearAlgebra for exact training covariance eigendecomposition. `output/mnist_pca/pca.jls` stores the mean, two components, projections and per-split agreement. `summary.txt` records the full threshold sweep; `demo.html` is self-contained and works offline.

Checks cover polygon clipping, valid retained-ancestor assignments, finite exported data, reconstruction/leaf consistency at polygon interior points, unchanged baseline classification, nested group counts, viewer controls and mobile layout.
