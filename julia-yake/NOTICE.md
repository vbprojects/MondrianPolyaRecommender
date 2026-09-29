# Provenance

This is a Julia adaptation of YAKE's default statistical algorithm, examined
against the installed Python `yake==0.6.0` sources:
https://github.com/INESCTEC/yake

Original YAKE copyright: © 2018 INESC TEC. Authors: Ricardo Campos, Vítor
Mangaravite, Arian Pasquali, Alípio Jorge, Célia Nunes, Adam Jatowt.
The upstream distribution's actual LICENSE specifies AGPL-3.0-or-later; its
complete license and attribution are preserved in LICENSE. This folder's
adapted implementation is distributed under those terms. This notice does
not relicense other repository files.

Word-boundary and contraction rules are adapted from segtok 1.5.11,
© Florian Leitner, MIT license; see notices/segtok-LICENSE.txt.
Sentence segmentation here is a smaller native approximation and is not
represented as a complete segtok replacement. Scoring and tokenization
parity are tested independently. English stopwords are copied from the
YAKE 0.6.0 distribution and covered by its accompanying license.

Julia adaptation: 2026. No Python, NumPy, NetworkX, or segtok runtime is used.
