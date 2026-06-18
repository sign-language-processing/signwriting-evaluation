# Project guide

Automatic evaluation metrics for SignWriting machine-learning output (BLEU, chrF, CLIP, and the
custom `SymbolsDistances` similarity metric).

## Similarity metric — mandatory documentation rule

We are actively iterating on `signwriting_evaluation/metrics/similarity_v2/similarity_v2.py` toward a new, better,
**data-driven** version, and we intend to publish it.

**Every modification to the similarity metric MUST be accompanied by an edit to
[`signwriting_evaluation/metrics/similarity_v2/similarity_v2.md`](signwriting_evaluation/metrics/similarity_v2/similarity_v2.md)** that
explains the change: what it does, the reasoning behind it, the alternatives considered, and the
evidence (numbers on the calibration signals) for the specific decisions made.

This is not optional bookkeeping — `similarity.md` is the running record we will turn into the paper.
If the reasoning ("why this exponent / weight / formulation and not another") is not written down at
the time of the change, it is effectively lost. Capture:

- the motivation / failure case the change addresses,
- the formulation (semi-mathematical),
- alternatives tried and why they were rejected,
- measured impact on the data-driven signals (text-Levenshtein correlation for fingerspelling /
  mouthing, human pairwise preferences, and the same/variation/mirror/different labels),
- any chosen constant's value and how it was tuned.

When you change a metric default or formula, also update the affected expected values in
`signwriting_evaluation/metrics/similarity_v2/test_similarity_v2.py`, and re-run the calibration/correlation checks.

## Calibration

The `calibration/` directory (git-ignored) holds the human-label collection harness and data used to
drive the metric: pairwise preferences, same/variation/mirror/different judgements, and the
ten-idea experiment write-up (`calibration/REPORT.md`). It is a research harness, not part of the
shipped library.
