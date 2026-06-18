"""Correlate the SignWriting similarity metric with a translation model's own prediction confidence.

External validation: for a set of model predictions (source_sign -> predicted_sign, with the model's
`confidence`), a metric that better captures "is this prediction good?" should track confidence more
closely. We plot the OLD metric (the `signwriting_similarity` field, pre-re-fit) and the NEW metric
side by side against confidence, with correlations.

    python -m signwriting_evaluation.evaluation.model_confidence_correlation [predictions.jsonl]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from signwriting.formats.fsw_to_sign import fsw_to_sign
from signwriting.formats.swu_to_fsw import swu2fsw

from signwriting_evaluation.evaluation.text_correlation import ASSETS_DIR
from signwriting_evaluation.metrics.similarity_v2 import SignWritingSimilarityV2Metric, get_symbol_attributes

DEFAULT_PREDICTIONS = Path.home() / "Downloads" / "eval_run20rt_prefix.jsonl"

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 14})


def reference_coverage(source: str, predicted: str) -> float:
    # Fraction of the reference's (canonical) symbols that the prediction also contains. The metric is
    # best calibrated when this is 1 (the prediction uses all the reference's symbols); below that, the
    # length / inventory penalty dominates and the score reflects missing content, not dissimilarity.
    try:
        ref = {get_symbol_attributes(s["symbol"]).shape for s in fsw_to_sign(swu2fsw(source))["symbols"]}
        hyp = {get_symbol_attributes(s["symbol"]).shape for s in fsw_to_sign(swu2fsw(predicted))["symbols"]}
    except (ValueError, KeyError, IndexError):
        return 0.0
    return len(ref & hyp) / len(ref) if ref else 0.0


def load_points(path: Path):
    metric = SignWritingSimilarityV2Metric()
    confidence, old_scores, new_scores, coverage = [], [], [], []
    with open(path, encoding="utf-8") as predictions_file:
        records = [json.loads(line) for line in predictions_file]
    for record in records:
        conf, source, predicted = record.get("confidence"), record.get("source_sign"), record.get("predicted_sign")
        old = record.get("signwriting_similarity")
        if conf is None or old is None or not source or not predicted:
            continue
        try:
            new = metric.score(source, predicted)
        except (ValueError, KeyError, IndexError):
            continue
        confidence.append(conf)
        old_scores.append(old)
        new_scores.append(new)
        coverage.append(reference_coverage(source, predicted))
    return confidence, old_scores, new_scores, coverage


def binned_trend(confidence, scores, low, num_bins=12):
    # Bin over the OBSERVED confidence range [low, max] (it clusters near 1), not [0, 1].
    confidence, scores = np.array(confidence), np.array(scores)
    edges = np.linspace(low, confidence.max(), num_bins + 1)
    centers, means, stds = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= lower) & (confidence <= upper if upper == edges[-1] else confidence < upper)
        if in_bin.any():
            centers.append((lower + upper) / 2)
            means.append(scores[in_bin].mean())
            stds.append(scores[in_bin].std())
    return np.array(centers), np.array(means), np.array(stds)


def plot_panel(axis, confidence, scores, title, low, color="tab:purple"):  # pylint: disable=too-many-arguments
    confidence, scores = np.asarray(confidence), np.asarray(scores)
    pearson = pearsonr(confidence, scores).statistic
    spearman = spearmanr(confidence, scores).statistic
    axis.scatter(confidence, scores, s=6, alpha=0.15, color=color)
    centers, means, stds = binned_trend(confidence, scores, low)
    axis.plot(centers, means, color="black", marker="o", linewidth=2, label="bin mean")
    axis.fill_between(centers, means - stds, means + stds, color="black", alpha=0.15, label="±1 std")
    axis.plot([low, 1], [low, 1], color="tab:red", linestyle="--", linewidth=1, label="y = x (45°)")
    axis.set_title(f"{title}\nn={len(scores)}  Pearson r={pearson:.3f}, Spearman ρ={spearman:.3f}")
    axis.set_xlabel("model confidence")
    axis.set_xlim(low, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper left", fontsize=9)


def plot_by_completeness(confidence, new_scores, coverage, low, out_path):
    # The metric tracks confidence best when the prediction covers all the reference's symbols; split
    # the cloud by that coverage to show the agreement is concentrated in the "complete" regime.
    confidence, new_scores, coverage = np.asarray(confidence), np.asarray(new_scores), np.asarray(coverage)
    segments = [("Complete (covers 100% of reference)", coverage >= 1.0, "tab:green"),
                ("Partial (50–99%)", (coverage >= 0.5) & (coverage < 1.0), "tab:orange"),
                ("Sparse (<50%)", coverage < 0.5, "tab:red")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("New metric vs model confidence, by prediction completeness")
    for axis, (title, mask, color) in zip(axes, segments):
        plot_panel(axis, confidence[mask], new_scores[mask], title, low, color=color)
    axes[0].set_ylabel("SignWriting similarity (new metric)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PREDICTIONS
    confidence, old_scores, new_scores, coverage = load_points(path)
    print(f"Loaded {len(confidence)} predictions from {path.name}")

    correlation_dir = ASSETS_DIR / "correlation"
    correlation_dir.mkdir(parents=True, exist_ok=True)
    low = min(confidence)  # confidence clusters near 1; start the x-axis at the observed minimum
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle("SignWriting similarity vs model confidence")
    plot_panel(axes[0], confidence, old_scores, "Original metric", low)
    plot_panel(axes[1], confidence, new_scores, "New metric", low)
    axes[0].set_ylabel("SignWriting similarity")
    plt.tight_layout()
    plt.savefig(correlation_dir / "model_confidence_correlation.png")
    plt.close()

    plot_by_completeness(confidence, new_scores, coverage, low,
                         correlation_dir / "model_confidence_by_completeness.png")

    for name, scores in [("original", old_scores), ("new", new_scores)]:
        print(f"{name:>9}: Pearson r={pearsonr(confidence, scores).statistic:.3f}, "
              f"Spearman ρ={spearmanr(confidence, scores).statistic:.3f}")
    coverage = np.asarray(coverage)
    for label, mask in [("complete", coverage >= 1.0), ("partial", (coverage >= 0.5) & (coverage < 1.0)),
                        ("sparse", coverage < 0.5)]:
        scores = np.asarray(new_scores)[mask]
        print(f"  new/{label:8s} n={int(mask.sum()):4d}: Pearson r="
              f"{pearsonr(np.asarray(confidence)[mask], scores).statistic:.3f}, mean={scores.mean():.3f}")


if __name__ == "__main__":
    main()
