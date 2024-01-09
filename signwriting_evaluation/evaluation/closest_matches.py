from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from signwriting.visualizer.visualize import signwriting_to_image

from signwriting_evaluation.metrics.base import SignWritingMetric
from signwriting_evaluation.metrics.bleu import SignWritingBLEU
from signwriting_evaluation.metrics.chrf import SignWritingCHRF
from signwriting_evaluation.metrics.clip import SignWritingCLIPScore


CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent.parent / "assets"


def load_signs(signs_file: Path):
    with open(signs_file, 'r', encoding='utf-8') as signs_f:
        signs = signs_f.read().splitlines()
    return list(dict.fromkeys(signs))  # unique signs, ordered


def find_closest_signs(signs: list[str], all_signs: list[str], metrics: list[SignWritingMetric]):
    matches_dir = ASSETS_DIR / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        print(f"Computing {metric.name}")
        all_scores = metric.score_all(signs, all_signs)

        for specific_sign, scores in zip(signs, all_scores):
            sign_dir = matches_dir / specific_sign
            sign_dir.mkdir(parents=True, exist_ok=True)
            signwriting_to_image(specific_sign).save(sign_dir / "ref.png")

            metric_dir = sign_dir / metric.name
            metric_dir.mkdir(parents=True, exist_ok=True)

            closest_signs = sorted(zip(all_signs, scores), key=lambda x: x[1], reverse=True)[1:11]
            print("Closest signs:")
            for i, (sign, score) in enumerate(closest_signs):
                print(f"{score}: {sign}")
                signwriting_to_image(sign).save(metric_dir / f"{i}.png")


def metrics_distribution(signs: list[str], metrics: list[SignWritingMetric]):
    distribution_dir = ASSETS_DIR / "distribution"
    distribution_dir.mkdir(parents=True, exist_ok=True)

    signs_subset = signs[:1000]

    metric_scores = {}

    for metric in metrics:
        print(f"Computing {metric.name}")
        all_scores = metric.score_all(signs_subset, signs_subset)

        scores = np.array(all_scores).flatten()
        scores = scores[scores < 0.999]  # Remove self scores
        metric_scores[metric.name] = scores

    for metric_name, scores in metric_scores.items():
        mean, std = np.mean(scores), np.std(scores)

        plt.title(f"{metric_name}: µ={mean:.3f}, σ={std:.3f}")
        plt.hist(scores, bins=100)
        plt.tight_layout()
        plt.savefig(distribution_dir / f"{metric_name}.png")
        plt.close()

    bins = np.linspace(0, 1, 100)
    for metric_name, scores in metric_scores.items():
        plt.hist(scores, bins=bins, alpha=0.5, label=metric_name)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(distribution_dir / "all.png")
    plt.close()


if __name__ == "__main__":
    single_signs = load_signs(ASSETS_DIR / "single_signs.txt")
    hello_signs = load_signs(ASSETS_DIR / "hello_signs.txt")
    print(f"Found {len(single_signs)} signs")

    all_metrics = [
        SignWritingBLEU(),
        SignWritingCHRF(),
        SignWritingCLIPScore(cache_directory=None),
    ]

    metrics_distribution(single_signs, all_metrics)

    find_closest_signs(hello_signs, single_signs, all_metrics)
