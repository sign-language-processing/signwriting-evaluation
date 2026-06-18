import random
import string
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from epitran import Epitran
from faker import Faker
from scipy.stats import pearsonr, spearmanr

from signwriting.fingerspelling.fingerspelling import spell
from signwriting.mouthing.mouthing import mouth_ipa

from signwriting_evaluation.metrics.base import SignWritingMetric
from signwriting_evaluation.metrics.similarity_v2 import SignWritingSimilarityV2Metric

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent.parent / "assets"

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 14})

# A renderer maps a word to (text_for_comparison, signwriting), or None when it can't be rendered.
# text_for_comparison is the orthographic word for fingerspelling, and the IPA for mouthing.
Renderer = Callable[[str], "tuple[str, str] | None"]


class Series(NamedTuple):
    label: str
    text_points: list[float]
    metric_points: list[float]
    color: str
    text_label: str


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1 - levenshtein(a, b) / max(len(a), len(b))


def perturb_word(word: str, rng: random.Random, num_edits: int) -> str:
    # Single-character edits keep the result close to the original, populating the high-similarity region.
    # The capitalized first letter is preserved so casing doesn't add spurious differences.
    chars = list(word)
    for _ in range(num_edits):
        operations = ["substitute", "insert"]
        if len(chars) > 2:
            operations += ["delete", "transpose"]
        operation = rng.choice(operations)
        if operation == "insert":
            chars.insert(rng.randint(1, len(chars)), rng.choice(string.ascii_lowercase))
        elif operation == "substitute":
            chars[rng.randint(1, len(chars) - 1)] = rng.choice(string.ascii_lowercase)
        elif operation == "delete":
            del chars[rng.randint(1, len(chars) - 1)]
        else:  # transpose two adjacent characters
            i = rng.randint(1, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def sample_words(render: Renderer, num_words: int, variants_per_word: int = 0,
                 max_edits: int = 3, seed: int = 42) -> list[tuple[str, str]]:
    fake = Faker("en_US")
    Faker.seed(seed)
    rng = random.Random(seed)

    seen: set[str] = set()
    records: list[tuple[str, str]] = []
    bases = 0
    while bases < num_words:
        name = fake.first_name()
        if name in seen:
            continue
        seen.add(name)
        rendered = render(name)
        if rendered is None:  # skip names that can't be rendered
            continue
        records.append(rendered)
        bases += 1

        for _ in range(variants_per_word):
            variant = perturb_word(name, rng, rng.randint(1, max_edits))
            if variant in seen:
                continue
            seen.add(variant)
            rendered_variant = render(variant)
            if rendered_variant is not None:
                records.append(rendered_variant)
    return records


def fingerspell_renderer(signed_language: str = "ase", seed: int = 42) -> Renderer:
    def render(word: str) -> "tuple[str, str] | None":
        fsw = spell(word, language=signed_language, vertical=True, seed=seed)
        return (word, fsw) if fsw is not None else None

    return render


def mouthing_renderer(language: str = "eng-Latn") -> Renderer:
    epitran = Epitran(language, ligatures=True)  # build once; library's mouth() rebuilds it per call

    def render(word: str) -> "tuple[str, str] | None":
        ipa = epitran.transliterate(word)
        fsw = mouth_ipa(ipa)
        return (ipa, fsw) if fsw is not None else None  # compare on IPA: mouthing is phonetic, not orthographic

    return render


def correlation_points(records: list[tuple[str, str]], metric: SignWritingMetric,
                       text_similarity: Callable[[str, str], float]) -> tuple[list[float], list[float]]:
    texts = [text for text, _ in records]
    signs = [sign for _, sign in records]
    metric_scores = metric.score_all(signs, signs)

    text_points, metric_points = [], []
    for i, text_i in enumerate(texts):
        for j in range(i + 1, len(texts)):  # unordered pairs, excluding self-comparisons
            text_points.append(text_similarity(text_i, texts[j]))
            metric_points.append(metric_scores[i][j])
    return text_points, metric_points


def binned_trend(text_points: list[float], metric_points: list[float], num_bins: int = 10):
    text_array, metric_array = np.array(text_points), np.array(metric_points)
    edges = np.linspace(0, 1, num_bins + 1)
    centers, means, stds = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (text_array >= lower) & (text_array <= upper if upper == 1 else text_array < upper)
        if in_bin.any():
            centers.append((lower + upper) / 2)
            means.append(metric_array[in_bin].mean())
            stds.append(metric_array[in_bin].std())
    return np.array(centers), np.array(means), np.array(stds)


def plot_correlations(series: list[Series], metric_name: str, title: str, file_name: str):
    correlation_dir = ASSETS_DIR / "correlation"
    correlation_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(series), figsize=(6 * len(series), 6), sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(title)

    for axis, item in zip(axes, series):
        pearson = pearsonr(item.text_points, item.metric_points).statistic
        spearman = spearmanr(item.text_points, item.metric_points).statistic

        axis.scatter(item.text_points, item.metric_points, s=6, alpha=0.12, color=item.color)
        centers, means, stds = binned_trend(item.text_points, item.metric_points)
        axis.plot(centers, means, color="black", marker="o", linewidth=2, label="bin mean")
        axis.fill_between(centers, means - stds, means + stds, color="black", alpha=0.15, label="±1 std")

        axis.set_title(f"{item.label}\nPearson r={pearson:.3f}, Spearman ρ={spearman:.3f}")
        axis.set_xlabel(item.text_label)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(loc="upper left", fontsize=10)
        print(f"{item.label}: Pearson r={pearson:.3f}, Spearman ρ={spearman:.3f} ({len(item.text_points)} pairs)")

    axes[0].set_ylabel(f"{metric_name} (SignWriting)")
    plt.tight_layout()
    plt.savefig(correlation_dir / f"{file_name}.png")
    plt.close()


if __name__ == "__main__":
    similarity_metric = SignWritingSimilarityV2Metric()

    fingerspelled = sample_words(fingerspell_renderer(), num_words=60, variants_per_word=3)
    print(f"Sampled {len(fingerspelled)} fingerspellable names (with edit perturbations)")
    mouthed = sample_words(mouthing_renderer(), num_words=60, variants_per_word=3)
    print(f"Sampled {len(mouthed)} mouthable names (with edit perturbations)")

    fingerspelling_xs, fingerspelling_ys = correlation_points(fingerspelled, similarity_metric, levenshtein_similarity)
    mouthing_xs, mouthing_ys = correlation_points(mouthed, similarity_metric, levenshtein_similarity)

    plot_correlations(
        series=[
            Series("Fingerspelling (ase)", fingerspelling_xs, fingerspelling_ys,
                   "tab:blue", "Levenshtein similarity (text)"),
            Series("Mouthing (eng)", mouthing_xs, mouthing_ys,
                   "tab:orange", "Levenshtein similarity (IPA)"),
        ],
        metric_name=similarity_metric.name,
        title="SignWriting similarity vs text Levenshtein similarity",
        file_name="text_similarity_correlation")
