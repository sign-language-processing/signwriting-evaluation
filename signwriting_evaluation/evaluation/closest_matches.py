from pathlib import Path

from signwriting.visualizer.visualize import signwriting_to_image

from signwriting_evaluation.metrics.bleu import SignWritingBLEU
from signwriting_evaluation.metrics.chrf import SignWritingCHRF
from signwriting_evaluation.metrics.clip import SignWritingCLIPScore


def load_signs(signs_file: Path):
    with open(signs_file, 'r', encoding='utf-8') as signs_f:
        signs = signs_f.read().splitlines()
    return list(dict.fromkeys(signs))  # unique signs, ordered


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    assets_dir = current_dir.parent.parent / "assets"
    all_signs = load_signs(assets_dir / "single_signs.txt")
    specific_signs = load_signs(assets_dir / "hello_signs.txt")
    print(f"Found {len(all_signs)} signs")

    matches_dir = assets_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        SignWritingBLEU(),
        SignWritingCHRF(),
        SignWritingCLIPScore(cache_directory=None),
    ]

    for metric in metrics:
        print(f"Computing {metric.name}")
        all_scores = metric.score_all(specific_signs, all_signs)

        for specific_sign, scores in zip(specific_signs, all_scores):
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
