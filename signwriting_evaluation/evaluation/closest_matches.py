from pathlib import Path

from signwriting.visualizer.visualize import signwriting_to_image

from signwriting_evaluation.metrics.bleu import SignWritingBLEU
from signwriting_evaluation.metrics.chrf import SignWritingCHRF


def load_signs(signs_file: Path):
    with open(signs_file, 'r', encoding='utf-8') as signs_f:
        signs = signs_f.read().splitlines()
    return list(set(signs))


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
    ]

    for specific_sign in specific_signs:
        sign_dir = matches_dir / specific_sign
        sign_dir.mkdir(parents=True, exist_ok=True)
        signwriting_to_image(specific_sign).save(sign_dir / "ref.png")

        other_signs = [sign for sign in all_signs if sign != specific_sign]
        for metric in metrics:
            metric_dir = sign_dir / metric.name
            metric_dir.mkdir(parents=True, exist_ok=True)

            print(f"Computing {metric.name} for {specific_sign}")
            scores = metric.score_all([specific_sign], other_signs)[0]
            closest_signs = sorted(zip(other_signs, scores), key=lambda x: x[1], reverse=True)[:10]
            print("Closest signs:")
            for i, (sign, score) in enumerate(closest_signs):
                print(f"{score}: {sign}")
                signwriting_to_image(sign).save(metric_dir / f"{i}.png")
