from tqdm import tqdm


class SignWritingMetric:
    """Base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, hypothesis: str, reference: str) -> float:
        return self.score(hypothesis, reference)

    def score(self, hypothesis: str, reference: str) -> float:
        raise NotImplementedError

    def score_max(self, hypothesis: str, references: list[str]) -> float:
        all_scores = self.score_all([hypothesis], references)
        return max(max(scores) for scores in all_scores)

    def corpus_score(self, hypotheses: list[str], references: list[list[str]]) -> float:
        # Default implementation: average over sentence scores
        return sum(self.score_max(h, r) for h, r in zip(hypotheses, references)) / len(references)

    def score_all(self, hypotheses: list[str], references: list[str]) -> list[list[float]]:
        # Default implementation: call the score function for each hypothesis-reference pair
        return [[self.score(h, r) for r in references] for h in tqdm(hypotheses, disable=len(hypotheses) == 1)]

    def __str__(self):
        return self.name
