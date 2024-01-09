from tqdm import tqdm


class SignWritingMetric:
    """Base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, hypothesis: str, reference: str) -> float:
        return self.score(hypothesis, reference)

    def score(self, hypothesis: str, reference: str) -> float:
        raise NotImplementedError

    def corpus_score(self, hypotheses: list[str], references: list[list[str]]) -> float:
        # Default implementation: average over sentence scores
        return sum(self.score(h, r) for h, r in zip(hypotheses, references)) / len(references)

    def score_all(self, hypotheses: list[str], references: list[str]) -> list[list[float]]:
        # Default implementation: call the score function for each hypothesis-reference pair
        return [[self.score(h, r) for r in references] for h in tqdm(hypotheses)]

    def __str__(self):
        return self.name
