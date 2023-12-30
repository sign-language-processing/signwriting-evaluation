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
        return sum(self.score(r, h) for r, h in zip(references, hypotheses)) / len(references)

    def __str__(self):
        return self.name
