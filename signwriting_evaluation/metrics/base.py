import itertools
from typing import Sequence

import numpy as np
from tqdm import tqdm


def validate_corpus_score_input(hypotheses: Sequence[str], references: Sequence[list[str]]):
    # This method is designed to avoid mistakes in the use of the corpus_score method
    assert isinstance(hypotheses, (list, tuple)), "Hypotheses must be a list or a touple"
    assert isinstance(references, (list, tuple)), "References must be a list or a tuple"
    if len(references) > 0:
        reference_type = type(references[0])
        assert reference_type in [list, tuple], \
            f"References must be a list of lists or tuples (found list of {reference_type})"

    for reference in references:
        assert len(hypotheses) == len(reference), \
            (f"Hypotheses ({len(hypotheses)}) and reference ({len(reference)}) "
             f"must have the same number of instances (references is ({len(references)}))")


class SignWritingMetric:
    """Base class for all metrics."""

    SYMMETRIC = False  # If True, the metric is symmetric (score(h, r) == score(r, h))

    def __init__(self, name: str):
        self.name = name

    def __call__(self, hypothesis: str, reference: str) -> float:
        return self.score(hypothesis, reference)

    def score(self, hypothesis: str, reference: str) -> float:
        raise NotImplementedError

    def score_max(self, hypothesis: str, references: list[str]) -> float:
        all_scores = self.score_all([hypothesis], references)
        return max(max(scores) for scores in all_scores)

    def corpus_score(self, hypotheses: Sequence[str], references: Sequence[list[str]]) -> float:
        # Default implementation: average over sentence scores
        # example: hypotheses=["hello"], references=[["hi"], ["hello"]]
        validate_corpus_score_input(hypotheses, references)
        transpose_references = list(zip(*references))
        return sum(self.score_max(h, r) for h, r in zip(hypotheses, transpose_references)) / len(hypotheses)

    def score_all(self, hypotheses: Sequence[str], references: Sequence[str], progress_bar=True) -> list[list[float]]:
        # Default implementation: call the score function for each hypothesis-reference pair
        total = len(hypotheses) * len(references)
        iterator = itertools.product(hypotheses, references)
        if progress_bar and total > 1:
            # tqdm for short iterators has a large overhead even with disable=True (~1 second per 100,000 calls)
            iterator = tqdm(iterator, total=total)

        scores = [self.score(h, r) for h, r in iterator]
        return [scores[i:i + len(references)] for i in range(0, total, len(references))]

    def score_self(self, hypotheses: Sequence[str], progress_bar=True) -> list[list[float]]:
        if not self.SYMMETRIC:
            return self.score_all(hypotheses, hypotheses, progress_bar)
        
        # For symmetric metrics, only compute upper triangle to avoid redundant calculations
        n = len(hypotheses)
        scores = np.eye(n, dtype=np.float16)  # Initialize with diagonal 1
        
        total = n * (n - 1) // 2  # Exclude diagonal
        iterator = tqdm([(i, j) for i in range(n) for j in range(i + 1, n)],
                        total=total, disable=not progress_bar or total == 1)
        
        for i, j in iterator:
            score = self.score(hypotheses[i], hypotheses[j])
            scores[i][j] = scores[j][i] = score

        return scores.tolist()

    def __str__(self):
        return self.name
