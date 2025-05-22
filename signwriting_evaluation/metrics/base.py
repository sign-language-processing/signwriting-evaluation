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

    def validate_corpus_score_input(self, hypotheses: list[str], references: list[list[str]]):
        # This method is designed to avoid mistakes in the use of the corpus_score method
        assert isinstance(hypotheses,list), "Hypotheses must be a list"
        assert isinstance(references, list), "References must be a list"
        if len(references) > 0:
            reference_type = type(references[0])
            assert reference_type in [list, tuple], \
                f"References must be a list of lists or tuples (found list of {reference_type})"

        for reference in references:
            assert len(hypotheses) == len(reference), \
                (f"Hypotheses ({len(hypotheses)}) and reference ({len(reference)}) "
                 f"must have the same number of instances (references is ({len(references)}))")

    def corpus_score(self, hypotheses: list[str], references: list[list[str]]) -> float:
        # Default implementation: average over sentence scores
        # example: hypotheses=["hello"], references=[["hi"], ["hello"]]
        self.validate_corpus_score_input(hypotheses, references)
        transpose_references = list(zip(*references))
        return sum(self.score_max(h, r) for h, r in zip(hypotheses, transpose_references)) / len(hypotheses)

    def score_all(self, hypotheses: list[str], references: list[str], progress_bar=True) -> list[list[float]]:
        # Default implementation: call the score function for each hypothesis-reference pair
        return [[self.score(h, r) for r in references]
                for h in tqdm(hypotheses, disable=not progress_bar or len(hypotheses) == 1)]

    def __str__(self):
        return self.name
