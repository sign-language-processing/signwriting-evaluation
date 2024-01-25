from sacrebleu.metrics import CHRF

from signwriting_evaluation.metrics.base import SignWritingMetric


class SignWritingCHRF(SignWritingMetric):
    """Wrapper for sacrebleu's CHRF metric."""

    chrf = CHRF()

    def __init__(self):
        super().__init__(name="CHRF")

    def score(self, hypothesis: str, reference: str) -> float:
        return self.chrf.sentence_score(hypothesis, [reference]).score / 100

    def corpus_score(self, hypotheses: list[str], references: list[list[str]]) -> float:
        self.validate_corpus_score_input(hypotheses, references)
        return self.chrf.corpus_score(hypotheses, references).score / 100
