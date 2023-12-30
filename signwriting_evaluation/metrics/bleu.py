from sacrebleu.metrics import BLEU

from signwriting.tokenizer import SignWritingTokenizer
from signwriting_evaluation.metrics.base import SignWritingMetric


class SignWritingBLEU(SignWritingMetric):
    """Wrapper for sacrebleu's BLEU metric with added tokenization."""

    bleu = BLEU()
    tokenizer = SignWritingTokenizer()

    def __init__(self):
        super().__init__(name="TokenizedBLEU")

    def score(self, hypothesis: str, reference: str) -> float:
        return self.corpus_score([hypothesis], [[reference]])

    def tokenize(self, text: str) -> list[str]:
        return " ".join(self.tokenizer.text_to_tokens(text))

    def corpus_score(self, hypotheses: list[str], references: list[list[str]]) -> float:
        hypotheses = [self.tokenize(h) for h in hypotheses]
        references = [[self.tokenize(r) for r in reference] for reference in references]
        return self.bleu.corpus_score(hypotheses, references).score / 100
