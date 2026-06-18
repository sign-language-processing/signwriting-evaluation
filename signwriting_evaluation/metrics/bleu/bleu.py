from sacrebleu.metrics import BLEU

from signwriting.tokenizer import SignWritingTokenizer

from signwriting_evaluation.metrics.base import SignWritingMetric, validate_corpus_score_input


class SignWritingBLEU(SignWritingMetric):
    """Wrapper for sacrebleu's BLEU metric with added tokenization."""

    bleu = BLEU(effective_order=True)
    tokenizer = SignWritingTokenizer()

    def __init__(self):
        super().__init__(name="TokenizedBLEU")

    def tokenize(self, text: str) -> str:
        return " ".join(self.tokenizer.text_to_tokens(text))

    def score(self, hypothesis: str, reference: str) -> float:
        hypothesis = self.tokenize(hypothesis)
        reference = self.tokenize(reference)
        return self.bleu.sentence_score(hypothesis, [reference]).score / 100

    def corpus_score(self, hypotheses: list[str], references: list[list[str]]) -> float:
        validate_corpus_score_input(hypotheses, references)
        hypotheses = [self.tokenize(h) for h in hypotheses]
        references = [[self.tokenize(r) for r in reference] for reference in references]
        return self.bleu.corpus_score(hypotheses, references).score / 100
