import unittest

from signwriting_evaluation.metrics.similarity import SignWritingSimilarityMetric
from signwriting_evaluation.metrics.bleu import SignWritingBLEU
from signwriting_evaluation.metrics.chrf import SignWritingCHRF


class TestMetricsReturnFloat(unittest.TestCase):
    """Test that all metrics return float values, never integers."""

    def setUp(self):
        self.similarity = SignWritingSimilarityMetric()
        self.bleu = SignWritingBLEU()
        self.chrf = SignWritingCHRF()

    def test_similarity_returns_float(self):
        """Test that similarity metric always returns float."""
        # Test with None values (edge case that was returning int)
        score = self.similarity.score(None, None)
        self.assertIsInstance(score, float, "Similarity score with None inputs should be float")

        # Test with valid FSW strings
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        score = self.similarity.score(hypothesis, reference)
        self.assertIsInstance(score, float, "Similarity score should be float")

        # Test with None hypothesis
        score = self.similarity.score(None, reference)
        self.assertIsInstance(score, float, "Similarity score with None hypothesis should be float")

        # Test with None reference
        score = self.similarity.score(hypothesis, None)
        self.assertIsInstance(score, float, "Similarity score with None reference should be float")

    def test_bleu_returns_float(self):
        """Test that BLEU metric always returns float."""
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        score = self.bleu.score(hypothesis, reference)
        self.assertIsInstance(score, float, "BLEU score should be float")

    def test_chrf_returns_float(self):
        """Test that CHRF metric always returns float."""
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        score = self.chrf.score(hypothesis, reference)
        self.assertIsInstance(score, float, "CHRF score should be float")

    def test_similarity_score_all_returns_float(self):
        """Test that score_all returns list of lists of floats."""
        hypotheses = ["M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"]
        references = ["M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"]
        scores = self.similarity.score_all(hypotheses, references, progress_bar=False)
        for row in scores:
            for score in row:
                self.assertIsInstance(score, float, "Each score in score_all should be float")

    def test_similarity_corpus_score_returns_float(self):
        """Test that corpus_score returns float."""
        hypotheses = ["M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"]
        references = [["M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"]]
        score = self.similarity.corpus_score(hypotheses, references)
        self.assertIsInstance(score, float, "Corpus score should be float")


if __name__ == '__main__':
    unittest.main()
