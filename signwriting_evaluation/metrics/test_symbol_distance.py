import unittest
from signwriting_evaluation.metrics.symbol_distance import SignWritingSimilarityMetric


class TestSignWritingSymbolDistance(unittest.TestCase):
    def setUp(self):
        self.metric = SignWritingSimilarityMetric()

    def test_score(self):
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 0.5509574768254414)

        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M530x538S22f03469x517S37602508x462S20e00488x510S15a11493x494"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 1)

        hypothesis = "M530x538S17600508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M530x538S17600508x462S12a11493x494S20e00488x510S22f13469x517"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.8326259781509948)


if __name__ == '__main__':
    unittest.main()
