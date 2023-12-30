import unittest

from signwriting_evaluation.metrics.bleu import SignWritingBLEU


class TestSignWritingBLEU(unittest.TestCase):
    def setUp(self):
        self.metric = SignWritingBLEU()

    def test_score(self):
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 0.126835469)


if __name__ == '__main__':
    unittest.main()
