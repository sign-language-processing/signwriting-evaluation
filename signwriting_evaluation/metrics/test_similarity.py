import unittest

from signwriting_evaluation.metrics.similarity import SignWritingSimilarityMetric


class TestSignWritingSymbolDistance(unittest.TestCase):
    def setUp(self):
        self.metric = SignWritingSimilarityMetric()

    def test_score(self):
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 0.5509574768254414)

    def test_score_jumbled_sign(self):
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M530x538S22f03469x517S37602508x462S20e00488x510S15a11493x494"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 1)

    def test_different_shape(self):
        hypothesis = "M530x538S17600508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M530x538S17600508x462S12a11493x494S20e00488x510S22f13469x517"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.8326259781509948)

    def test_corpus_score(self):
        hypothesis = "M530x538S17600508x462S15a11493x494S20e00488x510S22f03469x517"
        good_reference = "M530x538S17600508x462S12a11493x494S20e00488x510S22f13469x517"
        bad_reference = "M530x538S17600508x462"
        score = self.metric.corpus_score([hypothesis], [[good_reference], [bad_reference]])
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.8326259781509948)

    def test_multi_sign_score(self):
        hypothesis_single = "M530x538S17600508x462S15a11493x494S20e00488x510S22f03469x517"
        hypothesis = f"{hypothesis_single} {hypothesis_single}"
        reference = "M530x538S17600508x462S12a11493x494S20e00488x510S22f13469x517"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.8326259781509948 / 2)

    def test_multi_sign_score_is_order_invariant(self):
        sign_1 = "M530x538S17600508x462S15a11493x494S20e00488x510S22f03469x517"
        sign_2 = "M530x538S17600508x462S12a11493x494S20e00488x510S22f13469x517"
        hypothesis = f"{sign_1} {sign_2}"
        reference = f"{sign_2} {sign_1}"
        score = self.metric.score(hypothesis, reference)
        self.assertAlmostEqual(score, 1)

    def test_bad_fsw_equals_0(self):
        bad_fsw = "M<s><s>M<s>p483"
        score = self.metric.corpus_score([bad_fsw], [[bad_fsw]])
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0)

    def test_score_swu(self):
        hypothesis = "𝠃𝤤𝤬񎱃𝤎𝣠񂇒𝣿𝤀񆕁𝣺𝤐񇆤𝣧𝤗"
        reference = "𝠃𝤙𝤨񎵡𝤃𝣤񎲬𝤃𝣷񂈒𝣽𝤇񇆤𝣳𝤓"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 0.5509574768254414)

    def test_unknown_symbol_class_returns_zero_score(self):
        # Test that symbols with shapes outside defined class ranges are handled gracefully
        # When a symbol's shape doesn't match any defined symbol class ranges, the metric
        # should return maximum distance (resulting in zero similarity score) rather than crashing
        hypothesis = "M530x538S38c00508x462"  # S38c00 has shape 0x38c, outside all defined ranges
        reference = "M530x538S10000508x462"   # S10000 has shape 0x100, in hands_shapes range
        score = self.metric.score(hypothesis, reference)
        self.assertEqual(score, 0)


if __name__ == '__main__':
    unittest.main()
