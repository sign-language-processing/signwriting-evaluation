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
        self.assertAlmostEqual(score, 0.5555982772844742)

    def test_score_is_symemtric(self):
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        score1 = self.metric.score(hypothesis=hypothesis, reference=reference)
        score2 = self.metric.score(hypothesis=reference, reference=hypothesis)
        self.assertAlmostEqual(score1, score2, msg="The metric is not symmetric")

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
        self.assertAlmostEqual(score, 0.8210067817002714)

    def test_corpus_score(self):
        hypothesis = "M530x538S17600508x462S15a11493x494S20e00488x510S22f03469x517"
        good_reference = "M530x538S17600508x462S12a11493x494S20e00488x510S22f13469x517"
        bad_reference = "M530x538S17600508x462"
        score = self.metric.corpus_score([hypothesis], [[good_reference], [bad_reference]])
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.8210067817002714)

    def test_multi_sign_score(self):
        hypothesis_single = "M530x538S17600508x462S15a11493x494S20e00488x510S22f03469x517"
        hypothesis = f"{hypothesis_single} {hypothesis_single}"
        reference = "M530x538S17600508x462S12a11493x494S20e00488x510S22f13469x517"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.8210067817002714 / 2)

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

    def test_score_is_translation_invariant(self):
        # Same sign shifted in y; only absolute coordinates differ.
        hypothesis = "M518x553S10000500x523S2ff00482x483"
        reference = "M518x532S10000500x502S2ff00482x462"
        score = self.metric.score(hypothesis, reference)
        self.assertAlmostEqual(score, 1)

    def test_score_extra_symbol_keeps_hands_aligned(self):
        # Identical hands; reference adds a neutral face. Hands must still align,
        # leaving only the length penalty for the extra symbol.
        hypothesis = "M540x515S10000525x485S10008460x485"
        reference = "M540x542S10000525x512S10008460x512S2ff00482x483"
        score = self.metric.score(hypothesis, reference)
        self.assertAlmostEqual(score, 0.765625)

    def test_score_swu(self):
        hypothesis = "𝠃𝤤𝤬񎱃𝤎𝣠񂇒𝣿𝤀񆕁𝣺𝤐񇆤𝣧𝤗"
        reference = "𝠃𝤙𝤨񎵡𝤃𝣤񎲬𝤃𝣷񂈒𝣽𝤇񇆤𝣳𝤓"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 0.5555982772844742)

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
