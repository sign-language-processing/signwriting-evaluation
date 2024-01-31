import unittest

import numpy as np
from PIL import Image

from signwriting_evaluation.metrics.clip import SignWritingCLIPScore, signwriting_to_clip_image


class TestSignWritingCLIPScore(unittest.TestCase):
    def setUp(self):
        self.metric = SignWritingCLIPScore(cache_directory=None)

    def test_score(self):
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = "M519x534S37900497x466S3770b497x485S15a51491x501S22f03481x513"
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 0.96, places=2)  # Score is not fully deterministic

    def test_score_image(self):
        hypothesis = "M530x538S37602508x462S15a11493x494S20e00488x510S22f03469x517"
        reference = Image.new("RGB", (5, 50), (255, 255, 255))  # ends up as empty image
        score = self.metric.score(hypothesis, reference)
        self.assertIsInstance(score, float)  # Check if the score is a float
        self.assertAlmostEqual(score, 0.7759, places=2)

    def test_bad_fsw_is_not_an_empty_image(self):
        fsw = "M530x538S37602531x539"
        image = signwriting_to_clip_image(fsw)
        self.assertTrue(np.any(np.array(image) != 255))


if __name__ == '__main__':
    unittest.main()
