# SignWriting Evaluation

The lack of automatic SignWriting evaluation metrics is a major obstacle in the development of
SignWriting transcription and translation[^1] models.

## Goals

The primary objective of this repository is to house a suite of
automatic evaluation metrics specifically tailored for SignWriting.
This includes standard metrics like BLEU[^2], chrF[^3], and CLIPScore[^4],
as well as custom-developed metrics unique to our approach.
We recognize the distinct challenges in evaluating single signs versus continuous signing,
and our methods reflect this differentiation.

To qualitatively demonstrate the efficacy of these evaluation metrics,
we implement a nearest-neighbor search for selected signs from the SignBank corpus.
The rationale is straightforward: the closer the sign is to its nearest neighbor in the corpus,
the more effective the evaluation metric is in capturing the nuances of sign language transcription and translation.

## Evaluation Metrics

- ✅ [Tokenized BLEU](signwriting_evaluation/metrics/bleu.py) - BLEU score for tokenized SignWriting FSW strings.
- ✅ [chrF](signwriting_evaluation/metrics/chrf.py) - chrF score for untokenized SignWriting FSW strings.
- ❌ [CLIPScore](signwriting_evaluation/metrics/clipscore.py) - CLIPScore between SignWriting images.

## References

[^1]: Amit Moryossef, Zifan Jiang.
2023. [SignBank+: Multilingual Sign Language Translation Dataset](https://arxiv.org/abs/2309.11566).
[^2]: Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu.
2002. [Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/). In
Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia,
Pennsylvania, USA. Association for Computational Linguistics.
[^3]: Maja Popović.
2015. [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/). In Proceedings
of the Tenth Workshop on Statistical Machine Translation, pages 392–395, Lisbon, Portugal. Association for Computational
Linguistics.
[^4]: Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi.
2021. [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://aclanthology.org/2021.emnlp-main.595/).
In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, Online and
Punta Cana, Dominican Republic. Association for Computational Linguistics.