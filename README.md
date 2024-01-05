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
- ✅ [CLIPScore](signwriting_evaluation/metrics/clipscore.py) - CLIPScore between SignWriting images. (Using the original CLIP model)

## Qualitative Evaluation

It is well-known that the SignBank corpus contains many forms of the sign for "hello".
We carefully select some of these signs to evaluate our metrics, by looking for their closest matches in the corpus,
which contains around 230k single signs.

The problems of each metric are revealed when comparing the top 10 nearest neighbors for each sign.
For each sign and metric, either the first match is incorrect, or there is a more correct match further down the list.

<table style="text-align: center">
<thead>
<tr><td></td><td colspan='3'><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/ref.png' /></td><td colspan='3'><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/ref.png' /></td><td colspan='3'><img src='assets/matches/M520x520S14c20480x484S27106505x480/ref.png' /></td></tr>
<tr><td></td><td>CLIPScore</td><td>TokenizedBLEU</td><td>CHRF</td><td>CLIPScore</td><td>TokenizedBLEU</td><td>CHRF</td><td>CLIPScore</td><td>TokenizedBLEU</td><td>CHRF</td></tr>
</thead>
<tbody>
<tr><td>1</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/0.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/0.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/0.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/0.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/0.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/0.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/0.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/0.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/0.png' /></td></tr>
<tr><td>2</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/1.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/1.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/1.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/1.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/1.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/1.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/1.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/1.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/1.png' /></td></tr>
<tr><td>3</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/2.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/2.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/2.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/2.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/2.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/2.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/2.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/2.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/2.png' /></td></tr>
<tr><td>4</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/3.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/3.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/3.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/3.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/3.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/3.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/3.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/3.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/3.png' /></td></tr>
<tr><td>5</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/4.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/4.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/4.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/4.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/4.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/4.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/4.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/4.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/4.png' /></td></tr>
<tr><td>6</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/5.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/5.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/5.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/5.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/5.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/5.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/5.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/5.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/5.png' /></td></tr>
<tr><td>7</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/6.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/6.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/6.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/6.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/6.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/6.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/6.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/6.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/6.png' /></td></tr>
<tr><td>8</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/7.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/7.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/7.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/7.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/7.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/7.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/7.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/7.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/7.png' /></td></tr>
<tr><td>9</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/8.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/8.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/8.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/8.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/8.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/8.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/8.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/8.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/8.png' /></td></tr>
<tr><td>10</td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CLIPScore/9.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/TokenizedBLEU/9.png' /></td><td><img src='assets/matches/M533x518S2ff00482x483S15a11510x487S26500508x469/CHRF/9.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CLIPScore/9.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/TokenizedBLEU/9.png' /></td><td><img src='assets/matches/M528x557S14c21473x531S2890a499x527S30a00482x482S33e00482x482/CHRF/9.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CLIPScore/9.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/TokenizedBLEU/9.png' /></td><td><img src='assets/matches/M520x520S14c20480x484S27106505x480/CHRF/9.png' /></td></tr>
</tbody>
</table>

## References

[^1]: Amit Moryossef, Zifan Jiang.

2023. [SignBank+: Multilingual Sign Language Translation Dataset](https://arxiv.org/abs/2309.11566).
      [^2]: Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu.
2002. [Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/). In
      Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318,
      Philadelphia,
      Pennsylvania, USA. Association for Computational Linguistics.
      [^3]: Maja Popović.
2015. [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/). In Proceedings
      of the Tenth Workshop on Statistical Machine Translation, pages 392–395, Lisbon, Portugal. Association for
      Computational
      Linguistics.
      [^4]: Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi.
2021. [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://aclanthology.org/2021.emnlp-main.595/).
      In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, Online
      and
      Punta Cana, Dominican Republic. Association for Computational Linguistics.