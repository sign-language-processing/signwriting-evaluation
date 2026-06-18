# Manual annotation outputs

Durable, version-controlled record of the human annotation collected to calibrate the
`SymbolsDistances` similarity metric (`signwriting_evaluation/metrics/similarity.py`). The collection
harness lives in the git-ignored `calibration/` directory; this folder is where its **outputs** are
kept so they survive a clean checkout and can be reused on future calibration runs.

| file | task | format | feeds |
|---|---|---|---|
| `preferences.jsonl` | query + 2 candidates, pick the closer | `{query, candidates, human}` | preference agreement |
| `same_pairs.jsonl` | two near-identical signs: same / variation / mirror / different | `{sign_a, sign_b, label}` | label-rank, mirror |
| `spanning_pairs.jsonl` | same-symbol, different writing order | `{sign_a, sign_b, label}` | reordering / overlap |
| `implicit_pairs.jsonl` | implicit face / touch judgements | `{sign_a, sign_b, label}` | implicit symbols, touch_penalty |
| `triplet_pairs.jsonl` | confirm the metric's nearest-of-two choice | `{query, candidates, human}` | preference agreement |
| `outlier_pairs.jsonl` | high-confidence / low-score predictions: same / similar / different / mirrored | `{source_sign, predicted_sign, confidence, score, old_score, label}` | low-end diagnosis (§13) |
| `handshape_equivalences.json` | hand wall/floor `{fill}{rotation}` variants that render identically | `{pairs: [[code, code], …]}` | `WALL_FLOOR_EQUIVALENTS` (§11.1) |
| `heel_rotation_map.json` | heel-of-hand ↔ top-view rotation correspondence | `{heel, top, pairs: [[angle, angle], …]}` | `HEEL_TO_TOP` rotations (§11.1b) |

The annotation servers read from and write to this folder (`ANNOTATION_DIR`). Optimizer scratch
(`search_log.jsonl`, `tuned*.json`) stays in `calibration/` — it is regenerated, not hand-made.
