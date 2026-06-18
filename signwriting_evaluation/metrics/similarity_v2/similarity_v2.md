# SignWriting Similarity Metric v2 (`SymbolsDistancesV2`)

A reference-based similarity between a hypothesis and a reference Formal SignWriting (FSW)
transcription, returning a score in
`[0, 1]` (`1` = identical). Unlike generic string metrics (BLEU, chrF), it is tailored to the rules
of SignWriting: symbols carry typed, weighted meaning; the same sign may be written in different
symbol orders; and layout matters. It is built bottom-up — a distance between **symbols**, lifted to
a distance between **signs** via optimal symbol matching, lifted to a score between **sequences of
signs** via optimal sign matching. An optional **reordering factor** makes it sensitive to symbol
*arrangement*, which is what makes it usable for fingerspelling and mouthing.

Implementation: [`similarity_v2.py`](similarity_v2.py). This is **v2** (`SignWritingSimilarityV2Metric`);
the original published metric is frozen at `signwriting_evaluation.metrics.similarity` for
backward compatibility.

> **Current parameters are data-fit.** All weights/exponents below were jointly optimized against the
> calibration signals — see [§9 "Adopted re-fit"](#9-data-driven-calibration) for the production values
> and results. The constants shown inline in §2–§6 are the original hand-set values that *motivated*
> each mechanism; they remain useful for intuition but are superseded by the fit.
>
> **Name-aware extension.** Beyond the §9 fit, the symbol distance also exploits the ISWA *symbol
> names* — plane equivalences (hand and arrow), size variants, and a **name word-distance that is the
> identity term** (replacing the integer shape code) — see
> [§11](#11-plane-equivalent-and-name-aware-symbol-distance). The `shape` weight now scales the name
> distance; a `size` key and the `SIZE_CANON`/`PLANE_CANON`/`VARIANT_CANONICAL`/`HEEL_TO_TOP`
> canonicalizations were added.
>
> **Current results (production).** Fingerspelling Pearson **0.78**, mouthing **0.59** (vs text
> Levenshtein); contrastive AUC **0.84**; lexical preference **0.86**; label-rank **0.73**;
> model-confidence Spearman **0.42** (original metric 0.32). §9 records the joint re-fit; §11–§13 the
> name-aware identity term and external validation that lifted these from the re-fit baseline.

---

## 1. Symbol model

A SignWriting sign is a set of positioned symbols. Each symbol is an FSW key `S{shape}{facing}{angle}`
plus a 2-D position. We decode (`get_symbol_attributes`):

- `shape` = `int(key[1:4], 16)` — the glyph, in `[0x100, 0x38b]`.
- `facing` = `int(key[4], 16)` — palm orientation, in `[0, 15]`.
- `angle` = `int(key[5], 16)` — rotation, in `[0, 15]`.
- `parallel` = `facing > 2` — a coarse plane/parallelization flag.
- `position` = `(x, y)` — layout coordinates (origin ~`(500, 500)`).

Shapes are grouped into **classes** (`SYMBOL_CLASSES`): hands, contact, movement paths, a unified head/face class (head movement + facial
expressions), and etc. `class(shape)` is the index of the class containing `shape`,
or `None` if it falls outside every defined range.

---

## 2. Symbol-to-symbol distance

The cost of a pair splits into an **identity** term and a **position** term with *different curvature*
(`calculate_distance` for identity, `symbols_score` for the combination), with weights `w_*` from
`ERROR_WEIGHT`:

```
d_id(a, b)   = sqrt( (w_shape·N(a,b))² + (w_facing·Δfacing)² + (w_angle·Δangle)² + (w_par·[par_a≠par_b])² + (w_size·Z(a,b))² )
             + w_class · |class(a) − class(b)|
cost(a, b)   = ( d_id(a, b) / max_distance ) ^ (1 / 2.5)        # identity: concave (spreads small diffs)
             + ( ‖pos_a − pos_b‖₂ / position_scale )²           # position: convex (jitter ≈ 0)
```

(clamped to `[0, 1]`). The identity term is **`N(a,b)`, the word-level Levenshtein between the two
symbols' ISWA names** (§11.4) — *not* the integer shape-code difference, which is a poor proxy for
similarity (adjacent codes need not be similar symbols). `Z(a,b)` is the size difference (§11.3); both
`N` and `Z` are `0` for symbols that canonicalize to the same shape. Facing matters more than angle;
crossing a **symbol class** boundary (e.g. a hand vs a face) is heavily penalized (`w_class`).
`max_distance = d_id( S10000, S38b07 )` is a fixed worst-case identity reference. If a shape is outside
all classes, the identity term returns `max_distance`.

**Why identity and position have opposite curvature.** Earlier the position distance was *added to* the
identity distance and the sum was put through the single `^(1/2.5)` (5th-root) normalization. That root
is concave — it *inflates* tiny inputs — so a few px of layout jitter (the same sign written slightly
differently) became a ~0.2 cost, and a genuinely identical full sign scored ~0.69 instead of ~1.0
(observed on the calibration `hello` queries: two near-identical references scored 0.567 and 0.673).
Raising the *global* exponent toward squared fixes the near-matches but **destroys** fingerspelling /
mouthing (the spelling correlation needs the concave root to keep small *identity* differences
separable). The resolution is to treat the two separately: keep the concave root for **identity** (the
spelling signal), and make **position** convex — `(‖Δpos‖ / position_scale)²` — so sub-symbol jitter
costs ≈ 0 while a real relocation costs a lot. Measured impact (val, vs the lumped-root baseline):

| | lumped root (old) | separated (`position_scale = 90`) |
|---|---|---|
| Fingerspelling Spearman | 0.688 | **0.778** |
| Mouthing Spearman | 0.463 | **0.520** |
| Contrastive AUC | 0.836 | **0.845** |
| Combined `J` | 3.557 | **3.752** |
| near-identical `hello` ref | 0.673 | **0.997** |

Fingerspelling/mouthing *rise* because position jitter was previously polluting the identity signal
through the shared root; isolating it cleans both up. `position_scale` was swept (peak `J` at ≈90;
larger values forgive position more, smaller penalize it harder). Sweep in
`calibration/experiment_separated.py`.

---

## 3. Sign-to-sign matching (`error_rate`)

Two signs may list their symbols in any order and sit anywhere on the canvas. We want a distance
invariant to **writing order** and to **global translation**.

1. **Center** both signs on their centroid, so an extra/unmatched symbol can't drag the layout.
2. **Assign** symbols by minimizing total cost — the Hungarian algorithm
   (`linear_sum_assignment`) over the `cost(·,·)` matrix (`assignment`). This is an *optimal set
   matching*: order-independent.
3. **Re-align** on the matched pairs only: shift `hyp` by `mean(matched_hyp) − mean(matched_ref)`,
   then re-score. Two signs differing only by a translation now match perfectly.
4. **Mean matched cost** `c̄ = mean over matched pairs of cost(·,·)`.
5. **Length penalty** for differing symbol counts:

```
ℓ = |len_hyp − len_ref| / (max(len_hyp, len_ref) + 1)        # length_acc
λ = ℓ ^ 1.5                                                  # exp_factor = 1.5
error_rate = λ + c̄ · (1 − λ)
```

So `error_rate` blends "how many symbols are unmatched" (`λ`) with "how well the matched ones agree"
(`c̄`); when lengths are equal, `λ = 0` and `error_rate = c̄`.

The base per-sign similarity is then

```
sim_base(hyp, ref) = (1 − error_rate)²
```

(`score_single_sign`, before the reordering factor below). The `²` sharpens separation at the top.

### Implicit symbols (optional faces and touches)

Some symbols are routinely left unmarked, so a sign that omits one is not actually different from one
that draws it. `error_rate`'s length penalty should not punish such an omission. We model three
implicit symbols (`implicit_classes`, default on via the `implicit` flag):

1. **Implicit face (materialized).** If a sign draws no head/face and is compared against one that does,
   we materialize the missing face as a *real positioned symbol* (`add_implicit_face`): a head circle
   `S2ff00` at the canonical `(482, 482)`, with the rest of the sign shifted **down** by the face height
   + a small gap, reproducing the usual "head on top, hands below" layout. Both signs then carry a head
   anchor, so the matcher computes *positional* distances correctly instead of merely forgiving the
   missing face as a free leftover (which left near-identical face/no-face signs scoring ~0.69, and let
   a degenerate lone arrow look like a near-match). It is **conditional**: applied only when the other
   side has a face, so two face-less signs (e.g. fingerspelling) are untouched and never gain a free,
   dilutive face↔face match. Measured (val, vs. forgiving the face as a leftover): mean ideal-ranking
   Kendall τ **+0.29 → +0.37**, preference **0.88 → 0.89**, J **3.752 → 3.753**, and a lone arrow vs. a
   full face+hand+arrow sign drops **0.79 → 0.14** (it can no longer masquerade as a near-match).

   *Open issue — the head-circle privilege.* Because the materialized face is always a head circle, a
   face-less sign gets a *perfect* head-circle match against a head-circle sign, so e.g. hand+arrow can
   score closer to hand+head-circle than hand+mouth does (a head circle "should" prefer a facial
   expression over an arrow). Skipped in `test_head_and_facial_are_one_class` pending a fix.
2. **Implicit head-rim touch** — each head-rim symbol (`S300`–`S309`) implies a `contact`-class touch
   at the rim that is often not drawn.
3. **Implicit contact touch** — where a **hand's inked glyph overlaps** another **hand or a face**, a
   `contact`-class touch is implied. Restricted to hand-involved contacts: overlaps between facial
   expressions, or between movement symbols, are normal layout, not unmarked touches.

   *Touch detection is pixel-accurate (`symbols_share_ink`, `symbol_ink_mask`).* The earlier test used
   a bounding-box gap (`≤ 1px`), but two glyph **boxes** can intersect while the **ink** does not — a
   hand drawn beside a head circle would register a phantom touch purely from box proximity (observed
   on a calibration query). We now do the cheap bounding-box overlap test first (disjoint boxes can
   never share ink) and, only when boxes overlap, AND the two symbols' 1-bit ink masks over the overlap
   region — a real touch requires a genuinely shared painted pixel. The masks are rendered from the
   SignWriting line+fill fonts (mirrors signwriting's `canonicalize._symbol_mask`, replicated locally
   because that module is not yet on PyPI). Legitimate hand↔face contacts (overlapping ink) are kept;
   box-only near-misses are no longer counted.

**How touches are applied — leftover explanation, not extra matching.** The face is materialized as a
real symbol (above); the two **touches** are not. Adding a *zero-cost* touch as a matchable symbol would
let it *steal* a real symbol away from a genuine real↔real match. Instead we match real↔real exactly as
in §3, and only then look at the **leftover (unmatched) real symbols**: a leftover is *explained* (not
counted in the length penalty) if the other sign has an implicit touch of the **same class** to spare
(`implicit_length_acc`, `unexplained_penalty`). Each implicit touch explains at most one leftover. This
is symmetric and cannot disturb the real matching or the matched-cost term `c̄`. (The face is handled by
materialization rather than this leftover path, so a missing face is supplied with correct position
instead of merely waived.)

**Unexplained touches weigh more (`touch_penalty`).** A leftover that is a `contact` (touch) symbol
counts as `touch_penalty` units in the length penalty, not 1. A touch denotes an *interaction* the
other sign lacks — a lone hand cannot touch anything — so its absence is a real difference, not a
cosmetic extra symbol. Data-driven: of 161 labels on the implicit task, **97 were single-hand vs.
has-touch pairs, all labelled "different", yet all scored 0.766** (the implicit face explained the
face, leaving only a mild 1-symbol length penalty for the touch). Sweeping `touch_penalty` against the
labels, **2.0** is the balance — the "different" group's mean score drops 0.72 → 0.45 (clearly below
the "variation" mean of 0.88) while same stays 0.96, fingerspelling/preference are unchanged, and
mouthing barely moves; 3.0 starts to hurt mouthing and preference. The penalty is smooth: a lone hand
vs hand+face+touch scores ≈ 0.42 (the matching hand still earns credit) rather than 0.

Example: hands-only `M540x515S10000525x485S10008460x485` vs hands+face
`M540x542S10000525x512S10008460x512S2ff00482x483` scored `0.766` (length penalty for the extra face)
and now scores `1.0` — the face is implicit in the first sign.

**Status — implemented, default on, not yet validated by data.** On the current signals: fingerspelling
Pearson unchanged (0.543 → 0.543), mouthing unchanged (0.405 → 0.405, after excluding facial
expressions), but lexical preference drops 89.1% → 86.1% (≈3 cases). The drop is real but the relevant
test cases are under-sampled: our preference/same pairs are blocked by shape signature, so a
"face present vs absent" pair (different shape sets) is rarely even surfaced — the few that are seem to
be cases where the human *does* treat the optional symbol as meaningful. The mechanism is clearly
correct on its target case (the example above), but whether to keep it on is unresolved: it needs
targeted annotation (see §10) of optional-symbol-omission pairs — "is a sign with an explicit face the
same as the same sign without one?" Set `implicit=False` to disable.

---

## 4. The reordering factor — supporting fingerspelling & mouthing

### The problem

Step 3 matches symbols as a **set**, so it is blind to their *arrangement*. For lexical signs that
is mostly fine. But **fingerspelling** ("a-b-c", spelled as a vertical stack of letter handshapes)
and **mouthing** (a horizontal sequence of mouth shapes derived from a word's phonemes) are
*sequences* — order is meaning. The set matcher scores "abc" vs "cba" ≈ 1, because the same letters
are present. `d_pos` (weight `1/10`, and partly removed by the re-alignment in step 3) is too weak to
fix this.

We confirmed this empirically: against text **Levenshtein** similarity — the right ground truth for
spelling/mouthing, see [`../../evaluation/text_correlation.py`](../../evaluation/text_correlation.py) —
the set-only metric correlated only weakly (fingerspelling Pearson ≈ 0.35).

### The fix: rank-order inversions along the principal axis (`reordering_factor`)

After matching, we measure whether the matched symbols keep the same **relative order**:

1. Center the matched positions of each sign.
2. For each sign, take its **principal axis** `u` (first right singular vector of its centered
   matched positions) — the dominant writing direction (vertical for fingerspelling, horizontal for
   mouthing).
3. Project matched symbols onto `u` and count **discordant pairs** — matched pairs `(i, j)` whose
   order flips between hyp and ref — each comparison **weighted by match quality** `q = 1 − cost` of
   the two matched pairs involved:

```
disc(u) = Σ_{i<j} qᵢ·qⱼ·[ sign(uᵀ(hᵢ−hⱼ)) ≠ sign(uᵀ(rᵢ−rⱼ)) ]  /  Σ_{i<j} qᵢ·qⱼ   (over pairs that move)
```

   Axis orientation cancels (flipping `u` negates both sides), so `disc` is sign-invariant.
4. Average over both signs' axes (`u_hyp` and `u_ref`) so the factor stays **symmetric**.
5. The factor multiplies the score:

```
reorder(hyp, ref) = exp( −β · mean(disc(u_hyp), disc(u_ref)) ),     β = reordering_weight
sim(hyp, ref)     = sim_base(hyp, ref) · reorder(hyp, ref)
```

**Why quality-weighting.** Without it, a pair like fingerspelled *Veronica* vs *Vororich* (62%
text-similar) was crushed to 0.35: the *shared* letters all match in order, but the two *substituted*
letters have no natural counterpart and the matcher pairs them with distant hands, creating crossings
that fired the penalty. Those arbitrary matches are low-quality (very different hands → `q≈0`), so
weighting each comparison by `qᵢ·qⱼ` lets only confidently-matched symbols' order count. This lifts
*Veronica/Vororich*'s reorder factor 0.46 → 0.57 (and *Michelle/Mtichell* 0.65 → 0.80) while a genuine
reversal ("abc"↔"cba", all matches high-quality and crossed) stays fully penalized at 0.14. Measured:
fingerspelling Pearson 0.544 → 0.566, Spearman 0.378 → 0.406; mouthing, lexical preference, and the
same/variation/different bands unchanged.

### Why this lever and not others

Why **rank order along the principal axis** rather than comparing raw displacement vectors or
pairwise distances?

- Pairwise **scalar distances** are reflection-invariant — a reversal ("abc"↔"cba") preserves them
  exactly, so they cannot detect it.
- Raw **displacement vectors** detect it, but also penalize the small position **jitter** present in
  legitimate lexical variants — which hurts agreement with human judgement.
- **Rank order** is the sweet spot: jitter almost never flips the order of two symbols, but a true
  reordering flips it completely. It is invariant to translation, scale, and small noise.

This was selected by trying ten formulations and scoring each against both objectives
([`../../../calibration/REPORT.md`](../../../calibration/REPORT.md)). Result — fingerspelling /
mouthing correlation vs. lexical human-agreement (from 101 collected preferences):

| metric | FS Pearson | MO Pearson | lexical agreement |
| --- | --- | --- | --- |
| set-only (`β = 0`) | 0.354 | 0.179 | 92.1% |
| reordering (`β = 2`) | 0.540 | 0.395 | 90.1% |

i.e. the spelling/mouthing correlation ~doubles while lexical human-agreement barely moves.

### Configuration

`reordering_weight` (β) defaults to **2** in production — the arrangement-aware behavior is now the
standard metric. Set **β = 0** to recover the legacy set-only behavior (no reordering penalty).
Larger β increases spelling sensitivity but slowly erodes lexical agreement; β = 2 was chosen as the
data-driven balance (Section 9).

---

## 5. The overlap-order factor — z-order of overlapping symbols

### The problem

The set matcher (Section 3) and the reordering factor (Section 4) both ignore **writing order** — the
order symbols are listed, which is the order they are *drawn*. For non-overlapping symbols that is
correct: draw order is invisible. But when two symbols **overlap**, the later-drawn one occludes the
earlier, so a different draw order produces a different rendered glyph. Two signs with the same
symbols at the same positions but a swapped draw order of overlapping symbols are *not* the same sign,
yet `sim_base · reorder` scores them 1.0 (identical positions → no rank inversion). This surfaced in
calibration: e.g. the word "sala" had a top pair scoring 1.0 that was just such a swap.

### The fix (`overlap_order_factor`)

1. Take the symbol assignment (Section 3). For every matched **pair** of pairs, measure how *visible*
   their draw order is: `draw_order_change_fraction` renders the two glyphs in both orders and returns
   the fraction of pixels that change **color** when swapped (0 if they share no ink, or paint the same
   color where they do — two black lines crossing render identically either way). A flip is only
   meaningful where the pair **visibly occludes in BOTH signs**, so the pair weight is the *minimum* of
   its two occlusion fractions (`pair_occlusion`); below `OVERLAP_VISIBLE_MIN` (= 2%) it is treated as
   cosmetic and ignored.
2. Among occluding matched pairs, penalize the occlusion-weighted fraction whose **draw order is
   inverted** between hyp and ref:

```
overlap(hyp, ref) = exp( −γ · Σ(occlusionₚ · invertedₚ) / Σ occlusionₚ ),     γ = overlap_weight
sim(hyp, ref)     = sim_base · reorder · overlap
```

A pair the layout pulled apart in one sign (no overlap there → min occlusion 0) never contributes —
that is a *positional* difference, already scored by `error_rate`, not a z-order one. This matters for
the materialized implicit face (§3): the face is shifted clear of the other symbols, so it raises no
spurious draw-order penalty against a sign that does overlap its face.

### Why this formulation / weight

- **Why pixel color-change, not box overlap**: the earlier test used bounding-box intersection
  (`≥ 0.2` of the smaller box). But boxes can intersect while the ink does not, and even overlapping ink
  is invisible if both paint the same color — only a *fill covering a line* (or similar) actually
  changes the rendered glyph. Rendering both orders and counting changed pixels measures exactly that;
  the 2% floor drops cosmetic slivers (a hand clipping the edge of a head circle changed ~1.6% of pixels
  and was being penalized spuriously).
- **Why the minimum across both signs**: a draw-order flip is only a z-order difference where *both*
  signs actually occlude that pair; if only one does, the appearance differs because of position, which
  `error_rate` already captures.
- **Weight `γ = 0.25` (default)**: chosen data-driven. The penalty is a fraction in `[0, 1]`, so γ sets
  how hard a fully-inverted overlap is punished. A sweep over the three signals showed γ = 0.25 is the
  balance point — fingerspelling correlation is untouched (Pearson 0.540 → 0.543), mouthing improves
  (0.395 → 0.405), lexical agreement is nearly flat (90.1% → 89.1%), and a single wrong overlap (the
  "sala" pair) is penalized gently to ≈ 0.88 rather than crushed. Larger γ (1–2) over-penalized: a
  lone wrong overlap fell to 0.6–0.37 and both fingerspelling correlation and lexical agreement
  regressed. Set `overlap_weight = 0` to disable.

---

## 6. The mirror factor — a sign and its horizontal mirror are related

A horizontal mirror flips every symbol's facing and x-position, so the direct score between a sign
and its mirror can be very low (the reordering factor alone reverses the x-axis order). But a mirror
is *related*, not unrelated — often the same sign produced with the other hand. It should be
penalized, but mildly.

So `score` also scores the **mirrored reference** (`mirror_fsw`, via the library's `mirror_sign`,
applied per sign for sequences) and credits it at a fixed fraction:

```
score(hyp, ref) = max( directed_score(hyp, ref),  mirror_penalty · directed_score(hyp, mirror(ref)) )
```

`directed_score` is the §3–5 score; `score` is this wrapper. For an **exact** mirror, mirroring the
reference un-mirrors it, so `directed_score(hyp, mirror(ref)) = 1` and the pair scores exactly
`mirror_penalty`. The `max` means non-mirrors are unaffected (their discounted mirror term stays below
the direct score). Mirroring only the reference stays **symmetric** because the metric is invariant to
mirroring both signs, so `directed_score(a, mirror(b)) = directed_score(mirror(a), b)`.

**`mirror_penalty = 0.75`** (set to 0 to disable). Chosen directly: an exact mirror should land around
0.75, not the ~0.1–0.25 the direct score gives. Verified on the labelled data — the heavily-penalized
mirror pair ("m'attire") rose from a direct **0.20 to 0.62** (a *rough* mirror, so below the 0.75
exact-mirror ceiling), while already-high near-symmetric mirror pairs were unchanged. On the signals
it is a small trade: lexical preference improved 86.1% → 88.1% (≈2 cases), fingerspelling unchanged,
mouthing dipped 0.405 → 0.376 (mouth shapes are near mirror-symmetric, so the mirror term slightly
inflates some dissimilar pairs).

---

## 7. Sequence-of-signs scoring (`directed_score`)

Inputs may be multiple signs (a phrase). `text_to_signs` normalizes (SWU→FSW, tokenization) and
splits into signs.

- **Single vs single** → `score_single_sign` (Sections 3–4).
- **Otherwise**: pad the shorter side with `None`, build the full `score_single_sign` matrix between
  hyp-signs and ref-signs, run Hungarian on `1 − matrix`, take the **mean** matched score, and
  multiply by a **sign-reordering factor** (below).
- `None` / unparseable input → `0`.

**Sign-reordering factor (`sequence_reordering_factor`).** Matching signs as a set would make the
sequence score fully order-invariant, but sign order carries meaning (it is the order of the
utterance), so we penalize it — gently, since the *content* still matches. Over the matched signs we
compute the quality-weighted fraction `δ` of pairs whose sequence order flips (the same
`discordant_fraction` used for symbols, along the 1-D sequence index, weighted by each match's score
so a padding/weak match barely counts), and multiply the mean by `exp(−sequence_weight · δ)`. The
default `sequence_weight = 0.2` is deliberately small: a full reversal scores `e^{−0.2} ≈ 0.82`
(a ~18% penalty), an in-order sequence is unchanged, and single-sign scores — hence every calibration
signal — are untouched. This is the sign-level analogue of the symbol reordering factor (§4), but much
gentler.

---

## 8. Properties

- **Range** `[0, 1]`; `1` iff identical up to writing order + translation (+ symbol reordering when
  `β = 0`).
- **Symmetric** (`SYMMETRIC = True`): `score(a, b) = score(b, a)`, including the reordering factor.
- **Invariances**: writing order of symbols, global translation. With `β > 0`, deliberately *not*
  invariant to reordering/reflection along the writing axis.
- Corpus/aggregate scoring via `score_all`, `score_self`, `corpus_score` in [`base.py`](base.py).

---

## 9. Data-driven calibration

### Adopted re-fit (production parameters)

All 14 free parameters are jointly optimized (random search over a known-good neighbourhood, guarded so
no signal drops below the prior operating point) against a blended objective of **six** signals:
contrastive AUC (same-meaning > random), fingerspelling/mouthing Spearman, preference agreement,
label-rank, and a **tier-aware ranking** agreement with human cluster rankings (cross-tier pairwise
concordance; see [`../../../calibration/objective.py`](../../../calibration/objective.py), `optimize.py`).
Adopted production values:

| parameter | adopted | parameter | adopted |
|---|---|---|---|
| `shape` | 487.83 | `exp_factor` | 1.287 |
| `facing` | 1.636 | `facial_scale` | 0.910 |
| `angle` | 0.144 | `reordering_weight` | 1.277 |
| `parallel` | 3.004 | `overlap_weight` | 0.272 |
| `position_scale` | 80.40 | `touch_penalty` | 2.299 |
| `class_penalty` | 69.51 | `mirror_penalty` | 0.424 |
| `normalized_factor` | 0.159 | `movement_weight` | 0.506 |

This point was re-fit after collecting human **cluster rankings** (15 query signs, each grouped into
ordered tiers down to a "junk" tier). No change improves the
ranking agreement *without* lowering fingerspelling or mouthing — the operating point is Pareto-optimal
in that sense — so the adopted point trades a small amount (≈0.01 each) of fingerspelling, contrastive
AUC, preference and label-rank for clear gains in **ranking agreement** (0.67 → 0.71) and **mouthing**
(0.51 → 0.54). The notable shift is `exp_factor 1.59 → 1.29`: a steeper length penalty, so a sign that
omits half its symbols is penalized harder (a half-content match drops from 0.59 to 0.48), which the
cluster tiers reward. `facial_scale 0.91` gently pulls two facial expressions closer than the full
hand-vs-arrow distance. The harness is retained so the next batch of cluster rankings can re-fit.

### The signals

The original constants here (`ERROR_WEIGHT`, the exponents, β) were **hand-tuned before we had
data** — chosen for plausibility, not fit to evidence. We are now iterating toward a new version by
making every change earn its place against measured signals. A change is kept only if it improves a
target objective *without* regressing the others. The reordering factor (Section 4) was the first
result of this loop; β = 2 is a fit value, not a guess.

We deliberately use **three complementary signals**, because no single one covers the space:

1. **Text Levenshtein correlation — automatic, for fingerspelling & mouthing.**
   Spelling and mouthing have an objective ground truth: the edit distance of the underlying text /
   IPA. We sample words (names, via Faker), render them to SignWriting (fingerspelling and mouthing),
   and measure how well the metric correlates with normalized Levenshtein similarity over many pairs
   — Pearson/Spearman, plus a binned trend. No human labelling needed, so this is cheap to re-run on
   every change. See [`../../evaluation/text_correlation.py`](../../evaluation/text_correlation.py).

2. **Human pairwise preferences — for lexical signs.**
   Lexical similarity has no formula, so we collect judgement. For a query sign we surface its two
   nearest candidates (same spoken-language word, same SignBank puddle) and a human picks which is
   closer; we then measure how often the metric's ranking **agrees** with the human. Trials alternate
   "hard" (the metric's top-2) and "easy" (top-1 vs. a random variant). Stored as
   `assets/annotation/preferences.jsonl`.

3. **"Same / variation / different" labels — score-band calibration.**
   To pin down *what score means*, we surface the highest-scoring within-word pairs (after filtering
   exact/shift/reorder duplicates that are 1.0 by construction) and a human tags each
   "basically the same" / "slight variation" / "not the same". The metric should map these to
   descending score bands (≈1 / high / mid). Stored as `assets/annotation/same_pairs.jsonl`.

Collection runs through a small local web UI (the `calibration/` harness, not version-controlled — it
is not part of the library). Its **outputs** are kept in `assets/annotation/` (version-controlled; see
that folder's README) so the human work survives and future runs reuse it. Signals 2 and 3 grow over
time; signal 1 is regenerable on demand.

**The loop** (mirrors a benchmark optimization cycle): propose one change → score it on all three
signals → keep if it Pareto-improves, revert otherwise → record the result. The ten-way comparison
that selected the reordering formulation is written up in
[`../../../calibration/REPORT.md`](../../../calibration/REPORT.md). Guardrails: tune on samples but
re-check on fresh draws to avoid overfitting (label sets are still small — a handful of cases can move
a percentage point), and never trade away correctness for a metric bump.

The end state is a metric whose free parameters are **fit to these signals** rather than hand-set,
with the data and harness retained so future changes remain measurable.

## 10. Edge cases & planned improvements

This metric was hand-tuned before much data existed; we are now calibrating it against (a) text
Levenshtein for spelling/mouthing and (b) collected human judgements for lexical signs. Open items:

- **Overlapping symbols**: when symbols overlap, draw order can change the rendered glyph, but the
  set matcher (and rank order) ignore it. Needs a rendering-aware treatment.
- **Class-relative normalization**: `cost` normalizes by a single global `max_distance`; within-class
  differences (hand vs hand) use only a sliver of `[0, 1]`. A class-local scale could sharpen
  discrimination.
- **Data-driven re-fit**: the constants in `ERROR_WEIGHT` and the exponents (`1/2.5`, `1.5`, the
  final `²`, and `β`) are hand-set; they can be fit to maximize correlation/agreement on held-out
  data. A strong, *automatic* signal for this is **contrastive**: two signs with the same meaning
  (same puddle + text) score well above two random signs (measured AUC ≈ 0.79), giving unlimited
  weak supervision to jointly tune all free parameters alongside the human labels and the
  fingerspelling/mouthing correlation.
- **Direction (movement) penalty** — *implemented but disabled* (`movement_weight`, default 0). A
  per-symbol rotation/facing penalty (`direction_factor`) was added to catch arrow-direction flips.
  Swept against all signals it lowered the flip pairs as intended but **regressed lexical preference
  88% → 77% and "same" 0.91 → 0.75**, because rotation jitter pervades genuine variants — so it is
  kept off. It remains a tunable knob for the joint fit, which can weigh it against the harm.
- **Reordering on short signs**: the factor needs ≥2 matched symbols, so very short signs get no
  reordering signal.

---

## 11. Plane-equivalent and name-aware symbol distance

The raw symbol distance (§2) treats a symbol as `(shape, facing, angle, parallel)` integers and
measures `Δshape` as a raw code subtraction. Two refinements add *semantic* knowledge the integer
codes lack: (a) **plane equivalence** — SignWriting writes the same hand/arrow differently depending
on whether it lies in the wall plane (facing the reader) or the floor plane (seen from above), and at
the rotations where the planes intersect the two glyphs coincide; (b) **the ISWA symbol name** — each
base shape has a human name (e.g. `S22a` = "Single Straight Movement, Wall Plane Small") that encodes
identity, plane, and size far more meaningfully than the code's numeric neighborhood.

Names come from `metrics/similarity/base_symbol_names.json` (652 base symbols), extracted from the
Lessons-in-SignWriting ISWA 2010 listing. All four mechanisms below are driven by it.

### 11.0 Data provenance and the SignWriting plane model

**Provenance.** The International SignWriting Alphabet 2010 (ISWA 2010) organizes the script into 7
categories, 30 groups, and 652 named base symbols; the full glyph set (≈37,800) is each base symbol
crossed with up to 6 *fills* and 16 *rotations*, encoded in a key as `S{shape}{fill}{rotation}`
(`shape` = 3 hex, `fill`/`rotation` = 1 hex each; the code calls these `facing`/`angle`). We take the
base-symbol names from the Lessons-in-SignWriting project's `baseSymbolNames.ts`. They agree with the
official **Unicode** character names in the Sutton SignWriting block (U+1D800–U+1DAAF), which
independently encode plane and size — e.g. `U+1D998 SIGNWRITING MOVEMENT-WALLPLANE LOOP SMALL DOUBLE`
— giving an authoritative cross-check for the paper. (The expanded per-fill/rotation glyphs at
U+40001+ are *not* individually named, so the base-symbol name is the finest naming available.)

**The plane model.** SignWriting situates a sign in two reference planes. The **wall plane** is
parallel to the front wall — what an observer sees looking straight at the signer; movement in it goes
up/down/left/right within the picture. The **floor plane** is parallel to the floor — a bird's-eye
view from above; movement in it goes toward/away from the signer. The same hand or movement is drawn
differently in each plane (a different fill for hands, a different base symbol for arrows). Crucially,
where a hand *points* — or a straight movement *travels* — **along the line where the two planes
intersect**, its 2D projection is identical in both planes, so the two glyphs coincide. That shared
line of intersection is what every equivalence in this section exploits: it falls at rotations
**2, 6, a, e** for hands and at **2, 6** for straight arrows. This is a geometric identity, not a
stylistic preference — the glyphs are pixel-for-pixel the same — which is why collapsing them is
"free" (no signal regressed; see §11.1, §11.5).

### 11.1 Hand wall/floor variants (`VARIANT_CANONICAL`)

**Motivation.** For a hand `S{shape}{facing}{angle}`, the *facing* digit encodes the plane. At the
plane-intersection angles **2, 6, a, e**, certain facings render the identical hand — e.g. facing 0
and facing 4 at angle 2. These were marked by hand in a 6×4 grid annotator (`calibration/handshapes.py`,
fills 0–5 × rotations 2,6,a,e for shape `S100`), producing 8 equivalence pairs
(`assets/annotation/handshape_equivalences.json`): at angles {2, a} the floor facing = wall facing + 4
(0≡4, 1≡5); at angles {6, e} it is + 2 (1≡3, 2≡4).

**Formulation.** In `get_symbol_attributes`, for hand-class shapes only, canonicalize the equivalent
`(facing, angle)` to one representative before computing the distance, so `Δfacing = Δangle = 0` for
partners (and `parallel` agrees). Other classes are untouched (their digits 4–5 are not a hand plane).

**Evidence.** A targeted correctness fix: fingerspelling/mouthing correlations were **unchanged**
(the alphabet rarely uses these specific variants), and all other signals held (AUC 0.796, pref 0.852,
labrank 0.736). Model-confidence correlation nudged up (Pearson 0.383 → 0.385). It costs nothing and
fixes a class of false negatives confirmed by a native writer's domain knowledge.

### 11.1b Heel-of-hand vs top view (`HEEL_TO_TOP`)

**Motivation.** A flat hand or fist whose fingers/knuckles point straight forward (arm parallel to the
floor plane) can be written two equivalent ways — the **Heel of Hand "wrist view"** or the traditional
**top view** — for the same physical handshape. The ISWA assigns these *different* symbol keys, so the
metric would otherwise treat the two notations as different hands. The Lessons-in-SignWriting "Heel of
Hand or Top View?" section enumerates the **7** such symbols (5 flat hands + 2 fists).

**Formulation.** The 7 shapes pair heel `facing 1` → top `facing 5` with a near-adjacent shape code:
`15c→15a, 15e→15d, 14d→14c, 151→150, 14f→14e, 204→203, 1f6→1f5`. The two viewpoints' **rotations also
correspond**: mapped by hand over all 16 rotations in a two-column annotator
(`calibration/heel_rotations.py`, `S15c1X` vs `S15a5X`), the relationship is a **half-turn within each
8-rotation plane group** — top angle = (heel angle + 4) mod 8, separately for rotations 0–7 and 8–15
(`HEEL_TOP_ROTATIONS`). `HEEL_TO_TOP` is the cross product of the 7 shape pairs × 16 rotations = **112
entries**, mapping each heel `(shape, 1, angle)` to its top-view triple, applied first in
`get_symbol_attributes` (directional: heel → top, so both notations canonicalize to the top view).

**Evidence.** Like §11.1, a targeted correctness fix from a native-writer source: a heel-written sign
now scores 1.0 against its top-view spelling **at every rotation**, and all calibration signals are
unchanged (these symbols are rare in the corpora). Keyed on the exact full symbol, so nothing else is
affected.

### 11.2 Arrow wall/floor plane equivalence (`PLANE_CANON`)

**Motivation.** The same idea for movement arrows, but here the plane is a *different base symbol*
(different name, not a facing digit): `S22a` "Single Straight Movement, Wall Plane Small" vs `S265`
"…Floor Plane Small". When their names differ **only** in the plane word and the symbol's angle is at
a plane intersection (**2 or 6** for arrows), they are the same arrow.

**Formulation.** Group movement-class shapes by their plane-and-size-stripped name; shapes that have
both a Wall and a Floor member are plane partners, mapped to one representative. Applied in
`get_symbol_attributes` only when `angle ∈ {2, 6}`. Built over the size representatives (§11.3) so it
composes with size collapse. At non-intersection angles the partners stay distinct (penalized by the
ordinary `Δshape`).

### 11.3 Size variants (`SIZE_CANON` + the `size` term)

**Motivation.** Names differing only in a trailing size word — **Small < Medium < Large < Largest** —
are the same base symbol (`S22a`…`S22d`); people simply write a sign a little bigger or smaller. This
deserves a *small* penalty, not the large `Δshape` the consecutive codes would otherwise incur.

**Formulation.** Collapse all size variants of a `(name, plane)` family to the smallest-size
representative (so `Δshape = 0`) and carry a `size_index ∈ {0,1,2,3}`. Add a new distance term, gated
to symbols that are otherwise the same base symbol (equal canonical shape):

```
d_size = w_size · |size_index_a − size_index_b|      (only when shape_a == shape_b)
```

**Constant & tuning.** `w_size = 0.04`. Because the `^(1/normalized_factor)` step (§2) is steep, even
this tiny weight yields a perceptible-but-gentle penalty: with a shared hand anchor, a Δ1 size step
scores **0.90** and a Δ3 step **0.88**. Larger values over-penalize a cosmetic difference; `0.04` was
chosen for that "barely there" feel and can be folded into the joint fit.

### 11.4 Name word-distance — the identity term (replaces the integer code)

**Motivation.** The integer `Δshape` is a poor proxy for similarity — adjacent codes are not
necessarily similar symbols. The name is. "Index Bent on Circle" (`S107`) shares more of its words
with "Index on Circle" (`S101`, one word inserted → 0.25 word-distance) than with "Index Bent on Fist
Thumb Under" (`S108`, 0.50). The metric therefore uses the name word-distance as the **identity term
in §2, in place of the integer code difference** (the additive bias it started as was a stepping
stone; making it primary is the principled form).

**Formulation.** `name_word_distance(a, b)` is the token-level Levenshtein between the two symbols'
**full** ISWA names, normalized to `[0, 1]`, and `d_shape = w_shape · name_word_distance`:

```
d_shape = w_shape · name_word_distance(shape_a, shape_b)      (0 when canonical shapes are equal)
```

The **full** name (plane/size words kept) is used, not the stripped core: canonicalization
(§11.1–11.3) already collapses every truly-equivalent symbol to one shape, so `N = 0` for all of them
(size, plane-at-intersection, heel/top). The only same-core-name distinct shapes that remain are e.g.
wall vs floor at a *non-intersection* angle — and there the retained plane word yields a small, correct
distance instead of the `0` the stripped name would give.

**Constant & tuning.** With the integer term gone, `w_shape` now scales a distance in `[0,1]`, so it
re-tunes to a much larger value. Swept against all validation signals (others held at the §9 re-fit;
FS/MO here are the objective's Spearman on 600 pairs):

| `w_shape` | AUC | FS(ρ) | MO(ρ) | pref | labrank | J |
|---|---|---|---|---|---|---|
| 60 | 0.807 | 0.507 | 0.369 | 0.842 | 0.731 | 3.255 |
| 130 | 0.818 | 0.581 | 0.403 | 0.832 | 0.730 | 3.363 |
| 250 | 0.826 | 0.640 | 0.434 | 0.852 | 0.730 | 3.481 |
| **500** | **0.828** | **0.682** | **0.458** | **0.861** | 0.729 | **3.558** |
| 700 | 0.828 | 0.693 | 0.467 | 0.852 | 0.729 | 3.567 |

`J` rises steeply and plateaus near 500–700; `w_shape = 500` is the knee (best preference and AUC,
near-max correlations). `normalized_factor` was re-checked and stays at 0.20 — on the full-corpus
headline it beats 0.15 (FS Pearson 0.783 vs 0.774).

### 11.5 Combined measured impact (name-as-identity vs the additive-name baseline)

| signal | integer code + additive name | name *is* the identity term (`w_shape=500`) |
|---|---|---|
| Fingerspelling Pearson / Spearman | 0.691 / 0.532 | **0.783 / 0.682** |
| Mouthing Pearson / Spearman | 0.479 / 0.343 | **0.590 / 0.458** |
| Contrastive AUC | 0.792 | **0.828** |
| Lexical preference | 0.842 | **0.861** |
| Label rank (Spearman) | 0.734 | 0.729 |
| Model-confidence Pearson / Spearman | 0.389 / 0.413 | 0.384 / **0.426** |

A large, clean win: replacing the integer code with the name distance lifts every signal except a
negligible label-rank dip (−0.005). The data strongly prefers semantic naming to code-adjacency. The
`size` weight is not yet in the joint optimizer (`calibration/objective.py`); folding it in is the
natural next step.

### 11.6 Coverage, reproducibility, and limitations

**Coverage.** Of the 652 named base symbols: the hand wall/floor rule contributes **8** `(facing,
angle)` equivalence pairs across the 4 intersection angles (`VARIANT_CANONICAL`, applied to the 261
hand-class shapes); **52** movement shapes have a cross-plane partner collapsed at angles 2/6
(`PLANE_CANON`); **112** shapes fall into size families that collapse to a smaller representative
(`SIZE_CANON`); the name-distance term is active for *every* symbol pair.

**Reproducibility.** Names: `base_symbol_names.json`, re-derivable from the ISWA 2010 listing (and
checkable against Unicode names via `unicodedata.name` on the U+1D800-block base codepoint). Hand
pairs: the grid annotator `calibration/handshapes.py` renders fills 0-5 × rotations 2,6,a,e for a
chosen base shape and writes `assets/annotation/handshape_equivalences.json`; the production list is baked
into `WALL_FLOOR_EQUIVALENTS`. Arrow/size families are derived at import from the name table
(`SIZE_CANON`, `PLANE_CANON`). Canonicalization order is **size → plane** so a Floor-Medium arrow at
angle 2 maps onto the Wall-Small representative with `size_index = 1`.

**Parsing safeguards.** A *size* word counts only as the **last** token, so "Squeeze **Large** Single"
or "Flick **Small** Single" (where Large/Small are mid-name modifiers, not the size grade) are *not*
treated as size variants — the trailing word there is "Single". The *plane* phrase is matched as the
exact substring "Wall Plane"/"Floor Plane". Both the size word and the plane phrase are stripped
before the word-Levenshtein so neither leaks into the general name term.

**Limitations / future work** (paper "future directions"):

- **Hand equivalences are marked on `S100` only**, then assumed shape-independent. Geometrically the
  plane intersection is a property of the rotation, not the handshape, so this should hold for every
  hand; the annotator can re-render any base shape (`--shape`) to spot-check, and the list can be
  extended if a base proves exceptional. The marker also noted "there are even more" intersections
  than the 8 captured — completing the table is open work.
- **Arrow plane equivalence is restricted to the movement class at angles {2, 6}.** Head- and
  torso-movement symbols also carry "Wall Plane"/"Floor Plane" names (e.g. *Head Movement Curves*,
  *Shoulder Hip Move*), but their plane-intersection rotations may differ and were not confirmed, so
  they are deliberately **not** collapsed yet — a candidate for the next annotation round.
- **Identity rests on the ISWA naming.** The name word-distance is now the identity term (§11.4), so
  the metric inherits the names' granularity: symbols with identical names are treated as the same
  identity, and similarity tracks shared *words* rather than visual form directly. A learned name/glyph
  embedding could capture sub-word and visual similarity the word-Levenshtein misses.
- **The size penalty is class-agnostic and linear** in the size index; the perceptual step
  Large→Largest may not equal Small→Medium, and a few non-movement families (3 hand shapes) reuse the
  same words. A per-family or non-linear size scale is possible if the data warrants.
- **`name` and `size` weights are hand-/sweep-tuned, not jointly optimized.** Adding them to
  `objective.py`'s `WEIGHT_KEYS` would let the random search rebalance them against `shape`,
  `class_penalty`, and the factor weights in one fit.

---

## 12. A unified head/face class

**Motivation.** SignWriting writes the face as a **head circle** (`S2ff`) together with **facial
expressions** — mouths, eyes, brows — drawn on it. These were two *adjacent* symbol classes
(*head_movement*, *facial_expressions*), so a head circle and a mouth paid a full cross-class penalty
even though both mark the same face region — a head circle's best partner could end up an unrelated
symbol (in one lexical case it was forced onto a movement arrow).

**Formulation.** We **merge** the two into a single `head_face` class (`SYMBOL_CLASSES`), so a head
circle and a facial expression share a class and incur no cross-class penalty. Within the class they
are still separated by the name term (§11.4), so a head circle and a mouth are a *moderate* (not
identical) match — and a head circle preferentially matches facial content over an unrelated symbol,
because anything outside the class carries the class penalty.

**Evidence.** A strict simplification the data endorses: it replaces an earlier `face_bridge` special
case (a fixed small distance for head↔facial cross-pairs, with its own tuned constant and a branch in
`calculate_distance`) and *improves* the validation signals — contrastive **AUC 0.828 → 0.838**,
fingerspelling/mouthing slightly up, preference unchanged, label-rank 0.729 → 0.731 — while deleting a
parameter and a code path. Model-confidence (a monotonic validator, not a target) dips marginally
(Spearman 0.426 → 0.417).

**Alternatives considered.** (a) The `face_bridge` — it worked but was a special case with its own
constant; the merge subsumes it more simply and scores better. (b) Keeping the classes separate and
treating the head circle as implicit — rejected: both signs draw explicit face content, so the task is
*matching* present face symbols, not forgiving an absent one.

---

## 13. External validation: model confidence and the completeness regime

As a third, fully-automatic check (`evaluation/model_confidence_correlation.py`) we correlate the
metric with a translation model's own **prediction confidence** over 2,984 `source → predicted`
pairs (the model's confidence is a proxy for "is this a good prediction?"). The new metric tracks it
markedly better than the original: **Pearson 0.296 → 0.395, Spearman 0.318 → 0.423**.

**The completeness regime (key finding).** Human review of the high-confidence / low-score outliers
revealed that *the metric is best calibrated when the prediction contains all the symbols used in the
reference*. Segmenting the pairs by **reference coverage** (the fraction of the reference's symbols the
prediction also contains) confirms this quantitatively:

| subset | n | Pearson(metric, confidence) | mean metric |
|---|---|---|---|
| complete (coverage = 1) | 423 | **0.482** | **0.605** |
| partial (0.5–1.0) | 1365 | 0.392 | 0.351 |
| sparse (< 0.5) | 1196 | 0.281 | 0.200 |

So the metric agrees with the model strongly in its **operating regime** (the prediction realizes the
reference's symbol inventory); the disagreement is concentrated in incomplete predictions, where the
low score is *correct* — it reflects missing content, driven by the length/inventory penalty (§3), not
a mis-scoring of a complete-but-different sign. The chart `model_confidence_by_completeness.png` shows
the three regimes with a `y = x` reference.

**Why model confidence is not a calibration target.** We tested rescaling the metric to a 45° line
against confidence (a monotonic `g` fit on the complete subset so the binned trend hits `metric =
confidence`). It is measurably counterproductive, for a structural reason: **confidence has almost no
dynamic range** — it sits in `[0.81, 1.0]` with median 0.97. Matching the metric's `[0.3, 0.7]` spread
onto that sliver forces a near-step `g` that *collapses* the metric's variation. Measured effects:

- complete-subset Pearson **0.482 → 0.330** (the fit gets *worse* at the very thing it optimizes);
- scores inflate wholesale (complete mean 0.605 → 0.911; fingerspelling mean 0.16 → 0.44);
- fingerspelling Pearson 0.691 → 0.629, mouthing 0.479 → 0.396.

Because `g` is monotonic, every **rank-based** signal (AUC, preference, label-rank, and all Spearman
correlations — including the 0.50 on the complete subset) is **invariant** to it. So the model-agreement
that matters is already captured by the rank correlation and needs no rescaling. We therefore use
confidence as a **monotonic external validator**, not a calibration target. (Levenshtein, §11.5, *is*
a legitimate 45° target because it is a true similarity spanning `[0, 1]`; confidence is not.)

---

## 14. Rust backend (speed)

v2 is ~10× slower than v1 in Python (name distance, canonicalizations, factors, implicit handling, and
the mirror check, which alone doubles the work). To make large-scale scoring cheap, the per-pair
**core is ported to Rust** (the `signwriting_similarity_rs` crate, compiled by maturin into
`signwriting_evaluation._similarity_rs` and shipped inside the wheel); the metric routes through it when
constructed with `rust=True` (falling back to Python if the extension is somehow absent).

**What is in Rust:** FSW parse, symbol attributes + all canonicalizations (wall/floor, heel/top, size,
plane), the name identity distance (with the head/face `facial_scale`), the separated identity/position
cost, the conditional materialized implicit face, the Hungarian assignment, the length penalty, and the
reordering and direction factors. Data tables (names, canon maps, weights, and `get_symbol_size`) are
exported from Python into the crate and embedded. **What is approximated in Rust:** the two
*rendering-derived* factors — pixel-accurate touch and the color-change overlap-order weighting — cannot
render glyphs in Rust, so the kernel uses fast bounding-box tests for them. **What stays in Python:**
the mirror wrapper (`mirror_fsw`, cached and exact) and multi-sign *sequence* assembly.

**Speed (Apple M-class, 12 cores):**

| path | pairs/s | vs v2-python |
|---|---|---|
| v2 `score()` loop, python | 1,400 | 1× |
| v2 `score()` loop, **rust** | 90,000 | **65×** |
| v2 `score_all`, python | 1,500 | 1× |
| v2 `score_all`, **rust batch (rayon)** | 83,000–210,000 | **54–112×** |

`score_all` (and therefore `corpus_score`, `score_self`, and the eval/calibration scripts that go
through it) batches every single-sign `(hyp, ref)` and `(hyp, mirror(ref))` pair into **one parallel
Rust call with the GIL released**. The optimized single-sign kernel is ~24× over the Python core on
its own (precomputed pairwise name-distance table, per-symbol cached class/name index, `FxHashMap`
lookups, flat matrices); rayon adds the rest.

**Parity.** Close but not exact: the two rendering-derived factors use bounding-box approximations in
Rust (above), so per-pair scores match Python within ~0.02. Callers needing exact scores re-score the
shortlisted top candidates through Python (e.g. the calibration tools score the corpus with Rust, then
re-rank the top ~40 with the Python metric). Aggregate signals are unaffected. The package is built by
maturin (`pip install .` compiles the kernel; `maturin develop --release` for an editable dev build).
