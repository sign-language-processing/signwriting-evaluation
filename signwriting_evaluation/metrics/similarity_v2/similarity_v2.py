import json
import math
import os
from collections import Counter
from functools import cache
from itertools import combinations
from typing import NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from signwriting.formats.fsw_to_sign import fsw_to_sign
from signwriting.formats.swu_to_fsw import swu2fsw
from signwriting.tokenizer import normalize_signwriting
from signwriting.types import Sign, SignSymbol
from signwriting.utils.mirror import mirror_sign
from signwriting.visualizer.visualize import (
    get_font,
    get_symbol_size,
    key2id,
    signwriting_to_image,
    symbol_fill,
    symbol_line,
)

from signwriting_evaluation.metrics.base import SignWritingMetric


class SymbolAttributes(NamedTuple):
    shape: int
    facing: int
    angle: int
    parallel: bool
    size_index: int = 0


SYMBOL_CLASSES = {
    'hands_shapes': range(0x100, 0x205),
    'contact_symbols': range(0x205, 0x221),
    'movement_paths': range(0x221, 0x2FF),
    'head_face': range(0x2FF, 0x36A),   # head movement + facial expressions: one face/head region
    'etc': range(0x36A, 0x38C)
}

HAND_CLASS = list(SYMBOL_CLASSES).index('hands_shapes')
HEAD_CLASS = list(SYMBOL_CLASSES).index('head_face')          # the unified head/face class
CONTACT_CLASS = list(SYMBOL_CLASSES).index('contact_symbols')  # touch markers
# "face" for implicit purposes is the head circle or any facial expression (one merged class).
FACE_CLASSES = (HEAD_CLASS,)
HEAD_RIM_SHAPES = range(0x300, 0x30A)  # head-rim symbols imply an (often unmarked) touch location

# Implicit face materialization: a sign that draws no head/face is compared, against a sign that does,
# as if it had a head circle at the canonical position with the rest of the sign shifted down to make
# room (the typical "head on top, hands below" layout). This gives the matcher a shared face anchor so
# positional distances are computed correctly, rather than forgiving the missing face as a free leftover.
IMPLICIT_FACE_SYMBOL = "S2ff00"
IMPLICIT_FACE_POSITION = (482, 482)
IMPLICIT_FACE_SHIFT = get_symbol_size(IMPLICIT_FACE_SYMBOL)[1] + 5  # face height + a small gap (computed once)

# Wall-plane vs floor-plane handshape variants that render as the SAME hand. A hand symbol is
# S{shape}{facing}{angle}; the facing digit also encodes whether the hand lies in the wall plane
# (facing the reader) or the floor plane (seen from above). At the angles where the two planes
# intersect (2, 6, a, e) the glyphs coincide, so e.g. facing 0 and facing 4 at angle 2 are the same
# hand drawn from two viewpoints. We canonicalize each equivalent (facing, angle) to one representative
# so the symbol distance scores them identically. Marked by hand in the handshape grid annotator;
# see assets/annotation/handshape_equivalences.json and similarity.md §11. (Hand class only — for other
# symbol classes the 4th/5th hex digits do not encode the handshape plane.)
WALL_FLOOR_EQUIVALENTS = [
    ((0, 0x2), (4, 0x2)), ((1, 0x2), (5, 0x2)),  # angle 2: floor = wall + 4
    ((0, 0xa), (4, 0xa)), ((1, 0xa), (5, 0xa)),  # angle a: floor = wall + 4
    ((1, 0x6), (3, 0x6)), ((2, 0x6), (4, 0x6)),  # angle 6: floor = wall + 2
    ((1, 0xe), (3, 0xe)), ((2, 0xe), (4, 0xe)),  # angle e: floor = wall + 2
]
VARIANT_CANONICAL = {high: low for low, high in (sorted(pair) for pair in WALL_FLOOR_EQUIVALENTS)}

# Heel-of-hand ("wrist view") vs top view: a flat hand or fist with fingers/knuckles pointing straight
# forward (arm parallel to the floor plane) can be written either way — they are the same handshape.
# The Lessons-in-SignWriting "Heel of Hand or Top View?" section lists 7 such shapes (heel facing 1,
# top facing 5). The two viewpoints' ROTATIONS also correspond: within each 8-rotation plane group the
# top view is a half-turn (+4 mod 8) from the heel view, mapped by hand across all 16 rotations
# (calibration/heel_rotations.py; see similarity.md §11.1b). We canonicalize every heel symbol to its
# top-view counterpart so the two notations score identically. Directional (heel -> top only).
HEEL_TOP_SHAPES = [(0x15c, 0x15a), (0x15e, 0x15d), (0x14d, 0x14c), (0x151, 0x150),
                   (0x14f, 0x14e), (0x204, 0x203), (0x1f6, 0x1f5)]
HEEL_TOP_ROTATIONS = {0x4: 0x0, 0x5: 0x1, 0x6: 0x2, 0x7: 0x3, 0x0: 0x4, 0x1: 0x5, 0x2: 0x6, 0x3: 0x7,
                      0xc: 0x8, 0xd: 0x9, 0xe: 0xa, 0xf: 0xb, 0x8: 0xc, 0x9: 0xd, 0xa: 0xe, 0xb: 0xf}
HEEL_TO_TOP = {(heel_shape, 1, heel_angle): (top_shape, 5, top_angle)
               for heel_shape, top_shape in HEEL_TOP_SHAPES
               for heel_angle, top_angle in HEEL_TOP_ROTATIONS.items()}


@cache
def get_shape_class_index(shape: int) -> Optional[int]:
    return next((i for i, r in enumerate(SYMBOL_CLASSES.values()) if shape in r), None)


# ---- Name-aware symbol distance -----------------------------------------------------------------
# Every base symbol (the 3-hex shape) has a human ISWA name (base_symbol_names.json, extracted from
# the Lessons-in-SignWriting ISWA 2010 listing). The name carries semantics the raw shape code does
# not, which three mechanisms exploit (see similarity.md §11):
#   - size: names differing only in a trailing size word are the SAME base symbol; collapse them and
#     add a small size-difference penalty (people write the same sign at slightly different sizes).
#   - plane: a movement whose name differs only in plane (Wall vs Floor) renders the same arrow at the
#     angles where the planes intersect (2, 6); collapse those to one representative.
#   - name distance: word-level Levenshtein over the names biases otherwise-different symbols toward
#     each other by how much of their name they share ("Index Bent on Circle" vs "Index on Circle").
with open(os.path.join(os.path.dirname(__file__), "base_symbol_names.json"), encoding="utf-8") as _names_file:
    BASE_SYMBOL_NAMES = {int(key, 16): value for key, value in json.load(_names_file).items()}

_SIZE_WORDS = {"small": 0, "medium": 1, "large": 2, "largest": 3}


@cache
def name_tokens(shape: int) -> Tuple[str, ...]:
    # Lower-cased words of the name with the plane phrase and any trailing size word removed, so the
    # FAMILY GROUPINGS (size/plane) compare only the symbol's "core" identity.
    name = BASE_SYMBOL_NAMES.get(shape)
    if not name:
        return ()
    name = name.replace("Wall Plane", "").replace("Floor Plane", "")
    tokens = [token.strip(",").lower() for token in name.split()]
    tokens = [token for token in tokens if token]
    if tokens and tokens[-1] in _SIZE_WORDS:
        tokens.pop()
    return tuple(tokens)


@cache
def full_name_tokens(shape: int) -> Tuple[str, ...]:
    # Lower-cased words of the full name (plane and size words kept). Used for the symbol-distance name
    # term: canonicalization already collapses truly-equivalent symbols to one shape (name distance 0),
    # so the only same-core-name distinct shapes left are e.g. wall vs floor at a non-intersection
    # angle, where the retained plane word yields a correct small distance.
    name = BASE_SYMBOL_NAMES.get(shape)
    if not name:
        return ()
    return tuple(token.strip(",").lower() for token in name.split() if token.strip(","))


def _name_plane_size(shape: int) -> Tuple[Optional[str], Optional[int]]:
    name = BASE_SYMBOL_NAMES.get(shape, "")
    plane = "wall" if "Wall Plane" in name else "floor" if "Floor Plane" in name else None
    size = _SIZE_WORDS.get(name.split()[-1].lower()) if name else None
    return plane, size


def _build_canonical_maps() -> Tuple[dict, dict]:
    # SIZE_CANON: shapes whose name differs only in the trailing size word collapse to the
    #   smallest-size representative, carrying the size index (0..3).
    # PLANE_CANON: movement shapes with the same core name but Wall vs Floor collapse to one
    #   representative (built over the size reps so it composes); applied only at the intersection
    #   angles. See similarity.md §11.
    size_families: dict = {}
    for shape in BASE_SYMBOL_NAMES:
        plane, size = _name_plane_size(shape)
        if size is not None:
            size_families.setdefault((name_tokens(shape), plane), []).append((size, shape))
    size_canon = {}
    for members in size_families.values():
        rep = min(members)[1]
        for size, shape in members:
            size_canon[shape] = (rep, size)

    plane_families: dict = {}
    for shape in BASE_SYMBOL_NAMES:
        plane, _ = _name_plane_size(shape)
        if plane is not None and shape in SYMBOL_CLASSES["movement_paths"]:
            plane_families.setdefault(name_tokens(shape), {})[plane] = size_canon.get(shape, (shape, 0))[0]
    plane_canon = {}
    for by_plane in plane_families.values():
        if len(by_plane) >= 2:
            rep = min(by_plane.values())
            plane_canon.update({partner: rep for partner in by_plane.values()})
    return size_canon, plane_canon


PLANE_INTERSECTION_ANGLES = (0x2, 0x6)
SIZE_CANON, PLANE_CANON = _build_canonical_maps()


@cache
def name_word_distance(shape1: int, shape2: int) -> float:
    # Word-level Levenshtein between two symbols' full names, normalized to [0, 1] (0 = same name).
    if shape1 == shape2:
        return 0.0
    tokens1, tokens2 = full_name_tokens(shape1), full_name_tokens(shape2)
    if not tokens1 or not tokens2:
        return 1.0 if tokens1 != tokens2 else 0.0
    previous = list(range(len(tokens2) + 1))
    for i, word1 in enumerate(tokens1, start=1):
        current = [i]
        for j, word2 in enumerate(tokens2, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word1 != word2)))
        previous = current
    return previous[-1] / max(len(tokens1), len(tokens2))


@cache
def text_to_signs(text: str) -> tuple[str, ...]:
    text_as_fsw = swu2fsw(text)  # converts swu symbols to fsw, while keeping the fsw symbols if present
    return tuple(normalize_signwriting(text_as_fsw).split(" "))


@cache
def get_symbol_attributes(symbol: str) -> SymbolAttributes:
    shape = int(symbol[1:4], 16)
    facing = int(symbol[4], 16)
    angle = int(symbol[5], 16)
    shape, facing, angle = HEEL_TO_TOP.get((shape, facing, angle), (shape, facing, angle))
    if get_shape_class_index(shape) == HAND_CLASS:
        facing, angle = VARIANT_CANONICAL.get((facing, angle), (facing, angle))
    shape, size_index = SIZE_CANON.get(shape, (shape, 0))  # collapse size variants to one base symbol
    if angle in PLANE_INTERSECTION_ANGLES and shape in PLANE_CANON:  # wall/floor arrows coincide here
        shape = PLANE_CANON[shape]
    parallel = facing > 2
    return SymbolAttributes(shape, facing, angle, parallel, size_index)


@cache
def fast_positional_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    # Unbelievably, this is faster than using numpy or scipy for simple Euclidean distance
    # It reduces the overhead of converting to numpy arrays when calculating distances
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)


@cache
def symbol_ink_mask(symbol: str) -> np.ndarray:
    # Boolean ink mask (line + fill) of a symbol rendered at the origin. Mirrors signwriting's
    # canonicalize._symbol_mask; replicated here so the metric needs only the public glyph primitives
    # (the canonicalize module is not yet on PyPI).
    width, height = get_symbol_size(symbol)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.fontmode = "1"  # 1-bit edges; overlap is a crisp yes/no question
    symbol_id = key2id(symbol)
    draw.text((0, 0), symbol_fill(symbol_id), fill=(0, 0, 0, 255), font=get_font("SuttonSignWritingFill"))
    draw.text((0, 0), symbol_line(symbol_id), fill=(0, 0, 0, 255), font=get_font("SuttonSignWritingLine"))
    return np.asarray(image)[:, :, 3] > 0


def symbols_share_ink(symbol_a: str, pos_a, symbol_b: str, pos_b) -> bool:
    # A real touch: the two positioned glyphs paint a shared pixel. Cheap bounding-box overlap test
    # first (disjoint boxes can never share ink), then AND the ink masks over the overlap region.
    mask_a, mask_b = symbol_ink_mask(symbol_a), symbol_ink_mask(symbol_b)
    (ax, ay), (bx, by) = pos_a, pos_b
    left, top = max(ax, bx), max(ay, by)
    right = min(ax + mask_a.shape[1], bx + mask_b.shape[1])
    bottom = min(ay + mask_a.shape[0], by + mask_b.shape[0])
    if left >= right or top >= bottom:
        return False
    region_a = mask_a[top - ay:bottom - ay, left - ax:right - ax]
    region_b = mask_b[top - by:bottom - by, left - bx:right - bx]
    return bool(np.logical_and(region_a, region_b).any())


@cache
def draw_order_change_fraction(symbol_a: str, pos_a, symbol_b: str, pos_b) -> float:
    # Fraction of rendered pixels that change COLOR when the two symbols' draw order is swapped — i.e.
    # how visible their z-order is. Two glyphs that share no ink (or paint the same color where they do,
    # e.g. two black lines crossing) render identically in either order -> 0. A fill covering another's
    # line changes pixels. Returns 0 immediately for ink-disjoint pairs (the common case).
    if not symbols_share_ink(symbol_a, pos_a, symbol_b, pos_b):
        return 0.0

    def render(order):
        fsw = "M500x500" + "".join(f"{symbol}{x}x{y}" for symbol, (x, y) in order)
        return np.asarray(signwriting_to_image(fsw, trust_box=False).convert("RGB"))

    forward = render([(symbol_a, pos_a), (symbol_b, pos_b)])
    swapped = render([(symbol_b, pos_b), (symbol_a, pos_a)])
    if forward.shape != swapped.shape:
        return 0.0
    return float((forward != swapped).any(axis=2).mean())


# Data-fit values (joint optimization against the contrastive, fingerspelling/mouthing, preference,
# and label-rank signals; see calibration/optimize.py). Earlier hand-set values in parentheses.
ERROR_WEIGHT = {
    "shape": 487.83,  # weight on the name word-distance (the identity signal; replaced the integer code)
    "facing": 1.636,
    "angle": 0.144,
    "parallel": 3.004,
    "position_scale": 80.40,  # px; position cost is (euclidean / position_scale)^2 — see symbols_score
    "normalized_factor": 0.159,  # concave identity exponent; the main lever for spelling correlation
    "exp_factor": 1.287,  # length-penalty steepness (lower => missing symbols penalized harder)
    "class_penalty": 69.51,
    "size": 0.04,  # small penalty per size step (Small<Medium<Large<Largest) of the same base symbol
    "facial_scale": 1.0,  # scales the name distance between two head/face symbols (1.0 = no scaling);
    #   two facial expressions are the same class and should not reach the full hand-vs-arrow distance
}


@cache
def fast_symbol_distance(attributes1: SymbolAttributes, attributes2: SymbolAttributes) -> float:
    # The shape/identity distance is the word-Levenshtein between the two symbols' names (a far better
    # proxy than the integer code difference it replaces): same name -> 0, fully different -> w_shape.
    name_distance = name_word_distance(attributes1.shape, attributes2.shape)
    if (get_shape_class_index(attributes1.shape) == HEAD_CLASS
            and get_shape_class_index(attributes2.shape) == HEAD_CLASS):
        name_distance *= ERROR_WEIGHT["facial_scale"]  # same class -> never fully hand-vs-arrow distant
    d_shape = name_distance * ERROR_WEIGHT["shape"]
    d_facing = (attributes1.facing - attributes2.facing) * ERROR_WEIGHT["facing"]
    d_angle = (attributes1.angle - attributes2.angle) * ERROR_WEIGHT["angle"]
    d_parallel = (attributes1.parallel != attributes2.parallel) * ERROR_WEIGHT["parallel"]
    # size only differentiates two symbols that are otherwise the same base symbol.
    d_size = (abs(attributes1.size_index - attributes2.size_index) * ERROR_WEIGHT["size"]
              if attributes1.shape == attributes2.shape else 0.0)
    return math.sqrt(d_shape ** 2 + d_facing ** 2 + d_angle ** 2 + d_parallel ** 2 + d_size ** 2)


fsw_to_sign = cache(fsw_to_sign)


@cache
def mirror_fsw(fsw: str) -> Optional[str]:
    # Horizontal mirror of an FSW (or SWU) sign/sequence, or None if it can't be mirrored.
    try:
        return " ".join(mirror_sign(sign) for sign in swu2fsw(fsw).split(" "))
    except (ValueError, KeyError, IndexError):
        return None


def centroid(symbols: Tuple[SignSymbol, ...]) -> Tuple[float, float]:
    count = len(symbols)
    return (sum(s["position"][0] for s in symbols) / count,
            sum(s["position"][1] for s in symbols) / count)


def translate(symbol: SignSymbol, offset: Tuple[float, float]) -> SignSymbol:
    return {"symbol": symbol["symbol"],
            "position": (symbol["position"][0] - offset[0], symbol["position"][1] - offset[1])}


class SignWritingSimilarityV2Metric(SignWritingMetric):
    SYMMETRIC = True

    OVERLAP_VISIBLE_MIN = 0.02  # draw-order is "visible" only if swapping it changes >=2% of pixels

    def __init__(self, reordering_weight: float = 1.277, overlap_weight: float = 0.272,  # pylint: disable=too-many-arguments
                 implicit: bool = True, touch_penalty: float = 2.299, mirror_penalty: float = 0.424,
                 movement_weight: float = 0.506, sequence_weight: float = 0.2, rust: bool = False):
        super().__init__("SymbolsDistancesV2")
        # rust: route scoring through the compiled Rust kernel (signwriting_similarity_rs) for speed.
        # The kernel ports the full formula (separated identity/position cost, facial_scale, conditional
        # materialized implicit face, matching, length, reordering, direction). The two RENDERING-derived
        # refinements — pixel-accurate touch and color-change overlap weighting — cannot run in Rust, so
        # the kernel uses fast bounding-box approximations for them; it is therefore a fast APPROXIMATE
        # ranking kernel (parity with Python is close but not exact). Callers that need exact scores
        # re-score the shortlisted candidates through the Python path (see the calibration tools).
        self._rs = None
        if rust:
            try:
                import signwriting_similarity_rs as _rs  # noqa: PLC0415
                self._rs = _rs
            except ImportError:
                self._rs = None
        # Two opt-out factors (set weight to 0 to disable), each multiplied into the per-sign score:
        # - reordering_weight: penalizes relative-position reorderings (e.g. "abc" vs "cba").
        # - overlap_weight: penalizes a different draw order of *overlapping* symbols (their z-order
        #   changes the rendered glyph; for non-overlapping symbols, order does not matter).
        # implicit: don't penalize a symbol the other sign leaves implicit (a missing head/face, a
        #   head-rim touch, or a touch where two symbols overlap). See implicit_classes / error_rate.
        # touch_penalty: weight of an UNexplained touch in the length penalty. A touch denotes an
        #   interaction the other sign lacks (a lone hand can't touch anything), so it counts as more
        #   than a cosmetic extra symbol.
        # mirror_penalty: a sign and its horizontal mirror are related, not unrelated. Score the
        #   reference mirrored too; if that fits better, credit it at this fraction (an exact mirror
        #   scores mirror_penalty rather than the much lower direct score). Set to 0 to disable.
        # movement_weight: penalize direction (rotation/facing) flips of matched same-shape symbols,
        #   e.g. a flipped arrow, in their own rotation space so they aren't crushed by the global scale.
        # sequence_weight: for a SEQUENCE of signs, gently penalize signs appearing in a different order
        #   (sign order carries meaning, but less rigidly than symbol layout within a sign).
        self.reordering_weight = reordering_weight
        self.overlap_weight = overlap_weight
        self.implicit = implicit
        self.touch_penalty = touch_penalty
        self.mirror_penalty = mirror_penalty
        self.movement_weight = movement_weight
        self.sequence_weight = sequence_weight
        self.max_distance = self.calculate_distance({"symbol": "S10000"}, {"symbol": "S38b07"})

    def calculate_distance(self, hyp: SignSymbol, ref: SignSymbol) -> float:
        # The IDENTITY distance of two symbols (shape name + facing/angle/parallel/size + symbol class),
        # independent of where they sit. Position is handled separately in symbols_score.
        hyp_attributes = get_symbol_attributes(hyp['symbol'])
        ref_attributes = get_symbol_attributes(ref['symbol'])

        symbols_distance = fast_symbol_distance(hyp_attributes, ref_attributes)

        hyp_class = get_shape_class_index(hyp_attributes.shape)
        ref_class = get_shape_class_index(ref_attributes.shape)

        if hyp_class is None or ref_class is None:
            return self.max_distance

        class_penalty = abs(hyp_class - ref_class) * ERROR_WEIGHT["class_penalty"]

        return symbols_distance + class_penalty

    def normalized_distance(self, unnormalized: float) -> float:
        return pow(unnormalized / self.max_distance, ERROR_WEIGHT["normalized_factor"])

    def symbols_score(self, hyp: SignSymbol, ref: SignSymbol) -> float:
        # Identity and position contribute with DIFFERENT curvature. Identity uses the concave
        # normalized_distance (a small shape difference must still register — this is what keeps the
        # fingerspelling/mouthing correlation), while position is CONVEX: (euclidean / position_scale)^2,
        # so a few px of layout jitter (the same sign written slightly differently) costs ~0 yet a real
        # relocation costs a lot. Lumping both into one root previously inflated tiny shifts to ~0.2,
        # dragging near-identical signs far below 1.
        identity = self.normalized_distance(self.calculate_distance(hyp, ref))
        position = (fast_positional_distance(hyp["position"], ref["position"]) / ERROR_WEIGHT["position_scale"]) ** 2
        return min(1.0, identity + position)

    def length_acc(self, hyp: Sign, ref: Sign) -> float:
        hyp_len = len(hyp["symbols"])
        ref_len = len(ref["symbols"])
        # plus 1 for the box symbol
        return abs(hyp_len - ref_len) / (max(hyp_len, ref_len) + 1)

    @classmethod
    def implicit_classes(cls, symbols) -> Counter:
        # Symbol classes that may be present yet left unmarked, so an explicit one in the *other* sign
        # should not be penalized:
        #  - a head/face if the sign draws neither a head nor a facial expression (idea 1; this leaves
        #    mouthing, which is facial expressions, untouched). It would sit above the whole sign, so it
        #    can never create a touch and is not considered for contact;
        #  - a touch beside each head-rim symbol (idea 2);
        #  - a touch where a hand touches/overlaps another hand or a face (idea 3).
        shapes = [get_symbol_attributes(symbol["symbol"]).shape for symbol in symbols]
        symbol_classes = [get_shape_class_index(shape) for shape in shapes]
        classes: Counter = Counter()
        if not any(klass in FACE_CLASSES for klass in symbol_classes):
            classes[HEAD_CLASS] += 1
        classes[CONTACT_CLASS] += sum(1 for shape in shapes if shape in HEAD_RIM_SHAPES)
        classes[CONTACT_CLASS] += cls.hand_contact_count(symbols, symbol_classes)
        return classes

    @classmethod
    def hand_contact_count(cls, symbols, symbol_classes) -> int:
        # Number of pairs where a hand's INK touches (shares a pixel with) a hand or a face. Bounding-box
        # overlap is necessary but not sufficient — two glyph boxes can intersect while the ink does not —
        # so we confirm a real touch with symbols_share_ink (box-overlap prefilter, then mask overlap).
        count = 0
        for first, second in combinations(range(len(symbols)), 2):
            pair = (symbol_classes[first], symbol_classes[second])
            if HAND_CLASS not in pair or not all(klass in (HAND_CLASS,) + FACE_CLASSES for klass in pair):
                continue
            if symbols_share_ink(symbols[first]["symbol"], symbols[first]["position"],
                                 symbols[second]["symbol"], symbols[second]["position"]):
                count += 1
        return count

    def unexplained_penalty(self, leftover_indices, symbols, available: Counter) -> float:
        # Weighted count of leftover (unmatched) real symbols NOT explained by an implicit symbol of
        # the same class in the other sign. Each implicit symbol explains at most one leftover. A
        # leftover touch counts as `touch_penalty` (an interaction the other sign cannot have); any
        # other leftover counts as 1.
        available = available.copy()
        penalty = 0.0
        for index in leftover_indices:
            symbol_klass = get_shape_class_index(get_symbol_attributes(symbols[index]["symbol"]).shape)
            if available.get(symbol_klass, 0) > 0:
                available[symbol_klass] -= 1
            else:
                penalty += self.touch_penalty if symbol_klass == CONTACT_CLASS else 1.0
        return penalty

    def implicit_length_acc(self, hyp_symbols, ref_symbols, row_ind, col_ind) -> float:
        # Like length_acc, but unmatched real symbols that the other sign leaves implicit are free.
        leftover_hyp = [i for i in range(len(hyp_symbols)) if i not in set(row_ind)]
        leftover_ref = [j for j in range(len(ref_symbols)) if j not in set(col_ind)]
        unexplained = (self.unexplained_penalty(leftover_ref, ref_symbols, self.implicit_classes(hyp_symbols))
                       + self.unexplained_penalty(leftover_hyp, hyp_symbols, self.implicit_classes(ref_symbols)))
        return min(1.0, unexplained / (max(len(hyp_symbols), len(ref_symbols)) + 1))

    def assignment(self, hyp_symbols, ref_symbols) -> Tuple[np.ndarray, np.ndarray]:
        # Match on centroid-centered positions so an unmatched symbol (e.g. an added
        # face) can't bias the matching by dragging the centroid off the shared layout.
        hyp_centroid, ref_centroid = centroid(hyp_symbols), centroid(ref_symbols)
        cost_matrix = np.array(
            [self.symbols_score(translate(first, hyp_centroid), translate(second, ref_centroid))
             for first in hyp_symbols for second in ref_symbols])
        return linear_sum_assignment(cost_matrix.reshape(len(hyp_symbols), -1))

    def error_rate(self, hyp: Sign, ref: Sign, assignment=None) -> float:
        hyp_symbols, ref_symbols = hyp["symbols"], ref["symbols"]
        if not hyp_symbols or not ref_symbols:
            return 1.0

        row_ind, col_ind = assignment if assignment is not None else self.assignment(hyp_symbols, ref_symbols)

        # Re-align on the matched pairs alone, then score them, so signs that differ
        # only by a translation match perfectly regardless of absolute coordinates.
        offset = tuple(np.array([hyp_symbols[i]["position"] for i in row_ind], dtype=float).mean(axis=0)
                       - np.array([ref_symbols[j]["position"] for j in col_ind], dtype=float).mean(axis=0))
        mean_cost = float(np.mean([self.symbols_score(translate(hyp_symbols[i], offset), ref_symbols[j])
                                   for i, j in zip(row_ind, col_ind)]))

        length_acc = (self.implicit_length_acc(hyp_symbols, ref_symbols, row_ind, col_ind)
                      if self.implicit else self.length_acc(hyp, ref))
        length_weight = pow(length_acc, ERROR_WEIGHT["exp_factor"])
        return length_weight + mean_cost * (1 - length_weight)

    @staticmethod
    def principal_axis(centered_positions: np.ndarray):
        if not centered_positions.any():
            return None
        return np.linalg.svd(centered_positions, full_matrices=False)[2][0]

    @staticmethod
    def discordant_fraction(hyp_pos: np.ndarray, ref_pos: np.ndarray, axis: np.ndarray,
                            weights: np.ndarray) -> float:
        # Quality-weighted fraction of matched pairs whose rank order along `axis` flips between hyp
        # and ref. Each pairwise comparison is weighted by the match quality of the two matched pairs
        # involved, so an arbitrary (low-quality) leftover match barely contributes — only the order of
        # confidently-matched symbols counts. Axis orientation cancels, so this is sign-invariant.
        upper = np.triu_indices(len(hyp_pos), k=1)
        hyp_order = (hyp_pos @ axis)[upper[0]] - (hyp_pos @ axis)[upper[1]]
        ref_order = (ref_pos @ axis)[upper[0]] - (ref_pos @ axis)[upper[1]]
        moving = (np.abs(hyp_order) > 1e-6) & (np.abs(ref_order) > 1e-6)
        pair_weight = weights[upper[0]] * weights[upper[1]] * moving
        if pair_weight.sum() == 0:
            return 0.0
        discordant = np.sign(hyp_order) != np.sign(ref_order)
        return float((pair_weight * discordant).sum() / pair_weight.sum())

    def reordering_factor(self, hyp: Sign, ref: Sign, assignment=None) -> float:
        # Penalize reorderings the set-based matching misses (e.g. "abc" vs "cba") via rank-order
        # inversions of the matched symbols, averaged over each sign's own principal axis (which keeps
        # it symmetric). Position jitter rarely flips rank order, so lexical variants are spared, and
        # the discordance is weighted by match quality so arbitrary leftover matches don't fire it.
        if self.reordering_weight <= 0:
            return 1.0
        hyp_symbols, ref_symbols = hyp["symbols"], ref["symbols"]
        if len(hyp_symbols) < 2 or len(ref_symbols) < 2:
            return 1.0
        row_ind, col_ind = assignment if assignment is not None else self.assignment(hyp_symbols, ref_symbols)
        if len(row_ind) < 2:
            return 1.0
        hyp_centroid, ref_centroid = centroid(hyp_symbols), centroid(ref_symbols)
        weights = np.array([1.0 - self.symbols_score(translate(hyp_symbols[i], hyp_centroid),
                                                     translate(ref_symbols[j], ref_centroid))
                            for i, j in zip(row_ind, col_ind)])
        hyp_pos = np.array([hyp_symbols[i]["position"] for i in row_ind], dtype=float)
        ref_pos = np.array([ref_symbols[j]["position"] for j in col_ind], dtype=float)
        hyp_pos -= hyp_pos.mean(axis=0)
        ref_pos -= ref_pos.mean(axis=0)

        axes = [axis for axis in (self.principal_axis(hyp_pos), self.principal_axis(ref_pos)) if axis is not None]
        if not axes:
            return 1.0
        discordant = np.mean([self.discordant_fraction(hyp_pos, ref_pos, axis, weights) for axis in axes])
        return float(np.exp(-self.reordering_weight * discordant))

    @classmethod
    def pair_occlusion(cls, symbols, idx_a, idx_b) -> float:
        # How visible the draw order of two symbols is: the fraction of pixels that change when their
        # order is swapped (0 if they share no ink or paint the same color). Below OVERLAP_VISIBLE_MIN
        # the occlusion is cosmetic and ignored.
        fraction = draw_order_change_fraction(symbols[idx_a]["symbol"], tuple(symbols[idx_a]["position"]),
                                              symbols[idx_b]["symbol"], tuple(symbols[idx_b]["position"]))
        return fraction if fraction >= cls.OVERLAP_VISIBLE_MIN else 0.0

    def overlap_order_factor(self, hyp: Sign, ref: Sign, assignment=None) -> float:
        # When two matched symbols occlude each other, the order they are drawn changes the rendered
        # glyph, so a flipped draw order between hyp and ref is a real difference. A draw-order flip only
        # matters where the pair *visibly* occludes in BOTH signs (a fill covering a line, not merely
        # intersecting boxes, and not a pair the layout pulled apart in one sign — that is a positional
        # difference, scored elsewhere). Penalize the occlusion-weighted fraction of inverted pairs.
        if self.overlap_weight <= 0:
            return 1.0
        hyp_symbols, ref_symbols = hyp["symbols"], ref["symbols"]
        if len(hyp_symbols) < 2 or len(ref_symbols) < 2:
            return 1.0
        row_ind, col_ind = assignment if assignment is not None else self.assignment(hyp_symbols, ref_symbols)
        if len(row_ind) < 2:
            return 1.0

        first, second = np.triu_indices(len(row_ind), k=1)
        # draw order is visible only where the pair occludes in BOTH signs -> the weaker of the two
        occlusion = np.array([min(self.pair_occlusion(hyp_symbols, row_ind[a], row_ind[b]),
                                  self.pair_occlusion(ref_symbols, col_ind[a], col_ind[b]))
                              for a, b in zip(first, second)])
        if occlusion.sum() == 0:
            return 1.0
        inverted = (np.sign(row_ind[first] - row_ind[second]) != np.sign(col_ind[first] - col_ind[second]))
        return float(np.exp(-self.overlap_weight * (occlusion * inverted).sum() / occlusion.sum()))

    def direction_factor(self, hyp: Sign, ref: Sign, assignment=None) -> float:
        # Penalize direction (rotation/facing) flips of matched same-shape symbols, e.g. an arrow
        # pointing the other way. Measured in the symbol's own rotation(0-15, cyclic)/facing(0-5)
        # space and SUMMED (not averaged), so a single flip registers instead of being normalized to
        # ~0 by the global max_distance or diluted across the sign.
        if self.movement_weight <= 0:
            return 1.0
        hyp_symbols, ref_symbols = hyp["symbols"], ref["symbols"]
        if not hyp_symbols or not ref_symbols:
            return 1.0
        row_ind, col_ind = assignment if assignment is not None else self.assignment(hyp_symbols, ref_symbols)
        total = 0.0
        for i, j in zip(row_ind, col_ind):
            attr_hyp = get_symbol_attributes(hyp_symbols[i]["symbol"])
            attr_ref = get_symbol_attributes(ref_symbols[j]["symbol"])
            if attr_hyp.shape != attr_ref.shape:  # direction is only comparable for the same symbol
                continue
            angle = min(abs(attr_hyp.angle - attr_ref.angle), 16 - abs(attr_hyp.angle - attr_ref.angle)) / 8
            facing = abs(attr_hyp.facing - attr_ref.facing) / 5
            total += (angle + facing) / 2
        return float(np.exp(-self.movement_weight * total))

    @staticmethod
    def sign_has_face(sign: Sign) -> bool:
        return any(get_shape_class_index(get_symbol_attributes(symbol["symbol"]).shape) in FACE_CLASSES
                   for symbol in sign["symbols"])

    @staticmethod
    def add_implicit_face(sign: Sign) -> Sign:
        # Insert a head circle at the canonical position and shift the rest of the sign down to clear it.
        shifted = [{"symbol": symbol["symbol"],
                    "position": (symbol["position"][0], symbol["position"][1] + IMPLICIT_FACE_SHIFT)}
                   for symbol in sign["symbols"]]
        face = {"symbol": IMPLICIT_FACE_SYMBOL, "position": IMPLICIT_FACE_POSITION}
        return {"box": sign["box"], "symbols": [face] + shifted}

    def score_single_sign(self, hypothesis: str, reference: str) -> float:
        # Calculate the evaluate score for a given hypothesis and ref.
        hyp = fsw_to_sign(hypothesis)
        ref = fsw_to_sign(reference)
        # When exactly one side draws a face, materialize the other's implicit face so both are compared
        # with a head anchor (see IMPLICIT_FACE_*). Conditional: two face-less signs are left untouched,
        # so a shared free face match never dilutes face-less comparisons (e.g. fingerspelling).
        if self.implicit:
            hyp_face, ref_face = self.sign_has_face(hyp), self.sign_has_face(ref)
            if hyp_face and not ref_face:
                ref = self.add_implicit_face(ref)
            elif ref_face and not hyp_face:
                hyp = self.add_implicit_face(hyp)
        # The Hungarian matching depends only on the two symbol sets, so compute it once and share it
        # across the error rate and all three factors instead of re-running it four times.
        shared = (self.assignment(hyp["symbols"], ref["symbols"])
                  if hyp["symbols"] and ref["symbols"] else None)
        return (pow(1 - self.error_rate(hyp, ref, shared), 2)
                * self.reordering_factor(hyp, ref, shared)
                * self.overlap_order_factor(hyp, ref, shared)
                * self.direction_factor(hyp, ref, shared))

    def directed_score(self, hypothesis: Optional[str], reference: Optional[str]) -> float:
        if hypothesis is None or reference is None:
            return 0.0

        # Here, hypothesis and reference are both FSW strings of potentially different number of signs
        hypothesis_signs = text_to_signs(hypothesis)
        reference_signs = text_to_signs(reference)
        if len(hypothesis_signs) == 1 and len(reference_signs) == 1:
            return self.score_single_sign(hypothesis_signs[0], reference_signs[0])

        # Pad with empty signs so both sides have the same length (extra signs match a None -> 0).
        if len(hypothesis_signs) != len(reference_signs):
            max_length = max(len(hypothesis_signs), len(reference_signs))
            hypothesis_signs += tuple([None] * (max_length - len(hypothesis_signs)))
            reference_signs += tuple([None] * (max_length - len(reference_signs)))

        # Match each hypothesis sign with each reference sign (per-sign, no mirror — that wraps here).
        cost_matrix = np.array([[0.0 if h is None or r is None else self.score_single_sign(h, r)
                                 for r in reference_signs] for h in hypothesis_signs])
        row_ind, col_ind = linear_sum_assignment(1 - cost_matrix)
        matched = cost_matrix[row_ind, col_ind]
        return float(matched.mean()) * self.sequence_reordering_factor(row_ind, col_ind, matched)

    def sequence_reordering_factor(self, row_ind, col_ind, qualities) -> float:
        # Gently penalize matched signs that appear in a different sequence order: the quality-weighted
        # fraction of order inversions, so a weak (e.g. padding) match barely counts. Sign order carries
        # meaning, but less rigidly than symbol layout within a sign, hence the small sequence_weight.
        if self.sequence_weight <= 0 or len(row_ind) < 2:
            return 1.0
        hyp_pos = np.asarray(row_ind, dtype=float).reshape(-1, 1)
        ref_pos = np.asarray(col_ind, dtype=float).reshape(-1, 1)
        discordant = self.discordant_fraction(hyp_pos, ref_pos, np.array([1.0]), np.asarray(qualities, float))
        return float(np.exp(-self.sequence_weight * discordant))

    def _rust_directed(self, hypothesis: Optional[str], reference: Optional[str]) -> float:
        # Mirror of directed_score using the Rust single-sign kernel; sequences stay in Python.
        if hypothesis is None or reference is None:
            return 0.0
        hyp_signs, ref_signs = list(text_to_signs(hypothesis)), list(text_to_signs(reference))
        if len(hyp_signs) == 1 and len(ref_signs) == 1:
            return self._rs.score_single(hyp_signs[0], ref_signs[0])
        if len(hyp_signs) != len(ref_signs):
            length = max(len(hyp_signs), len(ref_signs))
            hyp_signs += [None] * (length - len(hyp_signs))
            ref_signs += [None] * (length - len(ref_signs))
        cost_matrix = np.array([[0.0 if h is None or r is None else self._rs.score_single(h, r)
                                 for r in ref_signs] for h in hyp_signs])
        row_ind, col_ind = linear_sum_assignment(1 - cost_matrix)
        matched = cost_matrix[row_ind, col_ind]
        return float(matched.mean()) * self.sequence_reordering_factor(row_ind, col_ind, matched)

    def score(self, hypothesis: Optional[str], reference: Optional[str]) -> float:
        directed = self._rust_directed if self._rs is not None else self.directed_score
        direct = directed(hypothesis, reference)
        if self.mirror_penalty <= 0 or reference is None:
            return direct
        # A sign and its horizontal mirror are related, not unrelated. Also score the mirrored
        # reference; credit it at mirror_penalty, so an exact mirror scores ~mirror_penalty instead of
        # the low direct score. Non-mirrors are unaffected (the discounted mirror score stays below).
        mirrored = mirror_fsw(reference)
        if mirrored is None:
            return direct
        return max(direct, self.mirror_penalty * directed(hypothesis, mirrored))

    def score_all(self, hypotheses, references, progress_bar=False) -> list[list[float]]:
        # Rust fast path: batch every single-sign (hyp, ref) and (hyp, mirror(ref)) pair into one
        # parallel Rust call (GIL released); multi-sign pairs fall back to per-pair score().
        if self._rs is None:
            return super().score_all(hypotheses, references, progress_bar)
        hyp_signs = [text_to_signs(h) for h in hypotheses]
        ref_signs = [text_to_signs(r) for r in references]
        do_mirror = self.mirror_penalty > 0
        mir_signs = [text_to_signs(m) if (do_mirror and (m := mirror_fsw(r))) else None for r in references]
        tasks: list[tuple[str, str]] = []
        direct_t = [-1] * (len(hypotheses) * len(references))
        mirror_t = [-1] * (len(hypotheses) * len(references))
        fallback = []
        for i, hs in enumerate(hyp_signs):
            for j, rs in enumerate(ref_signs):
                pid = i * len(references) + j
                if len(hs) == 1 and len(rs) == 1:
                    direct_t[pid] = len(tasks)
                    tasks.append((hs[0], rs[0]))
                    ms = mir_signs[j]
                    if do_mirror and ms is not None and len(ms) == 1:
                        mirror_t[pid] = len(tasks)
                        tasks.append((hs[0], ms[0]))
                else:
                    fallback.append((pid, i, j))
        scores = self._rs.score_single_many(tasks)
        flat = [0.0] * (len(hypotheses) * len(references))
        for pid in range(len(flat)):
            if direct_t[pid] < 0:
                continue
            val = scores[direct_t[pid]]
            if mirror_t[pid] >= 0:
                val = max(val, self.mirror_penalty * scores[mirror_t[pid]])
            flat[pid] = val
        for pid, i, j in fallback:
            flat[pid] = self.score(hypotheses[i], references[j])
        return [flat[i * len(references):(i + 1) * len(references)] for i in range(len(hypotheses))]
