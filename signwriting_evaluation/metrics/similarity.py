import math
from functools import cache
from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from signwriting.formats.fsw_to_sign import fsw_to_sign
from signwriting.formats.swu_to_fsw import swu2fsw
from signwriting.tokenizer import normalize_signwriting
from signwriting.types import Sign, SignSymbol

from signwriting_evaluation.metrics.base import SignWritingMetric


class SymbolAttributes(NamedTuple):
    shape: int
    facing: int
    angle: int
    parallel: bool


class EnrichedSymbol(NamedTuple):
    symbol: str
    attributes: SymbolAttributes
    shape_class: Optional[int]
    x: int
    y: int


SYMBOL_CLASSES = {
    'hands_shapes': range(0x100, 0x205),
    'contact_symbols': range(0x205, 0x221),
    'movement_paths': range(0x221, 0x2FF),
    'head_movement': range(0x2FF, 0x30A),
    'facial_expressions': range(0x30A, 0x36A),
    'etc': range(0x36A, 0x38C)
}


@cache
def get_shape_class_index(shape: int) -> Optional[int]:
    return next((i for i, r in enumerate(SYMBOL_CLASSES.values()) if shape in r), None)


@cache
def text_to_signs(text: str) -> tuple[str, ...]:
    text_as_fsw = swu2fsw(text)  # converts swu symbols to fsw, while keeping the fsw symbols if present
    return tuple(normalize_signwriting(text_as_fsw).split(" "))


@cache
def get_symbol_attributes(symbol: str) -> SymbolAttributes:
    shape = int(symbol[1:4], 16)
    facing = int(symbol[4], 16)
    angle = int(symbol[5], 16)
    parallel = facing > 2
    return SymbolAttributes(shape, facing, angle, parallel)


ERROR_WEIGHT = {
    "shape": 5,  # same weight as switching parallelization
    "facing": 5 / 3,  # more important than angle, not as much as shape and orientation
    "angle": 5 / 24,  # lowest importance out of the criteria
    "parallel": 5,  # parallelization is 3 columns compare to 1 for the facing direction
    "positional": 1 / 10,  # may be big values
    "normalized_factor": 1 / 2.5,  # fitting shape of function
    "exp_factor": 1.5,  # exponential distribution
    "class_penalty": 100,  # big penalty for each class type passed
}


@cache
def fast_symbol_distance(attributes1: SymbolAttributes, attributes2: SymbolAttributes) -> float:
    d_shape = (attributes1.shape - attributes2.shape) * ERROR_WEIGHT["shape"]
    d_facing = (attributes1.facing - attributes2.facing) * ERROR_WEIGHT["facing"]
    d_angle = (attributes1.angle - attributes2.angle) * ERROR_WEIGHT["angle"]
    d_parallel = (attributes1.parallel != attributes2.parallel) * ERROR_WEIGHT["parallel"]
    return math.sqrt(d_shape * d_shape + \
                     d_facing * d_facing + \
                     d_angle * d_angle + \
                     d_parallel * d_parallel)


fsw_to_sign = cache(fsw_to_sign)


class SignWritingSimilarityMetric(SignWritingMetric):
    SYMMETRIC = True

    def __init__(self):
        super().__init__("SymbolsDistances")
        self.max_distance = self.calculate_distance(
            self.enrich_symbol({"symbol": "S10000", "position": (250, 250)}),
            self.enrich_symbol({"symbol": "S38b07", "position": (750, 750)}),
            fallback_distance=0.0,
        )

    @staticmethod
    def enrich_symbol(symbol: SignSymbol) -> EnrichedSymbol:
        symbol_id = symbol["symbol"]
        x, y = symbol["position"]
        attributes = get_symbol_attributes(symbol_id)
        shape_class = get_shape_class_index(attributes.shape)
        return EnrichedSymbol(symbol_id, attributes, shape_class, x, y)

    @staticmethod
    def enrich_sign(sign: Sign) -> list[EnrichedSymbol]:
        return [SignWritingSimilarityMetric.enrich_symbol(symbol) for symbol in sign["symbols"]]

    @staticmethod
    def _coerce_symbol(symbol) -> EnrichedSymbol:
        if isinstance(symbol, tuple):
            return symbol
        return SignWritingSimilarityMetric.enrich_symbol(symbol)

    @staticmethod
    def _coerce_sign(sign) -> list[EnrichedSymbol]:
        if isinstance(sign, dict):
            return SignWritingSimilarityMetric.enrich_sign(sign)
        return sign

    def calculate_distance(self, hyp, ref, fallback_distance=None) -> float:
        hyp_symbol = self._coerce_symbol(hyp)
        ref_symbol = self._coerce_symbol(ref)

        if (hyp_symbol.symbol == ref_symbol.symbol and hyp_symbol.x == ref_symbol.x and hyp_symbol.y == ref_symbol.y
                and hyp_symbol.shape_class is not None and ref_symbol.shape_class is not None):
            return 0.0

        if hyp_symbol.shape_class is None or ref_symbol.shape_class is None:
            return self.max_distance if fallback_distance is None else fallback_distance

        symbols_distance = fast_symbol_distance(hyp_symbol.attributes, ref_symbol.attributes)

        dx = hyp_symbol.x - ref_symbol.x
        dy = hyp_symbol.y - ref_symbol.y
        position_euclidean = math.sqrt(dx * dx + dy * dy)
        position_distance = ERROR_WEIGHT["positional"] * position_euclidean

        class_penalty = abs(hyp_symbol.shape_class - ref_symbol.shape_class) * ERROR_WEIGHT["class_penalty"]

        return symbols_distance + position_distance + class_penalty

    def normalized_distance(self, unnormalized: float) -> float:
        return (unnormalized / self.max_distance) ** ERROR_WEIGHT["normalized_factor"]

    def symbols_score(self, hyp, ref) -> float:
        distance = self.calculate_distance(hyp, ref)
        return self.normalized_distance(distance)

    def length_acc(self, hyp, ref) -> float:
        hyp_symbols = self._coerce_sign(hyp)
        ref_symbols = self._coerce_sign(ref)
        hyp_len = len(hyp_symbols)
        ref_len = len(ref_symbols)
        return abs(hyp_len - ref_len) / (max(hyp_len, ref_len) + 1)

    def mean_symbol_cost(self, hyp_symbols, ref_symbols) -> float:
        hyp_len = len(hyp_symbols)
        ref_len = len(ref_symbols)

        if hyp_len == 1 and ref_len == 1:
            return self.symbols_score(hyp_symbols[0], ref_symbols[0])

        cost_matrix = np.empty((hyp_len, ref_len), dtype=np.float64)
        for i, hyp_symbol in enumerate(hyp_symbols):
            row = cost_matrix[i]
            for j, ref_symbol in enumerate(ref_symbols):
                row[j] = self.symbols_score(hyp_symbol, ref_symbol)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return float(cost_matrix[row_ind, col_ind].mean())

    def error_rate(self, hyp, ref) -> float:
        hyp_symbols = self._coerce_sign(hyp)
        ref_symbols = self._coerce_sign(ref)

        if not hyp_symbols or not ref_symbols:
            return 1.0

        mean_cost = self.mean_symbol_cost(hyp_symbols, ref_symbols)
        length_error = self.length_acc(hyp_symbols, ref_symbols)
        length_weight = length_error ** ERROR_WEIGHT["exp_factor"]
        return length_weight + mean_cost * (1.0 - length_weight)

    def score_single_sign(self, hypothesis: str, reference: str) -> float:
        hyp_symbols = self.enrich_sign(fsw_to_sign(hypothesis))
        ref_symbols = self.enrich_sign(fsw_to_sign(reference))

        if hypothesis == reference and hyp_symbols and all(symbol.shape_class is not None for symbol in hyp_symbols):
            return 1.0

        score = 1.0 - self.error_rate(hyp_symbols, ref_symbols)
        return score * score

    def score(self, hypothesis: Optional[str], reference: Optional[str]) -> float:
        if hypothesis is None or reference is None:
            return 0.0

        hypothesis_signs = text_to_signs(hypothesis)
        reference_signs = text_to_signs(reference)
        if len(hypothesis_signs) == 1 and len(reference_signs) == 1:
            return self.score_single_sign(hypothesis_signs[0], reference_signs[0])

        return self.mean_sign_score(
            [self.enrich_sign(fsw_to_sign(sign)) for sign in hypothesis_signs],
            [self.enrich_sign(fsw_to_sign(sign)) for sign in reference_signs],
        )

    def mean_sign_score(self, hyp_signs, ref_signs) -> float:
        matrix_size = max(len(hyp_signs), len(ref_signs))

        if matrix_size == 0:
            return 0.0

        cost_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)
        for i, hyp_sign in enumerate(hyp_signs):
            row = cost_matrix[i]
            for j, ref_sign in enumerate(ref_signs):
                score = 1.0 - self.error_rate(hyp_sign, ref_sign)
                row[j] = score * score

        row_ind, col_ind = linear_sum_assignment(1.0 - cost_matrix)
        mean_score = cost_matrix[row_ind, col_ind].mean()
        return float(mean_score)
