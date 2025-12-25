import math
from functools import cache
from typing import Tuple, Optional, NamedTuple

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


@cache
def fast_positional_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    # Unbelievably, this is faster than using numpy or scipy for simple Euclidean distance
    # It reduces the overhead of converting to numpy arrays when calculating distances
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)


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
        self.max_distance = self.calculate_distance({"symbol": "S10000", "position": (250, 250)},
                                                    {"symbol": "S38b07", "position": (750, 750)})

    def calculate_distance(self, hyp: SignSymbol, ref: SignSymbol) -> float:
        hyp_attributes = get_symbol_attributes(hyp['symbol'])
        ref_attributes = get_symbol_attributes(ref['symbol'])

        symbols_distance = fast_symbol_distance(hyp_attributes, ref_attributes)

        position_euclidean = fast_positional_distance(hyp["position"], ref["position"])
        position_distance = ERROR_WEIGHT["positional"] * position_euclidean

        hyp_class = get_shape_class_index(hyp_attributes.shape)
        ref_class = get_shape_class_index(ref_attributes.shape)

        if hyp_class is None or ref_class is None:
            return self.max_distance

        class_penalty = abs(hyp_class - ref_class) * ERROR_WEIGHT["class_penalty"]

        return symbols_distance + position_distance + class_penalty

    def normalized_distance(self, unnormalized: float) -> float:
        return pow(unnormalized / self.max_distance, ERROR_WEIGHT["normalized_factor"])

    def symbols_score(self, hyp: SignSymbol, ref: SignSymbol) -> float:
        distance = self.calculate_distance(hyp, ref)
        normalized = self.normalized_distance(distance)
        return normalized

    def length_acc(self, hyp: Sign, ref: Sign) -> float:
        hyp_len = len(hyp["symbols"])
        ref_len = len(ref["symbols"])
        # plus 1 for the box symbol
        return abs(hyp_len - ref_len) / (max(hyp_len, ref_len) + 1)

    def error_rate(self, hyp: Sign, ref: Sign) -> float:
        # Calculate the evaluate score for a given hypothesis and ref.
        if not hyp["symbols"] or not ref["symbols"]:
            return 1.0

        cost_matrix = np.array(
            [self.symbols_score(first, second) for first in hyp["symbols"] for second in ref["symbols"]])
        cost_matrix = cost_matrix.reshape(len(hyp["symbols"]), -1)
        # Find the lowest cost matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mean_cost = float(cost_matrix[row_ind, col_ind].mean())

        length_error = self.length_acc(hyp, ref)
        length_weight = pow(length_error, ERROR_WEIGHT["exp_factor"])
        return length_weight + mean_cost * (1 - length_weight)

    def score_single_sign(self, hypothesis: str, reference: str) -> float:
        # Calculate the evaluate score for a given hypothesis and ref.
        hyp = fsw_to_sign(hypothesis)
        ref = fsw_to_sign(reference)
        return pow(1 - self.error_rate(hyp, ref), 2)

    def score(self, hypothesis: Optional[str], reference: Optional[str]) -> float:
        if hypothesis is None or reference is None:
            return 0.0

        # Here, hypothesis and reference are both FSW strings of potentially different number of signs
        hypothesis_signs = text_to_signs(hypothesis)
        reference_signs = text_to_signs(reference)
        if len(hypothesis_signs) == 1 and len(reference_signs) == 1:
            return self.score_single_sign(hypothesis_signs[0], reference_signs[0])

        # Pad with empty strings to make sure the number of signs is the same
        if len(hypothesis_signs) != len(reference_signs):
            max_length = max(len(hypothesis_signs), len(reference_signs))
            hypothesis_signs += tuple([None] * (max_length - len(hypothesis_signs)))
            reference_signs += tuple([None] * (max_length - len(reference_signs)))

        # Match each hypothesis sign with each reference sign
        cost_matrix = self.score_all(hypothesis_signs, reference_signs, progress_bar=False)
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(1 - cost_matrix)
        mean_score = cost_matrix[row_ind, col_ind].mean()
        return float(mean_score)
