from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dis
from signwriting.formats.fsw_to_sign import fsw_to_sign
from signwriting.formats.swu_to_fsw import swu2fsw
from signwriting.tokenizer import normalize_signwriting
from signwriting.types import Sign, SignSymbol

from signwriting_evaluation.metrics.base import SignWritingMetric


class SignWritingSimilarityMetric(SignWritingMetric):
    def __init__(self):
        super().__init__("SymbolsDistances")
        self.symbol_classes = {
            'hands_shapes': range(0x100, 0x205),
            'contact_symbols': range(0x205, 0x221),
            'movement_paths': range(0x221, 0x2FF),
            'head_movement': range(0x2FF, 0x30A),
            'facial_expressions': range(0x30A, 0x36A),
            'etc': range(0x36A, 0x38C)
        }
        self.weight = {
            "shape": 5,  # same weight as switching parallelization
            "facing": 5 / 3,  # more important than angle, not as much as shape and orientation
            "angle": 5 / 24,  # lowest importance out of the criteria
            "parallel": 5,  # parallelization is 3 columns compare to 1 for the facing direction
            "positional": 1 / 10,  # may be big values
            "normalized_factor": 1 / 2.5,  # fitting shape of function
            "exp_factor": 1.5,  # exponential distribution
            "class_penalty": 100,  # big penalty for each class type passed
        }
        self.max_distance = self.calculate_distance({"symbol": "S10000", "position": (250, 250)},
                                                    {"symbol": "S38b07", "position": (750, 750)})

    def get_shape_class_index(self, symbol_attribute) -> int:
        shape = symbol_attribute[0]
        return next((i for i, r in enumerate(self.symbol_classes.values()) if shape in r), None)

    def get_attributes(self, symbol: SignSymbol) -> Tuple[int, int, int, bool]:
        shape = int(symbol['symbol'][1:4], 16)
        facing = int(symbol['symbol'][4], 16)
        angle = int(symbol['symbol'][5], 16)
        parallel = facing > 2
        return shape, facing, angle, parallel

    def weight_vector(self, vector: Tuple[int, int, int, bool]) -> Tuple[float, ...]:
        weights = [self.weight["shape"], self.weight["angle"], self.weight["facing"], self.weight["parallel"]]
        weighted_values = [float(val * weight) for val, weight in zip(vector, weights)]
        return tuple(weighted_values)

    # return to this

    def calculate_distance(self, hyp: SignSymbol, ref: SignSymbol) -> float:
        hyp_veq = self.get_attributes(hyp)
        ref_veq = self.get_attributes(ref)

        hyp_class = self.get_shape_class_index(hyp_veq)
        ref_class = self.get_shape_class_index(ref_veq)

        hyp_veq = self.weight_vector(hyp_veq)
        ref_veq = self.weight_vector(ref_veq)
        distance = (dis.euclidean(hyp_veq, ref_veq) +
                    self.weight["positional"] * dis.euclidean(hyp["position"], ref["position"]))
        distance = distance + abs(hyp_class - ref_class) * self.weight["class_penalty"]
        return distance

    def normalized_distance(self, unnormalized: float) -> float:
        return pow(unnormalized / self.max_distance, self.weight["normalized_factor"])

    def symbols_score(self, hyp: SignSymbol, ref: SignSymbol) -> float:
        distance = self.calculate_distance(hyp, ref)
        normalized = self.normalized_distance(distance)
        return normalized

    def length_acc(self, hyp: Sign, ref: Sign) -> float:
        hyp = hyp["symbols"]
        ref = ref["symbols"]
        # plus 1 for the box symbol
        return abs(len(hyp) - len(ref)) / (max(len(hyp), len(ref)) + 1)

    def error_rate(self, hyp: Sign, ref: Sign) -> float:
        # Calculate the evaluate score for a given hypothesis and ref.
        if not hyp["symbols"] or not ref["symbols"]:
            return 1
        cost_matrix = np.array(
            [self.symbols_score(first, second) for first in hyp["symbols"] for second in ref["symbols"]])
        cost_matrix = cost_matrix.reshape(len(hyp["symbols"]), -1)
        # Find the lowest cost matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs = list(zip(row_ind, col_ind))
        # Print the matching and total cost
        values = [cost_matrix[row, col] for row, col in pairs]
        mean_cost = sum(values) / len(values)
        length_error = self.length_acc(hyp, ref)
        length_weight = pow(length_error, self.weight["exp_factor"])
        return length_weight + mean_cost * (1 - length_weight)

    def score_single_sign(self, hypothesis: str, reference: str) -> float:
        # Calculate the evaluate score for a given hypothesis and ref.
        hyp = fsw_to_sign(hypothesis)
        ref = fsw_to_sign(reference)
        return pow(1 - self.error_rate(hyp, ref), 2)

    def _text_to_signs(self, text: str) -> list[str]:
        text_as_fsw = swu2fsw(text) # converts swu symbols to fsw, while keeping the fsw symbols if present
        return normalize_signwriting(text_as_fsw).split(" ")

    def score(self, hypothesis: str, reference: str) -> float:
        # Here, hypothesis and reference are both FSW strings of potentially different number of signs
        hypothesis_signs = self._text_to_signs(hypothesis)
        reference_signs = self._text_to_signs(reference)
        if len(hypothesis_signs) == 1 and len(reference_signs) == 1:
            return self.score_single_sign(hypothesis_signs[0], reference_signs[0])

        # Pad with empty strings to make sure the number of signs is the same
        if len(hypothesis_signs) != len(reference_signs):
            max_length = max(len(hypothesis_signs), len(reference_signs))
            hypothesis_signs += [""] * (max_length - len(hypothesis_signs))
            reference_signs += [""] * (max_length - len(reference_signs))

        # Match each hypothesis sign with each reference sign
        cost_matrix = self.score_all(hypothesis_signs, reference_signs, progress_bar=False)
        row_ind, col_ind = linear_sum_assignment(1 - np.array(cost_matrix))
        pairs = list(zip(row_ind, col_ind))
        values = [cost_matrix[row][col] for row, col in pairs]
        return sum(values) / len(values)
