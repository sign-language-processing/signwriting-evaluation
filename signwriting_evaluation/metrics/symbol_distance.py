from math import sqrt, exp
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dis
from signwriting.types import Sign, SignSymbol
from signwriting.formats.fsw_to_sign import fsw_to_sign
from signwriting_evaluation.metrics.base import SignWritingMetric


class SignWritingSimilarityMetric(SignWritingMetric):
    def __init__(self):
        super().__init__("SymbolsDistances")
        self.symbol_classes = {
            'hand_shapes': range(0x100, 0x205),
            'contact_symbols': range(0x205, 0x221),
            'movement_paths': range(0x221, 0x2FF),
            'head_movement': range(0x2FF, 0x30A),
            'facial_expressions': range(0x30A, 0x36A),
            'etc': range(0x36A, 0x38C)
        }
        self.weight = {
            "shape": 24,  # same weight as switching parallelization
            "facing": 8,  # one column (facing) equals 8 out of 16 lines of the table (like switching hands)
            "angle": 1,  # lowest importance out of the criteria
            "parallel": 24,  # parallelization is 3 columns compare to 1 for the facing direction
            "positional": 1,  # may be big numerical differance, balancing with 1
            "normalized_factor": 1 / 5,  # fitting shape of function
            "class_penalty": 84,  # big penalty for each class type passed
        }
        self.max_distance = self.calculate_distance({"symbol": "S10000", "position": (250, 250)},
                                                    {"symbol": "S38b07", "position": (750, 750)})

    def euc_distance(self, first: Tuple[int, int], second: Tuple[int, int]) -> float:
        return sqrt(pow(first[0] - second[0], 2) + pow(first[1] - second[1], 2))

    def get_attributes(self, symbol: SignSymbol) -> Tuple[int, int, int, bool]:
        shape = int(symbol['symbol'][1:4], 16)
        facing = int(symbol['symbol'][4], 16)
        angle = int(symbol['symbol'][5], 16)
        parallel = facing > 2  # left or right of sign table (0 vertical, 1 horizontal)
        return shape, facing, angle, parallel

    def calculate_distance(self, hyp: SignSymbol, ref: SignSymbol) -> float:
        shape1, angle1, facing1, parallel1 = self.get_attributes(hyp)
        shape2, angle2, facing2, parallel2 = self.get_attributes(ref)
        distance = (self.weight["shape"] * abs(shape1 - shape2) +
                    self.weight["facing"] * abs(facing1 - facing2) +
                    self.weight["angle"] * abs(angle1 - angle2) +
                    self.weight["parallel"] * (parallel1 != parallel2) +
                    self.weight["positional"] * dis.euclidean(hyp["position"], ref["position"]))
        hyp_class = next((i for i, r in enumerate(self.symbol_classes.values()) if shape1 in r), None)
        ref_class = next((i for i, r in enumerate(self.symbol_classes.values()) if shape2 in r), None)
        distance = distance + abs(ref_class - hyp_class) * self.weight["class_penalty"]
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
        # Calculate the evaluate score for a given hypiction and ref.
        if (not hyp["symbols"] and ref["symbols"]) or (hyp["symbols"] and not ref["symbols"]):
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
        length_weight = (1.05 / (1 + exp(-7 * length_error + 3.5))) - 0.025
        return length_weight + mean_cost * (1 - length_weight)

    def score(self, hypothesis: str, reference: str) -> float:
        # Calculate the evaluate score for a given hypiction and ref.
        hyp = fsw_to_sign(hypothesis)
        ref = fsw_to_sign(reference)
        return 1 - self.error_rate(hyp, ref)
