from math import pow, sqrt, exp
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from signwriting.types import Sign, SignSymbol
from signwriting.formats.fsw_to_sign import fsw_to_sign
from base import SignWritingMetric

HEX_BASE = 16


class SignWritingSimilarityMetric(SignWritingMetric):
    def __init__(self):
        super().__init__("SymbolsDistancesMetric")
        self.symbol_classes = self.symbol_classes = {
            'handshakes': range(0x100, 0x205),
            'contact_symbols': range(0x205, 0x221),
            'movement_paths': range(0x221, 0x2FF),
            'head_movement': range(0x2FF, 0x30A),
            'facial_expressions': range(0x30A, 0x36A),
            'etc': range(0x36A, 0x38C)
        }
        self.shape_weight = 24
        self.facing_weight = 8
        self.angle_weight = 1
        self.parallel_weight = 24
        self.positional_weight = 1
        self.normalized_factor = 1/5
        self.class_penalty = 84
        self.max_distance = self.calculate_distance({"symbol": "S10000", "position": (0, 0)},
                                                    {"symbol": "S38b07", "position": (999, 999)})

    def euc_distance(self, first: Tuple[int, int], second: Tuple[int, int]) -> float:
        return sqrt(pow(first[0] - second[0], 2) + pow(first[1] - second[1], 2))

    def get_attributes(self, symbol: SignSymbol) -> Tuple[int, int, int, bool]:  # S12345, 123x123
        shape = int(symbol['symbol'][1:4], HEX_BASE)  # only 123
        facing = int(symbol['symbol'][4], HEX_BASE)  # only 4
        angle = int(symbol['symbol'][5], HEX_BASE)  # only 5
        parallel = facing > 2  # left or right of sign table (0 vertical, 1 horizontal)
        return shape, facing, angle, parallel

    def calculate_distance(self, pred: SignSymbol, gold: SignSymbol) -> float:

        shape1, angle1, facing1, parallel1 = self.get_attributes(pred)
        shape2, angle2, facing2, parallel2 = self.get_attributes(gold)
        distance = (self.shape_weight * abs(shape1 - shape2) + self.facing_weight * abs(facing1 - facing2) +
                    self.angle_weight * abs(angle1 - angle2) + self.parallel_weight * (parallel1 != parallel2) +
                    self.positional_weight * self.euc_distance(pred["position"], gold["position"]))
        pred_class = next((i for i, r in enumerate(self.symbol_classes.values()) if shape1 in r), None)
        gold_class = next((i for i, r in enumerate(self.symbol_classes.values()) if shape2 in r), None)
        distance = distance + abs(gold_class - pred_class) * self.class_penalty
        return distance

    def normalized(self, unnormalized: float) -> float:
        return pow(unnormalized / self.max_distance, self.normalized_factor)

    def symbols_score(self, pred: SignSymbol, gold: SignSymbol) -> float:
        distance = self.calculate_distance(pred, gold)
        normalized = self.normalized(distance)
        return normalized

    def length_acc(self, pred: Sign, gold: Sign) -> float:
        pred = pred["symbols"]
        gold = gold["symbols"]
        # plus 1 for the box symbol
        return abs(len(pred) - len(gold)) / (max(len(pred), len(gold)) + 1)

    def error_rate(self, pred: Sign, gold: Sign) -> float:
        """
        Calculate the evaluate score for a given prediction and gold.
        :param pred: the prediction
        :param gold: the gold
        :return: the FWS score
        """
        if not pred['symbols']:
            return 1
        cost_matrix = np.array(
            [self.symbols_score(first, second) for first in pred["symbols"] for second in gold["symbols"]])
        cost_matrix = cost_matrix.reshape(len(pred["symbols"]), -1)
        # Find the lowest cost matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs = list(zip(row_ind, col_ind))
        # Print the matching and total cost
        values = [cost_matrix[row, col] for row, col in pairs]
        mean_cost = sum(values) / len(values)
        length_error = self.length_acc(pred, gold)
        length_weight = (1.05 / (1 + exp(-7 * length_error + 3.5))) - 0.025
        return length_weight + mean_cost * (1 - length_weight)

    def score(self, hypothesis: str, reference: str) -> float:
        """
        Calculate the evaluate score for a given prediction and gold.
        :param reference:
        :param hypothesis:
        :param pred: the prediction
        :param gold: the gold
        :return: the FWS score
        """
        pred = fsw_to_sign(hypothesis)
        gold = fsw_to_sign(reference)
        return 1 - self.error_rate(pred, gold)





