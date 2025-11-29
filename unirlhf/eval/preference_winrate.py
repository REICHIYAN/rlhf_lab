from dataclasses import dataclass
from typing import List

import numpy as np

from unirlhf.data.schemas import PairwiseComparison


@dataclass
class PreferenceMetrics:
    win_rate: float
    bt_score: float


def compute_win_rate(
    method: str,
    comparisons: List[PairwiseComparison],
    vs_method: str,
) -> float:
    wins = 0
    total = 0
    for c in comparisons:
        if {c.method_a, c.method_b} != {method, vs_method}:
            continue
        if c.method_a == method and c.winner == "A":
            wins += 1
            total += 1
        elif c.method_b == method and c.winner == "B":
            wins += 1
            total += 1
        elif c.winner in ("A", "B"):
            total += 1
    return wins / total if total > 0 else 0.0


def compute_bradley_terry_score(
    method: str,
    comparisons: List[PairwiseComparison],
    vs_method: str,
    max_iter: int = 100,
    lr: float = 0.1,
) -> float:
    theta = 0.0
    for _ in range(max_iter):
        grad = 0.0
        n = 0
        for c in comparisons:
            if {c.method_a, c.method_b} != {method, vs_method}:
                continue
            p = float(np.exp(theta) / (np.exp(theta) + 1.0))
            if (c.method_a == method and c.winner == "A") or (
                c.method_b == method and c.winner == "B"
            ):
                grad += 1 - p
                n += 1
            elif (c.method_a == method and c.winner == "B") or (
                c.method_b == method and c.winner == "A"
            ):
                grad += -p
                n += 1
        if n == 0:
            break
        theta += lr * grad / n
    return float(theta)
