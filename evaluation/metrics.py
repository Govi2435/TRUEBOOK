from __future__ import annotations

from math import log2
from typing import List, Set


def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for x in rec_k if x in relevant)
    return hits / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for x in rec_k if x in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    def dcg(scores: List[int]) -> float:
        return sum((score / log2(idx + 2)) for idx, score in enumerate(scores))

    rec_k = recommended[:k]
    gains = [1 if x in relevant else 0 for x in rec_k]
    ideal = sorted(gains, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg