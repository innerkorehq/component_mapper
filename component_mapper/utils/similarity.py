from difflib import SequenceMatcher
from component_mapper.models import RankedCandidate


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Intersection over union. Returns 1.0 if both sets are empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def skeleton_similarity(skeleton_a: str, skeleton_b: str) -> float:
    """Normalized edit distance on skeleton strings using SequenceMatcher."""
    if not skeleton_a and not skeleton_b:
        return 1.0
    if not skeleton_a or not skeleton_b:
        return 0.0
    return SequenceMatcher(None, skeleton_a, skeleton_b).ratio()


def composite_score(
    structural: float,
    class_tokens: float,
    type_compat: float,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Weighted sum of structural, class_tokens, type_compat scores."""
    w_s, w_c, w_t = weights
    result = w_s * structural + w_c * class_tokens + w_t * type_compat
    return max(0.0, min(1.0, result))


def tfidf_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    try:
        import numpy as np

        a = np.array(vec_a, dtype=float)
        b = np.array(vec_b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except ImportError:
        # Pure Python fallback
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = sum(x * x for x in vec_a) ** 0.5
        norm_b = sum(y * y for y in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def rank_candidates(
    candidates: list[RankedCandidate],
    top_k: int = 4,
    min_threshold: float = 0.40,
) -> list[RankedCandidate]:
    """Sort by composite_score descending, filter below threshold, return top_k."""
    filtered = [c for c in candidates if c.composite_score >= min_threshold]
    filtered.sort(key=lambda c: c.composite_score, reverse=True)
    return filtered[:top_k]
