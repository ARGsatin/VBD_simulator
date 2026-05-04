from __future__ import annotations

import numpy as np


def rms_norm(vectors: np.ndarray) -> float:
    vectors = np.asarray(vectors, dtype=float)
    return float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=1))))


def max_norm(vectors: np.ndarray) -> float:
    vectors = np.asarray(vectors, dtype=float)
    return float(np.max(np.linalg.norm(vectors, axis=1)))
