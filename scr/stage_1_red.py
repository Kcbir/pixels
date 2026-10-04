import numpy as np


def compensate_red(rgb: np.ndarray) -> np.ndarray:
    result = np.asarray(rgb, dtype=np.float64).copy()
    means = result.mean(axis=(0, 1))
    result[..., 0] = np.clip(result[..., 0] + means[1] - means[0], 0.0, 255.0)
    return result
