import numpy as np


def gray_world(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float64)
    means = image.mean(axis=(0, 1))
    gains = np.divide(means.mean(), means, out=np.ones_like(means), where=means > 0)
    return np.clip(image * gains, 0.0, 255.0)
