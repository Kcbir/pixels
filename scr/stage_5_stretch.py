import numpy as np


def stretch_channels(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float64)
    minima = image.min(axis=(0, 1))
    spans = image.max(axis=(0, 1)) - minima
    return np.divide(255.0 * (image - minima), spans, out=image.copy(), where=spans > 0)
