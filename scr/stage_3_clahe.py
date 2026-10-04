import cv2
import numpy as np


def equalize_channels(rgb: np.ndarray, clip_limit: float = 2.0, grid: tuple[int, int] = (10, 10)) -> np.ndarray:
    image = np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return np.stack([operator.apply(image[..., k]) for k in range(3)], axis=-1).astype(np.float64)
