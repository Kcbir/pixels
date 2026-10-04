import cv2
import numpy as np


def bilateral_filter(rgb: np.ndarray, diameter: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    image = np.ascontiguousarray(rgb, dtype=np.float32)
    result = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(result, 0.0, 255.0).astype(np.float64)
