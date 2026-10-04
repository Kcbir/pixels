import cv2
import numpy as np


def sharpen_channels(rgb: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float64)
    laplacian = np.stack([cv2.Laplacian(image[..., k], cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101) for k in range(3)], axis=-1)
    return np.clip(image - alpha * laplacian, 0.0, 255.0)
