import cv2
import numpy as np
from typing import List, Tuple, Optional

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB/BGR image to grayscale float32 in [0,1].
    Assumes input is uint8.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float32) / 255.0
    return gray

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur with given sigma.
    Kernel size is derived automatically.
    """
    if sigma <= 0:
        return img
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def compute_image_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial image gradients Ix, Iy using Sobel filters.
    """
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return (Ix, Iy)

def build_gaussian_pyramid(img: np.ndarray, num_levels: int, sigma: float=1.0) -> List[np.ndarray]:
    """
    Build Gaussian pyramid.
    Level 0 = original resolution.
    """
    pyramid = [gaussian_blur(img, sigma)]
    for _ in range(1, num_levels):
        img = cv2.pyrDown(img)
        img = gaussian_blur(img, sigma)
        pyramid.append(img)
    return pyramid

def preprocess_frame(frame: np.ndarray, *, blur_sigma: float=1.0) -> np.ndarray:
    """
    Full preprocessing for a single frame:
    - grayscale
    - float
    - blur
    """
    gray = to_grayscale(frame)
    gray = gaussian_blur(gray, blur_sigma)
    return gray

def preprocess_video(frames: List[np.ndarray], *, blur_sigma: float=1.0) -> List[np.ndarray]:
    """
    Preprocess all frames identically.
    """
    return [preprocess_frame(f, blur_sigma=blur_sigma) for f in frames]
