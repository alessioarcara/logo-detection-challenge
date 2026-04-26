from abc import ABC, abstractmethod

import cv2
import numpy as np


class Blend(ABC):
    def _alpha_blend_roi(
        self,
        bg_roi: np.ndarray,
        logo_rgb: np.ndarray,
        logo_alpha: np.ndarray,
    ) -> np.ndarray:
        # * 0-255 alpha to 0.0-1.0, add an extra dimension for broadcasting
        alpha = (logo_alpha.astype(np.float32) / 255.0)[..., None]
        foreground = logo_rgb.astype(np.float32)
        background = bg_roi.astype(np.float32)

        blended = alpha * foreground + (1.0 - alpha) * background
        return blended.astype(np.uint8)

    @abstractmethod
    def paste(
        self,
        bg: np.ndarray,
        logo_rgb: np.ndarray,
        logo_alpha: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        raise NotImplementedError


class AlphaBlend(Blend):
    def paste(
        self,
        bg: np.ndarray,
        logo_rgb: np.ndarray,
        logo_alpha: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        logo_h, logo_w = logo_rgb.shape[:2]
        image = bg.copy()
        roi = image[y : y + logo_h, x : x + logo_w]
        image[y : y + logo_h, x : x + logo_w] = self._alpha_blend_roi(
            bg_roi=roi,
            logo_rgb=logo_rgb,
            logo_alpha=logo_alpha,
        )
        return image


class GaussianBlend(Blend):
    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    def paste(
        self,
        bg: np.ndarray,
        logo_rgb: np.ndarray,
        logo_alpha: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        logo_h, logo_w = logo_rgb.shape[:2]
        image = bg.copy()
        roi = image[y : y + logo_h, x : x + logo_w]
        softened_alpha = cv2.GaussianBlur(
            logo_alpha,
            ksize=(0, 0),
            sigmaX=self._sigma,
            sigmaY=self._sigma,
        )
        image[y : y + logo_h, x : x + logo_w] = self._alpha_blend_roi(
            bg_roi=roi,
            logo_rgb=logo_rgb,
            logo_alpha=softened_alpha,
        )
        return image
