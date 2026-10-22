"""Readers for the HOD3K mosaic representation and multi-page TIFF cubes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile


def read_hod3k_mosaic(path: str | Path, tau: float = 384.0) -> np.ndarray:
    """Decode a HOD3K 4x4 raw mosaic to a normalized [16, H, W] array."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    height, width = image.shape
    height -= height % 4
    width -= width % 4
    if height == 0 or width == 0:
        raise ValueError(f"Image is too small for a 4x4 mosaic: {image.shape}")
    image = image[:height, :width]
    cube = image.reshape(height // 4, 4, width // 4, 4)
    cube = cube.transpose(0, 2, 1, 3).reshape(height // 4, width // 4, 16)
    cube = np.clip(cube.astype(np.float32), 0.0, tau) / tau
    return np.ascontiguousarray(np.moveaxis(cube, -1, 0))


def read_multipage_tiff(path: str | Path, expected_bands: int) -> np.ndarray:
    """Read a multi-page TIFF cube as normalized float32 [C, H, W]."""
    cube = tifffile.imread(path)
    if cube.ndim != 3:
        raise ValueError(f"Expected a three-dimensional TIFF stack, received {cube.shape}")
    if cube.shape[0] == expected_bands:
        channels_first = cube
    elif cube.shape[-1] == expected_bands:
        channels_first = np.moveaxis(cube, -1, 0)
    else:
        raise ValueError(f"Expected {expected_bands} TIFF pages, received {cube.shape}")
    return np.ascontiguousarray(channels_first.astype(np.float32) / 255.0)
