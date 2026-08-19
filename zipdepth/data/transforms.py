"""Paired image/depth transforms used by ZipDepth training."""

from typing import Optional, Tuple

import cv2
import numpy as np

cv2.setNumThreads(0)


class PairedShortestSideCrop:
    """Resize the shortest side, take a random square crop, then maybe flip.

    RGB and continuous pseudo depth share the exact same geometry.  Both resize
    operations use bilinear interpolation; in particular, pseudo depth is not
    passed through Albumentations' ``mask`` path (which would use nearest
    neighbour interpolation).
    """

    def __init__(self, size: int, flip_probability: float = 0.5):
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        self.size = int(size)
        self.height = self.size
        self.width = self.size
        self.flip_probability = float(flip_probability)

    def __call__(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB image [H,W,3], got {image.shape}")
        if depth is not None:
            depth = np.asarray(depth).squeeze()
            if depth.ndim != 2:
                raise ValueError(f"expected pseudo depth [H,W], got {depth.shape}")
            if depth.shape != image.shape[:2]:
                raise ValueError(
                    f"RGB/depth shape mismatch before paired transform: "
                    f"{image.shape[:2]} vs {depth.shape}"
                )

        source_h, source_w = image.shape[:2]
        scale = self.size / min(source_h, source_w)
        resized_h = max(self.size, int(round(source_h * scale)))
        resized_w = max(self.size, int(round(source_w * scale)))

        image = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        if depth is not None:
            depth = cv2.resize(
                depth.astype(np.float32, copy=False),
                (resized_w, resized_h),
                interpolation=cv2.INTER_LINEAR,
            )

        max_y = resized_h - self.size
        max_x = resized_w - self.size
        crop_y = int(np.random.randint(max_y + 1)) if max_y else 0
        crop_x = int(np.random.randint(max_x + 1)) if max_x else 0
        image = image[crop_y:crop_y + self.size, crop_x:crop_x + self.size]
        if depth is not None:
            depth = depth[crop_y:crop_y + self.size, crop_x:crop_x + self.size]

        if np.random.random() < self.flip_probability:
            image = image[:, ::-1]
            if depth is not None:
                depth = depth[:, ::-1]

        # Flips create negative strides, which torch.from_numpy cannot consume.
        image = np.ascontiguousarray(image)
        depth = None if depth is None else np.ascontiguousarray(depth, dtype=np.float32)
        return image, depth

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self.size}, "
            f"flip_probability={self.flip_probability})"
        )


def get_train_transforms(height: int = 512, width: int = 512) -> PairedShortestSideCrop:
    """Return the paper training geometry (square resolutions only)."""
    if height != width:
        raise ValueError(
            "Paper-faithful ZipDepth training requires a square crop; "
            f"received {height}x{width}"
        )
    return PairedShortestSideCrop(size=height, flip_probability=0.5)
