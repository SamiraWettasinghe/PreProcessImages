"""Pure perspective-correction logic. No GUI dependencies."""

from __future__ import annotations

import numpy as np
import cv2


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left].

    Uses the classic sum/diff trick:
      - top-left has the smallest x+y
      - bottom-right has the largest x+y
      - top-right has the smallest y-x  (smallest "diff" in numpy's sense)
      - bottom-left has the largest y-x
    This means the user does not have to click the corners in any particular
    order — we sort them automatically.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).ravel()  # y - x
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Crop and rectify a quadrilateral region of `image` so it appears
    perpendicular to the camera.

    Parameters
    ----------
    image : np.ndarray
        Source image (BGR or grayscale) as loaded by OpenCV.
    pts : array-like of shape (4, 2)
        Four (x, y) corners of the region to rectify, in any order.

    Returns
    -------
    warped : np.ndarray
        The rectified, axis-aligned crop.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Output size: take the larger of the two parallel sides, so we don't lose
    # detail to downsampling.
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(round(width_top)), int(round(width_bottom)), 1)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(round(height_left)), int(round(height_right)), 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC
    )
    return warped
