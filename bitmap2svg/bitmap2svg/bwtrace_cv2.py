from __future__ import annotations

import numpy as np
import cv2


def trace_bitmap(bitmap: np.ndarray) -> list[list[tuple[float, float]]]:
    """Trace a bitmap image into vector paths using OpenCV contours.

    Args:
        bitmap: A binary image array where non-zero pixels represent the foreground.

    Returns:
        A list of polylines, each represented as a list of ``(x, y)`` tuples.
    """

    bw_u8 = (bitmap > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bw_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths: list[list[tuple[float, float]]] = []
    for cnt in contours:
        pts = [(float(x), float(y)) for x, y in cnt.reshape(-1, 2)]
        if len(pts) >= 3 and pts[0] != pts[-1]:
            pts.append(pts[0])
        if len(pts) >= 3:
            paths.append(pts)
    return paths
