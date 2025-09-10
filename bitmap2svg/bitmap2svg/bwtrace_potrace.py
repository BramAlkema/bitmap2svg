from __future__ import annotations
import numpy as np
import potrace

def trace_bitmap(bitmap: np.ndarray) -> list[list[tuple[float, float]]]:
    """Trace a bitmap image into vector paths using Potrace.

    Args:
        bitmap (np.ndarray): A binary image array (0s and 1s) where 1 represents the foreground.

    Returns:
        list[list[tuple[float, float]]]: A list of polylines, each represented as a list of (x, y) tuples.
    """
    bmp = potrace.Bitmap(bitmap)
    paths = []
    for curve in bmp.trace().curves:
        polyline = []
        for segment in curve.segments:
            polyline.append((float(segment.c.x), float(segment.c.y)))
        if len(polyline) >= 3 and polyline[0] != polyline[-1]:
            polyline.append(polyline[0])  # Close the loop
        if len(polyline) >= 3:
            paths.append(polyline)
    return paths