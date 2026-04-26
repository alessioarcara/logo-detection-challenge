import numpy as np

from src.utils.constants import EMPTY_BBOXES
from src.utils.typings import BBox, Point


def bbox_from_alpha(alpha: np.ndarray) -> BBox | None:
    """
    Return the xywh bbox around foreground alpha pixels, or None if no foreground.
    """
    ys, xs = np.where(alpha > 0)

    if xs.size == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)


def to_array(bboxes: list[BBox]) -> np.ndarray:
    if not bboxes:
        return EMPTY_BBOXES

    return np.asarray(bboxes, dtype=np.float32)


def bbox_center(bbox: BBox) -> Point:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def shift_bbox(bbox: BBox, offset: Point) -> BBox:
    x, y, w, h = bbox
    dx, dy = offset
    return x + dx, y + dy, w, h


def shift_point(point: Point, offset: Point) -> Point:
    x, y = point
    dx, dy = offset
    return x + dx, y + dy
