import numpy as np

VALID_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
EMPTY_BBOXES = np.zeros((0, 4), dtype=np.float32)
