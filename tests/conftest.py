import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def cloud_texture(h, w, rng, scale=6.0):
    """Smooth random texture with cloud-like blobs, values 0..255 (float32)."""
    import cv2

    n = rng.standard_normal((h // 4 + 2, w // 4 + 2)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), scale)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    return (60 + 160 * n).astype(np.float32)
