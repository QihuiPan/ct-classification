from __future__ import annotations

import numpy as np

from ct_classifier.preprocessing import apply_window, center_crop_or_pad


def test_window_maps_to_unit_interval() -> None:
    volume = np.array([-1000.0, -600.0, 150.0, 500.0], dtype=np.float32)
    result = apply_window(volume, center=-600, width=1500)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
    assert result[0] < result[1] < result[2]


def test_center_crop_and_pad_has_requested_shape() -> None:
    volume = np.ones((5, 7, 9), dtype=np.float32)
    result = center_crop_or_pad(volume, [6, 6, 10], fill=0.0)
    assert result.shape == (6, 6, 10)
    assert np.isfinite(result).all()

