import json

import numpy as np
import pandas as pd

from sinacon import calibrate as cal


def test_choose_gap_stops_where_curve_flattens():
    counts = pd.Series({0: 430, 0.5: 131, 1: 122, 2: 119, 5: 117, 10: 115})
    assert cal.choose_gap(counts, 0.02) == 2.0


def test_warning_window_edges_where_density_falls_to_twice_background():
    rng = np.random.default_rng(0)
    background = rng.uniform(-600, 600, 1200)                      # 1 lần/giây ở mọi nơi
    burst = rng.uniform(-10, 110, 2400)                            # dày quanh t0
    before, after, base = cal.choose_warning_window(np.r_[background, burst], 10, 300, (300, 600), 2.0)
    assert (before, after) == (10.0, 110.0) and 0.8 < base < 1.2


def test_load_windows_from_file(tmp_path):
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"incident_gap_min": 2.0, "warn_after_s": 110.0}), encoding="utf-8")
    w = cal.load_windows(f)
    assert (w["incident_gap_min"], w["warn_before_s"], w["warn_after_s"]) == (2.0, 10.0, 110.0)
