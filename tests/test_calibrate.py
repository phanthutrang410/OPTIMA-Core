import json

import numpy as np
import pandas as pd

from sinacon import calibrate as cal


def test_choose_gap_stops_where_curve_flattens():
    counts = pd.Series({0: 430, 0.5: 131, 1: 122, 2: 119, 5: 117, 10: 115})
    assert cal.choose_gap(counts, 0.02) == 2.0


def test_warning_window_widest_with_at_most_5_percent_random():
    # nền 1 lần/giây; 1000 warning liên quan dồn trong 0…20 s sau t0, 400 trong 10 s trước t0
    background = np.arange(-599.5, 600, 1.0)
    related = np.r_[np.linspace(0.1, 20, 1000), np.linspace(-10, -0.1, 400)]
    before, after, base = cal.choose_warning_window(np.r_[background, related], 0.05, (300, 600), 300)
    assert abs(base - 1) < 1e-9
    assert after == 52.0          # A·1 ≤ 5 % × (1000 + A)  →  A ≤ 52,6
    assert before == 21.0         # B·1 ≤ 5 % × (400 + B)   →  B ≤ 21,05


def test_load_windows_from_file(tmp_path):
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"incident_gap_min": 2.0, "warn_after_s": 110.0}), encoding="utf-8")
    w = cal.load_windows(f)
    assert (w["incident_gap_min"], w["warn_before_s"], w["warn_after_s"]) == (2.0, 10.0, 110.0)
