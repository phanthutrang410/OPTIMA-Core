import numpy as np
import pandas as pd

from sinacon import validation as v
from sinacon.tier23 import K


def test_wilson_interval():
    lo, hi = v.wilson(11, 21)
    assert (lo, hi) == (0.32, 0.72)
    assert np.isnan(v.wilson(0, 0)[0])


def test_auc():
    assert v.auc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert v.auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5


def test_om_truth_mapping():
    om = pd.DataFrame({"Device": ["IGBT", "AC", "DC", "AC"], "Fault_device": ["PowerStack", "Harmonic, Capicitor",
                                                                            "DC Breaker", "AC Breaker"],
                       "Note": ["Nổ IGBT", "Hỏng 1 chì", "", ""]})
    assert list(v.om_truth(om)) == ["B3", "B7", "B2", "B2"]


def test_evaluate_om_scores_top_classes(tmp_path):
    om = tmp_path / "om.csv"
    pd.DataFrame({"Fault_ID": [1], "Inverter_ID": ["7.2"], "APS": ["APS-1"], "APU_ID": ["3,4"], "Device": ["IGBT"],
                  "Fault_device": ["PowerStack"], "Fault_date": ["2026-05-03"], "Fault_time": ["12:00:00"],
                  "Note": ["Nổ IGBT"]}).to_csv(om, index=False)
    p = {f"p_{k}": 0.0 for k in K} | {"p_B3": 0.7, "p_GRID": 0.3}
    tab = pd.DataFrame([{"inverter": "7.2", "t0": "2026-05-03 11:50", "tier1_verdict": "real", "event_type": "new",
                         "n_trip": 2, **p}])
    r = v.evaluate_om(tab, om)
    assert r.loc[0, "result"] == "chấm" and r.loc[0, "top1"] and r.loc[0, "independent"]
