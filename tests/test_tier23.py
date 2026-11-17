import numpy as np
import pandas as pd

from sinacon import tier23 as m

GROUPS = {"APU/error/4": "G01", "APU/error/78": "G06", "APS/error/7": "G07", "APS/error/30": "G09"}


def top(p):
    return m.K[int(np.argmax(p))]


def test_prior_and_thermal_evidence():
    pi = m.code_prior("APU/error/4 (Overtemperature inductance)", GROUPS)
    assert top(pi) == "B1"
    f = dict.fromkeys(m.LR, 0) | {"temp_near_limit": 1, "thermal_cycling": 1}
    p = m.posterior(pi, f)
    assert p[m.K.index("B1")] > 0.95 and abs(p.sum() - 1) < 1e-9
    s = m.summarize(p)
    assert s["scope"] == m.INSIDE and s["set90_size"] == 1


def test_several_cabinets_push_grid():
    pi = m.code_prior("APS/error/7 (Medium voltage switch gear off) + APS/error/30 (APU 1 Powerlink)", GROUPS)
    f = dict.fromkeys(m.LR, 0) | {"few_cabinets": 1}
    assert m.summarize(m.posterior(pi, f))["scope"] == "Lưới-trung thế"


def test_features_cycling_and_stuck_apu():
    t0 = pd.Timestamp("2026-01-26 12:00:00")
    tr = pd.DataFrame({"System": ["APU 3"] * 5 + ["APU 1"],
                       "t": pd.to_datetime(["2026-01-26 06:00:00", "2026-01-26 12:00:01", "2026-01-26 12:01:00",
                                            "2026-01-26 12:02:00", "2026-01-26 12:03:00", "2026-01-20 06:00:00"]),
                       "OpState": [3020, 500, 3020, 500, 3020, 30]})
    f = m.features(tr, None, t0, n_other=0, n_repeat_days=5)
    assert f["thermal_cycling"] == 1 and f["apu_stuck_off"] == 1 and f["only_this_cabinet"] == 1


def test_supply_voltage_low_alone_is_auxiliary_supply():
    alone = m.code_prior("APS/error/19 (Supply voltage low) + APU/error/56 (Supply error 24V driver pos. SW)", {})
    with_mv = m.code_prior("APS/error/19 (Supply voltage low) + APS/error/7 (Medium voltage switch gear off)", {})
    assert alone[m.K.index("B4")] > alone[m.K.index("GRID")]
    assert with_mv[m.K.index("GRID")] > with_mv[m.K.index("B4")]


def test_event_type_and_case_aggregation():
    assert m.event_type("false", "no_trip", 0, True) == "none"
    assert m.event_type("real", "long_trip", 2, False) == "new"
    assert m.event_type("real", "not_feeding", 0, False) == "persistent"
    assert m.event_type("real", "not_feeding", 0, True) == "new"
    pk = {f"p_{k}": 0.0 for k in m.K}
    explosion = {**pk, "p_B3": 1.0, "event_type": "new", "n_trip": 2}
    grid = {**pk, "p_GRID": 1.0, "event_type": "new", "n_trip": 0}
    evening = {**pk, "p_B4": 1.0, "event_type": "persistent", "n_trip": 0}
    p, how = m.aggregate_case(pd.DataFrame([explosion, grid, evening, evening, evening]))
    assert how == "new events" and top(p) == "B3"
    p, how = m.aggregate_case(pd.DataFrame([evening]))
    assert how == "all real" and top(p) == "B4"


def test_low_temperature_does_not_penalise_coolant_level():
    t0 = pd.Timestamp("2026-05-27 11:00:00")
    temp = pd.DataFrame({"System": "APU 1", "t": pd.date_range(t0 - pd.Timedelta(minutes=5), periods=6, freq="min"),
                         "TInd/°C": 100.0, "TL1/°C": 60.0, "TL2/°C": 60.0, "TL3/°C": 60.0})
    assert m.features(None, temp, t0, 0, 1, "APS/error/3 (APU 1 / 2 cooling fluid low)")["temp_low"] == 0
    assert m.features(None, temp, t0, 0, 1, "APU/error/4 (Overtemperature inductance)")["temp_low"] == 1


def test_codes_of_one_incident_count_once_and_plant_event_wins():
    groups = {f"APU/error/{c}": "G04" for c in (75, 79, 80, 81, 82)}
    codes = " + ".join(f"APU/error/{c} (x)" for c in (75, 79, 80, 81, 82))
    one, many = m.code_prior("APU/error/79 (x)", groups), m.code_prior(codes, groups)
    assert abs(one[m.K.index("B3")] - many[m.K.index("B3")]) < 1e-9
    f = m.features(None, None, pd.Timestamp("2026-05-02 14:25"), 20, 1, codes, n_other_15=50)
    assert f["plant_event_2min"] == 1 and f["few_cabinets"] == 0 and f["plant_event_15min"] == 0
    assert top(m.posterior(many, f)) == "GRID"
    f15 = m.features(None, None, pd.Timestamp("2026-05-02 14:14"), 0, 1, codes, n_other_15=47)
    assert f15["plant_event_15min"] == 1 and f15["only_this_cabinet"] == 0


def test_powerlink_of_out_of_service_apu_is_consequence_and_history_points_to_b3():
    groups = {"APS/error/30": "G09", "APS/error/31": "G09"}
    codes = "APS/error/30 (APU 1 Powerlink) + APS/error/31 (APU 2 Powerlink)"
    pi = m.code_prior(codes, groups, {"APU 1", "APU 2"})
    assert pi[m.K.index("B6")] == pi[m.K.index("B3")]
    f = m.features(None, None, pd.Timestamp("2026-05-20 10:00"), 0, 5, codes, history=True, n_apu_out=2)
    assert f["hardware_history"] == 1 and f["failure_frequency"] == 1 and f["apu_stuck_off"] == 0
    assert top(m.posterior(pi, f)) == "B3"


def test_island_fuse_and_local_ac_fault_point_to_ac_filter():
    assert top(m.code_prior("APS/error/55 (APU1/2 Island Fuses)", {"APS/error/55": "G07"})) == "B7"
    codes = "APU/error/31 (Undervoltage VL1 level 1)"
    pi = m.code_prior(codes, {"APU/error/31": "G07"})
    local = m.features(None, None, pd.Timestamp("2026-06-01 10:00"), 0, 0, codes)
    wide = m.features(None, None, pd.Timestamp("2026-06-01 10:00"), 8, 0, codes)
    assert local["local_ac_fault"] == 1 and wide["local_ac_fault"] == 0
    assert top(m.posterior(pi, local)) == "B7" and top(m.posterior(pi, wide)) == "GRID"


def test_failure_frequency_factors():
    assert m.LR["failure_frequency"]["B2"] > 2 and m.LR["failure_frequency"]["B1"] == 0.3
