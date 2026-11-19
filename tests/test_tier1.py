import pandas as pd

from sinacon import tier1

T0 = pd.Timestamp("2026-01-26 11:33:50")


def trig(rows):
    return pd.DataFrame(rows, columns=["System", "t", "OpState"]).assign(t=lambda d: pd.to_datetime(d["t"]))


def feeding_since_morning(day="2026-01-26", state=3000):
    return [(f"APU {a}", f"{day} 06:30:00", state) for a in range(1, 5)]


def test_trip_is_real_and_short_trip_recovers():
    """Tủ 13.4 ngày 26/01: APU 3, 4 ngắt lúc 11:34 trong khi APU 1, 2 vẫn phát."""
    tr = trig(feeding_since_morning() + [("APU 3", "2026-01-26 11:34:01", 200), ("APU 4", "2026-01-26 11:34:02", 200)])
    el = tier1.electrical(tr, T0)
    assert el["n_feed"] == 4 and el["n_trip"] == 2 and el["phi"] == 0.5
    assert el["apu_minutes_lost"] > 100                       # 2 APU × ~60 phút
    assert tier1.decide(False, el, {})[:2] == ("real", "long_trip")
    brief = trig(feeding_since_morning() + [("APU 3", "2026-01-26 11:34:01", 200), ("APU 3", "2026-01-26 11:36:01", 3000)])
    assert tier1.decide(False, tier1.electrical(brief, T0), {})[1] == "short_trip"


def test_prior_power_based_false_alarm_without_trip():
    el = tier1.electrical(trig(feeding_since_morning()), T0)
    assert el["n_trip"] == 0
    assert tier1.decide(False, el, {}, prior_verdict="false", prior_reason="Tủ vẫn phát 98% mức trước đợt.")[1] == "no_trip"
    assert tier1.decide(False, el, {})[:2] == ("real", "kept_prior")


def temps(values, sensor="TInd/°C"):
    t = pd.date_range(T0.floor("min") - pd.Timedelta(minutes=len(values) - 1), periods=len(values), freq="min")
    return pd.DataFrame({"System": "APU 1", "t": t, sensor: values})


def test_temperature_at_protection_limit_is_real():
    th = tier1.thermal(temps([150, 153, 155, 157, 158, 159]), T0, ["APU 1"], {"APU/error/4"})
    assert th["T_lim"] == 160 and th["at_limit"]
    normal = tier1.thermal(temps([120, 121, 122, 122, 123, 123]), T0, ["APU 1"], {"APU/error/4"})
    assert not normal["at_limit"]                              # 123 °C là mức chạy bình thường của cuộn cảm
    el = {"n_feed": 4, "n_trip": 0, "limited": False}
    assert tier1.decide(True, el, th)[:2] == ("real", "temp_at_limit")


def test_temperature_far_from_limit_and_sensor_jump_are_false():
    el = {"n_feed": 4, "n_trip": 0, "limited": False}
    far = tier1.thermal(temps([60, 61, 61, 62, 62, 62]), T0, ["APU 1"], {"APU/error/4"})
    assert tier1.decide(True, el, far)[1] == "temp_far_from_limit"
    jump = tier1.thermal(temps([60, 61, 62, 95, 62, 61]), T0, ["APU 1"], {"APU/error/4"})
    assert jump["nonphysical"] and tier1.decide(True, el, jump)[1] == "sensor_jump"


def test_cabinet_not_feeding_is_skipped():
    el = tier1.electrical(trig(feeding_since_morning(state=200)), T0)
    assert tier1.decide(True, el, {}, prior_verdict="false")[:2] == ("skip", "not_feeding")
    assert tier1.decide(False, el, {})[:2] == ("skip", "not_feeding")


def test_low_irradiance_shutdown_is_not_a_trip():
    """17:40 mọi APU về thẳng Off (50) -> dừng cuối ngày (giả); về 500 (bảo vệ) vẫn là ngắt thật; về 50 lúc trưa vẫn thật."""
    dusk = pd.Timestamp("2026-01-26 17:40:00")
    tr = trig(feeding_since_morning() + [(f"APU {a}", "2026-01-26 17:40:05", 50) for a in range(1, 5)])
    el = tier1.electrical(tr, dusk)
    assert el["all_to_off"] and tier1.decide(False, el, {}, dusk)[:2] == ("false", "low_irradiance_stop")
    prot = trig(feeding_since_morning() + [("APU 1", "2026-01-26 17:40:05", 500)])
    assert tier1.decide(False, tier1.electrical(prot, dusk), {}, dusk)[0] == "real"
    noon = trig(feeding_since_morning() + [("APU 1", "2026-01-26 11:34:01", 50)])
    assert tier1.decide(False, tier1.electrical(noon, T0), {}, T0)[0] == "real"


def test_apu_out_of_service_and_resume_after_dusk_stop():
    t0 = pd.Timestamp("2026-05-28 17:44:00")
    tr = trig([("APU 1", "2026-05-28 00:00:00", 30), ("APU 2", "2026-05-28 00:00:00", 50),
               ("APU 3", "2026-05-28 06:30:00", 3010), ("APU 4", "2026-05-28 06:30:00", 3010),
               ("APU 3", "2026-05-28 17:44:05", 50), ("APU 4", "2026-05-28 17:44:05", 50)])
    el = tier1.electrical(tr, t0)
    assert el["n_apu_out"] == 2 and el["apus_out"] == "APU 1 APU 2"
    assert tier1.decide(True, el, {}, t0, temp_code=False)[:2] == ("real", "apu_out_of_service")
    grid = pd.Timestamp("2026-05-26 17:00:00")
    tr2 = trig(feeding_since_morning("2026-05-26", 3010) + [(f"APU {a}", "2026-05-26 17:00:05", 50) for a in range(1, 5)]
               + [(f"APU {a}", "2026-05-26 17:02:30", 3010) for a in range(1, 5)])
    el2 = tier1.electrical(tr2, grid)
    assert not el2["all_to_off"] and tier1.decide(False, el2, {}, grid)[0] == "real"
    drv = trig(feeding_since_morning("2026-01-24", 3020) + [("APU 3", "2026-01-24 16:45:00", 50),
                                                           ("APU 4", "2026-01-24 16:45:00", 50)])
    t1 = pd.Timestamp("2026-01-24 16:44:50")
    assert tier1.decide(False, tier1.electrical(drv, t1), {}, t1, hardware_code=True)[0] == "real"
