"""Bước 3 — Tầng 1: cảnh báo thật hay giả, từ hậu quả điện (OpState) và khoảng cách tới ngưỡng bảo vệ nhiệt.

Hậu quả điện, từ chuỗi OpState s_a(t) của APU a (manual mục 9.1):
    F_a(t) = 1[s_a(t) ≥ 3000]                                   APU đang phát
    F_a⁻   = F_a(t0 − 1′)
    N_a    = F_a⁻ · 1[∃ t ∈ [t0 − 1′, t0 + 10′]: F_a(t) = 0]     APU ngắt quanh cảnh báo
    E      = Σ_a ∫_{t0}^{t0+60′} F_a⁻ (1 − F_a(t)) dt            thời lượng ngừng phát (APU·phút)
    3010–3030: đang phát nhưng giới hạn công suất (giảm tải).

Nhiệt, so với ngưỡng bảo vệ: IGBT T_lim = 107 °C (manual tr. 327–328, mã APU 1–3); cuộn cảm T_lim = 160 °C (mã APU 4;
manual ghi 95 °C nhưng mọi lần bảo vệ cuộn cảm tác động trong log TInd dừng ở 157–161 °C, giá trị ước lượng từ dữ liệu):
    T*  = max_{[t0 − 5′, t0 + 1′]} T                     h = T_lim − T*
    Δh  = 2 + max(0, r)/2 + σ_T, σ_T = 3 °C              (r: tốc độ tăng nhiệt, °C/phút; log là trung bình 1 phút)
    h ≤ Δh: nhiệt tới ngưỡng. |ΔT/Δt| > 15 °C/phút hoặc T ngoài [−20, 200] °C: cảm biến phi vật lý.

Quy tắc, xét theo thứ tự (dừng ở quy tắc đầu tiên thỏa):
    1. không APU nào phát lúc t0 − 1′ (đêm, tủ đã dừng)            -> KHÔNG XÉT (verdict "skip"; người dùng chọn 2026-11-19)
    2. mã không phải nhiệt, có APU không phát suốt [t0 − 180′, t0 + 30′] trong khi APU khác phát, không có ngắt mới
                                                                -> thật, APU hỏng/tách kéo dài
    3. ngoài 08:00–16:30, mọi APU bị ngắt về thẳng Off (50), không APU nào phát lại trong 30′, không có mã phần cứng
                                                                -> giả, dừng do bức xạ thấp
    4. có APU bị ngắt (Σ N_a ≥ 1)                                -> thật; E ≤ 10: ngắt ngắn tự phục hồi
    5. mã nhiệt, h ≤ Δh hoặc APU giảm tải                         -> thật
    6. mã nhiệt, cảm biến phi vật lý                              -> giả
    7. còn lại                                                  -> giả (mã nhiệt: xa ngưỡng; mã khác: xem PRIOR_POWER_BASED)

    python -m sinacon.tier1 --incidents incidents.csv --windows <thư mục> --out tier1.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sinacon.config import N_APU, codes_in, hardware_error_codes, is_thermal as thermal_cell
from sinacon.windows import load_windows

FEED = 3000
LIMITED = {3010, 3020, 3030}
OFF = 50                                   # dừng có điều khiển (500 = bảo vệ tác động, chờ phát lại)
PRE_MIN, POST_MIN, ENERGY_MIN = 1, 10, 60   # PRE_MIN: người dùng chọn 1 phút (2026-11-19), trước là 2
SHORT_TRIP_E = 10.0                        # APU·phút
OUT_BEFORE_MIN, OUT_AFTER_MIN = 180, 30    # cửa sổ xét APU hỏng/tách kéo dài
RESUME_MIN = 30
LOW_IRRADIANCE = (8 * 60, 16 * 60 + 30)    # phút trong ngày; ngoài khoảng này tủ tự dừng/khởi động
T_LIM = {"IGBT": 107.0, "IND": 160.0}
SIGMA_T = 3.0
JUMP = 15.0
IGBT_CODES = {"APU/error/1": "TL1/°C", "APU/error/2": "TL2/°C", "APU/error/3": "TL3/°C"}
IND_CODES = {"APU/error/4": "TInd/°C"}
TEMP_CODES = set(IGBT_CODES) | set(IND_CODES)
# Chỉ dùng khi đầu vào có kết luận cũ của hệ cảnh báo trước đây (cột prior_verdict, prior_reason): kết luận "giả" dựa
# trên công suất tổng của tủ bị bỏ khi OpState cho thấy không APU nào ngắt.
PRIOR_POWER_BASED = ("Công suất tủ giữ", "Tủ vẫn phát")

CLASS_LABEL = {
    "long_trip": "Thật: APU ngắt kéo dài", "short_trip": "Thật: APU ngắt ngắn, tự phục hồi",
    "temp_at_limit": "Thật: nhiệt độ tới ngưỡng bảo vệ", "derating": "Thật: giảm tải do nhiệt",
    "apu_out_of_service": "Thật: APU hỏng/tách kéo dài",
    "sensor_jump": "Giả: cảm biến nhiệt nhảy phi vật lý", "temp_far_from_limit": "Giả: nhiệt xa ngưỡng, không ngắt",
    "no_temp_data": "Giả: không có số đo nhiệt, không ngắt", "no_trip": "Giả: không APU nào ngắt",
    "low_irradiance_stop": "Giả: dừng do bức xạ thấp (đầu/cuối ngày)",
    "not_feeding": "Không xét: không APU nào đang phát", "kept_prior": "Giữ kết luận đầu vào",
}


def state_at(tr: pd.DataFrame, t: pd.Timestamp) -> dict:
    """OpState của từng APU tại thời điểm t (lần đổi cuối cùng ≤ t)."""
    return tr[tr["t"] <= t].groupby("System")["OpState"].last().to_dict()


def electrical(tr: pd.DataFrame, t0: pd.Timestamp) -> dict:
    pre_t = t0 - pd.Timedelta(minutes=PRE_MIN)
    pre = state_at(tr, pre_t)
    feeding = {a for a, s in pre.items() if s >= FEED}
    w = tr[(tr["t"] > pre_t) & (tr["t"] <= t0 + pd.Timedelta(minutes=POST_MIN))]
    tripped = {a for a in feeding if (w[w["System"] == a]["OpState"] < FEED).any()}
    limited = bool(w["OpState"].isin(LIMITED).any() or any(pre.get(a) in LIMITED for a in feeding))
    first_off = []                                     # trạng thái đầu tiên sau khi rời trạng thái phát
    for a in tripped:
        s = w[w["System"] == a]["OpState"].to_numpy()
        first_off.append(int(s[s < FEED][0]))
    w30 = tr[(tr["t"] > pre_t) & (tr["t"] <= t0 + pd.Timedelta(minutes=RESUME_MIN))]
    resumed = any((w30[(w30["System"] == a) & (w30["t"] > t0)]["OpState"] >= FEED).any() for a in tripped)
    all_off = bool(tripped) and all(x == OFF for x in first_off) and not resumed
    # APU hỏng/tách kéo dài: không phát lúc nào trong [t0 − 180′, t0 + 30′] (kể cả APU không còn ghi log) trong khi
    # ít nhất một APU khác của tủ đang phát lúc t0 − 1′.
    h0 = t0 - pd.Timedelta(minutes=OUT_BEFORE_MIN)
    wh = tr[(tr["t"] > h0) & (tr["t"] <= t0 + pd.Timedelta(minutes=OUT_AFTER_MIN))]
    ever = {a for a, s in state_at(tr, h0).items() if s >= FEED} | set(wh.loc[wh["OpState"] >= FEED, "System"])
    out = sorted({f"APU {i}" for i in range(1, N_APU + 1)} - ever) if feeding else []
    grid = pd.date_range(t0, t0 + pd.Timedelta(minutes=ENERGY_MIN), freq="30s")
    lost = 0.0
    for a in feeding:
        ta = tr[tr["System"] == a]
        idx = np.searchsorted(ta["t"].to_numpy(), grid.to_numpy(), side="right") - 1
        st = np.where(idx >= 0, ta["OpState"].to_numpy()[np.clip(idx, 0, None)], pre[a])
        lost += float((st < FEED).sum()) * 0.5
    return dict(n_feed=len(feeding), n_trip=len(tripped), phi=len(tripped) / len(feeding) if feeding else np.nan,
                limited=limited, apu_minutes_lost=lost, tripped_apus=" ".join(sorted(tripped)), all_to_off=all_off,
                n_apu_out=len(out), apus_out=" ".join(out))


def low_irradiance(t0: pd.Timestamp) -> bool:
    m = t0.hour * 60 + t0.minute
    return not (LOW_IRRADIANCE[0] <= m < LOW_IRRADIANCE[1])


def thermal(temp: pd.DataFrame | None, t0: pd.Timestamp, apus: list[str], codes: set[str]) -> dict:
    sens = {**{c: ("IGBT", s) for c, s in IGBT_CODES.items()}, **{c: ("IND", s) for c, s in IND_CODES.items()}}
    hit = [sens[c] for c in codes if c in sens]
    if not hit or temp is None or temp.empty:
        return {}
    w = temp[(temp["t"] >= t0 - pd.Timedelta(minutes=5)) & (temp["t"] <= t0 + pd.Timedelta(minutes=1))]
    w10 = temp[(temp["t"] >= t0 - pd.Timedelta(minutes=10)) & (temp["t"] <= t0 + pd.Timedelta(minutes=1))]
    if apus:
        w, w10 = w[w["System"].isin(apus)], w10[w10["System"].isin(apus)]
    best = None
    for kind, s in hit:
        if s not in w or w[s].dropna().empty:
            continue
        tmax = float(w[s].max())
        rate = (w10.sort_values("t").groupby("System")[s].diff()).dropna()     # °C/phút
        r = float(rate.max()) if len(rate) else 0.0
        tol = 2.0 + max(0.0, r) / 2.0 + SIGMA_T
        h = T_LIM[kind] - tmax
        cand = dict(sensor=s, T_max=round(tmax, 1), T_lim=T_LIM[kind], h=round(h, 1), tolerance=round(tol, 1),
                    max_rate=round(r, 1), nonphysical=bool((rate.abs() > JUMP).any() or not (-20 <= tmax <= 200)),
                    at_limit=h <= tol)
        if best is None or cand["h"] < best["h"]:
            best = cand
    return best or {}


def decide(is_thermal: bool, el: dict, th: dict, t0: pd.Timestamp | None = None,
           temp_code: bool | None = None, hardware_code: bool = False,
           prior_verdict: str = "real", prior_reason: str = "") -> tuple[str, str, str]:
    """(verdict "real"/"false"/"skip", lớp quyết định, căn cứ). "skip" = không xét (tủ không phát lúc t0).

    temp_code: có mã bảo vệ theo nhiệt độ (APU error 1–4); mặc định = is_thermal.
    hardware_code: có mã phần cứng công suất (ô quá dòng / hỏng driver của bảng hai tầng)."""
    if temp_code is None:
        temp_code = is_thermal
    if not el or not el.get("n_feed"):
        return ("skip", "not_feeding", "Không APU nào đang phát 1 phút trước t0 (ban đêm hoặc tủ đã dừng): "
                "không xem được hậu quả, không xét.")
    if not temp_code and el.get("n_apu_out", 0) > 0 and not (el["n_trip"] > 0 and not el.get("all_to_off")):
        return ("real", "apu_out_of_service",
                f"{el['n_apu_out']} APU không phát suốt 3 giờ trước tới 30 phút sau cảnh báo trong khi APU khác của tủ "
                "đang phát: APU hỏng hoặc bị tách khỏi vận hành (tình trạng tồn tại, không phải sự kiện mới).")
    if el["n_trip"] > 0 and el.get("all_to_off") and t0 is not None and low_irradiance(t0) and not hardware_code:
        a, b = LOW_IRRADIANCE
        return ("false", "low_irradiance_stop",
                f"Ngoài {a // 60:02d}:{a % 60:02d}–{b // 60:02d}:{b % 60:02d}, mọi APU ngắt về thẳng Off (50): dừng có "
                "điều khiển do bức xạ thấp, không phải bảo vệ tác động (bảo vệ đưa APU về 500).")
    if el["n_trip"] > 0:
        return ("real", "short_trip" if el["apu_minutes_lost"] <= SHORT_TRIP_E else "long_trip",
                f"{el['n_trip']}/{el['n_feed']} APU đang phát bị ngắt quanh cảnh báo ({el['tripped_apus']}); "
                f"mất {el['apu_minutes_lost']:.0f} APU·phút trong 60 phút sau.")
    if is_thermal:
        if th and th["nonphysical"]:
            return "false", "sensor_jump", f"{th['sensor']} nhảy > {JUMP:.0f} °C/phút: cảm biến, không phải nhiệt thật."
        if th and th["at_limit"]:
            return ("real", "temp_at_limit",
                    f"{th['sensor']} = {th['T_max']} °C, cách ngưỡng bảo vệ {th['T_lim']:.0f} °C "
                    f"{th['h']} °C (≤ sai số đo {th['tolerance']} °C).")
        if el["limited"]:
            return "real", "derating", "APU chuyển sang chế độ giới hạn công suất (OpState 3010–3030) quanh cảnh báo."
        if th:
            return ("false", "temp_far_from_limit",
                    f"{th['sensor']} = {th['T_max']} °C, còn cách ngưỡng {th['T_lim']:.0f} °C {th['h']} °C; "
                    "không APU nào ngắt.")
        return "false", "no_temp_data", "Không có số đo nhiệt; không APU nào ngắt."
    if prior_verdict == "false" and str(prior_reason).startswith(PRIOR_POWER_BASED):
        return "false", "no_trip", "Không APU nào ngắt quanh cảnh báo (theo OpState)."
    return prior_verdict, "kept_prior", prior_reason


def apu_names(text) -> list[str]:
    s = str(text)
    return [] if s in ("cả tủ", "nan", "") else [f"APU {int(x)}" for x in re.findall(r"\d+", s)]


def run(incidents: pd.DataFrame, win_dir: Path) -> pd.DataFrame:
    """Tầng 1 cho mọi sự cố. Cột tùy chọn prior_verdict / prior_reason: kết luận của hệ cảnh báo trước đây."""
    hw = hardware_error_codes()
    v = incidents.copy()
    v["inverter"] = v["inverter"].astype(str)
    v["t0"] = pd.to_datetime(v["t0"])
    v = v.sort_values(["inverter", "t0"], kind="stable").reset_index(drop=True)
    rows, cache = [], {}
    for e in v.itertuples():
        if e.inverter not in cache:
            cache = {e.inverter: load_windows(win_dir, e.inverter)}
        w = cache[e.inverter]
        codes = codes_in(e.codes)
        temp_code = bool(codes & TEMP_CODES)
        is_thermal = thermal_cell(getattr(e, "first_device", None), getattr(e, "first_fault_type", None)) or temp_code
        el = electrical(w["trig"], e.t0) if "trig" in w else {}
        th = thermal(w.get("temp"), e.t0, apu_names(e.apu), codes) if is_thermal else {}
        pv = getattr(e, "prior_verdict", "real")
        pr = getattr(e, "prior_reason", "")
        verdict, cls, why = decide(is_thermal, el, th, e.t0, temp_code=temp_code, hardware_code=bool(codes & hw),
                                   prior_verdict="real" if pd.isna(pv) else pv, prior_reason="" if pd.isna(pr) else pr)
        rows.append(dict(tier1_verdict=verdict, tier1_class=cls, tier1_reason=why, has_window=bool(w), **el,
                         **{f"temp_{k}": x for k, x in th.items()}))
    return pd.concat([v, pd.DataFrame(rows)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incidents", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inc = pd.read_csv(a.incidents, dtype={"inverter": str}, encoding="utf-8-sig")
    t = run(inc, Path(a.windows))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{len(t)} sự cố; có cửa sổ log: {int(t['has_window'].sum())}")
    print(t["tier1_class"].map(CLASS_LABEL).value_counts().to_string())


if __name__ == "__main__":
    main()
