"""Bước 4 — Tầng 2–3: phạm vi và bộ phận lỗi, có xác suất, cho từng sự cố (mô hình Bayes ngây thơ).

    P(k | mã, e) ∝ π_k(mã) · Π_j LR_j(k)^{e_j}

Lớp k (11 lớp): B1–B7 (phạm vi "Trong tủ") và Lưới-trung thế, Phía DC, Điều khiển nhà máy, Thao tác-bảo trì (mỗi lớp
là một phạm vi). B1–B6 gom 230 mã của manual theo hệ thống con; B7 (bộ lọc AC, cầu chì) thêm từ manual tr. 326, 442,
464 và sổ bảo hành 2022–2025.

1. Tiên nghiệm theo mã. Mỗi mã c thuộc nhóm g(c) của danh mục hãng (config/fault_codes.csv), nhóm ánh xạ sang lớp
   (GROUP_TO_CLASS; một số mã theo SPLIT). Trọng số w_c = 1 (error), 0,3 (warning); mã chỉ là hệ quả (mất Powerlink,
   mất mẫu log) 0,3 / 0,1. Mã mất Powerlink của chính APU đang hỏng/tách: trọng số 0. Các mã của một sự cố là triệu
   chứng của MỘT sự kiện nên tổng khối lượng chuẩn hóa về 1:  π_k ∝ α + Σ_c w_c m_c(k) / Σ_c w_c,  α = 0,05.
   "Supply voltage low" (APS 19) đi một mình -> B4; kèm "Medium voltage switch gear off" (APS 7) -> Lưới.
   Mã điện áp/tần số AC cấp APU (nhóm G07) chia đôi Lưới / B7.
2. Bằng chứng e_j ∈ {0, 1} đo từ log quanh t0 (hàm features) nhân xác suất lớp với LR_j(k) (bảng LR). Sự kiện nhiều
   tủ: tỉ lệ trùng ngẫu nhiên đo trên 5.000 thời điểm ban ngày (≥ 5 tủ khác trong ±2′: 0/5000) -> LR Lưới 300.
   Tần suất hỏng thật (khi có APU hỏng/tách): LR_k = 7·ρ_k cắt về [0,3; 3], ρ_k từ sổ bảo hành 2022–2025.
3. Chuẩn hóa. Tầng 2 = tổng xác suất theo phạm vi; tầng 3 = lớp có xác suất cao nhất. Tập 90 %: các lớp cao nhất
   cho tới khi tổng ≥ 0,9.
4. Loại sự kiện: "new" khi có APU ngắt, nhiệt tới ngưỡng, hoặc bộ mã chưa xuất hiện ở tủ trong 7 ngày trước;
   "persistent" khi bộ mã lặp lại mà không có sự kiện vật lý mới. Gộp nhiều cảnh báo của một lần hỏng: trung bình có
   trọng số (1 + số APU ngắt) của các sự cố "new" (hàm aggregate_case).

    python -m sinacon.tier23 --tier1 tier1_09.csv tier1_10.csv --windows <thư mục> --out tier23.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sinacon.config import code_groups, codes_in, hardware_error_codes
from sinacon.tier1 import TEMP_CODES
from sinacon.windows import load_windows

K = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "GRID", "DC", "PLANT_CTRL", "OPERATION"]
LABEL = {"B1": "B1 Nhiệt-làm mát", "B2": "B2 DC-link-nạp trước", "B3": "B3 Dòng-IGBT-driver",
         "B4": "B4 Nguồn phụ-liên động", "B5": "B5 Cách điện-nối đất", "B6": "B6 Điều khiển-truyền thông",
         "B7": "B7 Bộ lọc AC-cầu chì", "GRID": "Lưới-trung thế", "DC": "Phía DC", "PLANT_CTRL": "Điều khiển nhà máy",
         "OPERATION": "Thao tác-bảo trì"}
INSIDE = "Trong tủ"
SCOPE = {**{k: INSIDE for k in ("B1", "B2", "B3", "B4", "B5", "B6", "B7")},
         **{k: LABEL[k] for k in ("GRID", "DC", "PLANT_CTRL", "OPERATION")}}
GROUP_TO_CLASS = {"G01": "B1", "G02": "B1", "G03": "B2", "G12": "B2", "G04": "B3", "G06": "B3", "G05": "B4",
                  "G08": "B5", "G09": "B6", "G10": "B6", "G11": "B6", "G07": "GRID"}
SPLIT = {                                              # mã mà nhóm danh mục chưa phản ánh đúng nguồn
    "APS/error/1": {"PLANT_CTRL": 1.0},                # Modbus TCP value out of range: lệnh từ bộ điều khiển nhà máy
    "APS/error/5": {"B4": 0.5, "OPERATION": 0.5},      # No enable signal
    "APU/error/62": {"B4": 0.5, "OPERATION": 0.5},     # Enable 24V
    "APS/error/7": {"GRID": 1.0}, "APS/error/9": {"GRID": 1.0}, "APS/error/19": {"GRID": 1.0},
    "APU/error/10": {"B2": 0.5, "DC": 0.5}, "APU/error/85": {"B2": 0.5, "DC": 0.5}, "APU/error/87": {"B2": 0.5, "DC": 0.5},
    "APS/warning/43": {"OPERATION": 1.0},              # APU in manual Mode
    "APS/error/55": {"B7": 1.0}, "APS/error/56": {"B7": 1.0},   # Island Fuses: cầu chì bộ lọc sóng hài (tr. 326, 464)
}
AC_APU = {f"APU/error/{n}" for n in [16, *range(17, 37), 71, 72, 96, 97, 98, 99, 103, 104, 108, 109, 110]}
CONSEQUENCE = {f"APS/error/{n}" for n in range(30, 36)} | {f"APS/warning/{n}" for n in range(29, 35)} | {"APS/warning/47"}
POWERLINK_APU = {**{f"APS/error/{29 + i}": f"APU {i}" for i in range(1, 7)},
                 **{f"APS/warning/{28 + i}": f"APU {i}" for i in range(1, 7)}}
ALPHA = 0.05

# Số lần hỏng theo lớp, sổ bảo hành 2022–2025 (sinacon.warranty, gộp theo tủ-ngày-lớp).
FAILURES_2022_2025 = {"B1": 7, "B2": 114, "B3": 116, "B4": 31, "B5": 0, "B6": 25, "B7": 77}
_N = sum(FAILURES_2022_2025.values())
LR_FAILURE_FREQUENCY = {k: float(np.clip(7 * (n + 1) / (_N + 7), 0.3, 3.0)) for k, n in FAILURES_2022_2025.items()}

T_IND_NEAR, T_IGBT_NEAR = 155.0, 104.0
# Tỉ số hợp lý: bằng chứng -> {lớp: hệ số}; lớp không ghi có hệ số 1. Giá trị đặt trước khi chấm.
LR = {
    "temp_near_limit": {"B1": 10.0},                                  # TInd ≥ 155 °C hoặc IGBT ≥ 104 °C
    "temp_low": {"B1": 0.3},                                          # TInd < 130 và IGBT < 90 °C (chỉ với mã nhiệt độ)
    "thermal_cycling": {"B1": 5.0},                                   # ≥ 2 lần 30xx -> 500 trong 30′
    "many_restarts": {"B2": 2.0, "B3": 3.0, "B4": 3.0, "B1": 0.5},    # ≥ 20 lần đổi trạng thái/APU
    "cabinet_stopped": {"OPERATION": 5.0, "GRID": 3.0, "PLANT_CTRL": 2.0},   # mọi APU dừng, ≥ 30′ chưa phát lại
    "few_cabinets": {"GRID": 10.0, "PLANT_CTRL": 3.0},                # 3–4 tủ khác báo trong ±2′
    "only_this_cabinet": {"GRID": 0.5},                               # không tủ nào khác báo
    "apu_stuck_off": {"B3": 3.0},                                     # chỉ dùng khi không có thông tin APU hỏng/tách
    "plant_event_2min": {"GRID": 300.0, "PLANT_CTRL": 30.0},          # ≥ 5 tủ khác báo trong ±2′
    "plant_event_15min": {"GRID": 100.0, "PLANT_CTRL": 10.0},         # ≥ 20 tủ khác trong ±15′ (khi ±2′ < 5)
    "hardware_history": {"B3": 10.0},                                 # APU hỏng đã từng ngắt kèm mã phần cứng
    "local_ac_fault": {"B7": 5.0},                                    # mã AC cấp APU, chỉ tủ này báo
    "failure_frequency": LR_FAILURE_FREQUENCY,                        # có APU hỏng/tách kéo dài
}
HISTORY_DAYS = 180


def code_prior(codes_text: str, groups: dict, apus_out: set | None = None) -> np.ndarray:
    pi = np.full(len(K), ALPHA)
    mass = np.zeros(len(K))
    codes = codes_in(codes_text)
    for c in codes:
        if apus_out and POWERLINK_APU.get(c) in apus_out:
            continue
        err = "/error/" in c
        w = (0.3 if err else 0.1) if c in CONSEQUENCE else (1.0 if err else 0.3)
        target = SPLIT.get(c) or ({GROUP_TO_CLASS[groups[c]]: 1.0} if groups.get(c) in GROUP_TO_CLASS else {})
        if c in AC_APU:
            target = {"GRID": 0.5, "B7": 0.5}
        if c == "APS/error/19" and "APS/error/7" not in codes:
            target = {"B4": 1.0}
        for k, share in target.items():
            mass[K.index(k)] += w * share
    if mass.sum() > 0:
        mass = mass / mass.sum()
    pi = pi + mass
    return pi / pi.sum()


def features(tr: pd.DataFrame | None, temp: pd.DataFrame | None, t0: pd.Timestamp, n_other: int,
             n_repeat_days: int, codes_text: str | None = None, n_other_15: int = 0, history: bool = False,
             n_apu_out: int | None = None) -> dict:
    f = dict.fromkeys(LR, 0)
    temp_alarm = codes_text is None or bool(codes_in(codes_text) & TEMP_CODES)
    if temp is not None and not temp.empty:
        w = temp[(temp["t"] >= t0 - pd.Timedelta(minutes=5)) & (temp["t"] <= t0 + pd.Timedelta(minutes=1))]
        if len(w):
            ti = float(w["TInd/°C"].max())
            tg = float(w[["TL1/°C", "TL2/°C", "TL3/°C"]].max().max())
            f["temp_near_limit"] = int(ti >= T_IND_NEAR or tg >= T_IGBT_NEAR)
            f["temp_low"] = int(ti < 130 and tg < 90 and temp_alarm)
    if tr is not None and not tr.empty:
        win = tr[(tr["t"] > t0 - pd.Timedelta(minutes=2)) & (tr["t"] <= t0 + pd.Timedelta(minutes=30))]
        pre = tr[tr["t"] <= t0 - pd.Timedelta(minutes=2)].groupby("System")["OpState"].last()
        feeding = [a for a, s in pre.items() if s >= 3000]
        cyc = 0
        for a, g in win.groupby("System"):
            s = g["OpState"].to_numpy()
            cyc = max(cyc, int(((s[:-1] >= 3000) & (s[1:] == 500)).sum()
                               + (pre.get(a, 0) >= 3000 and len(s) and s[0] == 500)))
        f["thermal_cycling"] = int(cyc >= 2)
        f["many_restarts"] = int(win.groupby("System").size().max() >= 20) if len(win) else 0
        if feeding:
            last = tr[tr["t"] <= t0 + pd.Timedelta(minutes=30)].groupby("System")["OpState"].last()
            stopped = [a for a in feeding if a in last and last[a] < 3000]
            any_stop = [a for a in feeding if (win[win["System"] == a]["OpState"] < 3000).any()]
            f["cabinet_stopped"] = int(len(stopped) == len(feeding) and len(any_stop) == len(feeding)
                                       and 8 <= t0.hour < 16)
        dead = [a for a, s in pre.items() if s in (30, 50) and win[win["System"] == a].empty]
        f["apu_stuck_off"] = int(bool(dead) and n_repeat_days >= 3 and 8 <= t0.hour < 16)
    if n_apu_out is not None:
        f["apu_stuck_off"] = 0
        f["failure_frequency"] = int(n_apu_out > 0)
    f["few_cabinets"] = int(3 <= n_other < 5)
    f["plant_event_2min"] = int(n_other >= 5)
    f["plant_event_15min"] = int(n_other < 5 and n_other_15 >= 20)
    f["hardware_history"] = int(bool(history))
    f["only_this_cabinet"] = int(n_other == 0 and n_other_15 < 20)
    f["local_ac_fault"] = int(f["only_this_cabinet"] == 1 and bool(codes_in(codes_text or "") & AC_APU))
    return f


def posterior(pi: np.ndarray, f: dict) -> np.ndarray:
    p = pi.copy()
    for j, on in f.items():
        if on:
            for k, r in LR[j].items():
                p[K.index(k)] *= r
    return p / p.sum()


def summarize(p: np.ndarray) -> dict:
    order = np.argsort(-p)
    s, cover = 0.0, []
    for i in order:
        cover.append(K[i])
        s += p[i]
        if s >= 0.9:
            break
    scope = {}
    for k, v in zip(K, p):
        scope[SCOPE[k]] = scope.get(SCOPE[k], 0) + v
    top_scope = max(scope, key=scope.get)
    return dict(
        top1=K[order[0]], p_top1=round(float(p[order[0]]), 3), top2=K[order[1]], p_top2=round(float(p[order[1]]), 3),
        set90=" | ".join(cover), set90_size=len(cover), scope=top_scope, p_scope=round(scope[top_scope], 3),
        **{f"p_{k}": round(float(v), 3) for k, v in zip(K, p)},
        p_inside=round(float(sum(p[K.index(k)] for k in K if SCOPE[k] == INSIDE)), 3),
    )


def event_type(verdict: str, tier1_class: str, n_trip: float, first_in_7_days: bool) -> str:
    if verdict != "real":
        return "none"
    if (n_trip or 0) > 0 or tier1_class in ("temp_at_limit", "derating"):
        return "new"
    return "new" if first_in_7_days else "persistent"


def aggregate_case(x: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Phân bố của một lần hỏng từ các sự cố thật x (cột p_<lớp>, event_type, n_trip)."""
    pk = [f"p_{k}" for k in K]
    m = x[x["event_type"] == "new"]
    if m.empty:
        return x[pk].mean().to_numpy(), "all real"
    w = 1 + m["n_trip"].fillna(0).to_numpy()
    return (m[pk].to_numpy() * w[:, None]).sum(0) / w.sum(), "new events"


def count_other_cabinets(times: np.ndarray, invs: np.ndarray, inv: str, t0: pd.Timestamp, minutes: int) -> int:
    lo = np.searchsorted(times, (t0 - pd.Timedelta(minutes=minutes)).to_datetime64())
    hi = np.searchsorted(times, (t0 + pd.Timedelta(minutes=minutes)).to_datetime64(), side="right")
    return len(set(invs[lo:hi]) - {inv})


def hardware_trip_events(t: pd.DataFrame) -> pd.DataFrame:
    """Ngắt kèm mã phần cứng công suất, không tự phục hồi (≥ 30 APU·phút), không phải sự kiện nhiều tủ."""
    hw = hardware_error_codes()
    x = t[t["codes"].map(lambda s: bool(codes_in(s) & hw))
          & (t["n_trip"].fillna(0) > 0) & (t["apu_minutes_lost"].fillna(0) >= 30)]
    ts, iv = t["t0"].to_numpy(), t["inverter"].to_numpy()
    keep = [count_other_cabinets(ts, iv, r.inverter, r.t0, 2) < 5 and count_other_cabinets(ts, iv, r.inverter, r.t0, 15) < 20
            for r in x.itertuples()]
    return x[keep][["inverter", "t0", "tripped_apus"]]


def has_history(ev: pd.DataFrame, inv: str, t0: pd.Timestamp, apus: set) -> bool:
    e = ev[(ev["inverter"] == inv) & (ev["t0"] < t0 - pd.Timedelta(hours=1))
           & (ev["t0"] >= t0 - pd.Timedelta(days=HISTORY_DAYS))]
    return any(apus & set(re.findall(r"APU \d", str(a))) for a in e["tripped_apus"])


def run(tier1: pd.DataFrame, win_dir: Path | list[Path], context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tầng 2–3 cho mọi sự cố trong tier1. context: mọi sự cố đã có tầng 1 (kể cả các tháng khác), dùng đếm tủ khác
    báo trong ±15′ và tìm tiền sử ngắt phần cứng; mặc định là chính tier1."""
    dirs = [Path(d) for d in (win_dir if isinstance(win_dir, list) else [win_dir])]
    groups = code_groups()
    t = tier1.copy()
    t["t0"] = pd.to_datetime(t["t0"])
    t["inverter"] = t["inverter"].astype(str)
    t["code_set"] = t["codes"].map(lambda s: " ".join(sorted(codes_in(s))))
    t = t.sort_values(["inverter", "t0"], kind="stable").reset_index(drop=True)
    ctx = (t if context is None else context.assign(t0=pd.to_datetime(context["t0"]),
                                                     inverter=context["inverter"].astype(str)))
    ctx = ctx.drop_duplicates(["cabinet", "t0"]).sort_values("t0", kind="stable").reset_index(drop=True)
    cts, civ = ctx["t0"].to_numpy(), ctx["inverter"].to_numpy()
    ev = hardware_trip_events(ctx)
    own = t[["inverter", "t0"]].sort_values("t0", kind="stable")
    rows, cache = [], {}
    for e in t.itertuples():
        if e.inverter not in cache:
            cache = {e.inverter: next((w for d in dirs if (w := load_windows(d, e.inverter))), {})}
        w = cache[e.inverter]
        near = own[(own["t0"] - e.t0).abs() <= pd.Timedelta(minutes=2)]
        n_other = near.loc[near["inverter"] != e.inverter, "inverter"].nunique()
        same = t[(t["inverter"] == e.inverter) & ((t["t0"] - e.t0).abs() <= pd.Timedelta(days=7))
                 & (t["code_set"] == e.code_set)]
        repeat_days = same["t0"].dt.normalize().nunique()
        first = not ((same["t0"] < e.t0.normalize()) & (same["t0"] >= e.t0 - pd.Timedelta(days=7))).any()
        out = set(re.findall(r"APU \d", str(getattr(e, "apus_out", "") or "")))
        n15 = count_other_cabinets(cts, civ, e.inverter, e.t0, 15)
        hist = has_history(ev, e.inverter, e.t0, out) if out else False
        nout = getattr(e, "n_apu_out", None)
        f = features(w.get("trig"), w.get("temp"), e.t0, n_other, repeat_days, e.codes, n_other_15=n15,
                     history=hist, n_apu_out=None if nout is None or pd.isna(nout) else int(nout))
        p = posterior(code_prior(e.codes, groups, out), f)
        rows.append(dict(**summarize(p), **{f"e_{k}": v for k, v in f.items()}, n_other_2min=n_other,
                         n_other_15min=n15, repeat_days=repeat_days,
                         event_type=event_type(e.tier1_verdict, e.tier1_class, e.n_trip, first)))
    return pd.concat([t.drop(columns="code_set"), pd.DataFrame(rows)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier1", nargs="+", required=True, help="kết quả tầng 1 cần phân loại")
    ap.add_argument("--context", nargs="*", default=[], help="kết quả tầng 1 các tháng khác (tiền sử, sự kiện nhiều tủ)")
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    def read(p):
        t = pd.read_csv(p, dtype={"inverter": str}, encoding="utf-8-sig", low_memory=False)
        return t if "period" in t else t.assign(period=Path(p).stem)     # giai đoạn log = tên file

    t1 = pd.concat([read(p) for p in a.tier1], ignore_index=True)
    ctx = pd.concat([t1, *[read(p) for p in a.context]], ignore_index=True) if a.context else None
    t = run(t1, [Path(d) for d in a.windows], ctx)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False, encoding="utf-8-sig")
    real = t[t["tier1_verdict"] == "real"]
    print(f"{len(t)} sự cố, {len(real)} thật. Lớp cao nhất (sự cố thật):")
    print(real["top1"].map(LABEL).value_counts().to_string())


if __name__ == "__main__":
    main()
