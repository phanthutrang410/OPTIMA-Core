"""Kiểm chứng tầng 3.

(a) Sổ thay thế linh kiện (config/om_replacements.csv): mỗi lần hỏng (tủ, ngày, linh kiện) có log, lấy các sự cố THẬT
    của tủ đó trong [ngày − 3, ngày + 1], gộp phân bố (tier23.aggregate_case), so lớp cao nhất và hai lớp cao nhất với
    linh kiện đã thay:  IGBT, driver -> B3; tụ, cầu chì, bộ lọc sóng hài -> B7; máy cắt, công tắc, cơ cấu nạp -> B2;
    ultracap -> B4; nước làm mát -> B1; mức dầu MBA -> Lưới. Sổ chỉ ghi lần thay linh kiện: sự cố tự phục hồi không có.
    Phiếu trước 01/04/2026 đã được xem khi xây dựng phương pháp (cột independent = False).

(b) Hậu quả ngày sau, đo bằng SCADA (không cần nhãn). Hỏng phần cứng (B3) thì APU nằm chờ sửa, tủ phát thiếu ở các
    ngày sau; quá nhiệt (B1), sự kiện lưới thì APU tự phát lại. Mô hình chỉ thấy log tới t0 + 60′.
    Đơn vị: sự cố thật, loại "new", ≥ 1 APU bị ngắt, t0 trong 08:00–16:30 (sự cố cùng tủ cùng ngày gộp một, lấy p_B3
    lớn nhất). R_X(d) = trung vị r_X (sinacon.scada.block_ratios) 10:00–14:00 ngày d, tham chiếu > 1500 kW, ≥ 60 phút.
    Thiếu kéo dài: R ngày gần nhất trong [d − 3, d − 1] ≥ 0,90 và trung vị R trong [d + 1, d + 2] ≤ 0,85.

    python -m sinacon.validation om --tier23 tier23.csv --out om_check.csv
    python -m sinacon.validation next-day --tier23 tier23.csv --scada scada_*.pkl --out next_day.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sinacon import scada as sc
from sinacon.config import CONFIG
from sinacon.tier23 import LABEL, K, aggregate_case

OM_INDEPENDENT_FROM = pd.Timestamp("2026-04-01")
HEALTHY, SHORT = 0.90, 0.85


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson (1927) cho tỉ lệ k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 2), round(float(c + h), 2))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- (a) sổ thay thế
def om_truth(om: pd.DataFrame) -> pd.Series:
    txt = (om["Device"].fillna("") + " " + om["Fault_device"].fillna("") + " " + om["Note"].fillna("")).str.lower()
    return pd.Series(np.select([txt.str.contains("igbt|driver"), txt.str.contains("harmonic|capicitor|capacitor|tụ|chì"),
                                txt.str.contains("breaker|switch|nạp tải"), txt.str.contains("ultracap"),
                                txt.str.contains("nước làm mát"), txt.str.contains("dầu")],
                               ["B3", "B7", "B2", "B4", "B1", "GRID"], default="?"), index=om.index)


def evaluate_om(tab: pd.DataFrame, replacements: Path | None = None) -> pd.DataFrame:
    om = pd.read_csv(replacements or CONFIG / "om_replacements.csv", dtype=str, encoding="utf-8-sig")
    om["d"] = pd.to_datetime(om["Fault_date"])
    om["truth"] = om_truth(om)
    om = om.drop_duplicates(["Inverter_ID", "Fault_date", "truth"])
    t = tab.copy()
    t["t0"] = pd.to_datetime(t["t0"])
    t["inverter"] = t["inverter"].astype(str)
    period = t.get("period", pd.Series("all", index=t.index))
    spans = t.groupby(period)["t0"].agg(["min", "max"])
    pk = [f"p_{k}" for k in K]
    rows = []
    for r in om.itertuples():
        base = dict(inverter=r.Inverter_ID, date=r.Fault_date, truth=r.truth, independent=bool(r.d >= OM_INDEPENDENT_FROM))
        if not ((spans["min"] - pd.Timedelta(days=1) <= r.d) & (r.d <= spans["max"] + pd.Timedelta(days=1))).any():
            rows.append(dict(**base, result="ngoài khoảng log"))
            continue
        x = t[(t["inverter"] == r.Inverter_ID) & (t["tier1_verdict"] == "real")
              & (t["t0"] >= r.d - pd.Timedelta(days=3)) & (t["t0"] < r.d + pd.Timedelta(days=2))]
        if x.empty:
            rows.append(dict(**base, result="không có sự cố thật"))
            continue
        m, how = aggregate_case(x)
        order = [K[i] for i in np.argsort(-m)]
        rows.append(dict(**base, note=r.Note, n_incidents=len(x), n_new=int((x["event_type"] == "new").sum()),
                         aggregation=how, pred_1=order[0], p_1=round(float(m.max()), 2), pred_2=order[1],
                         top1=order[0] == r.truth, top2=r.truth in order[:2], result="chấm"))
    return pd.DataFrame(rows)


def report_om(r: pd.DataFrame) -> None:
    s = r[r["result"] == "chấm"]
    print(r["result"].value_counts().to_string())
    for name, x in (("tất cả", s), ("độc lập (từ 01/04/2026)", s[s["independent"]])):
        k1, k2, n = int(x["top1"].sum()), int(x["top2"].sum()), len(x)
        print(f"{name}: lớp cao nhất {k1}/{n} {wilson(k1, n)}; hai lớp cao nhất {k2}/{n} {wilson(k2, n)}")
    if len(s):
        g = s.astype({"top1": int, "top2": int}).groupby("truth").agg(n=("top1", "size"), top1=("top1", "sum"),
                                                                          top2=("top2", "sum"))
        print(g.rename(index=lambda k: LABEL.get(k, k)).to_string())


# ---------------------------------------------------------------- (b) hậu quả ngày sau
def daily_ratio(scada_files: list[Path]) -> pd.DataFrame:
    out = []
    for p in scada_files:
        r, ref = sc.block_ratios(sc.ac_power(sc.load_scada(p)))
        h = r.index.hour
        keep = (h >= 10) & (h < 14)
        rr = r[keep].where(ref[keep] > sc.F_CAP)
        day = rr.index.normalize()
        out.append(rr.groupby(day).median().where(rr.notna().groupby(day).sum() >= 60))
    d = pd.concat(out)
    return d[~d.index.duplicated(keep="last")].sort_index()


def evaluate_next_day(tab: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    t = tab.copy()
    t["t0"] = pd.to_datetime(t["t0"])
    t["inverter"] = t["inverter"].astype(str)
    m = t["t0"].dt.hour * 60 + t["t0"].dt.minute
    t = t[(t["tier1_verdict"] == "real") & (t["event_type"] == "new") & (t["n_trip"].fillna(0) >= 1)
          & (m >= 480) & (m < 990)]
    t = t.drop_duplicates(["cabinet", "t0"])
    t["day"] = t["t0"].dt.normalize()
    t = t.sort_values("p_B3", ascending=False).drop_duplicates(["inverter", "day"])
    rows = []
    for e in t.itertuples():
        if e.inverter not in R.columns:
            continue
        s = R[e.inverter]
        before = s.loc[e.day - pd.Timedelta(days=3): e.day - pd.Timedelta(days=1)].dropna()
        after = s.loc[e.day + pd.Timedelta(days=1): e.day + pd.Timedelta(days=2)].dropna()
        if before.empty or after.empty:
            continue
        rows.append(dict(inverter=e.inverter, day=e.day, top1=e.top1, p_B3=e.p_B3, R_before=before.iloc[-1],
                         R_after=after.median()))
    x = pd.DataFrame(rows, columns=["inverter", "day", "top1", "p_B3", "R_before", "R_after"])
    x = x[x["R_before"] >= HEALTHY].copy()
    x["short_after"] = (x["R_after"] <= SHORT).astype(int)
    return x


def report_next_day(x: pd.DataFrame) -> None:
    print(f"{len(x)} sự cố (tủ đủ công suất trước đó), thiếu công suất ngày sau: {int(x['short_after'].sum())}")
    for k, g in x.groupby("top1"):
        k1 = int(g["short_after"].sum())
        print(f"  {LABEL[k]:28s}: {k1:3d}/{len(g):3d} = {k1 / len(g):5.1%}  {wilson(k1, len(g))}")
    print(f"  AUC của p_B3: {auc(x['p_B3'].to_numpy(), x['short_after'].to_numpy()):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("om")
    o.add_argument("--tier23", nargs="+", required=True)
    o.add_argument("--replacements", default=None)
    o.add_argument("--out", required=True)
    n = sub.add_parser("next-day")
    n.add_argument("--tier23", nargs="+", required=True)
    n.add_argument("--scada", nargs="+", required=True)
    n.add_argument("--out", required=True)
    a = ap.parse_args()
    tab = pd.concat([pd.read_csv(p, dtype={"inverter": str}, encoding="utf-8-sig", low_memory=False)
                     for p in a.tier23], ignore_index=True)
    if a.cmd == "om":
        r = evaluate_om(tab, Path(a.replacements) if a.replacements else None)
        r.to_csv(a.out, index=False, encoding="utf-8-sig")
        report_om(r)
    else:
        x = evaluate_next_day(tab, daily_ratio([Path(p) for p in a.scada]))
        x.to_csv(a.out, index=False, encoding="utf-8-sig")
        report_next_day(x)


if __name__ == "__main__":
    main()
