"""Kiểm chứng tầng 1 bằng công suất SCADA (hệ đo của nhà máy, độc lập với log inverter).

Quy tắc định trước:
  1. P_X(t): công suất AC từng phút của inverter X. Tham chiếu: trung vị các tủ KHÁC CÙNG KHỐI (đặt cạnh nhau, chịu
     cùng mây):  r_X(t) = P_X(t) / (k_X · median_{Y cùng khối, Y ≠ X} P_Y(t)),  k_X = trung vị tỉ số khi tham chiếu
     > 1500 kW. r_X ≈ 1 khi X phát như các tủ cùng khối.
  2. Với sự cố lúc t0: đo được khi tham chiếu ≥ 250 kW ở ≥ 80 % số phút của W = [t0 − 2′, t0 + 30′] và trời ổn định
     (hệ số biến thiên của tham chiếu trên [t0 − 10′, t0 + 30′] < 0,05).
     Mức nền b = trung vị r_X trên [t0 − 60′, t0 − 5′] (cần ≥ 20 phút, không thì b = 1).
     Phần mất L = 1 − min_W (trung bình trượt 3 phút của r_X) / b.
  3. L ≥ 0,15 -> "mất công suất"; L ≤ 0,05 -> "giữ công suất"; giữa -> "không rõ". Mất 1 trên 4 APU: L ≈ 0,25.
  Cảnh báo bị kết luận "giả" mà SCADA ghi "mất công suất" là kết luận sai.
Đồng hồ SCADA chậm hơn log khoảng 5 phút; cửa sổ W đã đủ rộng để chứa độ lệch này.

    python -m sinacon.scada read-xls --source "<power report/2026>" --start 2026-04-01 --end 2026-10-31 --out scada.pkl
    python -m sinacon.scada check --tier1 tier1.csv --scada scada.pkl --out scada_check.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

F_CAP = 1500.0                 # kW: phút dùng ước lượng k_X
F_MIN = 500.0                  # kW: tham chiếu tối thiểu = F_MIN / 2
W_BEFORE, W_AFTER = 2, 30
BASE = (60, 5)
LOSS_YES, LOSS_NO = 0.15, 0.05
CV_MAX = 0.05
LOST, KEPT = "mất công suất", "giữ công suất"


# ---------------------------------------------------------------- đọc
def read_power_report_xls(path: Path) -> pd.DataFrame:
    """Một file "Power reports*.xls" (SICAM, 1 phút) -> bảng cột "RADIATION (W/m2)", "BLOCK b INV#i AC|DC"."""
    import xlrd
    sh = xlrd.open_workbook(str(path)).sheet_by_index(0)
    head = next((i for i in range(min(15, sh.nrows)) if "Date Time" in [str(x).strip() for x in sh.row_values(i)]), None)
    if head is None:
        raise ValueError("không thấy dòng 'Date Time'")
    blocks, units = sh.row_values(head), sh.row_values(head + 1)
    tcol = [str(x).strip() for x in blocks].index("Date Time")
    names, cur = {}, ""
    for j in range(tcol + 1, sh.ncols):
        b = str(blocks[j]).strip()
        if b.startswith("BLOCK"):
            cur = b
        u = str(units[j]).strip()
        if b == "RADIATION" or (not cur and u == "(W/m2)"):
            names[j] = "RADIATION (W/m2)"
        elif cur and u.startswith("INV#"):
            names[j] = f"{cur} {u}"
    rows = [[str(r[tcol]).strip()] + [r[j] for j in names]
            for r in (sh.row_values(i) for i in range(head + 2, sh.nrows)) if str(r[tcol]).strip()[:1].isdigit()]
    d = pd.DataFrame(rows, columns=["t", *names.values()])
    d.index = pd.to_datetime(d.pop("t"), format="%d/%m/%Y %H:%M", errors="coerce")
    return d[d.index.notna()].apply(pd.to_numeric, errors="coerce").astype("float32")


def read_power_reports(source: Path, start: str, end: str) -> pd.DataFrame:
    files = sorted(Path(source).rglob("Power reports*.xls"))

    def safe(p):
        try:
            return read_power_report_xls(p)
        except Exception as e:                                       # noqa: BLE001
            print(f"  {p.name}: LỖI {e}", flush=True)
            return None

    with ThreadPoolExecutor(6) as pool:
        parts = [d for d in pool.map(safe, files) if d is not None and len(d)]
    d = pd.concat(parts)
    return d[~d.index.duplicated(keep="first")].sort_index().loc[start:end]


def ac_power(scada: pd.DataFrame) -> pd.DataFrame:
    """Cột công suất AC, đổi tên "BLOCK 1 INV#1 AC" -> "1.1"."""
    ac = scada[[c for c in scada.columns if c.endswith(" AC")]].copy()
    ac.columns = [c.replace("BLOCK ", "").replace(" INV#", ".").replace(" AC", "") for c in ac.columns]
    return ac.clip(lower=0)


def load_scada(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path, compression="gzip" if str(path).endswith(".pkl") else "infer")


# ---------------------------------------------------------------- tỉ số và nhãn
def block_ratios(ac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """r_X so với trung vị các tủ khác cùng khối, và công suất tham chiếu đó."""
    blk = {c: c.split(".")[0] for c in ac.columns}
    r, ref_all = {}, {}
    for c in ac.columns:
        peers = [d for d in ac.columns if blk[d] == blk[c] and d != c]
        if not peers:
            continue
        ref = ac[peers].median(axis=1)
        big = ref > F_CAP
        k = (ac.loc[big, c] / ref[big]).median()
        r[c] = (ac[c] / (k * ref)).where(ref >= F_MIN / 2)
        ref_all[c] = ref
    return pd.DataFrame(r), pd.DataFrame(ref_all)


def sky_cv(ref: pd.Series, t0: pd.Timestamp) -> float:
    x = ref.loc[t0 - pd.Timedelta(minutes=10): t0 + pd.Timedelta(minutes=W_AFTER)]
    return float(x.std() / x.mean()) if len(x) and x.mean() > 0 else np.nan


def power_loss(r: pd.Series, measurable: pd.Series, t0: pd.Timestamp) -> tuple[str, float]:
    w = slice(t0 - pd.Timedelta(minutes=W_BEFORE), t0 + pd.Timedelta(minutes=W_AFTER))
    okw = measurable.loc[w]
    if len(okw) == 0 or okw.mean() < 0.8:
        return "không đo được", np.nan
    base = r.loc[t0 - pd.Timedelta(minutes=BASE[0]): t0 - pd.Timedelta(minutes=BASE[1])].dropna()
    b = float(base.median()) if len(base) >= 20 and base.median() > 0.2 else 1.0
    m = r.loc[w].rolling(3, min_periods=2).mean().min()
    if not np.isfinite(m):
        return "không đo được", np.nan
    x = 1 - m / b
    return (LOST if x >= LOSS_YES else KEPT if x <= LOSS_NO else "không rõ"), round(float(x), 3)


def label_incidents(incidents: pd.DataFrame, scada: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột scada (nhãn), scada_loss (L), sky_cv cho các sự cố trong khoảng thời gian của SCADA."""
    ac = ac_power(scada)
    fleet = ac.median(axis=1)
    measurable = fleet >= F_MIN
    rb, refb = block_ratios(ac)
    v = incidents.copy()
    v["t0"] = pd.to_datetime(v["t0"])
    v["inverter"] = v["inverter"].astype(str)
    v = v[(v["t0"] >= ac.index.min()) & (v["t0"] <= ac.index.max())].copy()
    rows = []
    for e in v.itertuples():
        if e.inverter not in rb.columns:
            rows.append(("không có tủ cùng khối", np.nan, np.nan))
            continue
        cv = sky_cv(refb[e.inverter], e.t0)
        lab, x = power_loss(rb[e.inverter], measurable, e.t0)
        if lab != "không đo được" and not (cv < CV_MAX):
            lab = "không đo được (mây)"
        rows.append((lab, x, round(cv, 3) if pd.notna(cv) else np.nan))
    v[["scada", "scada_loss", "sky_cv"]] = pd.DataFrame(rows, index=v.index)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read-xls")
    r.add_argument("--source", required=True)
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tier1", required=True)
    c.add_argument("--scada", nargs="+", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "read-xls":
        d = read_power_reports(Path(a.source), a.start, a.end)
        d.to_pickle(a.out, compression="gzip")
        print(f"{d.shape}, {d.index.min()} → {d.index.max()} -> {a.out}")
        return
    t = pd.read_csv(a.tier1, dtype={"inverter": str}, encoding="utf-8-sig", low_memory=False)
    parts = [label_incidents(t, load_scada(Path(p))) for p in a.scada]
    v = pd.concat(parts).drop_duplicates(["cabinet", "t0"], keep="last")
    v.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(pd.crosstab(v["tier1_verdict"], v["scada"], margins=True).to_string())
    m = v[v["scada"] == LOST]
    print(f"\nCảnh báo mất công suất (SCADA) bị kết luận giả: {int((m['tier1_verdict'] == 'false').sum())}/{len(m)}")


if __name__ == "__main__":
    main()
