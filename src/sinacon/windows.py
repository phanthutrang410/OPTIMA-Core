"""Bước 2 — Trích số đo quanh từng sự cố từ log (chỉ đọc thư mục log).

Với mỗi tủ có sự cố, lấy:
    trig   chuỗi OpState từng APU (bản ghi APU Stat Trig, mili giây), chỉ giữ các lần đổi trạng thái
    temp   nhiệt độ TInd, TL1–TL3 từng APU theo phút (APU Stat 60s) trong [t0 − 180′, t0 + 60′] của mọi sự cố
Đọc được hai dạng log:
    - bảng đã tách: <tủ>/log/APU_Stat_Trig.csv, APU_Stat_60s.csv (lô 01–03/2026);
    - file ngày gộp APS-xxxxxx_YYYYMMDD_HHMMSS.csv (các lô khác); chỉ đọc các ngày chạm cửa sổ.
Mỗi tủ ghi một file windows_<inverter>.pkl; tủ đã có file thì bỏ qua (chạy lại sau khi bị ngắt).

    python -m sinacon.windows --logs "<thư mục log>" --incidents incidents.csv --out <thư mục>
"""
from __future__ import annotations

import argparse
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BEFORE_MIN, AFTER_MIN = 180, 60
TEMPS = ["TInd/°C", "TL1/°C", "TL2/°C", "TL3/°C"]
CABINET_DIR_RE = re.compile(r"(?:INV\s*|log\s*)?(\d+\.\d+)", re.I)
RAW_KEEP = {b"APU Stat Trig": "trig", b"APU Stat 60s": "temp"}


def _retry(fn, tries: int = 3, wait_s: float = 5.0):
    for k in range(tries):
        try:
            return fn()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(wait_s)


def read_split_table(path: Path, cols: list[str]) -> pd.DataFrame:
    """Đọc bảng đã tách. Một số lô có dòng đầu là tiêu đề chung "Log Type,System,Time Stamp,Values..." và tên cột thật
    ở dòng 2; đọc được cả hai kiểu."""
    def read():
        with open(path, encoding="utf-8-sig") as f:
            first = f.readline()
        if "Values..." in first:
            d = pd.read_csv(path, skiprows=1, encoding="utf-8-sig", low_memory=False)
            d = d.rename(columns={d.columns[0]: "Log Type", d.columns[1]: "System", d.columns[2]: "Time Stamp"})[cols]
            d = d[~d["Time Stamp"].astype(str).isin(["TimeStamp", "Time Stamp"])]     # tiêu đề lặp khi ghép ngày
            for c in cols:
                if c not in ("System", "Time Stamp"):
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            return d
        return pd.read_csv(path, usecols=cols, encoding="utf-8-sig", low_memory=False)
    return _retry(read)


def in_windows(ts: pd.Series, t0s: np.ndarray) -> np.ndarray:
    """ts nằm trong ít nhất một cửa sổ [t0 − BEFORE_MIN, t0 + AFTER_MIN]."""
    lo = np.sort(t0s - np.timedelta64(BEFORE_MIN, "m"))
    hi = np.sort(t0s + np.timedelta64(AFTER_MIN, "m"))
    v = ts.to_numpy()
    i = np.searchsorted(lo, v, side="right") - 1
    ok = i >= 0
    out = np.zeros(len(v), bool)
    out[ok] = v[ok] <= hi[i[ok]]
    return out


def read_raw_days(log: Path, t0s: np.ndarray) -> dict[str, pd.DataFrame]:
    """File ngày gộp: mọi loại log trộn chung, mỗi loại có dòng tiêu đề riêng. Chỉ đọc các ngày chạm cửa sổ."""
    days = set()
    for t in pd.to_datetime(t0s):
        days |= {(t - pd.Timedelta(minutes=BEFORE_MIN)).strftime("%Y%m%d"), t.strftime("%Y%m%d"),
                 (t + pd.Timedelta(minutes=AFTER_MIN)).strftime("%Y%m%d")}
    files = sorted(p for p in log.glob("APS-*_*.csv") if p.stem.split("_")[-2] in days)
    buf = {k: [] for k in RAW_KEEP.values()}
    hdr = {}
    for p in files:
        raw = _retry(p.read_bytes)
        for line in raw.splitlines():
            name = RAW_KEEP.get(line.split(b";", 1)[0])
            if name is None:
                continue
            if b";TimeStamp;" in line[:40]:
                hdr[name] = line.decode("latin-1").replace("�", "°").replace("\xb0", "°")
            else:
                buf[name].append(line.decode("latin-1"))
    out = {}
    for name, rows in buf.items():
        if not rows or name not in hdr:
            continue
        cols = hdr[name].split(";")
        cols[:3] = ["Log Type", "System", "Time Stamp"]
        d = pd.read_csv(io.StringIO("\n".join(rows)), sep=";", header=None, low_memory=False)
        d = d.iloc[:, :len(cols)]
        d.columns = cols[:d.shape[1]]
        out[name] = d.drop_duplicates()
    return out


def _state_changes(tr: pd.DataFrame) -> pd.DataFrame:
    tr["t"] = (pd.to_datetime(tr["Time Stamp"]) +
               pd.to_timedelta(pd.to_numeric(tr["Milliseconds/ms"], errors="coerce").fillna(0), unit="ms"))
    tr["OpState"] = pd.to_numeric(tr["OpState"], errors="coerce")
    tr = tr[["System", "t", "OpState"]].dropna().sort_values("t", kind="stable")
    tr = tr[tr.groupby("System")["OpState"].transform(lambda s: s.ne(s.shift()))]
    return tr.reset_index(drop=True)


def _temps(th: pd.DataFrame, t0s: np.ndarray) -> pd.DataFrame:
    th["t"] = pd.to_datetime(th["Time Stamp"])
    for c in TEMPS:
        th[c] = pd.to_numeric(th[c], errors="coerce")
    return th.loc[in_windows(th["t"], t0s), ["System", "t", *TEMPS]].sort_values("t", kind="stable").reset_index(drop=True)


def log_dir_of(folder: Path) -> Path:
    """Thư mục chứa log của một tủ: <tủ>/log, hoặc thư mục con đầu tiên có file ngày (tên không thống nhất)."""
    log = folder / "log"
    if not log.is_dir():
        cands = [folder] + [d for d in folder.iterdir() if d.is_dir() and d.name.upper() != "TR"]
        log = next((d for d in cands if any(d.glob("APS-*_*.csv"))), log)
    return log


def extract_cabinet(inverter: str, folder: Path, t0s: np.ndarray, out_dir: Path) -> str:
    dst = out_dir / f"windows_{inverter}.pkl"
    if dst.exists():
        return f"[{inverter}] đã có"
    log = log_dir_of(folder)
    res: dict = {"inverter": inverter}
    if not (log / "APU_Stat_Trig.csv").exists() and any(log.glob("APS-*_*.csv")):
        try:
            raw = read_raw_days(log, t0s)
        except Exception as e:                                                       # noqa: BLE001
            raw, res["error"] = {}, str(e)
        if "trig" in raw:
            res["trig"] = _state_changes(raw["trig"])
        if "temp" in raw and all(c in raw["temp"] for c in TEMPS):
            res["temp"] = _temps(raw["temp"], t0s)
    else:
        try:
            res["trig"] = _state_changes(read_split_table(
                log / "APU_Stat_Trig.csv", ["System", "Time Stamp", "Milliseconds/ms", "OpState"]))
        except Exception as e:                                                       # noqa: BLE001
            res["error_trig"] = str(e)
        try:
            res["temp"] = _temps(read_split_table(log / "APU_Stat_60s.csv", ["System", "Time Stamp", *TEMPS]), t0s)
        except Exception as e:                                                       # noqa: BLE001
            res["error_temp"] = str(e)
    pd.to_pickle(res, dst)
    missing = [k for k in ("trig", "temp") if k not in res]
    return f"[{inverter}] xong ({len(t0s)} sự cố){' thiếu: ' + ', '.join(missing) if missing else ''}"


def cabinet_folders(root: Path) -> dict[str, Path]:
    folders = {}
    for p in list(root.glob("*/*")) + list(root.glob("*/*/*")):
        m = CABINET_DIR_RE.fullmatch(p.name.strip())
        if p.is_dir() and m:
            folders.setdefault(m.group(1), p)
    return folders


def load_windows(win_dir: Path, inverter: str) -> dict:
    p = Path(win_dir) / f"windows_{inverter}.pkl"
    return pd.read_pickle(p) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--incidents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inc = pd.read_csv(a.incidents, dtype={"inverter": str}, encoding="utf-8-sig")
    inc["t0"] = pd.to_datetime(inc["t0"])
    folders = cabinet_folders(Path(a.logs))
    jobs = []
    for inv, g in inc.groupby("inverter"):
        if inv not in folders:
            print(f"[{inv}] không có thư mục log", flush=True)
            continue
        jobs.append((inv, folders[inv], np.sort(g["t0"].to_numpy()), out))
    with ThreadPoolExecutor(a.workers) as pool:
        for msg in pool.map(lambda j: extract_cabinet(*j), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
