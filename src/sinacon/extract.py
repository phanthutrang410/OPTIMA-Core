"""Đọc log thô MỘT lần, cất phần cần dùng ra file nhỏ (bảng trung gian) để các bước sau tính nhanh.

Mỗi tủ một file `<tủ>.pkl.gz` gồm:
  - errors:   scope (APU/APS), unit (số APU, 0 = cả tủ), t, opstate, codes (mã error đang bật)  ← APU/APS Stat Trig
  - warnings: t, codes (warning APS đang bật)                                                  ← APS Stat Trig
  - temps:    System (APU 1–4), t, TInd, TL1, TL2, TL3, TPCB, Hum                               ← APU Stat 60s
  - meta:     tủ, mã APS, thư mục nguồn, số file / số ngày, thời điểm đầu / cuối
Đọc được cả log ngày (APS-xxxxxx_YYYYMMDD_hhmmss.csv) lẫn bảng tách gộp cả đợt (APU_Stat_Trig.csv…).
Tủ đã có file thì bỏ qua (chạy lại sau khi Drive ngắt sẽ làm tiếp chỗ dở).
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import pandas as pd

from sinacon.config import cabinet_to_inverter
from sinacon.daily_log import INVERTER_DIR_RE, LOG_RE, _codes, _read_retry, iter_log_files

TEMP_COLS = ["TInd", "TL1", "TL2", "TL3", "TPCB", "Hum"]


def _rows_daily(cab_dir: Path):
    """(loại dòng, các ô) của mọi dòng cần dùng trong các log ngày của tủ; và số file, mã APS."""
    n, aps = 0, ""
    for name, read in iter_log_files(cab_dir):
        data = _read_retry(read)
        if data is None:
            print(f"   BỎ QUA (không đọc được): {name}", flush=True)
            continue
        n += 1
        aps = aps or LOG_RE.search(name).group(1)
        for line in io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace"):
            if line.startswith(("APU Stat Trig;", "APS Stat Trig;", "APU Stat 60s;")):
                yield line.rstrip("\r\n").split(";")
    _rows_daily.last = (n, aps)


def _rows_split(log_dir: Path):
    """Như _rows_daily nhưng đọc bảng tách (dấu phẩy, có thể có tiêu đề 'Values...' và tiêu đề lặp)."""
    for table in ("APU_Stat_Trig", "APS_Stat_Trig", "APU_Stat_60s"):
        f = log_dir / f"{table}.csv"
        if not f.is_file():
            continue
        with f.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            for r in csv.reader(fh):
                if len(r) > 3 and r[2][:1] == "2":                  # dòng số liệu (thời điểm 2xxx-…)
                    yield r


def _parse(rows) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    err, warn, temp = [], [], []
    for p in rows:
        kind = p[0]
        if len(p) < 5 or p[2][:1] != "2":
            continue
        if kind == "APU Stat Trig" and p[1].startswith("APU "):
            err.append(("APU", int(p[1][4:]), p[2], int(p[3] or 0), int(p[4]), _codes(p[15:23])))
        elif kind == "APS Stat Trig" and p[1] == "APS":
            err.append(("APS", 0, p[2], int(p[3] or 0), int(p[4]), _codes(p[5:13])))
            warn.append((p[2], int(p[3] or 0), frozenset(_codes(p[13:21]))))
        elif kind == "APU Stat 60s" and p[1].startswith("APU "):
            temp.append((p[1], p[2], *p[3:9]))
    errors = pd.DataFrame(err, columns=["scope", "unit", "ts", "ms", "opstate", "codes"]).drop_duplicates()
    errors["t"] = pd.to_datetime(errors["ts"]) + pd.to_timedelta(errors["ms"], unit="ms")
    errors = errors.sort_values(["scope", "unit", "t"])[["scope", "unit", "t", "opstate", "codes"]].reset_index(drop=True)
    w = pd.DataFrame(warn, columns=["ts", "ms", "codes"]).drop_duplicates(["ts", "ms"])
    w["t"] = pd.to_datetime(w["ts"]) + pd.to_timedelta(w["ms"], unit="ms")
    w = w.sort_values("t")[["t", "codes"]].reset_index(drop=True)
    tp = pd.DataFrame([r + (None,) * (8 - len(r)) for r in temp], columns=["System", "t", *TEMP_COLS])
    tp["t"] = pd.to_datetime(tp["t"])
    for c in TEMP_COLS:
        tp[c] = pd.to_numeric(tp[c], errors="coerce").astype("float32")
    tp = tp.drop_duplicates(["System", "t"]).sort_values(["t", "System"]).reset_index(drop=True)
    return errors, w, tp


def extract_cabinet(cab_dir: Path, inverter: str) -> dict:
    """Bảng trung gian của một tủ (đọc thẳng thư mục trên Drive, không dựng lại log ngày)."""
    cab_dir = Path(cab_dir)
    split = next((d for d in (cab_dir / "log", cab_dir) if (d / "APU_Stat_Trig.csv").is_file()), None)
    expected = {v: k for k, v in cabinet_to_inverter().items()}.get(inverter, "")
    if split is not None:
        errors, warnings, temps = _parse(_rows_split(split))
        n_files, aps = 3, expected
    else:
        _rows_daily.last = (0, "")
        errors, warnings, temps = _parse(_rows_daily(cab_dir))
        n_files, aps = _rows_daily.last
    if aps and expected and aps != expected:
        print(f"   !!! thư mục {inverter} chứa log {aps}, nhưng {inverter} là {expected} (thư mục xếp nhầm?)")
    meta = dict(inverter=inverter, aps=aps, expected_aps=expected, source=str(cab_dir), n_files=n_files,
                format="bảng tách" if split is not None else "log ngày",
                t_first=errors["t"].min() if len(errors) else None, t_last=errors["t"].max() if len(errors) else None,
                n_days=int(errors["t"].dt.normalize().nunique()) if len(errors) else 0)
    return dict(errors=errors, warnings=warnings, temps=temps, meta=meta)


def cabinet_dirs(period_root: Path) -> dict[str, Path]:
    """{tủ: thư mục} cho mọi thư mục tủ dưới thư mục đợt (tên dạng "9.3", "INV 9.3")."""
    out = {}
    for d in sorted(Path(period_root).rglob("*")):
        m = INVERTER_DIR_RE.fullmatch(d.name.strip()) if d.is_dir() else None
        if m and m.group(1) not in out and (any(d.glob("log")) or any(d.glob("*.csv"))):
            out[m.group(1)] = d
    return dict(sorted(out.items(), key=lambda kv: [int(x) for x in kv[0].split(".")]))


def extract_period(period_root: Path, out_dir: Path, cabinets: list[str] | None = None) -> pd.DataFrame:
    """Trích mọi tủ của một đợt vào out_dir (bỏ qua tủ đã có). Trả về bảng tóm tắt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = cabinet_dirs(period_root)
    if cabinets:
        dirs = {k: v for k, v in dirs.items() if k in set(cabinets)}
    print(f"Tìm thấy {len(dirs)} tủ trong {period_root}", flush=True)
    for i, (inv, d) in enumerate(dirs.items(), 1):
        f = out_dir / f"{inv}.pkl.gz"
        if f.is_file():
            continue
        t = time.time()
        try:
            x = extract_cabinet(d, inv)
            pd.to_pickle(x, f)
            m = x["meta"]
            print(f"[{i}/{len(dirs)}] {inv} {m['aps']}: {m['n_days']} ngày, {len(x['errors'])} dòng trạng thái, "
                  f"{len(x['temps'])} dòng nhiệt ({time.time() - t:.0f} s)", flush=True)
        except Exception as e:                                       # Drive ngắt, file hỏng: ghi lại, làm tủ khác
            print(f"[{i}/{len(dirs)}] {inv}: LỖI {e!r:.200}", flush=True)
    return summary(out_dir)


def load(out_dir: Path) -> dict[str, dict]:
    return {f.name[:-len(".pkl.gz")]: pd.read_pickle(f) for f in sorted(Path(out_dir).glob("*.pkl.gz"))}


def summary(out_dir: Path) -> pd.DataFrame:
    rows = []
    for inv, x in load(out_dir).items():
        m = x["meta"]
        rows.append(dict(tu=inv, aps=m["aps"], dung_ma_aps=(not m["expected_aps"]) or m["aps"] == m["expected_aps"],
                         dang=m["format"], so_ngay=m["n_days"], tu_ngay=m["t_first"], den_ngay=m["t_last"],
                         dong_trang_thai=len(x["errors"]), lan_warning=len(x["warnings"]), dong_nhiet=len(x["temps"])))
    return pd.DataFrame(rows)
