"""Bước 1 — Gom mã lỗi trong log ngày thành sự cố (đợt cảnh báo).

Quy tắc (giống sổ lỗi Fault_log của nhóm):
  1. Đợt mã: mỗi (cấp, đơn vị, mã error) tính từ bản ghi đầu tiên có mã tới bản ghi đầu tiên không còn mã.
  2. Sự cố: các đợt mã nối nhau, đợt sau bật trong vòng INCIDENT_GAP sau khi đợt trước tắt.
  3. Thời điểm sự cố t0 = lúc mã error đầu tiên bật (làm tròn xuống giây). Thiết bị và loại lỗi chính = ô của mã đó trong bảng hai tầng.
  4. Bổ sung warning APS: mọi warning APS chuyển sang bật trong [t0 − WARN_BEFORE, t0 + WARN_AFTER].
INCIDENT_GAP, WARN_BEFORE, WARN_AFTER đọc từ config/windows.json (chọn bằng công thức, xem sinacon/calibrate.py).

Đầu ra mỗi dòng một sự cố: cabinet, inverter, apu ("1 2", hoặc "cả tủ"), t0, codes, first_device, first_fault_type.

    python -m sinacon.incidents --logs "<thư mục log của tháng>" --out incidents.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import pandas as pd

from sinacon.calibrate import load_windows
from sinacon.config import code_device, fault_codes
from sinacon.daily_log import find_inverter_dirs, parse_trigger_records, warning_activations

_W = load_windows()
INCIDENT_GAP = timedelta(minutes=_W["incident_gap_min"])
WARN_BEFORE, WARN_AFTER = pd.Timedelta(seconds=_W["warn_before_s"]), pd.Timedelta(seconds=_W["warn_after_s"])
NOT_IN_MANUAL = "Không có trong manual 01/2020"
COLUMNS = ["cabinet", "inverter", "apu", "t0", "codes", "first_device", "first_fault_type"]


def code_episodes(errors: pd.DataFrame) -> pd.DataFrame:
    """Mỗi (cấp, đơn vị, mã): từ bản ghi đầu tiên có mã tới bản ghi đầu tiên không còn mã (end = NaT nếu chưa tắt)."""
    eps = []
    for (scope, unit), g in errors.groupby(["scope", "unit"], sort=False):
        active = {}
        for r in g.itertuples():
            codes = set(r.codes)
            for c in codes - active.keys():
                active[c] = (r.t, r.opstate)
            for c in list(active.keys() - codes):
                t0, op = active.pop(c)
                eps.append((scope, unit, c, t0, r.t, op))
        for c, (t0, op) in active.items():
            eps.append((scope, unit, c, t0, pd.NaT, op))
    return (pd.DataFrame(eps, columns=["scope", "unit", "code", "start", "end", "opstate_start"])
            .sort_values("start").reset_index(drop=True))


def group_incidents(eps: pd.DataFrame) -> pd.Series:
    """Số thứ tự sự cố cho từng đợt mã (eps đã sắp theo start)."""
    inc, cur, cur_end = [], -1, None
    for r in eps.itertuples():
        end = r.end if pd.notna(r.end) else r.start
        if cur_end is None or r.start > cur_end + INCIDENT_GAP:
            cur += 1
            cur_end = end
        else:
            cur_end = max(cur_end, end)
        inc.append(cur)
    return pd.Series(inc, index=eps.index)


def _manual_names() -> dict[tuple[str, int], str]:
    t = fault_codes()
    t = t[t["type"] == "error"]
    return {(r.scope, int(r.number)): r.name for r in t.itertuples()}


def incidents_of_cabinet(inv_dir: Path, inverter: str) -> pd.DataFrame:
    cabinet, errors, warnings = parse_trigger_records(inv_dir)
    if not errors["codes"].map(len).any():
        return pd.DataFrame(columns=COLUMNS)
    names, cells = _manual_names(), code_device()
    eps = code_episodes(errors)
    eps["incident"] = group_incidents(eps)
    eps["key"] = eps["scope"] + "/error/" + eps["code"].astype(str)
    on = warning_activations(warnings)
    rows = []
    for _, g in eps.groupby("incident", sort=True):
        # mã đầu tiên: sớm nhất theo giây (như sổ lỗi), cùng giây thì theo cấp, số mã, số APU
        g = g.assign(start_s=g["start"].dt.floor("s")).sort_values(["start_s", "scope", "code", "unit"], kind="stable")
        t0 = g["start_s"].iloc[0]
        units = sorted({int(u) for u in g.loc[g["scope"] == "APU", "unit"]})
        named = sorted({(k, names.get((s, int(c)), NOT_IN_MANUAL)) for k, s, c in zip(g["key"], g["scope"], g["code"])})
        codes = " + ".join(f"{k} ({n})" for k, n in named)
        w = on.loc[(on["t"] >= t0 - WARN_BEFORE) & (on["t"] <= t0 + WARN_AFTER), "code"]
        codes += "".join(f" + {c}" for c in sorted(set(w), key=lambda c: int(c.rsplit("/", 1)[1])))
        dev, ftype = cells.get(g["key"].iloc[0], (None, None))
        rows.append(dict(cabinet=cabinet, inverter=inverter, apu=" ".join(map(str, units)) or "cả tủ",
                         t0=t0, codes=codes, first_device=dev, first_fault_type=ftype))
    return pd.DataFrame(rows, columns=COLUMNS)


def _job(args) -> pd.DataFrame:
    inverter, d = args
    t = incidents_of_cabinet(Path(d), inverter)
    print(f"[{inverter}] {len(t)} sự cố", flush=True)
    return t


def build_incidents(log_root: Path, workers: int = 4) -> pd.DataFrame:
    dirs = [(inv, str(d)) for inv, d in find_inverter_dirs(log_root)]
    with ProcessPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(_job, dirs) if len(p)]
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    return t.drop_duplicates(["cabinet", "t0"]).sort_values(["t0", "cabinet"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, help="thư mục gốc chứa thư mục từng tủ (chỉ đọc)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t = build_incidents(Path(a.logs), a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{len(t)} sự cố, {t['cabinet'].nunique()} tủ -> {a.out}")


if __name__ == "__main__":
    main()
