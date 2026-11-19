"""Chọn các khoảng thời gian của Bước 1 bằng CÔNG THỨC trên dữ liệu, không đặt tay.

Hai tham số:
  - INCIDENT_GAP: mã bật trong vòng bao lâu sau khi nhóm mã trước tắt thì gom chung một sự cố.
  - WARN_BEFORE / WARN_AFTER: lấy warning APS bật từ bao lâu trước t0 tới bao lâu sau t0.

Công thức (tham số của công thức ghi trong config/windows.json, đổi được):
  1. Ngưỡng gom. Thử các ngưỡng g trong `gap_candidates_min`; N(g) = số sự cố (cộng mọi tủ). Chọn g NHỎ NHẤT
     mà đi lên ngưỡng kế tiếp chỉ bớt thêm < `gap_tol` (vd 2 %) số sự cố: tới đó chuỗi đã gom xong,
     ngưỡng lớn hơn chỉ gộp thêm chuyện khác.
  2. Cửa sổ warning. Với mỗi lần warning bật, d = thời gian tới t0 của sự cố gần nhất (giây, âm = trước t0).
     Mật độ m(bin) = số lần / giây trong từng ô `bin_s` giây. Mức nền = mật độ ở xa sự cố
     (|d| trong `far_s`). Từ ô chứa t0 đi ra hai phía, giữ các ô liền nhau có m ≥ `k_background` × nền;
     biên cửa sổ = mép ngoài cùng của các ô đó.

Chỉ chạy trên đợt XÂY PHƯƠNG PHÁP (04/04–03/06/2026); kết quả ghi config/windows.json kèm dữ liệu đã dùng.
Không chạy lại trên đợt kiểm định rồi dùng kết quả để chấm (mất khách quan).

    python -m sinacon.calibrate --root "<01_log/2026-04-04_2026-06-04>" --cabinets 1.1 1.2 4.1 ... --out config/windows.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sinacon.config import CONFIG

WINDOWS_FILE = CONFIG / "windows.json"
RULE = {"gap_candidates_min": [0, 0.5, 1, 2, 5, 10, 15, 30, 60], "gap_tol": 0.02,
        "bin_s": 10, "max_s": 300, "far_s": [300, 600], "k_background": 2.0}
DEFAULT = {"incident_gap_min": 5.0, "warn_before_s": 10.0, "warn_after_s": 90.0}   # giá trị đặt tay cũ


def load_windows(path: Path = WINDOWS_FILE) -> dict:
    """Giá trị đang dùng: từ config/windows.json nếu có, không thì giá trị đặt tay cũ."""
    if Path(path).is_file():
        return {**DEFAULT, **json.loads(Path(path).read_text(encoding="utf-8"))}
    return dict(DEFAULT)


def incident_counts(eps_list: list[pd.DataFrame], candidates_min) -> pd.Series:
    """N(g): tổng số sự cố trên mọi tủ khi gom với từng ngưỡng g (phút)."""
    from sinacon import incidents as inc
    keep, out = inc.INCIDENT_GAP, {}
    try:
        for g in candidates_min:
            inc.INCIDENT_GAP = timedelta(minutes=g)
            out[g] = sum(inc.group_incidents(e).nunique() for e in eps_list if len(e))
    finally:
        inc.INCIDENT_GAP = keep
    return pd.Series(out, name="so_su_co")


def choose_gap(counts: pd.Series, tol: float) -> float:
    """Ngưỡng nhỏ nhất mà bước tiếp theo bớt < tol số sự cố."""
    g = list(counts.index)
    for a, b in zip(g, g[1:]):
        if counts[a] > 0 and (counts[a] - counts[b]) / counts[a] < tol:
            return float(a)
    return float(g[-1])


def density(offsets_s: np.ndarray, bin_s: float, max_s: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(-max_s, max_s + bin_s, bin_s)
    cnt, _ = np.histogram(offsets_s, edges)
    return edges, cnt / bin_s


def choose_warning_window(offsets_s: np.ndarray, bin_s: float, max_s: float, far_s, k: float) -> tuple[float, float, float]:
    """(trước_s, sau_s, mức nền lần/giây) theo công thức ở đầu file."""
    offsets_s = np.asarray(offsets_s, float)
    far = np.abs(offsets_s)
    base = ((far >= far_s[0]) & (far < far_s[1])).sum() / (2 * (far_s[1] - far_s[0]))
    edges, m = density(offsets_s, bin_s, max_s)
    i0 = int(np.searchsorted(edges, 0, side="right")) - 1          # ô [0, bin_s)
    thr = k * base
    lo = i0
    while lo - 1 >= 0 and m[lo - 1] >= thr:
        lo -= 1
    hi = i0
    while hi + 1 < len(m) and m[hi + 1] >= thr:
        hi += 1
    return float(-edges[lo]), float(edges[hi + 1]), float(base)


def calibrate(cabinet_dirs: dict[str, Path], rule: dict = RULE) -> dict:
    """Đọc từng tủ, áp công thức, trả về dict để ghi windows.json."""
    from sinacon import explore as ex
    from sinacon import incidents as inc
    eps_list, offs = [], []
    for name, d in cabinet_dirs.items():
        _, errors, warnings = ex.read_cabinet(d)
        eps_list.append(inc.code_episodes(errors))
        offs.append(ex.warning_offsets(errors, warnings)["giay"].to_numpy())
        print(f"  {name}: {len(eps_list[-1])} đợt mã, {len(offs[-1])} lần warning bật", flush=True)
    counts = incident_counts(eps_list, rule["gap_candidates_min"])
    return result_from(counts, np.concatenate(offs) if offs else np.array([]), rule, list(cabinet_dirs))


def result_from(counts: pd.Series, offsets_s, rule: dict, cabinets: list[str], period: str = "") -> dict:
    """counts: N(g) theo ngưỡng (phút); offsets_s: d của mọi lần warning bật (giây)."""
    gap = choose_gap(counts, rule["gap_tol"])
    before, after, base = choose_warning_window(offsets_s, rule["bin_s"], rule["max_s"], rule["far_s"],
                                                rule["k_background"])
    return {"incident_gap_min": gap, "warn_before_s": before, "warn_after_s": after,
            "rule": rule, "background_per_s": round(base, 4),
            "incident_counts": {f"{k:g}": int(v) for k, v in counts.items()},
            "data": {"period": period, "cabinets": cabinets, "n_warning_activations": int(len(offsets_s))},
            "computed_on": date.today().isoformat()}


def main() -> None:
    from sinacon import explore as ex
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="thư mục đợt xây phương pháp, vd 01_log/2026-04-04_2026-06-04")
    ap.add_argument("--cabinets", nargs="+", required=True)
    ap.add_argument("--out", default=str(WINDOWS_FILE))
    a = ap.parse_args()
    dirs = {c: ex.find_cabinet(Path(a.root), c) for c in a.cabinets}
    res = calibrate(dirs)
    res["data"]["period"] = Path(a.root).name
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("incident_gap_min", "warn_before_s", "warn_after_s")}))


if __name__ == "__main__":
    main()
