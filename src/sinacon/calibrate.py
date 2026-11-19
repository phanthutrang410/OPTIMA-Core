"""Chọn các khoảng thời gian của Bước 1 bằng CÔNG THỨC trên dữ liệu, không đặt tay.

Hai tham số:
  - INCIDENT_GAP: mã bật trong vòng bao lâu sau khi nhóm mã trước tắt thì gom chung một sự cố.
  - WARN_BEFORE / WARN_AFTER: lấy warning APS bật từ bao lâu trước t0 tới bao lâu sau t0.

Công thức (tham số của công thức ghi trong config/windows.json, đổi được):
  1. Ngưỡng gom. Thử các ngưỡng g trong `gap_candidates_min`; N(g) = số sự cố (cộng mọi tủ). Chọn g NHỎ NHẤT
     mà đi lên ngưỡng kế tiếp chỉ bớt thêm < `gap_tol` (vd 2 %) số sự cố: tới đó chuỗi đã gom xong,
     ngưỡng lớn hơn chỉ gộp thêm chuyện khác.
  2. Cửa sổ warning ("xô hứng mưa"). Với mỗi lần warning bật, d = thời gian tới t0 của sự cố gần nhất
     (giây, âm = trước t0). Mức nền r = số warning bật mỗi giây ở xa sự cố (|d| trong `far_s`): warning
     ngẫu nhiên, không liên quan sự cố ("mưa phùn"). Cửa sổ sau t0 dài A giây gắn N(A) warning, trong đó
     khoảng r·A là ngẫu nhiên. Chọn A LỚN NHẤT (≤ `max_s`) mà r·A / N(A) ≤ `max_random_share` (vd 5 %).
     Phía trước t0 làm y hệt với B.
     Mức chấp nhận duy nhất người chọn: `max_random_share` (tỉ lệ warning gắn nhầm chấp nhận được).

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
        "max_random_share": 0.05, "max_s": 300, "far_s": [300, 600]}
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


def background_rate(offsets_s, far_s) -> float:
    """Warning bật mỗi giây ở xa sự cố (cả hai phía), tức là warning ngẫu nhiên."""
    far = np.abs(np.asarray(offsets_s, float))
    return float(((far >= far_s[0]) & (far < far_s[1])).sum() / (2 * (far_s[1] - far_s[0])))


def window_curve(offsets_s, far_s, max_s: int) -> pd.DataFrame:
    """Với mỗi độ dài cửa sổ x = 1…max_s giây: số warning gắn được sau t0 / trước t0 và số ngẫu nhiên ước tính."""
    d = np.sort(np.asarray(offsets_s, float))
    r = background_rate(d, far_s)
    x = np.arange(1, max_s + 1)
    after = np.searchsorted(d, x, side="right") - np.searchsorted(d, 0, side="left")
    before = np.searchsorted(d, 0, side="left") - np.searchsorted(d, -x, side="left")
    return pd.DataFrame({"giay": x, "gan_sau": after, "gan_truoc": before, "ngau_nhien": r * x})


def choose_warning_window(offsets_s, max_random_share: float, far_s, max_s: int) -> tuple[float, float, float]:
    """(B trước t0, A sau t0, mức nền lần/giây): cửa sổ rộng nhất mà phần ngẫu nhiên ≤ max_random_share."""
    c = window_curve(offsets_s, far_s, max_s)

    def widest(n):
        ok = c["giay"][c["ngau_nhien"] <= max_random_share * np.maximum(n, 1)]
        return float(ok.max()) if len(ok) else 0.0
    return widest(c["gan_truoc"]), widest(c["gan_sau"]), background_rate(offsets_s, far_s)


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
    before, after, base = choose_warning_window(offsets_s, rule["max_random_share"], rule["far_s"], rule["max_s"])
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
