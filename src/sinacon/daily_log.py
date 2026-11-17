"""Đọc log ngày của inverter SINACON PV.

Mỗi tủ ghi một (vài) file mỗi ngày, tên APS-xxxxxx_YYYYMMDD_HHMMSS.csv, phân cách ";", nhiều loại bản ghi trộn chung
(cột đầu là loại bản ghi). Module này chỉ lấy:
  - "APU Stat Trig" / "APS Stat Trig": bản ghi khi trạng thái hoặc mã lỗi đổi, có mili giây. Cột Error1–8 (APU: cột
    15–22; APS: cột 5–12) và Warning1–8 của APS (cột 13–20).
File có thể nằm ở mọi độ sâu dưới thư mục tủ, kể cả trong file zip; file trùng tên chỉ đọc một lần; bỏ thư mục "TR"
(file Transient Recorder).
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

LOG_RE = re.compile(r"(APS-\d+)_(\d{8})_(\d{6})\.csv$", re.I)
INVERTER_DIR_RE = re.compile(r"(?:INV\s*|log\s*)?(\d+\.\d+)", re.I)   # "1.1", "INV 10.1", "log 14.1"


def find_inverter_dirs(root: Path) -> list[tuple[str, Path]]:
    """[(số hiệu inverter, thư mục)] cho mọi thư mục tủ dưới root (tên thư mục không thống nhất giữa các đợt tải)."""
    out = {}
    for d in sorted(Path(root).rglob("*")):
        if not d.is_dir():
            continue
        m = INVERTER_DIR_RE.fullmatch(d.name.strip())
        if m and any(d.rglob("APS-*_*.csv")):
            out.setdefault(m.group(1), d)
    return sorted(out.items(), key=lambda kv: [int(x) for x in kv[0].split(".")])


def iter_log_files(inv_dir: Path) -> Iterator[tuple[str, Callable[[], bytes]]]:
    """(tên file, hàm đọc nội dung) cho mọi log ngày của một tủ, kể cả trong zip; bỏ trùng tên và thư mục TR."""
    seen = set()
    for f in sorted(inv_dir.rglob("*.csv")):
        if LOG_RE.search(f.name) and "TR" not in f.relative_to(inv_dir).parts and f.name not in seen:
            seen.add(f.name)
            yield f.name, (lambda f=f: f.read_bytes())
    for z in sorted(inv_dir.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if LOG_RE.search(Path(n).name)]
        for n in names:
            if Path(n).name not in seen:
                seen.add(Path(n).name)
                yield Path(n).name, (lambda z=z, n=n: zipfile.ZipFile(z).read(n))


def _read_retry(read: Callable[[], bytes], tries: int = 3, wait_s: float = 5.0) -> bytes | None:
    """Google Drive đôi khi báo OSError khi file đang được tải về: thử lại, vẫn lỗi thì bỏ qua file."""
    for _ in range(tries):
        try:
            return read()
        except OSError:
            time.sleep(wait_s)
    return None


def _codes(fields: list[str]) -> tuple[int, ...]:
    return tuple(int(x) for x in fields if x not in ("", "0"))


def parse_trigger_records(inv_dir: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """(mã tủ, bản ghi mã lỗi, bản ghi warning APS) của một tủ.

    errors:   scope (APU/APS), unit (số APU, 0 = cả tủ), t, opstate, codes (tuple mã error đang bật)
    warnings: t, codes (frozenset mã warning APS đang bật)
    """
    cabinet, err, warn = "", [], []
    for name, read in iter_log_files(inv_dir):
        cabinet = cabinet or LOG_RE.search(name).group(1)
        data = _read_retry(read)
        if data is None:
            print(f"BỎ QUA (không đọc được): {inv_dir / name}", flush=True)
            continue
        for line in io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace"):
            p = line.rstrip("\r\n").split(";")
            if p[0] == "APU Stat Trig" and len(p) > 4 and p[1].startswith("APU "):
                err.append(("APU", int(p[1][4:]), p[2], int(p[3] or 0), int(p[4]), _codes(p[15:23])))
            elif p[0] == "APS Stat Trig" and len(p) > 4 and p[1] == "APS" and p[2] != "TimeStamp":
                err.append(("APS", 0, p[2], int(p[3] or 0), int(p[4]), _codes(p[5:13])))
                if p[2][:1] == "2":                       # dòng dữ liệu (thời điểm bắt đầu bằng năm 2xxx)
                    warn.append((p[2], int(p[3] or 0), frozenset(_codes(p[13:21]))))
    errors = pd.DataFrame(err, columns=["scope", "unit", "ts", "ms", "opstate", "codes"]).drop_duplicates()
    errors["t"] = pd.to_datetime(errors["ts"]) + pd.to_timedelta(errors["ms"], unit="ms")
    errors = errors.sort_values(["scope", "unit", "t"]).reset_index(drop=True)
    w = pd.DataFrame(warn, columns=["ts", "ms", "codes"]).drop_duplicates(["ts", "ms"])
    w["t"] = pd.to_datetime(w["ts"]) + pd.to_timedelta(w["ms"], unit="ms")
    return cabinet, errors, w.sort_values("t")[["t", "codes"]].reset_index(drop=True)


def warning_activations(warnings: pd.DataFrame) -> pd.DataFrame:
    """Các lần warning APS chuyển từ tắt sang bật: bảng (t, code)."""
    on, prev = [], frozenset()
    for t, w in zip(warnings["t"], warnings["codes"]):
        on += [(t, f"APS/warning/{c}") for c in w - prev]
        prev = w
    return pd.DataFrame(on, columns=["t", "code"])
