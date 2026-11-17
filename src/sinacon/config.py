"""Đường dẫn và bảng cấu hình dùng chung."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"

# Mã lỗi trong chuỗi mã của một sự cố, ví dụ "APU/error/4 (Overtemperature inductance)".
CODE_RE = re.compile(r"AP[SU]/(?:error|warning)/\d+")
ERROR_CODE_RE = re.compile(r"AP[SU]/error/\d+")
# Phân nhóm hai tầng (config/fault_codes.csv, cột device = tầng 1 thiết bị, fault_type = tầng 2 lỗi gì).
# Mã phần cứng công suất: các ô quá dòng và hỏng driver.
HARDWARE_FAULT_TYPES = ("Quá dòng", "Quá dòng pha AC", "Driver pha hỏng", "Không tìm thấy driver", "PWM lỗi",
                        "Driver hỏng")
N_APU = 4                     # mọi tủ ở Trung Nam có 4 APU


def codes_in(text) -> set[str]:
    return set(CODE_RE.findall(str(text)))


@lru_cache
def fault_codes() -> pd.DataFrame:
    """Danh mục 230 mã của Siemens (manual SINACON-PV_BA 01/2020, mục 10) kèm nhóm hệ thống con G01–G12."""
    return pd.read_csv(CONFIG / "fault_codes.csv", encoding="utf-8")


@lru_cache
def code_groups() -> dict[str, str]:
    t = fault_codes()
    return dict(zip(t["code_key"], t["group"]))


@lru_cache
def code_device() -> dict[str, tuple[str, str]]:
    """Mã -> (thiết bị, loại lỗi) theo bảng hai tầng; chỉ mã có trong bảng."""
    t = fault_codes().dropna(subset=["device"])
    return {k: (d, f) for k, d, f in zip(t["code_key"], t["device"], t["fault_type"])}


def is_thermal(device, fault_type) -> bool:
    """Mã nhiệt: loại lỗi có chữ "nhiệt" hoặc thiết bị Môi trường tủ (làm mát, nhiệt độ, độ ẩm tủ)."""
    return device == "ENCLOSURE" or "nhiệt" in str(fault_type).lower()


@lru_cache
def hardware_error_codes() -> frozenset[str]:
    t = fault_codes()
    return frozenset(t.loc[t["fault_type"].isin(HARDWARE_FAULT_TYPES) & (t["type"] == "error"), "code_key"])


@lru_cache
def cabinet_to_inverter() -> dict[str, str]:
    """Mã tủ APS-xxxxxx -> số hiệu inverter "khối.số" (ví dụ APS-000370 -> 1.1)."""
    t = pd.read_csv(CONFIG / "cabinets.csv", dtype=str, encoding="utf-8")
    return dict(zip(t["cabinet"], t["inverter"]))
