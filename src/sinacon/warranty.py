"""Tần suất hỏng thật theo lớp từ sổ bảo hành / thay thiết bị của nhà máy (sheet "Lỗi theo từng INV").

Mỗi lần thay được xếp vào lớp bộ phận theo từ khóa (mục khớp đầu tiên được dùng). Một lần hỏng = (tủ, ngày, lớp):
một phiếu thay tụ + 2 cầu chì chỉ tính một lần. Kết quả 2022–2025 được dùng làm FAILURES_2022_2025 trong tier23.

    python -m sinacon.warranty --register "<THỐNG KÊ THIẾT BỊ ... .xlsx>" --out <thư mục>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# thứ tự quan trọng: powerstack trước "24V", tụ/cầu chì trước "AC"
KEYWORDS = [
    ("B3", r"igbt|driver|drive|dtiver|powerstack|power stack"),
    ("B7", r"harmonic|capacitor|capicitor|tụ|chì|fuse|filter|lọc"),
    ("B2", r"dc[- ]?link|breaker|switch|nạp tải|recharge|motor|contactor|máy cắt"),
    ("B4", r"ultracap|sitop|psu|24 ?v|nguồn"),
    ("B6", r"card|control|cpu|truyền thông|comm|fiber|quang"),
    ("B1", r"nước|pump|bơm|quạt|fan|cool|làm mát"),
    ("B5", r"cách điện|insul|imd|ground|nối đất"),
    ("GRID", r"dầu|mba|transformer|trung thế"),
]


def read_register(path: Path) -> pd.DataFrame:
    x = pd.read_excel(path, sheet_name="Lỗi theo từng INV", header=1)
    x.columns = [str(c).split("\n")[0].strip() for c in x.columns]
    x = x[x["Mô tả lỗi"].notna() | x["Tên thiết bị lỗi"].notna()].copy()
    out = pd.DataFrame({
        "inverter": x["INVERTER"].astype(str).str.strip(),
        "date": pd.to_datetime(x["Ngày bị lỗi"], errors="coerce"),
        "description": x["Mô tả lỗi"].fillna("").astype(str).str.strip(),
        "device": x["Tên thiết bị lỗi"].fillna("").astype(str).str.strip(),
    })
    out["date"] = out.groupby("inverter")["date"].ffill()       # dòng không có ngày: phần tiếp theo của cùng phiếu
    txt = (out["description"] + " " + out["device"]).str.lower()
    out["cls"] = np.select([txt.str.contains(p) for _, p in KEYWORDS], [k for k, _ in KEYWORDS], default="?")
    return out


def failure_counts(reg: pd.DataFrame, years: tuple[int, int] = (2022, 2025)) -> pd.DataFrame:
    x = reg[reg["date"].dt.year.between(*years) & (reg["cls"] != "?")].drop_duplicates(["inverter", "date", "cls"])
    t = x["cls"].value_counts().rename("n").to_frame()
    t["share"] = (t["n"] / t["n"].sum()).round(4)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = read_register(Path(a.register))
    reg.to_csv(out / "warranty_register.csv", index=False, encoding="utf-8-sig")
    t = failure_counts(reg)
    t.to_csv(out / "failures_2022_2025.csv", encoding="utf-8-sig")
    print(t.to_string())
    print("không xếp được:", int((reg["cls"] == "?").sum()))


if __name__ == "__main__":
    main()
