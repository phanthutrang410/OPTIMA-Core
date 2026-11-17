"""Khóa dự đoán trước khi có phiếu sửa chữa, rồi chấm khi có phiếu (kiểm định trên dữ liệu tương lai).

`lock` ghi dự đoán từng sự cố vào locked/predictions_<tên>.csv kèm mã băm SHA-256 (commit git ngay sau đó).
`score` kiểm mã băm rồi chấm bằng đúng hàm validation.evaluate_om. File khóa bị sửa -> mã băm không khớp -> không chấm.

    python -m sinacon.lock lock --tier23 tier23.csv --name 2026_09_10
    python -m sinacon.lock score --name 2026_09_10 --replacements phieu_moi.csv --out <thư mục>
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from sinacon.config import REPO
from sinacon.tier23 import K
from sinacon.validation import evaluate_om, report_om

LOCKED = REPO / "locked"
COLUMNS = (["period", "cabinet", "inverter", "apu", "t0", "codes", "tier1_verdict", "tier1_class", "scope", "p_scope",
            "top1", "p_top1", "top2", "p_top2", "event_type", "n_trip"] + [f"p_{k}" for k in K])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lock(tier23: Path, name: str) -> None:
    t = pd.read_csv(tier23, dtype={"inverter": str}, low_memory=False)
    t = t[[c for c in COLUMNS if c in t]].sort_values(["t0", "cabinet"])
    LOCKED.mkdir(parents=True, exist_ok=True)
    f = LOCKED / f"predictions_{name}.csv"
    t.to_csv(f, index=False, encoding="utf-8", float_format="%.4f", lineterminator="\n")
    (LOCKED / f"predictions_{name}.sha256").write_text(f"{sha256(f)}  {f.name}\n", encoding="utf-8")
    print(f"{len(t)} sự cố khóa -> {f}\nSHA-256 {sha256(f)}")


def score(name: str, replacements: Path, out: Path) -> None:
    f = LOCKED / f"predictions_{name}.csv"
    expected = (LOCKED / f"predictions_{name}.sha256").read_text(encoding="utf-8").split()[0]
    if sha256(f) != expected:
        raise SystemExit(f"File khóa đã bị sửa (SHA-256 {sha256(f)} ≠ {expected}): không chấm.")
    r = evaluate_om(pd.read_csv(f, dtype={"inverter": str}), replacements)
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / f"score_{name}.csv", index=False, encoding="utf-8-sig")
    report_om(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("lock")
    k.add_argument("--tier23", required=True)
    k.add_argument("--name", required=True)
    s = sub.add_parser("score")
    s.add_argument("--name", required=True)
    s.add_argument("--replacements", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "lock":
        lock(Path(a.tier23), a.name)
    else:
        score(a.name, Path(a.replacements), Path(a.out))


if __name__ == "__main__":
    main()
