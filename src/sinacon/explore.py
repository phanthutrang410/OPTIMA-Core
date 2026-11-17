"""Xem dữ liệu và chạy từng bước cho MỘT tủ (dùng trong notebook SINACON_tung_buoc.ipynb).

Mọi hàm chỉ đọc thư mục log; kết quả ghi vào thư mục out do người dùng chọn.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sinacon import incidents as inc_mod
from sinacon import tier1, windows
from sinacon.config import code_device, fault_codes
from sinacon.daily_log import INVERTER_DIR_RE, LOG_RE, iter_log_files, parse_trigger_records

RECORD_MEANING = {
    "APU Stat Trig": ("Trạng thái (OpState) và mã lỗi từng APU, ghi mỗi khi có thay đổi", "DÙNG"),
    "APS Stat Trig": ("Trạng thái, mã lỗi và warning của cả tủ, ghi mỗi khi có thay đổi", "DÙNG"),
    "APU Stat 60s": ("Nhiệt độ cuộn cảm (TInd) và IGBT (TL1–TL3) từng APU, mỗi phút", "DÙNG"),
    "APU Stat 10s": ("Điện áp, dòng, công suất từng APU, mỗi 10 giây", "không dùng"),
    "APU Ctrl Trig": ("Lệnh điều khiển gửi xuống APU", "không dùng"),
    "APU Energy": ("Điện năng từng APU", "không dùng"),
    "APS Stat 60s": ("Số đo cả tủ mỗi phút", "không dùng"),
    "APS Energy": ("Điện năng cả tủ", "không dùng"),
    "APS Ctrl Trig": ("Lệnh điều khiển cả tủ", "không dùng"),
    "APS Switching Cycles": ("Số lần đóng cắt quạt, bơm", "không dùng"),
}
DEVICE_ORDER = ["IGBT", "DRIVER", "BOOSTER", "DC", "INDUCTOR", "INSULATION", "ENCLOSURE", "CONTROL", "SWITCHGEAR",
                "GRID", "OPERATION"]


def opstate_label(s) -> str:
    s = int(s)
    if s in (3010, 3020, 3030):
        return "Đang phát, giới hạn công suất"
    if s >= 3000:
        return "Đang phát điện"
    if s == 500:
        return "Bảo vệ tác động, chờ chạy lại"
    if s in (30, 50):
        return "Tắt"
    return "Đang khởi động / chuyển trạng thái"


LEVELS = [0, 1, 2, 3, 3.6]
LEVEL_NAMES = ["Tắt (30, 50)", "Bảo vệ tác động (500)", "Khởi động / chờ", "Phát giới hạn (3010–3030)", "Phát điện (3000)"]


def opstate_level(s: pd.Series) -> pd.Series:
    """OpState -> mức vẽ: 0 tắt, 1 bảo vệ, 2 khởi động, 3 phát giới hạn, 3,6 phát."""
    s = s.astype(int)
    return pd.Series(np.select([s.isin([30, 50]), s == 500, s.isin([3010, 3020, 3030]), s >= 3000],
                               [0, 1, 3, 3.6], default=2), index=s.index)


def _state_axis(ax):
    ax.set_yticks(LEVELS)
    ax.set_yticklabels(LEVEL_NAMES, fontsize=8)
    ax.set_ylim(-0.4, 4.0)


# ------------------------------------------------------------------ tìm thư mục, xem file
def find_log_root(drive: Path, link: str, sub: str) -> Path:
    """Thư mục `sub` (vd "Log file/10_2026") bên trong thư mục Drive có link `link` (chia sẻ, chỉ đọc).

    Colab thấy thư mục được chia sẻ qua lối tắt trong Drive của tôi: /content/drive/.shortcut-targets-by-id/<id>/...
    Không thấy thì tìm thư mục cuối của `sub` trong MyDrive (sâu tối đa 4 cấp)."""
    drive = Path(drive)
    fid = link.rstrip("/").split("folders/")[-1].split("?")[0]
    base = drive / ".shortcut-targets-by-id" / fid
    for cand in [base / sub, *[c / sub for c in (sorted(base.iterdir()) if base.is_dir() else []) if c.is_dir()]]:
        if cand.is_dir():
            return cand
    last = Path(sub).name
    for d, subdirs, _ in os.walk(drive / "MyDrive"):
        if Path(d).relative_to(drive / "MyDrive").parts.__len__() >= 4:
            subdirs[:] = []
            continue
        if last in subdirs:
            return Path(d) / last
    raise FileNotFoundError(f"Không thấy '{sub}'. Kiểm tra đã thêm lối tắt thư mục dữ liệu vào Drive của tôi chưa.")


def find_cabinet(root: Path, inverter: str) -> Path:
    """Thư mục log của tủ `inverter` (vd "9.3") dưới root; tên thư mục có thể là "9.3", "INV 9.3", "inv 9.3"."""
    for d, subdirs, _ in os.walk(root):
        for s in sorted(subdirs):
            m = INVERTER_DIR_RE.fullmatch(s.strip())
            if m and m.group(1) == inverter:
                return Path(d) / s
    raise FileNotFoundError(f"Không thấy thư mục tủ {inverter} dưới {root}")


def list_files(cab_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(cab_dir.rglob("*.csv")):
        m = LOG_RE.search(f.name)
        if m and "TR" not in f.relative_to(cab_dir).parts:
            rows.append(dict(file=f.name, ngay=pd.to_datetime(m.group(2)).date(), MB=round(f.stat().st_size / 1e6, 1)))
    return pd.DataFrame(rows)


def _first_file(cab_dir: Path, day: str | None = None) -> tuple[str, str]:
    for name, read in iter_log_files(cab_dir):
        if day is None or LOG_RE.search(name).group(2) == pd.Timestamp(day).strftime("%Y%m%d"):
            return name, read().decode("utf-8", errors="replace")
    raise FileNotFoundError(f"Không có file log ngày {day}")


def record_types(cab_dir: Path, day: str | None = None) -> pd.DataFrame:
    """Các loại dòng trong một file log ngày: số dòng, ý nghĩa, có dùng không."""
    name, text = _first_file(cab_dir, day)
    n = Counter(line.split(";", 1)[0] for line in text.splitlines() if ";" in line)
    t = pd.DataFrame([(k, v, *RECORD_MEANING[k]) for k, v in n.most_common() if k in RECORD_MEANING],
                     columns=["loai_dong", "so_dong", "y_nghia", "mo_hinh"])
    t.attrs["file"] = name
    return t


def sample_lines(cab_dir: Path, kind: str, n: int = 5, day: str | None = None, only_codes: bool = False) -> pd.DataFrame:
    """n dòng đầu (hoặc n dòng có mã lỗi) của loại `kind`, có tiêu đề cột như trong file."""
    _, text = _first_file(cab_dir, day)
    lines = [ln.split(";") for ln in text.splitlines() if ln.startswith(kind + ";")]
    head = next((ln for ln in lines if ln[2] == "TimeStamp"), None)
    data = [ln for ln in lines if ln[2] != "TimeStamp"]
    if only_codes:
        data = [ln for ln in data if any(x not in ("", "0") for x in ln[-8:])]
    t = pd.DataFrame(data[:n])
    if head is not None and t.shape[1] == len(head):
        t.columns = [c.replace("�", "°") for c in head]
    return t


# ------------------------------------------------------------------ mã lỗi
def device_table() -> pd.DataFrame:
    """10 thiết bị (tầng 1) + Vận hành: số mã và các loại lỗi (tầng 2)."""
    t = fault_codes().dropna(subset=["device"])
    g = t.groupby("device").agg(thiet_bi=("device_name", "first"), so_ma=("code_key", "size"),
                                loai_loi=("fault_type", lambda s: " · ".join(dict.fromkeys(s))))
    return g.reindex([d for d in DEVICE_ORDER if d in g.index]).reset_index(drop=True)


def read_cabinet(cab_dir: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    return parse_trigger_records(cab_dir)


def code_events(errors: pd.DataFrame) -> pd.DataFrame:
    """Mỗi lần một mã error BẬT: thời điểm, ở đâu, OpState lúc đó, mã, thiết bị và loại lỗi theo bảng hai tầng."""
    cells, names = code_device(), fault_codes().set_index("code_key")
    rows = []
    for (scope, unit), g in errors.groupby(["scope", "unit"], sort=False):
        prev = set()
        for r in g.itertuples():
            for c in sorted(set(r.codes) - prev):
                k = f"{scope}/error/{c}"
                dev = cells.get(k, (None, None))
                rows.append(dict(t=r.t, noi=f"APU {unit}" if scope == "APU" else "cả tủ", opstate=r.opstate,
                                 trang_thai=opstate_label(r.opstate), ma_log=k,
                                 ma_bang=names["doc_code"].get(k), ten_siemens=names["name"].get(k),
                                 thiet_bi=names["device_name"].get(k), loai_loi=dev[1]))
            prev = set(r.codes)
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True) if rows else pd.DataFrame()


def code_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    return (events.groupby(["ma_bang", "ma_log", "ten_siemens", "thiet_bi", "loai_loi"], dropna=False)
            .size().rename("so_lan_bat").reset_index().sort_values("so_lan_bat", ascending=False, ignore_index=True))


# ------------------------------------------------------------------ hình một ngày
def apu_temps(cab_dir: Path, day: str) -> pd.DataFrame:
    _, text = _first_file(cab_dir, day)
    rows = [ln.split(";") for ln in text.splitlines() if ln.startswith("APU Stat 60s;APU ")]
    t = pd.DataFrame([(r[1], r[2], *r[3:7]) for r in rows], columns=["System", "t", "TInd", "TL1", "TL2", "TL3"])
    t["t"] = pd.to_datetime(t["t"])
    for c in ["TInd", "TL1", "TL2", "TL3"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def plot_day(cab_dir: Path, errors: pd.DataFrame, events: pd.DataFrame, day: str):
    """Một ngày: OpState 4 APU (trên), nhiệt độ IGBT/cuộn cảm (dưới), vạch đỏ = lúc mã lỗi bật."""
    import matplotlib.pyplot as plt
    d0, d1 = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    e = errors[(errors["scope"] == "APU") & (errors["t"] >= d0) & (errors["t"] < d1)]
    for u, g in e.groupby("unit"):
        ax[0].step(g["t"], opstate_level(g["opstate"]) + (u - 2.5) * 0.06, where="post", label=f"APU {u}")
    _state_axis(ax[0])
    ax[0].set_title("Trạng thái (OpState) từng APU; vạch đỏ = lúc một mã lỗi bật")
    tp = apu_temps(cab_dir, day)
    for u, g in tp.groupby("System"):
        ax[1].plot(g["t"], g[["TL1", "TL2", "TL3"]].max(axis=1), label=f"{u} IGBT")
        ax[1].plot(g["t"], g["TInd"], ls="--", label=f"{u} cuộn cảm")
    for y, txt in ((107, "ngưỡng bảo vệ IGBT 107 °C"), (160, "ngưỡng bảo vệ cuộn cảm ~160 °C")):
        ax[1].axhline(y, color="k", lw=0.8)
        ax[1].text(d0, y + 1, txt, fontsize=8)
    ax[1].set_ylabel("°C")
    ax[1].set_title("Nhiệt độ (IGBT nóng nhất; cuộn cảm nét đứt)")
    ev = events[(events["t"] >= d0) & (events["t"] < d1)] if len(events) else events
    for a in ax:
        for t in ev.get("t", []):
            a.axvline(t, color="red", alpha=0.3, lw=1)
        a.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ các bước mô hình
def incidents_of(cab_dir: Path, inverter: str) -> pd.DataFrame:
    t = inc_mod.incidents_of_cabinet(cab_dir, inverter)
    names = fault_codes().drop_duplicates("device").set_index("device")["device_name"]
    t["thiet_bi_ma_dau"] = t["first_device"].map(names)
    return t


def step_a(inc: pd.DataFrame, cab_dir: Path, inverter: str, out_dir: Path) -> pd.DataFrame:
    """Bước A cho một tủ: trích OpState, nhiệt quanh từng sự cố rồi áp quy tắc giả/thật."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"windows_{inverter}.pkl").unlink(missing_ok=True)
    print(windows.extract_cabinet(inverter, cab_dir, np.sort(pd.to_datetime(inc["t0"]).to_numpy()), out_dir))
    t = tier1.run(inc, out_dir)
    t["ket_luan"] = t["tier1_verdict"].map({"real": "THẬT", "false": "GIẢ"})
    t["quy_tac"] = t["tier1_class"].map({**tier1.CLASS_LABEL,
                                         "kept_prior": "Chưa đủ căn cứ: tạm để thật",
                                         "not_feeding": "Tủ không phát lúc t0: tạm để thật"})
    t["ly_do"] = t["tier1_reason"].fillna("")
    t.loc[(t["tier1_class"] == "not_feeding") & (t["ly_do"] == ""), "ly_do"] = (
        "Lúc t0 không APU nào đang phát (đêm, hoặc tủ đã dừng từ trước): không đo được hậu quả; chưa có kết luận "
        "nào khác nên tạm để THẬT.")
    t.loc[(t["tier1_class"] == "kept_prior") & (t["ly_do"] == ""), "ly_do"] = (
        "Không APU nào ngắt và mã không phải mã nhiệt: Bước A không đủ căn cứ gọi là giả; tạm để THẬT.")
    return t


def plot_incident(win_dir: Path, inverter: str, t0, before_min: int = 30, after_min: int = 60):
    """OpState và nhiệt độ quanh một sự cố (đúng dữ liệu Bước A đã nhìn)."""
    import matplotlib.pyplot as plt
    w = windows.load_windows(win_dir, inverter)
    t0 = pd.Timestamp(t0)
    a, b = t0 - pd.Timedelta(minutes=before_min), t0 + pd.Timedelta(minutes=after_min)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    tr = w.get("trig")
    if tr is not None:
        for u, g in tr.groupby("System"):
            pre = g[g["t"] <= a].tail(1).assign(t=a)
            g = pd.concat([pre, g[(g["t"] > a) & (g["t"] <= b)]])
            if len(g):
                g = pd.concat([g, g.tail(1).assign(t=b)])
                k = int(str(u).split()[-1])
                ax[0].step(g["t"], opstate_level(g["OpState"]) + (k - 2.5) * 0.06, where="post", label=u)
    _state_axis(ax[0])
    temp = w.get("temp")
    if temp is not None:
        x = temp[(temp["t"] >= a) & (temp["t"] <= b)]
        for u, g in x.groupby("System"):
            ax[1].plot(g["t"], g[["TL1/°C", "TL2/°C", "TL3/°C"]].max(axis=1), label=f"{u} IGBT")
            ax[1].plot(g["t"], g["TInd/°C"], ls="--", label=f"{u} cuộn cảm")
    for y in (107, 160):
        ax[1].axhline(y, color="k", lw=0.6, ls=":")
    for x_ in ax:
        x_.axvline(t0, color="red", lw=1.5)
        x_.legend(fontsize=7, ncol=4, loc="upper left")
    ax[0].set_title(f"Tủ {inverter}, sự cố lúc {t0:%d/%m %H:%M:%S} (vạch đỏ)")
    ax[1].set_ylabel("°C")
    fig.tight_layout()
    return fig
