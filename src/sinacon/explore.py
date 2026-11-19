"""Xem dữ liệu và chạy từng bước cho MỘT tủ (dùng trong notebook SINACON_tung_buoc.ipynb).

Mọi hàm chỉ đọc thư mục log; kết quả ghi vào thư mục out do người dùng chọn.
"""
from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sinacon import incidents as inc_mod
from sinacon import tier1, windows
from sinacon.config import cabinet_to_inverter, code_device, fault_codes
from sinacon.daily_log import INVERTER_DIR_RE, LOG_RE, iter_log_files, parse_trigger_records

# loại dòng -> (các cột chính, mô hình hiện tại, có thể dùng cho). Tên cột lấy từ dòng tiêu đề trong log;
# "(?)" = ý nghĩa đoán theo ký hiệu, chưa có tài liệu Siemens xác nhận.
RECORD_MEANING = {
    "APU Stat Trig": ("OpState, Error1–8 của từng APU; ghi khi có thay đổi (có mili giây)",
                      "ĐANG DÙNG", "mã B, APU có ngắt không"),
    "APS Stat Trig": ("OpState, Error1–8, Warning1–8 của cả tủ; ghi khi có thay đổi",
                      "ĐANG DÙNG", "mã A, warning chế độ tay/bảo trì"),
    "APU Stat 60s": ("TInd cuộn cảm, TL1–TL3 IGBT 3 pha, TPCB nhiệt bo mạch, Hum độ ẩm; mỗi phút",
                     "ĐANG DÙNG (TInd, TL1–3)", "Quá nhiệt IGBT, cuộn cảm; TPCB, Hum cho Môi trường tủ"),
    "APU Stat 10s": ("Điện áp, dòng, P, Q 3 pha phía AC; tần số; Vdc, Idc, Pdc phía DC; mỗi 10 giây",
                     "chưa dùng – nên xét", "APU có tụt công suất không; Phía DC (Vdc), Truyền tải-MBA (áp, f)"),
    "APS Stat 60s": ("Tamb nhiệt trong tủ, Ttrans (?), Riso điện trở cách điện, Cleak điện dung rò; mỗi phút",
                     "chưa dùng – nên xét", "Cách điện (Riso, Cleak), Môi trường tủ (Tamb)"),
    "APS Stat 10s": ("Irr = bức xạ mặt trời (W/m²); chỉ có dòng tiêu đề, KHÔNG có số đo (tủ 9.3, tháng 8)",
                     "không có dữ liệu", "nếu có số đo: thay quy tắc giờ 08:00–16:30 khi xét bức xạ thấp"),
    "APU Ctrl Trig": ("Lệnh gửi xuống APU: P, Q đặt từng pha, OpMode, giới hạn Vdc/Idc; ghi khi đổi",
                      "chưa dùng", "biết APU bị ra lệnh giảm/tắt (không phải hỏng)"),
    "APS Ctrl Trig": ("Trạng thái quạt/điều hòa, bơm nước, sưởi, hút ẩm, lệnh bật APU 1–6, Enable; ghi khi đổi",
                      "chưa dùng", "Môi trường tủ (làm mát), Đóng cắt (Enable)"),
    "APU Energy": ("Bộ đếm điện tích (Ah) theo 12 kênh, chiều dương/âm (?)", "chưa dùng", "chưa rõ"),
    "APS Energy": ("Điện năng vào/ra (kWh) cộng dồn của từng APU và cả tủ", "chưa dùng", "sản lượng; kiểm tra chéo"),
    "APS Switching Cycles": ("Số lần đóng cắt phía AC và DC của từng APU (cộng dồn)",
                             "chưa dùng", "Đóng cắt-bảo vệ: contactor/máy cắt đóng cắt nhiều"),
}
DEVICE_ORDER = ["IGBT", "DRIVER", "BOOSTER", "DC", "INDUCTOR", "INSULATION", "ENCLOSURE", "CONTROL", "SWITCHGEAR",
                "GRID", "OPERATION"]


# OpState của APU theo manual SINACON PV 01/2020, tr. 212–213 (bảng "inverter subunit").
# 20 không có trong manual; thấy ngay trước 30 khi bộ điều khiển khởi động lại.
# OpState của cả tủ (APS) ghi trong log là 20000, cũng không có trong manual.
OPSTATE = {
    0: "Tiết kiệm năng lượng (bộ điều khiển APU tắt)", 20: "Bộ điều khiển khởi động lại (20, ngoài manual)",
    30: "Bộ điều khiển đang khởi động (booting)", 50: "Tắt (PWM tắt, AC và DC hở)",
    100: "Chuẩn bị nối DC", 130: "Chờ nối DC", 160: "Nạp trước phía DC", 200: "Đã nối DC (PWM tắt, AC hở)",
    300: "Đang thử nối AC (chờ áp, tần số lưới)", 330: "Chuẩn bị nối AC", 360: "Chờ nối AC", 400: "Nạp trước phía AC",
    500: "Đã nối AC nhưng PWM tắt: KHÔNG phát", 600: "Đã nối AC, lưới cô lập",
    3000: "Đang phát (MPPT)", 3010: "Đang phát, giới hạn công suất AC", 3020: "Đang phát, giới hạn dòng AC",
    3030: "Đang phát, giới hạn áp DC", 3040: "Đang phát (MPPT dò lên)", 3050: "Đang phát (MPPT dò xuống)",
    3060: "Đang phát (MPPT chu kỳ)", 3070: "PWM tắt 1 phút vì dòng DC thấp", 4000: "Lưới cô lập: khởi động",
    4100: "Lưới cô lập: sẵn sàng", 10000: "Chế độ thử (chỉ cho người bảo trì)",
}


def opstate_label(s) -> str:
    s = int(s)
    return OPSTATE.get(s, f"{s} (không có trong manual)")


LEVELS = [0, 1, 2, 3, 3.6]
LEVEL_NAMES = ["Tắt / khởi động lại (0–50)", "Nối AC, không phát (500)", "Đang nối DC/AC (100–400)",
               "Phát giới hạn (3010–3030)", "Phát điện (3000…)"]


def opstate_level(s: pd.Series) -> pd.Series:
    """OpState -> mức vẽ: 0 tắt/khởi động lại, 1 nối AC không phát, 2 đang nối, 3 phát giới hạn, 3,6 phát."""
    s = s.astype(int)
    return pd.Series(np.select([s <= 50, s == 500, s.isin([3010, 3020, 3030, 3070]), s >= 3000],
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
        subdirs[:] = [s for s in subdirs if not s.startswith("SINACON_")]   # bỏ thư mục kết quả của mình
        if Path(d).relative_to(drive / "MyDrive").parts.__len__() >= 4:
            subdirs[:] = []
            continue
        if last in subdirs:
            return Path(d) / last
    raise FileNotFoundError(f"Không thấy '{sub}'. Kiểm tra đã thêm lối tắt thư mục dữ liệu vào Drive của tôi chưa.")


# Lựa chọn trong notebook -> ("01_log", tên thư mục đợt, tên thư mục kết quả; vị trí [2] giữ như bản cũ). Từ 18/11/2026 nhóm xếp lại Drive:
# "Trung Nam/01_log/<đợt>/<tuyến>/<tủ>/log/" và mỗi tủ là các BẢNG TÁCH (APU_Stat_Trig.csv… gộp cả đợt), không còn
# file log ngày. Tên cũ (Inv_log) để dự phòng. Tháng 8, 9, 10/2026 nay chung một thư mục 13/08–31/10.
PERIODS = {
    "Tháng 10/2025": ("01_log", ["2025-10"], "thang_10_2025"),
    "Tháng 1–3/2026 (12/01–12/03)": ("01_log", ["2026-01-12_2026-03-12"], "thang_1_3"),
    "Tháng 4–6/2026 (04/04–03/06)": ("01_log", ["2026-04-04_2026-06-04", "4-4-2026-6-4-2026"], "thang_4_6"),
    "Tháng 6–8/2026 (13/06–13/08)": ("01_log", ["2026-06-13_2026-08-13", "6-13-2026-8-13-2026"], "thang_6_8"),
    "Tháng 8–10/2026 (13/08–31/10)": ("01_log", ["2026-08-13_2026-10-31"], "thang_8_10"),
}
for _old in ("Tháng 8/2026", "Tháng 9/2026", "Tháng 10/2026"):      # tab Colab cũ còn giữ lựa chọn cũ
    PERIODS[_old] = PERIODS["Tháng 8–10/2026 (13/08–31/10)"]

# Bảng tách -> log ngày (dựng trên đĩa máy Colab, không ghi lên Drive)
SPLIT_TABLES = ("APU_Stat_Trig", "APS_Stat_Trig", "APU_Stat_60s", "APU_Stat_10s", "APS_Stat_60s", "APS_Stat_10s",
                "APU_Ctrl_Trig", "APS_Ctrl_Trig", "APU_Energy", "APS_Energy", "APS_Switching_Cycles")
CACHE = Path("/content/sinacon_cache") if Path("/content").is_dir() else Path(tempfile.gettempdir()) / "sinacon_cache"


def _find_by_name(bases: list[Path], name: str, max_depth: int = 5) -> Path | None:
    for base in bases:
        if not base.is_dir():
            continue
        for d, subdirs, _ in os.walk(base):
            subdirs[:] = [s for s in subdirs if not s.startswith("SINACON_")]   # bỏ thư mục kết quả của mình
            if name in subdirs:
                folder = Path(d) / name
                for sub in ("processed data", "inprocessing"):   # có thư mục đã xử lý thì dùng nó
                    if (folder / sub).is_dir():
                        return folder / sub
                return folder
            if len(Path(d).relative_to(base).parts) >= max_depth:
                subdirs[:] = []
            subdirs[:] = [s for s in subdirs if not s.startswith(".") or base.name == ".shortcut-targets-by-id"]
    return None


def find_period(drive: Path, link: str, choice: str) -> Path:
    """Thư mục log của lựa chọn `choice` (khóa của PERIODS). Tìm nhanh ở <lối tắt>/<thư mục>/01_log/<đợt>,
    không thấy thì dò theo tên trong các lối tắt, Drive của tôi, Drive dùng chung. `link` giữ để tương thích."""
    drive = Path(drive)
    _, names, _ = PERIODS[choice]
    sc = drive / ".shortcut-targets-by-id"
    tops = [c for t in (sorted(sc.iterdir()) if sc.is_dir() else []) if t.is_dir() for c in t.iterdir() if c.is_dir()]
    tops += [c for c in (drive / "MyDrive").iterdir() if c.is_dir()] if (drive / "MyDrive").is_dir() else []
    for name in names:
        for top in tops:
            if (top / "01_log" / name).is_dir():
                return top / "01_log" / name
    for name in names:
        found = _find_by_name([sc, drive / "MyDrive", drive / "Shareddrives"], name)
        if found is not None:
            return found
    raise FileNotFoundError(f"Không thấy thư mục '{names[0]}'. Thêm lối tắt thư mục 'Trung Nam' (có 01_log) vào "
                            "Drive của tôi: chuột phải -> Sắp xếp -> Thêm lối tắt -> Drive của tôi, rồi chạy lại ô 2.")


def split_to_daily(log_dir: Path, aps: str, out_dir: Path) -> int:
    """Dựng lại log ngày `<aps>_YYYYMMDD_000000.csv` (dấu ;) từ các bảng tách của một tủ; trả về số ngày.
    Nhận cả hai dạng tiêu đề: 'Log Type,System,Time Stamp,<cột…>' và 'Log Type,System,Time Stamp,Values...'
    + dòng tiêu đề thật ở dòng sau (có thể lặp lại giữa file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    headers, handles = [], {}
    try:
        for table in SPLIT_TABLES:
            src = log_dir / f"{table}.csv"
            if not src.is_file():
                continue
            kind = table.replace("_", " ")
            with src.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
                rows = csv.reader(fh)
                head = next(rows, None)
                if head and len(head) > 3 and head[3].startswith("Values"):
                    head = next(rows, None)
                if not head or len(head) < 4:
                    continue
                headers.append(";".join([kind, kind.split()[0], "TimeStamp", *head[3:]]))
                for r in rows:
                    if len(r) < 3 or r[2] in ("TimeStamp", "Time Stamp"):
                        continue
                    day = r[2][:10].replace("-", "")
                    if len(day) != 8 or not day.isdigit():
                        continue
                    h = handles.get(day)
                    if h is None:
                        h = handles[day] = (out_dir / f".{day}.part").open("w", encoding="utf-8", newline="")
                    h.write(";".join(r) + "\r\n")
    finally:
        for h in handles.values():
            h.close()
    for day in sorted(handles):
        part = out_dir / f".{day}.part"
        with (out_dir / f"{aps}_{day}_000000.csv").open("w", encoding="utf-8", newline="") as out:
            out.write("Log Type;System;Time Stamp;Column Header...\r\n" + "".join(x + "\r\n" for x in headers))
            out.write(part.read_text(encoding="utf-8"))
        part.unlink()
    return len(handles)


def prepare_cabinet(cab_dir: Path, inverter: str) -> Path:
    """Thư mục có log ngày của tủ. Tủ dạng bảng tách thì dựng log ngày vào CACHE (một lần mỗi phiên Colab).
    Báo khi mã APS trong file không khớp tủ theo config/cabinets.csv."""
    expected = {v: k for k, v in cabinet_to_inverter().items()}.get(inverter)
    split = next((d for d in (cab_dir / "log", cab_dir) if (d / "APU_Stat_Trig.csv").is_file()), None)
    if split is None:
        found = sorted({m.group(1) for f, _ in iter_log_files(cab_dir) if (m := LOG_RE.search(f))})
        if expected and found and expected not in found:
            print(f"!!! CẢNH BÁO: thư mục tủ {inverter} chứa log của {', '.join(found)}, nhưng tủ {inverter} là "
                  f"{expected}. Thư mục trên Drive có thể bị xếp nhầm tủ.")
        return cab_dir
    aps = expected or "APS-000000"
    sig = "|".join(f"{f.name}:{f.stat().st_size}" for f in sorted(split.glob("*.csv")))
    out = CACHE / hashlib.sha1(str(cab_dir).encode()).hexdigest()[:10] / inverter
    if (out / ".xong").is_file() and (out / ".xong").read_text(encoding="utf-8") == sig:
        return out
    print(f"Tủ {inverter} ở dạng bảng tách (gộp cả đợt): đang dựng lại log ngày, khoảng 1–5 phút…", flush=True)
    for old in out.glob("*.csv"):
        old.unlink()
    n = split_to_daily(split, aps, out)
    (out / ".xong").write_text(sig, encoding="utf-8")
    print(f"Xong: {n} ngày, mã tủ {aps}.", flush=True)
    return out


def find_cabinet(root: Path, inverter: str) -> Path:
    """Thư mục log ngày của tủ `inverter` (vd "9.3") dưới root; tên thư mục có thể là "9.3", "INV 9.3", "inv 9.3".
    Tủ dạng bảng tách thì trả về thư mục log ngày đã dựng lại (xem prepare_cabinet)."""
    for d, subdirs, _ in os.walk(root):
        for s in sorted(subdirs):
            m = INVERTER_DIR_RE.fullmatch(s.strip())
            if m and m.group(1) == inverter:
                return prepare_cabinet(Path(d) / s, inverter)
    raise FileNotFoundError(f"Không thấy thư mục tủ {inverter} dưới {root}")


def list_files(cab_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(cab_dir.rglob("*.csv")):
        m = LOG_RE.search(f.name)
        if m and "TR" not in [x.upper() for x in f.relative_to(cab_dir).parts]:
            rows.append(dict(file=f.name, ngay=pd.to_datetime(m.group(2)).date(), MB=round(f.stat().st_size / 1e6, 1)))
    if not rows:
        print(f"!!! Không có file log nào trong {cab_dir}. Nếu đường dẫn nằm trong SINACON_mo_hinh_lai/SINACON_so_loi "
              "thì đó là thư mục KẾT QUẢ, không phải log: tab Colab đang giữ ô 2 cũ, mở lại notebook từ GitHub.")
    return pd.DataFrame(rows, columns=["file", "ngay", "MB"])


def _first_file(cab_dir: Path, day: str | None = None) -> tuple[str, str]:
    for name, read in iter_log_files(cab_dir):
        if day is None or LOG_RE.search(name).group(2) == pd.Timestamp(day).strftime("%Y%m%d"):
            return name, read().decode("utf-8", errors="replace")
    raise FileNotFoundError(f"Không có file log ngày {day}")


def record_types(cab_dir: Path, day: str | None = None) -> pd.DataFrame:
    """Các loại dòng trong một file log ngày: số dòng, ý nghĩa, có dùng không."""
    name, text = _first_file(cab_dir, day)
    n = Counter(line.split(";", 1)[0] for line in text.splitlines() if ";" in line)
    t = pd.DataFrame([(k, v, *RECORD_MEANING.get(k, ("loại dòng chưa biết", "chưa dùng", "")))
                      for k, v in n.most_common() if k[:3] in ("APU", "APS")],
                     columns=["loai_dong", "so_dong", "cot_chinh", "mo_hinh_hien_tai", "co_the_dung_cho"])
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
    if next(iter_log_files(cab_dir), None) is None:
        print(f"!!! Thư mục tủ có nhưng KHÔNG có file log nào (APS-xxxxxx_YYYYMMDD_hhmmss.csv): {cab_dir}\n"
              "    Có thể tủ này không có log ở đợt đã chọn, hoặc Drive bị ngắt: chạy lại ô 1, 2, 3.")
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


# ------------------------------------------------------------------ vì sao chọn các khoảng thời gian
GAP_CHOICES_MIN = (0, 0.5, 1, 2, 5, 10, 15, 30, 60)


def gap_choice(errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Số sự cố khi gom với từng ngưỡng (GAP_CHOICES_MIN) và khoảng cách (phút) từ lúc nhóm mã trước tắt
    tới lúc mã kế tiếp bật. Dùng để thấy vì sao chọn INCIDENT_GAP."""
    from datetime import timedelta
    eps = inc_mod.code_episodes(errors)
    keep = inc_mod.INCIDENT_GAP
    rows = []
    try:
        for m in GAP_CHOICES_MIN:
            inc_mod.INCIDENT_GAP = timedelta(minutes=m)
            rows.append((m, inc_mod.group_incidents(eps).nunique()))
    finally:
        inc_mod.INCIDENT_GAP = keep
    gaps, cur = [], None
    for s, e in zip(eps["start"], eps["end"].fillna(eps["start"])):
        if cur is not None and s > cur:
            gaps.append((s - cur).total_seconds() / 60)
        cur = e if cur is None else max(cur, e)
    return pd.DataFrame(rows, columns=["gom_neu_cach_duoi_phut", "so_su_co"]), pd.Series(gaps, name="phut")


def plot_gap_choice(errors: pd.DataFrame):
    """Trái: số sự cố theo ngưỡng gom. Phải: mã kế tiếp bật cách nhóm trước bao lâu."""
    import matplotlib.pyplot as plt
    tab, gaps = gap_choice(errors)
    chosen = inc_mod.INCIDENT_GAP.total_seconds() / 60
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = range(len(tab))
    ax[0].plot(x, tab["so_su_co"], "o-")
    for i, (m, n) in enumerate(tab.itertuples(index=False)):
        ax[0].annotate(str(n), (i, n), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax[0].set_xticks(list(x), [f"{m:g}" for m in tab["gom_neu_cach_duoi_phut"]])
    i5 = list(tab["gom_neu_cach_duoi_phut"]).index(chosen) if chosen in GAP_CHOICES_MIN else None
    if i5 is not None:
        ax[0].axvline(i5, color="red", ls="--", label=f"đang dùng: {chosen:g} phút")
    ax[0].set_xlabel("gom các mã bật cách nhau dưới … phút")
    ax[0].set_ylabel("số sự cố")
    ax[0].set_title("Khoảng gom càng rộng → càng ít sự cố.\nChọn chỗ đường gần nằm ngang: chuỗi đã gom xong, chưa gộp nhầm")
    ax[0].legend(fontsize=8)
    bins = [0, 0.5, 1, 2, 5, 10, 15, 30, 60, 180, np.inf]
    labels = ["≤0,5", "0,5–1", "1–2", "2–5", "5–10", "10–15", "15–30", "30–60", "1–3 h", ">3 h"]
    cnt = pd.cut(gaps, bins, labels=labels).value_counts().reindex(labels)
    colors = ["tab:blue" if b <= chosen else "tab:gray" for b in bins[1:]]
    ax[1].bar(labels, cnt.values, color=colors)
    for i, v in enumerate(cnt.values):
        ax[1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax[1].set_xlabel("mã kế tiếp bật sau khi nhóm trước tắt (phút)")
    ax[1].set_ylabel("số lần")
    ax[1].set_title(f"Xanh = bị gom vào sự cố trước (cách < {chosen:g} phút); xám = thành sự cố mới")
    ax[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig, tab


def warning_offsets(errors: pd.DataFrame, warnings: pd.DataFrame) -> pd.DataFrame:
    """Mỗi lần warning bật: cách t0 của sự cố gần nhất bao nhiêu giây (âm = trước t0)."""
    from sinacon.daily_log import warning_activations
    eps = inc_mod.code_episodes(errors)
    eps["inc"] = inc_mod.group_incidents(eps)
    t0 = np.sort(eps.groupby("inc")["start"].min().to_numpy())
    on = warning_activations(warnings)
    if not len(t0) or on.empty:
        return pd.DataFrame(columns=["giay", "code"])
    t = on["t"].to_numpy()
    k = np.clip(np.searchsorted(t0, t), 1, len(t0) - 1) if len(t0) > 1 else np.zeros(len(t), int)
    cand = np.stack([t0[k - 1 if len(t0) > 1 else k], t0[k]])
    d = (t[None, :] - cand) / np.timedelta64(1, "s")
    pick = np.abs(d).argmin(axis=0)
    return pd.DataFrame({"giay": d[pick, np.arange(len(t))], "code": on["code"].to_numpy()})


def plot_warning_window(errors: pd.DataFrame, warnings: pd.DataFrame):
    """Trái: warning bật mỗi giây quanh t0 ("mưa rào" quanh sự cố, "mưa phùn" ở xa).
    Phải: cửa sổ sau t0 dài A giây gắn bao nhiêu warning và bao nhiêu trong đó là ngẫu nhiên."""
    import matplotlib.pyplot as plt
    from sinacon import calibrate as cal
    r = cal.RULE
    o = warning_offsets(errors, warnings)
    d = o["giay"].to_numpy(float)
    before, after = inc_mod.WARN_BEFORE.total_seconds(), inc_mod.WARN_AFTER.total_seconds()
    b_here, a_here, base = cal.choose_warning_window(d, r["max_random_share"], r["far_s"], r["max_s"])
    edges = np.array([-300, -120, -60, -30, -10, 0, 10, 30, 60, 90, 120, 300])
    cnt, _ = np.histogram(d, edges)
    rate = cnt / np.diff(edges)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
    ax[0].stairs(rate, edges, fill=True, alpha=0.6, label="warning bật mỗi giây")
    ax[0].axhline(base, color="k", ls=":", lw=1.2,
                  label=f"mức nền (đo ở xa sự cố {r['far_s'][0]}–{r['far_s'][1]} s) = {base:.2f}/giây")
    ax[0].axvspan(-before, after, color="red", alpha=0.12, label=f"cửa sổ đang dùng −{before:g} … +{after:g} s")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("giây so với t0 (âm = trước t0)"); ax[0].set_ylabel("lần / giây")
    ax[0].set_title("Mưa rào quanh sự cố (cột cao) trên nền mưa phùn (đường chấm)")
    ax[0].legend(fontsize=7)
    c = cal.window_curve(d, r["far_s"], r["max_s"])
    ax[1].plot(c["giay"], c["gan_sau"], label="warning gắn vào sự cố (cửa sổ 0 … A)")
    ax[1].plot(c["giay"], c["ngau_nhien"], label="trong đó ước tính là ngẫu nhiên (mức nền × A)")
    ax[1].plot(c["giay"], r["max_random_share"] * c["gan_sau"], ls="--", color="gray",
               label=f"{r['max_random_share']:.0%} số warning gắn vào (mức chấp nhận)")
    ax[1].axvline(after, color="red", ls="--", lw=1, label=f"đang dùng A = {after:g} s (chọn trên đợt 04–06)")
    ax[1].axvline(a_here, color="green", ls=":", lw=1.2, label=f"áp công thức lên tủ/đợt này: A = {a_here:g} s")
    ax[1].set_xlabel("độ dài cửa sổ sau t0, A (giây)"); ax[1].set_ylabel("số warning (cộng mọi sự cố)")
    ax[1].set_title("Xô càng rộng hứng càng nhiều, nhưng mưa phùn cũng tăng.\n"
                    f"Chọn A lớn nhất mà phần ngẫu nhiên ≤ {r['max_random_share']:.0%}")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    inside = o[(o["giay"] >= -before) & (o["giay"] <= after)]
    names = fault_codes().set_index("code_key")["name"]
    top = lambda s: (s.value_counts().head(6).rename_axis("warning").reset_index(name="so_lan")
                     .assign(ten=lambda x: x["warning"].map(names)))
    pick = [10, 30, 60, 90, 120, 150, 200, 300]
    bang = c[c["giay"].isin(pick)].assign(ty_le_ngau_nhien=lambda x: (x["ngau_nhien"] / x["gan_sau"].clip(lower=1))
                                          .map("{:.0%}".format), ngau_nhien=lambda x: x["ngau_nhien"].round(0))
    bang = bang[["giay", "gan_sau", "ngau_nhien", "ty_le_ngau_nhien"]]
    print(f"{len(o)} lần warning bật; {len(inside)} lần rơi vào cửa sổ đang dùng. Mức nền {base:.2f} lần/giây.")
    print(f"Nếu áp công thức ({r['max_random_share']:.0%}) lên riêng tủ/đợt này: −{b_here:g} s … +{a_here:g} s "
          "(chỉ để so, không dùng để sửa).")
    return fig, bang, top(inside["code"]), top(o.loc[o["giay"].abs() > 600, "code"])


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
    t["ket_luan"] = t["tier1_verdict"].map({"real": "THẬT", "false": "GIẢ", "skip": "KHÔNG XÉT"})
    t["quy_tac"] = t["tier1_class"].map({**tier1.CLASS_LABEL,
                                         "kept_prior": "Chưa đủ căn cứ: tạm để thật",
                                         "not_feeding": "Không xét: tủ không phát lúc t0"})
    t["ly_do"] = t["tier1_reason"].fillna("")
    t.loc[(t["tier1_class"] == "not_feeding") & (t["ly_do"] == ""), "ly_do"] = (
        "Không APU nào đang phát 1 phút trước t0 (đêm, hoặc tủ đã dừng từ trước): không xem được hậu quả, không xét.")
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
