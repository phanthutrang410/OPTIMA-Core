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
        if Path(d).relative_to(drive / "MyDrive").parts.__len__() >= 4:
            subdirs[:] = []
            continue
        if last in subdirs:
            return Path(d) / last
    raise FileNotFoundError(f"Không thấy '{sub}'. Kiểm tra đã thêm lối tắt thư mục dữ liệu vào Drive của tôi chưa.")


# Lựa chọn trong notebook -> (nơi tìm, tên thư mục, tên thư mục kết quả). "data": dưới thư mục dữ liệu nhóm (DATA_LINK);
# "any": tìm theo tên ở mọi chỗ trong Drive đã kết nối; "mine": thư mục riêng trong Drive của tôi.
PERIODS = {
    "Tháng 1–3/2026 (12/01–12/03)": ("mine", "SINACON_so_loi/log_goc/thang_1_3", "thang_1_3"),
    "Tháng 4–6/2026 (04/04–03/06)": ("any", "4-4-2026-6-4-2026", "thang_4_6"),
    "Tháng 6–8/2026 (13/06–13/08)": ("any", "6-13-2026-8-13-2026", "thang_6_8"),
    "Tháng 8/2026": ("data", "Log file/08_2026", "thang_8"),
    "Tháng 9/2026": ("data", "Log file/September 2026", "thang_9"),
    "Tháng 10/2026": ("data", "Log file/10_2026", "thang_10"),
}


def _find_by_name(bases: list[Path], name: str, max_depth: int = 5) -> Path | None:
    for base in bases:
        if not base.is_dir():
            continue
        for d, subdirs, _ in os.walk(base):
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
    """Thư mục log của lựa chọn `choice` (khóa của PERIODS)."""
    drive = Path(drive)
    where, sub, _ = PERIODS[choice]
    if where == "data":
        return find_log_root(drive, link, sub)
    if where == "mine":
        p = drive / "MyDrive" / sub
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Chưa có {p}. Tháng 1–3 trên Drive nhóm chỉ có dạng bảng tách; bản log ngày được "
                                "dựng lại bằng SINACON_colab.ipynb (ô 2b, giai đoạn thang_1_3).")
    fid = link.rstrip("/").split("folders/")[-1].split("?")[0]
    found = _find_by_name([drive / ".shortcut-targets-by-id" / fid, drive / "MyDrive",
                           drive / ".shortcut-targets-by-id", drive / "Shareddrives"], sub)
    if found is None:
        raise FileNotFoundError(f"Không thấy thư mục '{sub}'. Nếu nó chỉ nằm ở 'Được chia sẻ với tôi': chuột phải "
                                "thư mục Inv_log -> Sắp xếp -> Thêm lối tắt -> Drive của tôi, rồi chạy lại ô 2.")
    return found


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
