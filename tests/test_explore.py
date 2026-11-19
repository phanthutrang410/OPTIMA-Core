import pandas as pd

from sinacon import explore


def test_opstate_level_orders_states():
    lv = explore.opstate_level(pd.Series([50, 500, 130, 3010, 3000]))
    assert list(lv) == [0, 1, 2, 3, 3.6]


def test_find_cabinet_and_log_root(tmp_path):
    d = tmp_path / ".shortcut-targets-by-id" / "ABC" / "Raw data all" / "Log file" / "10_2026" / "TUYEN 9" / "INV 9.3"
    d.mkdir(parents=True)
    (tmp_path / "MyDrive").mkdir()
    root = explore.find_log_root(tmp_path, "https://drive.google.com/drive/folders/ABC", "Log file/10_2026")
    assert root.name == "10_2026"
    assert explore.find_cabinet(root, "9.3") == d


def test_find_period_new_01_log_and_old_inv_log(tmp_path):
    new = tmp_path / ".shortcut-targets-by-id" / "QQ" / "Trung Nam" / "01_log" / "2026-08-13_2026-10-31"
    (new / "9" / "9.3" / "log").mkdir(parents=True)
    inv = tmp_path / "MyDrive" / "Trung Nam T5" / "data" / "Inv_log" / "4-4-2026-6-4-2026" / "processed data"
    (inv / "9" / "9.1" / "log").mkdir(parents=True)
    link = "https://drive.google.com/drive/folders/XYZ"
    assert explore.find_period(tmp_path, link, "Tháng 8–10/2026 (13/08–31/10)") == new
    assert explore.find_period(tmp_path, link, "Tháng 9/2026") == new          # lựa chọn cũ vẫn chạy
    assert explore.find_period(tmp_path, link, "Tháng 4–6/2026 (04/04–03/06)") == inv


def test_split_tables_rebuilt_as_daily_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(explore, "CACHE", tmp_path / "cache")
    log = tmp_path / "01_log" / "p" / "9" / "9.3" / "log"
    log.mkdir(parents=True)
    (log / "APS_Stat_Trig.csv").write_text(
        "﻿Log Type,System,Time Stamp,Milliseconds/ms,OpState," + ",".join(f"Error{i}" for i in range(1, 9))
        + "," + ",".join(f"Warning{i}" for i in range(1, 9)) + "\n"
        "APS Stat Trig,APS,2026-08-13 00:00:00,005,20000," + ",".join(["0"] * 16) + "\n"
        "APS Stat Trig,APS,2026-08-14 05:00:00,010,0,42,43," + ",".join(["0"] * 14) + "\n", encoding="utf-8")
    head = "APU Stat Trig,APU,TimeStamp,Milliseconds/ms,OpState," + ",".join(["x"] * 10) + "," + ",".join(
        f"Error{i}" for i in range(1, 9))
    (log / "APU_Stat_Trig.csv").write_text(            # dạng 'Values...' + tiêu đề thật ở dòng 2, lặp lại giữa file
        "﻿Log Type,System,Time Stamp,Values...\n" + head + "\n"
        "APU Stat Trig,APU 3,2026-08-14 10:03:11,200,500," + ",".join(["0"] * 10) + ",3,0,0,0,0,0,0,0\n"
        + head + "\n", encoding="utf-8")
    cab = explore.find_cabinet(tmp_path / "01_log" / "p", "9.3")
    assert sorted(f.name for f in cab.glob("*.csv")) == ["APS-000344_20260813_000000.csv",
                                                         "APS-000344_20260814_000000.csv"]
    name, errors, _ = explore.read_cabinet(cab)
    assert name == "APS-000344"
    assert {(r.scope, r.unit, r.codes) for r in errors.itertuples() if r.codes} == {("APS", 0, (42, 43)),
                                                                                   ("APU", 3, (3,))}
    assert explore.find_cabinet(tmp_path / "01_log" / "p", "9.3") == cab   # lần sau dùng lại


def test_device_table_has_ten_devices_and_operation():
    t = explore.device_table()
    assert len(t) == 11 and t["so_ma"].iloc[:10].sum() == 168


def test_find_log_root_skips_own_output_folders(tmp_path):
    (tmp_path / "MyDrive" / "SINACON_mo_hinh_lai" / "tung_buoc" / "10_2026" / "9.3").mkdir(parents=True)
    real = tmp_path / "MyDrive" / "Log file" / "10_2026"
    (real / "TUYEN 9" / "INV 9.3").mkdir(parents=True)
    assert explore.find_log_root(tmp_path, "https://drive.google.com/drive/folders/NONE", "Log file/10_2026") == real
    assert list(explore.list_files(real).columns) == ["file", "ngay", "MB"]


def test_gap_choice_and_warning_offsets():
    t = pd.to_datetime(["2026-08-14 10:00:00", "2026-08-14 10:00:01", "2026-08-14 10:00:20", "2026-08-14 14:00:00",
                        "2026-08-14 14:00:05"])
    errors = pd.DataFrame({"scope": "APU", "unit": 3, "t": t, "opstate": 500,
                           "codes": [(32,), (32, 33), (), (3,), ()]})
    tab, gaps = explore.gap_choice(errors)
    assert tab.set_index("gom_neu_cach_duoi_phut").loc[5, "so_su_co"] == 2
    assert tab.set_index("gom_neu_cach_duoi_phut").loc[0, "so_su_co"] == 2
    warnings = pd.DataFrame({"t": pd.to_datetime(["2026-08-14 10:00:30", "2026-08-14 11:00:00", "2026-08-14 13:59:50"]),
                             "codes": [frozenset({24}), frozenset(), frozenset({24})]})
    o = explore.warning_offsets(errors, warnings)
    assert list(o["giay"]) == [30.0, -10.0]
