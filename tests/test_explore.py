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


def test_find_period_inv_log_and_converted(tmp_path):
    inv = tmp_path / "MyDrive" / "Trung Nam T5" / "data" / "Inv_log" / "4-4-2026-6-4-2026" / "processed data"
    (inv / "9" / "9.1" / "log").mkdir(parents=True)
    conv = tmp_path / "MyDrive" / "SINACON_so_loi" / "log_goc" / "thang_1_3" / "9" / "9.3" / "log"
    conv.mkdir(parents=True)
    link = "https://drive.google.com/drive/folders/XYZ"
    assert explore.find_period(tmp_path, link, "Tháng 4–6/2026 (04/04–03/06)") == inv
    root = explore.find_period(tmp_path, link, "Tháng 1–3/2026 (12/01–12/03)")
    assert explore.find_cabinet(root, "9.3") == conv.parent
    assert {v[2] for v in explore.PERIODS.values()} >= {"thang_1_3", "thang_10"}


def test_device_table_has_ten_devices_and_operation():
    t = explore.device_table()
    assert len(t) == 11 and t["so_ma"].iloc[:10].sum() == 168
