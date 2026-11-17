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


def test_device_table_has_ten_devices_and_operation():
    t = explore.device_table()
    assert len(t) == 11 and t["so_ma"].iloc[:10].sum() == 167
