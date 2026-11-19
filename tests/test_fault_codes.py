"""Bảng phân nhóm hai tầng (phan-nhom-ma-loi-hai-tang.docx): 167 mã error + A52 (log có, danh mục Siemens không có), 10 thiết bị, mỗi mã đúng một ô."""
from sinacon.config import fault_codes

DEVICES = {"IGBT", "DRIVER", "BOOSTER", "DC", "INDUCTOR", "INSULATION", "ENCLOSURE", "CONTROL", "SWITCHGEAR", "GRID"}


def test_every_error_code_has_one_device():
    t = fault_codes()
    e = t[t["type"] == "error"]
    assert len(e) == 168 and (e["scope"] == "APU").sum() == 111 and (e["scope"] == "APS").sum() == 57
    assert e["device"].notna().all() and e["fault_type"].notna().all()
    assert set(e["device"]) == DEVICES
    assert e["code_key"].is_unique


def test_doc_examples():
    t = fault_codes().set_index("code_key")
    assert t.loc["APU/error/4", "device"] == "INDUCTOR"          # B4 quá nhiệt cuộn cảm
    assert t.loc["APU/error/1", "device"] == "IGBT"              # B1 quá nhiệt IGBT
    assert t.loc["APS/error/5", "device"] == "SWITCHGEAR"        # A5 không có enable / cửa tủ mở
    assert t.loc["APU/error/75", "device"] == "GRID"             # B75 sai lệch tổng dòng


def test_warnings_only_mark_operation():
    t = fault_codes()
    w = t[(t["type"] == "warning") & t["device"].notna()]
    assert set(w["device"]) == {"OPERATION"} and len(w) == 8
