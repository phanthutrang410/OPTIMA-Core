from pathlib import Path

import pandas as pd

from sinacon import daily_log, incidents


def aps_line(ts, ms, errors=(), warnings=(), op=20000):
    e = (list(errors) + [0] * 8)[:8]
    w = (list(warnings) + [0] * 8)[:8]
    return f"APS Stat Trig;APS;{ts};{ms};{op};" + ";".join(map(str, e)) + ";" + ";".join(map(str, w))


def apu_line(apu, ts, ms, errors=(), op=3000):
    e = (list(errors) + [0] * 8)[:8]
    return f"APU Stat Trig;APU {apu};{ts};{ms};{op};" + ";".join(["0"] * 10) + ";" + ";".join(map(str, e))


def write_log(folder: Path, lines):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "APS-000123_20260901_000000.csv").write_text("x;y\n" + "\n".join(lines) + "\n", encoding="utf-8")


def test_warning_activations_only_count_switch_on(tmp_path):
    write_log(tmp_path / "1.1" / "LOG", [aps_line("2026-09-01 10:00:00", 0, warnings=[13]),
                                        aps_line("2026-09-01 10:01:00", 0, warnings=[13, 47]),
                                        aps_line("2026-09-01 10:02:00", 0, warnings=[47]),
                                        aps_line("2026-09-01 10:03:00", 0, warnings=[13, 47])])
    cab, _, warn = daily_log.parse_trigger_records(tmp_path / "1.1")
    on = daily_log.warning_activations(warn)
    assert cab == "APS-000123"
    got = sorted((str(t.time()), c) for t, c in zip(on["t"], on["code"]))
    assert got == [("10:00:00", "APS/warning/13"), ("10:01:00", "APS/warning/47"), ("10:03:00", "APS/warning/13")]


def test_tr_folder_is_ignored(tmp_path):
    write_log(tmp_path / "1.1" / "TR", [aps_line("2026-09-01 10:00:00", 0, warnings=[13])])
    _, err, warn = daily_log.parse_trigger_records(tmp_path / "1.1")
    assert err.empty and warn.empty


def test_codes_within_five_minutes_form_one_incident(tmp_path, monkeypatch):
    # kiểm logic gom với ngưỡng 5 phút; giá trị đang dùng nằm trong config/windows.json
    from datetime import timedelta
    monkeypatch.setattr(incidents, "INCIDENT_GAP", timedelta(minutes=5))
    write_log(tmp_path / "INV 1.1", [
        apu_line(3, "2026-09-01 10:00:00", 500, errors=[4]),
        apu_line(3, "2026-09-01 10:02:00", 0, errors=[]),
        apu_line(4, "2026-09-01 10:05:00", 0, errors=[78]),        # bật 3′ sau khi mã trước tắt: cùng sự cố
        apu_line(4, "2026-09-01 10:06:00", 0, errors=[]),
        aps_line("2026-09-01 10:00:30", 0, warnings=[47]),        # warning bật trong 90 s sau t0
        apu_line(1, "2026-09-01 13:00:00", 0, errors=[78]),        # sự cố khác
        apu_line(1, "2026-09-01 13:01:00", 0, errors=[]),
    ])
    assert [inv for inv, _ in daily_log.find_inverter_dirs(tmp_path)] == ["1.1"]
    t = incidents.incidents_of_cabinet(tmp_path / "INV 1.1", "1.1")
    assert len(t) == 2
    first = t.iloc[0]
    assert first["t0"] == pd.Timestamp("2026-09-01 10:00:00") and first["apu"] == "3 4"
    assert first["codes"].startswith("APU/error/4 (") and "APU/error/78 (" in first["codes"]
    assert first["codes"].endswith(" + APS/warning/47") and (first["first_device"], first["first_fault_type"]) == ("INDUCTOR", "Quá nhiệt")
