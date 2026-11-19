import pandas as pd

from sinacon import extract


def test_extract_split_tables_and_summary(tmp_path):
    log = tmp_path / "p" / "9" / "9.3" / "log"
    log.mkdir(parents=True)
    (log / "APS_Stat_Trig.csv").write_text(
        "\ufeffLog Type,System,Time Stamp,Milliseconds/ms,OpState," + ",".join(f"Error{i}" for i in range(1, 9))
        + "," + ",".join(f"Warning{i}" for i in range(1, 9)) + "\n"
        "APS Stat Trig,APS,2026-08-14 05:00:00,010,0,42,43," + ",".join(["0"] * 6) + ",24," + ",".join(["0"] * 7)
        + "\n", encoding="utf-8")
    (log / "APU_Stat_Trig.csv").write_text(
        "\ufeffLog Type,System,Time Stamp,Values...\nAPU Stat Trig,APU,TimeStamp,x\n"
        "APU Stat Trig,APU 3,2026-08-14 10:03:11,200,500," + ",".join(["0"] * 10) + ",3,0,0,0,0,0,0,0\n",
        encoding="utf-8")
    (log / "APU_Stat_60s.csv").write_text(
        "\ufeffLog Type,System,Time Stamp,TInd,TL1,TL2,TL3,TPCB,Hum\n"
        "APU Stat 60s,APU 3,2026-08-14 10:03:00,40,74,73,72,39,41\n", encoding="utf-8")
    s = extract.extract_period(tmp_path / "p", tmp_path / "out")
    assert list(s["tu"]) == ["9.3"] and s.loc[0, "aps"] == "APS-000344"
    x = extract.load(tmp_path / "out")["9.3"]
    assert {(r.scope, r.unit, r.codes) for r in x["errors"].itertuples()} == {("APS", 0, (42, 43)), ("APU", 3, (3,))}
    assert x["warnings"].loc[0, "codes"] == frozenset({24})
    assert x["temps"].loc[0, "TL1"] == 74
