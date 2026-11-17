# SINACON PV — kiểm chứng và chẩn đoán cảnh báo lỗi biến tần

Nhà máy điện mặt trời Trung Nam có 100 tủ biến tần Siemens SINACON PV (APS), mỗi tủ 4 khối công suất (APU). Mỗi khi
một tủ báo lỗi, code này trả lời ba câu hỏi cho từng **sự cố** (một nhóm mã lỗi bật gần nhau, bắt đầu lúc t0):

| Tầng | Câu hỏi | Cách làm | Module |
|---|---|---|---|
| 1 | Cảnh báo thật hay giả? | Tiêu chí vật lý: APU có ngừng phát quanh t0 (OpState), nhiệt độ có chạm ngưỡng bảo vệ | `tier1` |
| 2 | Lỗi trong tủ, ngoài lưới, phía DC, điều khiển nhà máy hay thao tác? | Tổng xác suất hậu nghiệm theo phạm vi | `tier23` |
| 3 | Bộ phận nào, xác suất bao nhiêu? | Bayes: tiên nghiệm từ mã lỗi × tỉ số hợp lý của dấu hiệu trong log | `tier23` |

Phương pháp chi tiết (công thức, ngưỡng, nguồn của từng tham số): [`docs/METHOD.md`](docs/METHOD.md).

## Cài đặt

```bash
pip install -e .            # hoặc: pip install -r requirements.txt
python -m pytest            # kiểm tra
```

## Chạy cho một tháng log mới

Dữ liệu gốc (log, SCADA, sổ bảo hành) nằm trên Drive của nhóm và **chỉ đọc**; mọi kết quả ghi ra thư mục riêng.

```bash
# 1. Gom mã lỗi thành sự cố (đọc log ngày, kể cả trong zip)
python -m sinacon.incidents --logs "<Log file/10_2026>" --out out/10/incidents.csv

# 2. Trích OpState và nhiệt độ quanh từng sự cố
python -m sinacon.windows --logs "<Log file/10_2026>" --incidents out/10/incidents.csv --out out/10/windows

# 3. Tầng 1: thật / giả
python -m sinacon.tier1 --incidents out/10/incidents.csv --windows out/10/windows --out out/10/tier1.csv

# 4. Tầng 2–3: phạm vi, bộ phận, xác suất (--context: tầng 1 các tháng trước, dùng cho tiền sử hỏng của APU)
python -m sinacon.tier23 --tier1 out/10/tier1.csv --context out/09/tier1.csv \
    --windows out/10/windows out/09/windows --out out/10/tier23.csv
```

## Kiểm chứng

| Nguồn | Kiểm gì | Lệnh |
|---|---|---|
| SCADA công suất (đo độc lập với log) | Cảnh báo làm tủ mất công suất có bị kết luận giả không | `python -m sinacon.scada check --tier1 ... --scada ...` |
| SCADA những ngày sau | Sự cố xếp B3 có kéo theo thiếu công suất kéo dài hơn B1 không | `python -m sinacon.validation next-day ...` |
| Sổ thay thế linh kiện (`config/om_replacements.csv`) | Bộ phận dự đoán có trùng linh kiện đã thay không | `python -m sinacon.validation om ...` |
| Phiếu sửa chữa tương lai | Dự đoán đã khóa trước bằng SHA-256 | `python -m sinacon.lock lock / score ...` |

Đọc SCADA gốc (file `Power reports*.xls`): `python -m sinacon.scada read-xls --source ... --start ... --end ... --out scada.pkl`.
Tần suất hỏng từ sổ bảo hành: `python -m sinacon.warranty --register ... --out ...`.

## Cấu trúc

```
config/
  fault_codes.csv        230 mã error/warning của manual SINACON-PV_BA 01/2020 (mục 10); phân nhóm hai tầng:
                         device (thiết bị) + fault_type (lỗi gì) cho 167 mã error; cột group cũ G01–G12
                         chỉ còn dùng ở tier23 (đang chuyển)
  cabinets.csv           mã tủ APS-xxxxxx -> số hiệu inverter "khối.số"
  om_replacements.csv    34 phiếu thay linh kiện 03/01–27/05/2026 (nhãn thật duy nhất); cột device = thiết bị
src/sinacon/
  config.py              đường dẫn, đọc bảng cấu hình
  daily_log.py           đọc log ngày (APU/APS Stat Trig), tìm thư mục tủ
  incidents.py           bước 1: mã lỗi -> sự cố
  windows.py             bước 2: OpState và nhiệt độ quanh sự cố
  tier1.py               bước 3: tầng 1
  tier23.py              bước 4: tầng 2–3
  scada.py               đọc SCADA, nhãn mất công suất, kiểm chứng tầng 1
  validation.py          kiểm chứng tầng 3 (sổ thay thế, SCADA ngày sau), khoảng tin cậy Wilson, AUC
  lock.py                khóa dự đoán bằng SHA-256 và chấm khi có phiếu
  warranty.py            tần suất hỏng theo lớp từ sổ bảo hành
tests/
docs/METHOD.md
```

## Cột đầu ra chính

| Cột | Ý nghĩa |
|---|---|
| `cabinet`, `inverter`, `apu`, `t0`, `codes`, `first_device`, `first_fault_type` | sự cố: tủ, APU báo lỗi, thời điểm, các mã, thiết bị và loại lỗi của mã đầu tiên (bảng hai tầng) |
| `tier1_verdict` | `real` / `false` |
| `tier1_class`, `tier1_reason` | quy tắc đã áp dụng và căn cứ (xem `tier1.CLASS_LABEL`) |
| `n_feed`, `n_trip`, `apu_minutes_lost` | số APU đang phát, số APU bị ngắt, E (APU·phút ngừng phát trong 60′) |
| `n_apu_out`, `apus_out` | APU hỏng/tách kéo dài (không phát suốt 3 giờ trước t0) |
| `scope`, `p_scope` | tầng 2: phạm vi có xác suất cao nhất |
| `top1`, `p_top1`, `top2`, `p_top2` | tầng 3: hai lớp có xác suất cao nhất |
| `p_B1` … `p_OPERATION` | phân bố hậu nghiệm đầy đủ trên 11 lớp |
| `set90` | các lớp cao nhất cộng dồn tới ≥ 90 % |
| `event_type` | `new` (sự kiện vật lý mới) / `persistent` (tình trạng tồn tại) / `none` (giả) |

`p_*` là xác suất mô hình gán, **không phải độ chính xác**. Độ chính xác chỉ đo được bằng các phép kiểm chứng ở trên.
