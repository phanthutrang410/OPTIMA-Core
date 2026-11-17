# Phương pháp

## 1. Dữ liệu

| Nguồn | Vai trò | Nội dung |
|---|---|---|
| Log biến tần | đầu vào | APU/APS Stat Trig: trạng thái vận hành (OpState) và mã lỗi, độ phân giải mili giây; APU Stat 60s: nhiệt độ cuộn cảm (TInd) và IGBT (TL1–TL3) theo phút |
| SCADA nhà máy | kiểm chứng | công suất AC từng tủ, mỗi phút; hệ đo độc lập với log |
| Sổ thay thế linh kiện | kiểm chứng | tủ, APU, linh kiện đã thay; chỉ ghi lần thay linh kiện, không ghi sự cố tự phục hồi |
| Sổ bảo hành 2022–2025 | tham số | số lần hỏng theo lớp bộ phận (tần suất hỏng thật) |

Đồng hồ SCADA chậm hơn log khoảng 5 phút; các cửa sổ so sánh đã đủ rộng để chứa độ lệch này.

## 2. Sự cố (`incidents.py`)

1. Đợt mã: mỗi (cấp, đơn vị, mã error) từ bản ghi đầu tiên có mã tới bản ghi đầu tiên không còn mã.
2. Sự cố: các đợt mã nối nhau, đợt sau bật trong vòng 5 phút sau khi đợt trước tắt.
3. t0 = lúc mã error đầu tiên bật. Thêm các warning APS chuyển sang bật trong [t0 − 10 s, t0 + 90 s].

## 3. Tầng 1 — thật hay giả (`tier1.py`)

Hậu quả điện, từ chuỗi OpState s_a(t) của APU a (manual mục 9.1: ≥ 3000 đang phát, 500 bảo vệ vừa tác động, 50 tắt
có điều khiển):

- F_a(t) = 1[s_a(t) ≥ 3000]
- N_a = F_a(t0 − 2′) · 1[∃ t ∈ [t0 − 2′, t0 + 10′]: F_a(t) = 0] — APU đang phát bị ngắt quanh cảnh báo
- E = Σ_a ∫ từ t0 tới t0 + 60′ của F_a(t0 − 2′)·(1 − F_a(t)) dt — thời lượng ngừng phát (APU·phút); E ≤ 10: ngắt ngắn tự phục hồi

Nhiệt độ, so với ngưỡng bảo vệ:

- T* = nhiệt độ lớn nhất trong [t0 − 5′, t0 + 1′]; h = T_lim − T*
- T_lim IGBT = 107 °C (manual tr. 327–328). T_lim cuộn cảm = 160 °C: manual ghi 95 °C, nhưng ở mọi lần bảo vệ cuộn cảm tác
  động trong log TInd dừng ở 157–161 °C, trong khi lúc vận hành bình thường TInd thường 100–150 °C (giá trị ước lượng
  từ dữ liệu, cần hãng xác nhận)
- Δh = 2 + max(0, r)/2 + σ_T: r/2 là độ trễ của giá trị trung bình 1 phút khi nhiệt độ tăng tuyến tính với tốc độ r
  (°C/phút); σ_T = 3 °C là chênh lệch giữa cảm biến bảo vệ và cảm biến ghi log; 2 °C dự phòng
- h ≤ Δh: nhiệt độ đã chạm ngưỡng. |ΔT/Δt| > 15 °C/phút: cảm biến phi vật lý (khối nhiệt có hằng số thời gian vài phút)

Bảy quy tắc, xét theo thứ tự, dừng ở quy tắc đầu tiên thỏa:

| # | Điều kiện | Kết luận |
|---|---|---|
| 1 | Không APU nào phát lúc t0 − 2′ (tủ đang dừng) | giữ kết luận đầu vào |
| 2 | Mã không phải nhiệt; có APU không phát suốt [t0 − 180′, t0 + 30′] trong khi APU khác phát; không có ngắt mới | thật · APU hỏng/tách kéo dài |
| 3 | Ngoài 08:00–16:30; mọi APU ngắt về thẳng 50, không phát lại trong 30′; không có mã phần cứng | giả · dừng do bức xạ thấp |
| 4 | Σ N_a ≥ 1 | thật · ngắt ngắn (E ≤ 10) / kéo dài |
| 5 | Mã nhiệt; h ≤ Δh hoặc APU chuyển sang giới hạn công suất (3010–3030) | thật |
| 6 | Mã nhiệt; cảm biến phi vật lý | giả |
| 7 | Còn lại | giả |

Mô hình nhiệt dạng mạng Foster (phần dư nhiệt độ; theo dõi nhiệt trở R_th bằng CUSUM) đã được thử và không dùng:
kết luận "giả" khi phần dư nhỏ bị SCADA bác ở 163/540 cảnh báo mất công suất, vì bảo vệ vẫn tác động khi diễn biến
nhiệt khớp mô hình; R_th không tăng trong 6 ngày trước 6 vụ nổ IGBT (hỏng đột ngột), chỉ phát hiện 1/11 ca.

## 4. Tầng 2–3 — phạm vi và bộ phận (`tier23.py`)

P(k | mã, e) ∝ π_k(mã) · Π_j LR_j(k)^{e_j}

Lớp: B1 Nhiệt-làm mát, B2 DC-link-nạp trước, B3 Dòng-IGBT-driver, B4 Nguồn phụ-liên động, B5 Cách điện-nối đất,
B6 Điều khiển-truyền thông, B7 Bộ lọc AC-cầu chì (phạm vi "Trong tủ"); Lưới-trung thế; Phía DC; Điều khiển nhà máy;
Thao tác-bảo trì. B1–B6 gom 230 mã của manual (mục 10, tr. 312–347) theo hệ thống con; đối chiếu sổ thay thế 2026:
31/34 phiếu thuộc một nhóm, 3 phiếu hỏng bộ lọc AC không có mã tương ứng nên thêm B7 (manual tr. 326, 442, 464).

**Tiên nghiệm** π_k ∝ α + Σ_c w_c m_c(k) / Σ_c w_c, α = 0,05: mỗi mã thuộc một nhóm danh mục, nhóm ánh xạ sang lớp;
w_c = 1 (error), 0,3 (warning); mã chỉ là hệ quả (mất Powerlink, mất mẫu log) 0,3 / 0,1. Các mã của một sự cố là
triệu chứng của một sự kiện nên tổng khối lượng chuẩn hóa về 1.

**Tỉ số hợp lý** LR_j(k) = P(e_j = 1 | k) / P(e_j = 1 | ¬k):

| Dấu hiệu e_j | LR | Cơ sở |
|---|---|---|
| ≥ 5 tủ khác báo trong ±2′ | Lưới ×300 | ở 5.000 thời điểm ngẫu nhiên ban ngày: 0/5.000; quá nhiệt thật: tối đa 3 tủ |
| ≥ 20 tủ khác báo trong ±15′ | Lưới ×100 | ngẫu nhiên 0/5.000; quá nhiệt thật: tối đa 10 tủ |
| APU đang hỏng từng ngắt kèm mã phần cứng (180 ngày trước) | B3 ×10 | APU ngẫu nhiên 7,2 %; tủ có phiếu hỏng IGBT 90,5 % |
| Nhiệt độ ≥ 155 °C (TInd) hoặc ≥ 104 °C (IGBT) | B1 ×10 | gần ngưỡng bảo vệ |
| ≥ 2 lần 30xx → 500 trong 30′ | B1 ×5 | bảo vệ nhiệt tác động lặp |
| Mã AC cấp APU, không tủ nào khác báo | B7 ×5 | lưới là chung cho cả nhà máy |
| Có APU hỏng/tách kéo dài | 7·ρ_k, cắt về [0,3; 3] | ρ_k: tần suất hỏng theo lớp, sổ bảo hành 2022–2025 |

Tầng 2 là tổng xác suất theo phạm vi; tầng 3 là lớp có xác suất cao nhất. Nhiều cảnh báo của một lần hỏng được gộp
bằng trung bình có trọng số (1 + số APU ngắt) của các sự cố "new".

## 5. Kiểm chứng và kết quả (01–10/2026)

Quy tắc được cố định (git tag) trước khi chấm từng tập. 01–03 và 04–06/2026 dùng để xây dựng; 14/06–13/08 và 09–10/2026
dùng để kiểm định.

| Nguồn | Phép kiểm | Kết quả |
|---|---|---|
| SCADA tại t0, 01–08/2026 | cảnh báo mất công suất bị kết luận giả | 0/540 (cách kết luận theo mã lỗi + mô hình nhiệt: 163/540) |
| SCADA tại t0, 09–10/2026 | như trên | 0/58 (KTC 95 % 0–6 %) |
| SCADA ngày sau, 01–08/2026 | tủ thiếu công suất 1–2 ngày sau: lớp B3 / lớp B1 | 9,3 % (5–18 %) / 1,1 % (1–2 %); AUC của p_B3 0,741 |
| SCADA ngày sau, 09–10/2026 | như trên | 31 % / 5 %; Fisher một phía p = 0,019 |
| Sổ thay thế 01–05/2026 | lớp cao nhất / hai lớp cao nhất | 11/21 (32–72 %) / 14/21 (45–83 %); riêng 04–05 (độc lập): 2/5 / 4/5 |

Theo linh kiện (lớp cao nhất): IGBT/driver 7/10, máy cắt-nạp trước 2/5, nguồn 24 V 2/2, bộ lọc AC 0/3, lưới 0/1.
16 phiếu 01–03/2026 đã được xem khi xây dựng phương pháp. Khoảng tin cậy: Wilson (1927), doi:10.1080/01621459.1927.10502953.
