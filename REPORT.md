# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**06/11

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python: 3.13.16 PyTorch: 2.11.0+cu130 Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
 -      "class_id": 468,
        "class_name": "cab",
        "rank": 1,
        "score": 0.510915
        "taxonomy_name": "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
 - Record này nói về cả bức ảnh . Vì Prediction không chứa bounding box(khung giới hạn bao quanh vật thể) hoặc tọa độ vị trí 
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
 - Class list đi kèm taxonomy/checkpoint. Model không tự nghĩ ra tên lớp mới mà chỉ dự đoán trên tập lớp đã được định nghĩa trước.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
 - Không nên chỉ ghi class_name vì cùng một tên lớp có thể được sử dụng trong nhiều taxonomy khác nhau. Việc lưu đồng thời class_id, class_name và taxonomy_name giúp xác định chính xác lớp được dự đoán và bộ nhãn mà lớp đó thuộc về.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
 - Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ ảnh, ví dụ:
    Chọn đối tượng chiếm diện tích lớn nhất trong ảnh.
    Chọn đối tượng nằm ở trung tâm ảnh.
    Chọn chủ thể chính mà ảnh muốn mô tả.
    Quy định cách xử lý khi nhiều đối tượng xuất hiện với mức độ quan trọng tương đương.
    Quy định các trường hợp đặc biệt cần escalation nếu không xác định được lớp chính.
6. Vì sao không được lấy `score` của model làm ground truth?
 - vì score chỉ phản ánh mức độ tự tin của model đối với dự đoán, không chứng minh rằng dự đoán đó là đúng.
 - Ground truth là nhãn chuẩn được tạo bởi con người dựa trên guideline của dự án. Đây là dữ liệu tham chiếu để huấn luyện và đánh giá model. Ngược lại, score chỉ là đầu ra của model sau khi dự đoán.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
  {
    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  }
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 - Bounding box có góc trên-trái tại tọa độ (385.33, 69.24) và góc dưới-phải tại tọa độ (498.92, 348.92). Kích thước của box là khoảng 113.58 pixel chiều rộng và 279.68 pixel chiều cao. Box này biểu diễn vùng ảnh mà model dự đoán chứa đối tượng.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
 - Threshold Analysis
    Threshold = 0.20
    Số prediction có score ≥ 0.20: 11
    Threshold = 0.60
    Số prediction có score ≥ 0.60: 06
  - Khi tăng threshold từ 0.20 lên 0.60, số prediction giảm từ 11 xuống 6. Điều này cho thấy các prediction có score thấp hơn 0.60 đã bị loại bỏ.Threshold thấp (0.20) giữ lại nhiều prediction hơn, giúp giảm khả năng bỏ sót đối tượng nhưng có thể bao gồm nhiều dự đoán kém tin cậy. Ngược lại, threshold cao (0.60) chỉ giữ các prediction mà model tự tin hơn, giúp giảm số dự đoán yếu nhưng có thể làm bỏ sót một số đối tượng thực sự tồn tại trong ảnh.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 - phải làm thêm công việc kiểm tra và lọc kết quả.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
 - Luôn vẽ Bounding Box bao trọn toàn bộ phần vật thể nhìn thấy trong ảnh, bám sát mép ngoài của vật thể, không lấy thừa vùng nền xung quanh và không tự suy đoán phần bị che khuất. Mỗi vật thể riêng biệt phải có một Bounding Box riêng và được gán đúng nhãn (class).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
 - Vật thể ở góc trên bên trái bị cắt một phần bởi mép trên của ảnh. Tôi chọn escalation lên Lab Coach để xác nhận cách gán nhãn Bounding Box cho vật thể bị cắt mép, tránh việc mỗi annotator xử lý khác nhau
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

    "instance_id": "kitchen-001",
    "class_name": "person",
    "score": 0.899318,
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ] ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 - Polygon cung cấp thông tin về đường viền và hình dáng thực tế của vật thể chính xác hơn Bounding Box
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
 - dùng để phân biệt các vật thể riêng biệt với nhau, chứ không dùng để xác định loại vật thể hay theo dõi vật thể qua nhiều khung hình.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
 - Người gán nhãn phải vẽ đường biên Mask bám sát đường viền thực tế của vật thể, bao phủ toàn bộ vùng pixel thuộc vật thể và không lấy lẫn nền xung quanh. Với vật thể bị che khuất hoặc cắt mép ảnh, chỉ vẽ phần thực sự nhìn thấy, không tự suy đoán hay vẽ thêm phần bị khuất. 
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
 - người (person) đứng trước khu vực bếp/lò nướng, cơ thể người che khuất một phần các vật thể phía sau.
    Đây là trường hợp che khuất thông thường (occlusion).
    Chỉ vẽ Mask theo phần cơ thể người thực sự nhìn thấy.
    Không vẽ thêm phần cơ thể bị vật khác che khuất.
    Không đưa vùng nền hoặc lò nướng phía sau vào Mask của người.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh (ví dụ: `dog`, `car`, `person`). | Ảnh chứa nhiều đối tượng hoặc đối tượng chính không rõ ràng. | Chọn đúng class theo taxonomy và guideline. | Kiểm tra class đã phù hợp với nội dung ảnh chưa. |
| Phát hiện vật thể | Mỗi đối tượng gồm `class + bbox` (tọa độ vùng bao quanh vật thể). | Vật thể bị che khuất, quá nhỏ hoặc các box chồng lấn nhau. | Vẽ bounding box đúng vị trí và gán đúng class. | Kiểm tra box có bao kín vật thể, đúng class và không bỏ sót đối tượng. |
| Instance segmentation | Mỗi đối tượng gồm `class + polygon/mask` mô tả chính xác đường biên. | Biên vật thể phức tạp, nhiều đối tượng dính nhau hoặc khó tách nền. | Vẽ polygon/mask bám sát đường viền vật thể. | Kiểm tra độ chính xác của đường biên và sự tách biệt giữa các instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ sử dụng dữ liệu được cung cấp trong phạm vi bài lab, không chia sẻ, sao chép hoặc đưa thông tin cá nhân hay dữ liệu nhạy cảm vào báo cáo và file đầu ra.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

- Tôi sẽ dừng xử lý và báo cho Lab Coach (giảng viên hướng dẫn) để được hướng dẫn tiếp, không tự ý xử lý hoặc gán nhãn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
