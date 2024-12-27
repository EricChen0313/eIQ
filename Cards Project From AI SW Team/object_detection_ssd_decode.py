"""Command"""
#python3 object_detection_ssd_decode.py ssd_mobilenet_v3-2024-02-22T09-33-00.234Z_in-uint8_out-float32_channel_ptq_vela.tflite labels.txt ssd_anchors.npy xxx.jpg image

# Import required libraries
import tflite_runtime.interpreter as tflite
import cv2
import re
import numpy as np
import sys
import time
import os

def decode_ssd(anchor_box, ssd_box):
    """
    解碼 SSD 模型輸出的檢測框 (bounding boxes)，
    將偏移量轉換為實際的座標。
    """
    ay = anchor_box[:, 0]
    ax = anchor_box[:, 1]
    ah = anchor_box[:, 2]
    aw = anchor_box[:, 3]

    sy = ssd_box[:, 0]
    sx = ssd_box[:, 1]
    sh = ssd_box[:, 2]
    sw = ssd_box[:, 3]

    px = (sx / 10) * aw + ax
    py = (sy / 10) * ah + ay
    pw = (np.exp(sw / 5)) * aw
    ph = (np.exp(sh / 5)) * ah

    return np.stack(
        [(py - ph / 2) * video_height, (px - pw / 2) * video_width,
         (py + ph / 2) * video_height, (px + pw / 2) * video_width], axis=1
    )

def sigmoid(x):
    """
    計算 Sigmoid 值，用於將模型輸出的分數轉換為 [0, 1] 範圍的概率。
    """
    return 1 / (1 + np.exp(-x))

def nms(rect_box, num_anchor_box, nms_threshold, scores, classes):
    """
    Non-Maximum Suppression (NMS) 過濾重疊的框選，保留最高置信度的框。
    """
    x1 = rect_box[:, 0]
    y1 = rect_box[:, 1]
    x2 = rect_box[:, 2]
    y2 = rect_box[:, 3]

    # 計算框的面積
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 按分數降序排序
    temp_iou = []

    while order.size > 0:
        i = order[0]  # 當前最高分數的索引
        temp_iou.append(i)

        # 計算交集區域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 計算 IoU (交併比)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小於閾值的框
        inds = np.where(ovr <= nms_threshold)[0]
        order = order[inds + 1]

    return temp_iou

# 讀取標籤檔案
label_path = sys.argv[2]
p = re.compile(r'\s*(\d+)(.+)')
with open(label_path, 'r') as f:
    labels = {int(num): text.strip() for num, text in (p.match(line).groups() for line in f.readlines())}

# 載入 SSD Anchors
ssd_anchors = np.load(sys.argv[3])

# 加載 TFLite 模型
model_path = sys.argv[1]
ext_delegate = [tflite.load_delegate("/usr/lib/libethosu_delegate.so", {})]  # 載入 NPU 的 delegate
interpreter = tflite.Interpreter(model_path, experimental_delegates=ext_delegate)
interpreter.allocate_tensors()

# 獲取模型的輸入和輸出資訊
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height, input_width = input_details[0]['shape'][1:3]
floating_model = input_details[0]['dtype'] == np.float32  # 判斷模型是否為浮點型輸入

# 讀取輸入影像
image_path = sys.argv[4]
frame = cv2.imread(image_path)  # 使用 OpenCV 載入影像
video_height, video_width = frame.shape[:2]

# 調整影像大小並轉換為 RGB 格式
resized_frame_RGB = cv2.resize(frame, (input_width, input_height))
resized_frame_RGB = cv2.cvtColor(resized_frame_RGB, cv2.COLOR_BGR2RGB)

# 如果模型是浮點型，將影像標準化
if floating_model:
    resized_frame_RGB = np.float32(resized_frame_RGB) / 128 - 1

# 推論
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(resized_frame_RGB, axis=0))
start_time = time.time()
interpreter.invoke()  # 執行推論
inference_time = time.time() - start_time
print(f"Inference took {inference_time * 1000:.2f} ms")

# 解析模型輸出
output_data = interpreter.get_tensor(output_details[0]['index'])
conf, loc = np.split(np.squeeze(output_data), [len(labels) + 1], axis=1)
conf = sigmoid(conf)  # 應用 Sigmoid 將分數轉為概率
conf = np.split(conf, [1], axis=1)[1]  # 移除背景類別

# 過濾置信度低於閾值的檢測結果
index = np.where(conf > 0.5)
num_anchor_box = len(index[0])

if num_anchor_box > 0:
    # 解碼檢測框
    anchor_box = ssd_anchors[index[0], :]
    rect_box = decode_ssd(anchor_box, loc[index[0], :])
    scores = conf[index[0], index[1]]
    classes = index[1]

    # 如果檢測框多於一個，執行 NMS 過濾
    if num_anchor_box > 1:
        results = nms(rect_box, num_anchor_box, 0.6, scores, classes)
        rect_box = rect_box[results]
        scores = scores[results]
        classes = classes[results]

    # 在影像上繪製檢測結果
    for i in range(len(classes)):
        ymin, xmin, ymax, xmax = rect_box[i].astype(int)
        label = f"{labels[classes[i]]} ({scores[i] * 100:.2f}%)"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 儲存結果圖片
    input_filename = os.path.basename(image_path)
    input_dir = os.path.dirname(image_path)
    result_path = os.path.join(input_dir, f"detection_result_{input_filename}")
    cv2.imwrite(result_path, frame)
    print(f"Detection result saved to {result_path}")
else:
    print("No detections above threshold.")