"""Command"""
#python3 object_detection_ssd_decode.py detection-balanced-npu-2021-12-28T06-14-12.529Z_in-uint8_out-float32_channel_toco_ptq.tflite VOC2008_labels.txt ssd_anchors.npy bike.jpg image

# Import required libraries
import tflite_runtime.interpreter as tflite 
import cv2 
import re  
import numpy as np  
import sys 
import time  

def decode_ssd(anchor_box, ssd_box):
    """
    解碼 SSD 模型輸出的檢測框 (bounding boxes)，
    將偏移量轉換為實際的座標。
    
    Args:
        anchor_box: Predefined anchor box dimensions.
        ssd_box: Model-predicted offsets for each anchor box.
    Returns:
        Actual bounding box coordinates for the detected objects.
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

    return np.squeeze(np.dstack(
        [
            (py - ph / 2) * video_height,  # Top
            (px - pw / 2) * video_width,   # Left
            (py + ph / 2) * video_height,  # Bottom
            (px + pw / 2) * video_width    # Right
        ]
    ))

def nms(rect_box, num_anchor_box, score_threshold, scores, classes):
    """
    Non-Maximum Suppression (NMS) 過濾重疊的框選，保留最高置信度的框。
    
    Args:
        rect_box: Bounding box coordinates.
        num_anchor_box: Total number of detected anchor boxes.
        score_threshold: Minimum confidence score to consider.
        scores: Confidence scores for each box.
        classes: Class indices for each box.
    Returns:
        Indices of the remaining boxes after applying NMS.
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


def sigmoid(x):
    """
    計算 Sigmoid 值，用於將模型輸出的分數轉換為 [0, 1] 範圍的概率。
    """
    return 1 / (1 + np.exp(-x))

print('tflite version : %s' % tflite.__version__)
print('opencv version : %s' % cv2.__version__)

# 設置閾值
nms_threshold = 0.6  
score_threshold = 0.5  

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
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
floating_model = input_details[0]['dtype'] == np.float32  # 判斷模型是否為浮點型輸入
output_details = interpreter.get_output_details()

src_fmt = sys.argv[5]
use_image = False
if src_fmt == 'image':
    use_image = True
if use_image != True:
    if src_fmt == 'camera':
        video_num = int(sys.argv[4])
        video_width = 640
        video_height = 480
        cap = cv2.VideoCapture(f'v4l2src device=/dev/video{video_num} ! video/x-raw,width={video_width},height={video_height} ! videoconvert ! appsink sync=false')
    elif src_fmt == 'video':
        video_file = sys.argv[4]
        cap = cv2.VideoCapture(f'filesrc location={video_file} ! decodebin ! videoconvert ! appsink sync=false')
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        print("src_fmt should be camera or image")
        sys.exit()

while True:
    if use_image:
        frame = cv2.imread(sys.argv[4])
        img_size = frame.shape
        video_width = img_size[1]
        video_height = img_size[0]
    else:
        ret, frame = cap.read()
        if ret != True:  
            break

    resized_frame_BGR = cv2.resize(frame, dsize=(input_width, input_height))
    resized_frame_RGB = cv2.cvtColor(resized_frame_BGR, cv2.COLOR_BGR2RGB)

    if floating_model:
        resized_frame_RGB = np.float32(resized_frame_RGB) / 128 - 1

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(resized_frame_RGB, axis=0))
    time2 = time.time()
    interpreter.invoke()
    time3 = time.time()

    print("Inference time is :", (time3 - time2) * 1000, "ms")

    # 處理模型輸出
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if 'quantization' in output_details[0]['quantization']:
        scale, zero_point = output_details[0]['quantization']
        if scale == 0:
            output_data = output_data - zero_point
        else:
            output_data = scale * (output_data - zero_point)

    conf, loc = np.split(np.squeeze(output_data), [len(labels) + 1], axis=1)
    conf = sigmoid(conf)
    conf = np.split(conf, [1], axis=1)[1]  # 移除背景類別

	# 過濾置信度低於閾值的檢測結果
    index = np.where(conf > score_threshold)
    num_anchor_box = len(index[0])

    if num_anchor_box > 0:
        loc = loc[index[0], :]
        anchor_box = ssd_anchors[index[0], :]
        rect_box = decode_ssd(anchor_box, loc)
        scores = conf[index[0], index[1]]
        classes = index[1]

    	# 如果檢測框多於一個，執行 NMS 過濾
        if num_anchor_box > 1:
            results = nms(rect_box, num_anchor_box, nms_threshold, scores, classes)
            rect_box = rect_box[results]
            scores = scores[results]
            classes = classes[results]

    	# 在影像上繪製檢測結果
        for i in range(len(classes)):
            top = int(rect_box[i][0])
            left = int(rect_box[i][1])
            bottom = int(rect_box[i][2])
            right = int(rect_box[i][3])
            score = scores[i]
            class_id = classes[i]
            print(labels[class_id], "(", round(scores[i] * 100, 2), "%)")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
            cv2.putText(frame, f'{labels[class_id]}: {int(score * 100)}%', (left + 4, top + 4
