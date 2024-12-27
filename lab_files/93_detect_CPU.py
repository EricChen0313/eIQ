from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# Settings
PICTURE_FILE = "/run/media/MICRO_YHC-sda1/lab_files/small_dog.jpg"
MODEL_FILE_CPU = "/run/media/MICRO_YHC-sda1/lab_files/detect.tflite"
LABEL_FILE = "/run/media/MICRO_YHC-sda1/lab_files/labels.txt"
RESULT_FILE = "/run/media/MICRO_YHC-sda1/lab_files/detection_result.jpg"

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_FILE_CPU)
interpreter.allocate_tensors()

# Get model input/output details
input_info = interpreter.get_input_details()
output_info = interpreter.get_output_details()
input_shape = input_info[0]['shape']
width = input_shape[1]
height = input_shape[2]

# Load and preprocess the image
img = Image.open(PICTURE_FILE).convert("RGB").resize((width, height))
input_data = np.expand_dims(img, axis=0)

# Load the labels
label_names = []
with open(LABEL_FILE, 'r') as label_file:
    for line in label_file.readlines():
        label_names.append(line.strip())

# Set input tensor and run inference
interpreter.set_tensor(input_info[0]['index'], input_data)
start_time = time.perf_counter()
interpreter.invoke()
total_time = time.perf_counter() - start_time
print(f"Inference took {total_time * 1000:.2f} ms")

# Retrieve output data
boxes = interpreter.get_tensor(output_info[0]['index'])[0]
labels = interpreter.get_tensor(output_info[1]['index'])[0]
scores = interpreter.get_tensor(output_info[2]['index'])[0]
num_detections = int(interpreter.get_tensor(output_info[3]['index'])[0])

# Find the detection with the highest confidence
max_score_index = np.argmax(scores[:num_detections])
if scores[max_score_index] < 0.3:  # Ignore if highest score is below 30%
    print("No detection with high confidence.")
else:
    # Get details of the highest confidence detection
    ymin, xmin, ymax, xmax = boxes[max_score_index]
    label = label_names[int(labels[max_score_index])]
    score = scores[max_score_index] * 100

    # Scale bounding box to original image size
    original_img = Image.open(PICTURE_FILE).convert("RGB")
    original_width, original_height = original_img.size
    left = int(xmin * original_width)
    top = int(ymin * original_height)
    right = int(xmax * original_width)
    bottom = int(ymax * original_height)

    # Draw bounding box and label
    draw = ImageDraw.Draw(original_img)
    draw.rectangle([left, top, right, bottom], outline="red", width=5)

    # Set font size for label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    draw.text((left, top - 30), f"{label} ({score:.2f}%)", fill="red", font=font)

    # Save the result image
    original_img.save(RESULT_FILE)
    print(f"Detection result saved to {RESULT_FILE}")