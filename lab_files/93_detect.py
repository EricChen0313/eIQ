# TFLite Interpreter
import tflite_runtime.interpreter as tflite

# Inference API
import ethosu.interpreter as ethosu

# Allows us to manipulate data
from PIL import Image
import numpy as np

# Timer
import time

# Set values based on what you want to do
USE_CPU_INFERENCE = True
USE_VELA_INFERENCE = False

# Location of required files
PICTURE_FILE = "/run/media/Transcend-sda1/lab_files/small_dog.jpg"
MODEL_FILE_CPU = "/run/media/Transcend-sda1/lab_files/detect.tflite"
MODEL_FILE_VELA = "/run/media/Transcend-sda1/lab_files/output/detect_vela.tflite"
LABEL_FILE = "/run/media/Transcend-sda1/lab_files/labels.txt"

# Loads the model picked above 
if USE_VELA_INFERENCE:
    MODEL = MODEL_FILE_VELA
else:
    MODEL = MODEL_FILE_CPU

# Loads the inference engine picked above
if USE_CPU_INFERENCE:
    interpreter = tflite.Interpreter(model_path=MODEL)
    interpreter.allocate_tensors()
else:
    #interpreter = tflite.Interpreter(model_path=MODEL)
    ext_delegate = [ tflite.load_delegate("/usr/lib/libethosu_delegate.so", {}) ]
    interpreter = tflite.Interpreter(model_path=MODEL, experimental_delegates=ext_delegate)
    interpreter.allocate_tensors()
# Get information from the model
input_info = interpreter.get_input_details() 
output_info = interpreter.get_output_details()
input_shape = input_info[0]['shape']
width = input_shape[1]
height = input_shape[2]

# Load the picture
img = Image.open(PICTURE_FILE).resize((width,height))
input_data = np.expand_dims(img, axis=0)

# Load the labels
label_names = []
with open(LABEL_FILE, 'r') as label_file:
    for line in label_file.readlines():
        label_names.append(line)

# Set the image as the input
if USE_CPU_INFERENCE:
    interpreter.set_tensor(input_info[0]['index'], input_data)
else:
    interpreter.set_tensor(input_info[0]['index'], input_data)
# Time and invoke the model
start_time = time.perf_counter()
interpreter.invoke()
total_time = time.perf_counter() - start_time
print("Inference took " + str(total_time*1000) + " ms")
# Read the data from the output
if USE_CPU_INFERENCE:
    boxes = interpreter.get_tensor(output_info[0]['index'])[0]
    labels = interpreter.get_tensor(output_info[1]['index'])[0]
    scores = interpreter.get_tensor(output_info[2]['index'])[0]
    number = interpreter.get_tensor(output_info[3]['index'])[0]
else:
    boxes = interpreter.get_tensor(output_info[0]['index'])[0]
    labels = interpreter.get_tensor(output_info[1]['index'])[0]
    scores = interpreter.get_tensor(output_info[2]['index'])[0]
    number = interpreter.get_tensor(output_info[3]['index'])[0]
# Print that data
for i in range(int(number)):
    if (labels[i]%1.0 != 0.0):
        continue
    if scores[i] > 1.0:
        continue
    for j in range(4):
        if boxes[i][j] > 1.0:
            continue
    print(label_names[int(labels[i])][:-1] + " (" + str(scores[i]*100) + "%)")
