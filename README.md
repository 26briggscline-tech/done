import sys
sys.stdout.reconfigure(errors='replace')

import cv2
import os
import time
from picamera2 import Picamera2
import onnxruntime as ort
import numpy as np
from adafruit_servokit import ServoKit

# --- Camera Setup ---
cam_info = Picamera2.global_camera_info()
print(f"Available cameras: {str(cam_info).encode('utf-8', errors='replace').decode('utf-8')}")
if len(cam_info) == 0:
    raise RuntimeError("No cameras detected.")

picam2 = Picamera2(cam_info[0]['Num'])
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

FRAME_W = 640
FRAME_H = 480
FRAME_CX = FRAME_W // 2
FRAME_CY = FRAME_H // 2

# --- Rule of Thirds Target (top-left intersection) ---
TARGET_X = FRAME_W // 3   # 213
TARGET_Y = FRAME_H // 3   # 160

# --- Model Setup ---
session = ort.InferenceSession("/home/lick/version-RFB-320.onnx")

# --- Servo Setup ---
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(500, 2500)  # Tilt
kit.servo[1].set_pulse_width_range(500, 2500)  # Pan

TILT_MIN, TILT_MAX = 30, 150
PAN_MIN,  PAN_MAX  = 30, 150

tilt_angle = 75.0
pan_angle  = 45.0
kit.servo[0].angle = tilt_angle
kit.servo[1].angle = pan_angle

# --- Tuning ---
PAN_SPEED  = 0.005
TILT_SPEED = 0.005
DEADZONE   = 20

# --- Video Writer Setup ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
desktop_path = os.path.expanduser("~/Desktop/")
CHUNK_FRAMES = 900  # 30 seconds Ã 30 fps

def new_writer():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(desktop_path, f'face_tracking_{timestamp}.mp4')
    print(f"Recording to: {str(filename).encode('utf-8', errors='replace').decode('utf-8')}")
    return cv2.VideoWriter(filename, fourcc, 30.0, (FRAME_W, FRAME_H))

out = new_writer()
frame_count = 0

# --- Face Detection ---
def detect_faces(frame, threshold=0.7, nms_threshold=0.4):
    h, w = frame.shape[:2]

    img = cv2.resize(frame, (320, 240))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img - 127.0) / 128.0
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0).astype(np.float32)

    confidences, boxes = session.run(None, {session.get_inputs()[0].name: img})

    detection_boxes  = []
    detection_scores = []

    for i in range(confidences.shape[1]):
        confidence = confidences[0, i, 1]
        if confidence > threshold:
            box = boxes[0, i]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                detection_boxes.append([x1, y1, x2, y2])
                detection_scores.append(float(confidence))

    if detection_boxes:
        indices = cv2.dnn.NMSBoxes(detection_boxes, detection_scores, threshold, nms_threshold)
        faces = []
        for i in indices:
            x1, y1, x2, y2 = detection_boxes[i]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            faces.append((x1, y1, x2, y2, cx, cy, detection_scores[i]))
        return faces

    return []

# --- Servo Control (target = top-left rule-of-thirds intersection) ---
def update_servos(face_cx, face_cy):
    global tilt_angle, pan_angle

    error_x = face_cx - TARGET_X
    error_y = face_cy - TARGET_Y

    if abs(error_x) > DEADZONE:
        pan_angle += error_x * PAN_SPEED

    if abs(error_y) > DEADZONE:
        tilt_angle -= error_y * TILT_SPEED

    pan_angle  = max(PAN_MIN,  min(PAN_MAX,  pan_angle))
    tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle))

    kit.servo[0].angle = round(tilt_angle)
    kit.servo[1].angle = round(pan_angle)

# --- Main Loop ---
print("Face tracking started. Press 'q' to quit.")
try:
    while True:
        frame = picam2.capture_array()

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        faces = detect_faces(frame)

        if faces:
            best = max(faces, key=lambda f: f[6])
            x1, y1, x2, y2, cx, cy, conf = best
            update_servos(cx, cy)

        out.write(display)
        frame_count += 1

        # Rotate video file every 30 seconds
        if frame_count >= CHUNK_FRAMES:
            out.release()
            out = new_writer()
            frame_count = 0

        cv2.imshow("Face Tracking", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    out.release()
    cv2.destroyAllWindows()
    kit.servo[0].angle = 90
    kit.servo[1].angle = 90
    print("Stopped. Servos centered.")
    




