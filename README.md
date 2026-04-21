import cv2
from picamera2 import Picamera2
import onnxruntime as ort
import numpy as np
from adafruit_servokit import ServoKit

# --- Camera Setup ---
cam_info = Picamera2.global_camera_info()
print(f"Available cameras: {cam_info}")
if len(cam_info) == 0:
    raise RuntimeError("No cameras detected.")

picam2 = Picamera2(cam_info[0]['Num'])
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

FRAME_W = 640
FRAME_H = 480
FRAME_CX = FRAME_W // 2  # 320
FRAME_CY = FRAME_H // 2  # 240

# --- Model Setup ---
session = ort.InferenceSession("/home/lick/version-RFB-320.onnx")

# --- Servo Setup ---
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(500, 2500)  # Tilt
kit.servo[1].set_pulse_width_range(500, 2500)  # Pan

TILT_MIN, TILT_MAX = 30, 150
PAN_MIN,  PAN_MAX  = 30, 150

tilt_angle = 90.0
pan_angle  = 90.0
kit.servo[0].angle = tilt_angle
kit.servo[1].angle = pan_angle

# --- Tuning ---
PAN_SPEED  = 0.0005  # increase if too slow, decrease if too twitchy
TILT_SPEED = 0.0005
DEADZONE   = 20    # pixels - don't move if error is smaller than this

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

# --- Servo Control ---
def update_servos(face_cx, face_cy):
    global tilt_angle, pan_angle

    # Error = how far face center is from frame center
    error_x = face_cx - FRAME_CX   # positive = face is right of center
    error_y = face_cy - FRAME_CY   # positive = face is below center

    # Pan: move toward face to shorten the line to center
    # If panning the wrong way, flip += to -=
    if abs(error_x) > DEADZONE:
        pan_angle += error_x * PAN_SPEED

    # Tilt: face below center -> tilt down (decrease angle)
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
        faces   = detect_faces(frame)

        if faces:
            # Pick highest confidence face
            best = max(faces, key=lambda f: f[6])
            x1, y1, x2, y2, cx, cy, conf = best

            update_servos(cx, cy)

            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw face center dot
            cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
            # Draw line from face center to frame center
            cv2.line(display, (cx, cy), (FRAME_CX, FRAME_CY), (0, 255, 255), 1)
            cv2.putText(display, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw frame center crosshair
        cv2.drawMarker(display, (FRAME_CX, FRAME_CY), (0, 0, 255),
                       cv2.MARKER_CROSS, 30, 2)

        # HUD
        cv2.putText(display, f"Tilt: {round(tilt_angle)}  Pan: {round(pan_angle)}",
                    (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Tracking", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    kit.servo[0].angle = 90
    kit.servo[1].angle = 90
    print("Stopped. Servos centered.")
    

