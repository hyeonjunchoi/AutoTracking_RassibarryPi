#!/usr/bin/env python3
import time
import pathlib
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO
import platform

try:
    import pigpio
    PI = pigpio.pi()
    if not PI.connected:
        raise RuntimeError("pigpio 데몬에 연결할 수 없습니다.")
    PI_GPIO_OK = True
except Exception:
    PI_GPIO_OK = False

SERVO_PIN_X = 5
SERVO_PIN_Y = 6
PWM_MIN_PW = 500
PWM_MAX_PW = 2500
START_ANGLE_X = 90.0
START_ANGLE_Y = 5.0
K_PAN = 0.00002
K_TILT = 0.00002
DEAD_ZONE = 8

def angle_to_pulsewidth(angle: float) -> int:
    a = max(0.0, min(180.0, angle))
    pw = PWM_MIN_PW + (a / 180.0) * (PWM_MAX_PW - PWM_MIN_PW)
    return int(pw)

def set_servo_angle(pin: int, angle: float):
    if not PI_GPIO_OK:
        return
    pw = angle_to_pulsewidth(angle)
    PI.set_servo_pulsewidth(pin, pw)

def servo_off(pin: int):
    if not PI_GPIO_OK:
        return
    PI.set_servo_pulsewidth(pin, 0)

ROOT = pathlib.Path(__file__).resolve().parent
YOLO_W = ROOT / "yolov11n-face-metadata.onnx"
EDGE_ONNX = ROOT / "edgeface_xxs.onnx"

yolo = YOLO(model=str(YOLO_W), task="detect")

cuda_ok = torch.cuda.is_available()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_ok else ["CPUExecutionProvider"]
sess = ort.InferenceSession(str(EDGE_ONNX), providers=providers)

xfm = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def emb_normalized(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = xfm(Image.fromarray(rgb)).unsqueeze(0).numpy().astype(np.float32)
    out = sess.run(None, {"images": inp})[0][0]
    return out / (np.linalg.norm(out) + 1e-8)

def create_tracker():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("OpenCV KCF 트래커 미지원")

def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)

DETECT_EVERY = 20
THR = 0.70
IOU_THR = 0.60
SAVE_DIR = ROOT / "collected_images"
SAVE_DIR.mkdir(exist_ok=True)

class FaceTrack:
    def __init__(self, tracker, vec_norm):
        self.tracker = tracker
        self.vec = vec_norm
        self.bbox = None

def main():
    tracks = []
    db = {}
    saved = 0
    mode = "TRAINING"
    fid = 0
    prev_time = time.time()
    fps = 0.0
    ALPHA = 0.2

    cap = (cv2.VideoCapture(1, cv2.CAP_DSHOW) if platform.system() == "Windows"
           else cv2.VideoCapture(0, cv2.CAP_V4L2))
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_cx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    frame_cy = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

    global cur_ang_x, cur_ang_y
    cur_ang_x = START_ANGLE_X
    cur_ang_y = START_ANGLE_Y
    set_servo_angle(SERVO_PIN_X, cur_ang_x)
    set_servo_angle(SERVO_PIN_Y, cur_ang_y)
    time.sleep(0.05)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        now = time.time()
        fps = (1 - ALPHA) * fps + ALPHA / (now - prev_time)
        prev_time = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            mode = "TRAINING" if mode != "TRAINING" else "TRACKING"
            tracks.clear()
            fid = 0
            if mode == "TRAINING" and PI_GPIO_OK:
                cur_ang_x = START_ANGLE_X
                cur_ang_y = START_ANGLE_Y
                set_servo_angle(SERVO_PIN_X, cur_ang_x)
                set_servo_angle(SERVO_PIN_Y, cur_ang_y)
                time.sleep(0.05)
            elif mode == "TRACKING" and PI_GPIO_OK:
                set_servo_angle(SERVO_PIN_X, cur_ang_x)
                set_servo_angle(SERVO_PIN_Y, cur_ang_y)
                time.sleep(0.05)

        elif key == 27:
            break

        if mode == "TRAINING":
            pred = yolo.predict(frame, conf=0.5, verbose=False)[0]
            boxes = [tuple(map(int, b.xyxy[0])) for b in pred.boxes]
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if key == 32:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        name = f"face_{saved}.jpg"
                        cv2.imwrite(str(SAVE_DIR / name), crop)
                        vec_norm = emb_normalized(crop)
                        db[name] = vec_norm
                        saved += 1
                        print(f"[TRAINING] Saved {name}.")
        else:
            need_det = (fid % DETECT_EVERY == 0)
            det = []
            if tracks:
                ok_t, bb = tracks[0].tracker.update(frame)
                if ok_t:
                    tracks[0].bbox = tuple(map(int, bb))
                else:
                    tracks.clear()
            if need_det:
                pred = yolo.predict(frame, conf=0.5, verbose=False)[0]
                det = [tuple(map(int, b.xyxy[0])) for b in pred.boxes]
                if tracks and det:
                    tx, ty, tw, th = tracks[0].bbox
                    tr_box = (tx, ty, tx + tw, ty + th)
                    if max(iou(tr_box, d) for d in det) < IOU_THR:
                        tracks.clear()
                        print("Low IoU: clear tracker")

            if not tracks and db and need_det and det:
                sims = []
                det_info = []
                for (x1, y1, x2, y2) in det:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        v = emb_normalized(crop)
                        s = max(np.dot(v, e) for e in db.values())
                        sims.append(s)
                        det_info.append((x1, y1, x2, y2, v))
                    else:
                        sims.append(-1)
                        det_info.append((x1, y1, x2, y2, None))
                idx = int(np.argmax(sims))
                best_sim = sims[idx]
                bx1, by1, bx2, by2, v = det_info[idx]
                tr = create_tracker()
                tr.init(frame, (bx1, by1, bx2 - bx1, by2 - by1))
                ft = FaceTrack(tr, v)
                ft.bbox = (bx1, by1, bx2 - bx1, by2 - by1)
                tracks = [ft]
                cv2.putText(disp, f"sim:{best_sim:.2f}", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if PI_GPIO_OK and tracks:
                x, y, w, h = tracks[0].bbox
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_cx = x + w // 2
                face_cy = y + h // 2
                err_x = face_cx - frame_cx
                err_y = face_cy - frame_cy
                if abs(err_x) > DEAD_ZONE:
                    delta_pan = K_PAN * err_x * abs(err_x)
                    cur_ang_x -= delta_pan
                if abs(err_y) > DEAD_ZONE:
                    delta_tilt = K_TILT * err_y * abs(err_y)
                    cur_ang_y += delta_tilt
                cur_ang_x = float(np.clip(cur_ang_x, 0.0, 180.0))
                cur_ang_y = float(np.clip(cur_ang_y, 0.0, 180.0))
                set_servo_angle(SERVO_PIN_X, cur_ang_x)
                set_servo_angle(SERVO_PIN_Y, cur_ang_y)

        cv2.putText(disp, f"{mode} FPS:{fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("EdgeFace USB Webcam (pigpio Servo)", disp)

    cap.release()
    cv2.destroyAllWindows()

    if PI_GPIO_OK:
        servo_off(SERVO_PIN_X)
        servo_off(SERVO_PIN_Y)
        PI.stop()

if __name__ == "__main__":
    main()
