#!/usr/bin/env python3
# run_edgeface_usb_gray_lock_toggle_v2_onnx_opt_iou_clear_servo_arduino_style.py
# “Arduino 방식(데드존 + 오차 제곱 제어)”을 적용한 버전

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

##### ───── Servo 제어용 상수 (Arduino 방식) ──────────────────────────
try:
    import RPi.GPIO as GPIO
    PI_GPIO_OK = True
except (ImportError, RuntimeError):
    print("[WARN] RPi.GPIO 모듈이 없으므로 서보 제어 비활성화.")
    PI_GPIO_OK = False

# Raspberry Pi GPIO(PWM) 핀 설정
SERVO_PIN_X   = 2    # Pan(좌우) 서보: BCM 2
SERVO_PIN_Y   = 3    # Tilt(상하) 서보: BCM 3
SERVO_FREQ    = 50   # 50 Hz (서보 표준)

# 듀티사이클(%) 범위: 0° → 2.5%, 180° → 12.5%
SERVO_MIN_DCY = 2.5
SERVO_MAX_DCY = 12.5

# 시작 각도(중립) → 90°
START_ANGLE   = 90.0

# Arduino 코드와 동일한 오차 제곱 제어 상수
# (값이 너무 크면 과도하게 튈 수 있으므로, 필요 시 조정 권장)
K_pan  = 0.00002
K_tilt = 0.00002

# 데드존(threshold): 이내 오차면 서보를 움직이지 않음 (픽셀 단위)
DEAD_ZONE = 8

def angle_to_duty(angle: float) -> float:
    """0–180° → 듀티사이클(%) 변환"""
    return SERVO_MIN_DCY + (angle / 180.0) * (SERVO_MAX_DCY - SERVO_MIN_DCY)

# GPIO(PWM) 초기화
if PI_GPIO_OK:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN_X, GPIO.OUT)
    GPIO.setup(SERVO_PIN_Y, GPIO.OUT)

    pwm_x = GPIO.PWM(SERVO_PIN_X, SERVO_FREQ)
    pwm_y = GPIO.PWM(SERVO_PIN_Y, SERVO_FREQ)

    pwm_x.start(angle_to_duty(START_ANGLE))
    pwm_y.start(angle_to_duty(START_ANGLE))

    # 현재 서보 각도 상태를 저장
    cur_ang_x = float(START_ANGLE)
    cur_ang_y = float(START_ANGLE)
##### ───── Servo 제어 초기화 끝 ────────────────────────────────────

# ────────── 경로 & 모델 ──────────
ROOT      = pathlib.Path(__file__).resolve().parent
YOLO_W    = ROOT / "yolov11n-face-metadata.onnx"
EDGE_ONNX = ROOT / "edgeface_xxs.onnx"

# ── 1) YOLO 얼굴 탐지 초기화 (ONNX) ───────────────────────────────
yolo = YOLO(model=str(YOLO_W), task="detect")

# ── 2) EdgeFace ONNX 세션 초기화 ───────────────────────────────────
cuda_ok   = torch.cuda.is_available()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_ok else ["CPUExecutionProvider"]
sess      = ort.InferenceSession(str(EDGE_ONNX), providers=providers)

# ────────── 전처리 & 임베딩 함수 ──────────
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

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def create_tracker():
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    if hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("OpenCV MOSSE/KCF 트래커 미지원")

def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)

# ────────── 파라미터 ──────────
DETECT_EVERY = 20
THR          = 0.70
IOU_THR      = 0.60
SAVE_DIR     = ROOT / "collected_images"
SAVE_DIR.mkdir(exist_ok=True)

class FaceTrack:
    def __init__(self, tracker, vec_norm):
        self.tracker = tracker
        self.vec     = vec_norm
        self.bbox    = None  # (x, y, w, h)

def main():
    tracks, db, saved = [], {}, 0
    mode, fid = "TRAINING", 0
    prev_time, fps, ALPHA = time.time(), 0.0, 0.2

    # 카메라
    cap = (cv2.VideoCapture(1, cv2.CAP_DSHOW) if platform.system()=="Windows"
           else cv2.VideoCapture(0, cv2.CAP_V4L2))
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_cx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//2)
    frame_cy = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//2)

    global cur_ang_x, cur_ang_y

    while True:
        ok, frame = cap.read()
        if not ok: break
        fid += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        now = time.time()
        fps = (1-ALPHA)*fps + ALPHA/(now-prev_time)
        prev_time = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            mode = "TRAINING" if mode!="TRAINING" else "TRACKING"
            tracks.clear(); fid = 0
        elif key == 27:
            break

        # ───────── TRAINING ─────────
        if mode == "TRAINING":
            boxes = [tuple(map(int,b.xyxy[0])) for b in yolo.predict(frame,conf=0.5,verbose=False)[0].boxes]
            for (x1,y1,x2,y2) in boxes:
                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)
                if key == 32:  # Space
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        name=f"face_{saved}.jpg"; cv2.imwrite(str(SAVE_DIR/name),crop)
                        db[name] = emb_normalized(crop); saved += 1
        # ───────── TRACK / LOCK ─────
        else:
            if tracks:
                ft = tracks[0]
                ok_t, bb = ft.tracker.update(frame)
                if ok_t:
                    ft.bbox = tuple(map(int,bb))
                else:
                    tracks.clear()

            need_det = (fid % DETECT_EVERY == 0)
            if need_det:
                det = [tuple(map(int,b.xyxy[0])) for b in yolo.predict(frame,conf=0.5,verbose=False)[0].boxes]
                if tracks and det:
                    tx,ty,tw,th = tracks[0].bbox
                    tr_box = (tx,ty,tx+tw,ty+th)
                    if max(iou(tr_box,d) for d in det) < IOU_THR:
                        tracks.clear()
                        print("Low IoU: clear tracker")

            if not tracks and db:
                det = [tuple(map(int,b.xyxy[0])) for b in yolo.predict(frame,conf=0.5,verbose=False)[0].boxes]
                sims, det_info = [], []
                for (x1,y1,x2,y2) in det:
                    crop=frame[y1:y2, x1:x2]
                    if crop.size:
                        v=emb_normalized(crop); s=max(np.dot(v,e) for e in db.values())
                        sims.append(s); det_info.append((x1,y1,x2,y2,v))
                    else:
                        sims.append(-1); det_info.append((x1,y1,x2,y2,None))
                if sims:
                    idx=int(np.argmax(sims)); best_sim=sims[idx]
                    bx1,by1,bx2,by2,v = det_info[idx]
                    tr=create_tracker(); tr.init(frame,(bx1,by1,bx2-bx1,by2-by1))
                    ft=FaceTrack(tr,v); ft.bbox=(bx1,by1,bx2-bx1,by2-by1); tracks=[ft]
                    cv2.putText(disp,f"sim:{best_sim:.2f}",(bx1,by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

            # ───────── Servo 제어 (Arduino 방식) ─────────
            if tracks and PI_GPIO_OK:
                x,y,w,h = tracks[0].bbox
                cv2.rectangle(disp,(x,y),(x+w,y+h),(0,255,0),2)

                face_cx = x + w//2
                face_cy = y + h//2

                # 화면 중심과 얼굴 중심 간 오차(픽셀)
                err_x = face_cx - frame_cx
                err_y = face_cy - frame_cy

                # ─── 데드존 처리 ───
                if abs(err_x) > DEAD_ZONE:
                    # 오류 제곱 제어: error * |error|
                    delta_pan = K_pan * err_x * abs(err_x)
                    cur_ang_x -= delta_pan

                if abs(err_y) > DEAD_ZONE:
                    delta_tilt = K_tilt * err_y * abs(err_y)
                    cur_ang_y += delta_tilt

                # 각도 범위를 0~180°로 제한
                cur_ang_x = np.clip(cur_ang_x, 0.0, 180.0)
                cur_ang_y = np.clip(cur_ang_y, 0.0, 180.0)

                # PWM 듀티 사이클을 갱신하여 서보 이동
                pwm_x.ChangeDutyCycle(angle_to_duty(cur_ang_x))
                pwm_y.ChangeDutyCycle(angle_to_duty(cur_ang_y))
            # ───────── Servo 제어 끝 ─────────

        cv2.putText(disp,f"{mode} FPS:{fps:.1f}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.imshow("EdgeFace USB Webcam (Arduino-Style Servo)", disp)

    cap.release()
    cv2.destroyAllWindows()

    # GPIO(PWM) 정리
    if PI_GPIO_OK:
        pwm_x.stop()
        pwm_y.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
