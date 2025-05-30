#!/usr/bin/env python3
# run_edgeface_usb_gray_lock_toggle_v2_onnx.py
# EdgeFace 부분을 ONNX Runtime으로 교체한 버전
#
# 흐름 ────────────────────────────────────────────────
# ① TRAINING  : 매 프레임 YOLO → 트래커·임베딩 생성 + Space로 DB 저장
# ② m 키      : TRACKING ↔ TRAINING 토글
# ③ TRACKING  : DETECT_EVERY 프레임마다 YOLO → 트래커·임베딩 생성
# ④ LOCK      : TRACKING 중 db 유사도 ≥ THR & IoU ≥ IOU_THR → LOCK
# ⑤ UNLOCK    : LOCK 중 IoU < IOU_THR → TRACKING 해제
# ----------------------------------------------------

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

# ────────── 경로 & 모델 ──────────
ROOT      = pathlib.Path(__file__).resolve().parent
YOLO_W    = ROOT / "yolov11n-face.onnx"
EDGE_ONNX = ROOT / "edgeface_xxs.onnx"

# YOLO 얼굴 탐지 초기화
yolo = YOLO(model=str(YOLO_W), task="detect")

# EdgeFace ONNX 세션 초기화
cuda_ok   = torch.cuda.is_available()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_ok else ["CPUExecutionProvider"]
sess      = ort.InferenceSession(str(EDGE_ONNX), providers=providers)

# ────────── 전처리 & 임베딩 함수 ──────────
xfm = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def emb(bgr: np.ndarray) -> np.ndarray:
    """BGR 이미지 → 512-dim EdgeFace 임베딩 (ONNX Runtime)"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = xfm(Image.fromarray(rgb)).unsqueeze(0).numpy()  # (1,3,112,112)
    out = sess.run(None, {"images": inp})[0]              # (1,512)
    return out[0]                                        # (512,)

def cos(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도 계산"""
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def create_tracker():
    if hasattr(cv2, "TrackerMOSSE_create"):
        print("MOSSE")
        return cv2.TrackerMOSSE_create()
    if hasattr(cv2.legacy, "TrackerMOSSE_create"):
        print("MOSSE")
        return cv2.legacy.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerKCF_create"):    
        print("KCF")          
        return cv2.TrackerKCF_create()
    if hasattr(cv2.legacy, "TrackerKCF_create"):       
        print("KCF")
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("OpenCV MOSSE/KCF 트래커 미지원")

#----두 박스 (x1,y1,x2,y2) 간 IoU 계산----
def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = areaA + areaB - interArea

    return interArea / (unionArea + 1e-8)

# ────────── 파라미터 ──────────
DETECT_EVERY = 20    # TRACKING 모드에서 YOLO 재검출 주기
THR          = 0.70  # 유사도 LOCK 진입 임계값
IOU_THR      = 0.30  # IoU LOCK 유지 임계값
SAVE_DIR     = ROOT / "collected_images"
SAVE_DIR.mkdir(exist_ok=True)

class FaceTrack:
    """트래커 인스턴스와 임베딩을 묶어 관리"""
    def __init__(self, tracker, vec):
        self.tracker = tracker
        self.vec     = vec

def main():
    # 초기 상태
    tracks = []    # FaceTrack 리스트
    db     = {}    # {filename: embedding}
    fid    = 0
    saved  = 0
    mode   = "TRAINING"  # TRAINING / TRACKING / LOCK
    prev, fps, ALPHA = time.time(), 0.0, 0.2

    # 카메라 초기화
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    # 프레임 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ 카메라 프레임을 읽을 수 없습니다.")
            break

        # 그레이스케일 변환 → BGR 복원
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fid += 1

        # FPS 계산 (지수이동평균)
        now = time.time()
        fps = (1 - ALPHA) * fps + ALPHA / (now - prev)
        prev = now

        # 키 입력 - 모드 변환(m) - 종료(esc)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            mode = "TRAINING" if mode != "TRAINING" else "TRACKING"
            tracks.clear()
            fid = 0 # frame 숫자 초기화
        elif key == 27:
            break

        # TRAINING 모드: 매 프레임 YOLO → 트래커 재생성 + Space로 DB 저장
        if mode == "TRAINING":
            res    = yolo.predict(frame, conf=0.5, verbose=False)
            boxes  = [tuple(map(int, b.xyxy[0])) for b in res[0].boxes]

            for (x1, y1, x2, y2) in boxes: # 2) 박스 그리기 및 Space로 샘플 저장
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Space 키로 저장
                if key == 32:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        name = f"face_{saved}.jpg"
                        cv2.imwrite(str(SAVE_DIR/name), crop)
                        db[name] = emb(crop)
                        saved += 1
                        print(f"Saved {name} and its embedding.")

        # TRACKING / LOCK 모드
        else:
            # 재검출 주기 확인
            need_det = (fid % DETECT_EVERY == 0)
            det_boxes = []

            if need_det:
                print("재탐지")
                res       = yolo.predict(frame, conf=0.5, verbose=False)
                det_boxes = [tuple(map(int, b.xyxy[0])) for b in res[0].boxes]
                tracks.clear()
                for (x1, y1, x2, y2) in det_boxes:
                    tr  = create_tracker()
                    tr.init(frame, (x1, y1, x2 - x1, y2 - y1))
                    vec = emb(frame[y1:y2, x1:x2])
                    tracks.append(FaceTrack(tr, vec))

            # 트래커 업데이트 및 faces 리스트 구성
            faces = []
            for ft in tracks[:]:
                ok, bb = ft.tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, bb)
                    faces.append((x, y, x + w, y + h, ft)) # 트래커 박스 얼굴로 옯김
                else:
                    tracks.remove(ft)
                    mode = "TRACKING"

            # DB가 있고 det_boxes가 있을 때 유사도/IoU 기반 LOCK/UNLOCK
            if db and det_boxes and faces:
                sims = [
                    (ft, max(cos(ft.vec, e) for e in db.values()), (x1, y1, x2, y2))
                    for (x1, y1, x2, y2, ft) in faces
                ]
                ft_best, sim_best, (bx1, by1, bx2, by2) = max(sims, key=lambda t: t[1])

                ok, bb = ft_best.tracker.update(frame)
                if ok:
                    tx, ty, tw, th = map(int, bb)
                    track_box      = (tx, ty, tx + tw, ty + th)

                    # IoU 계산
                    ious = [(det, iou(track_box, det)) for det in det_boxes]
                    _, best_iou = max(ious, key=lambda x: x[1], default=(None, 0.0))

                    # LOCK/UNLOCK 판단
                    if sim_best >= THR and best_iou >= IOU_THR:
                        mode = "LOCK"
                    else:
                        mode = "TRACKING"

                    # sim, IoU 레이블 표시
                    print("sim, iou 표시")
                    cv2.putText(frame, f"sim:{sim_best:.2f}", (bx1, by1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, f"IoU:{best_iou:.2f}", (bx1, by1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 모든 얼굴 박스 그리기
            for (x1, y1, x2, y2, _) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 모드 & FPS 표시
        cv2.putText(frame, f"{mode}  FPS:{fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("EdgeFace USB Webcam (Gray, toggle+lock)", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
