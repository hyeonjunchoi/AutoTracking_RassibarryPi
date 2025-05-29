#!/usr/bin/env python3
# run_edgeface_usb_gray_lock_toggle_v2.py
# “Detection 프레임에서만 EdgeFace 임베딩” 버전
#
# 흐름 ────────────────────────────────────────────────
# ① TRAINING  : 매 프레임 YOLO → 트래커·임베딩 생성
# ② m 키      : TRACKING ↔ TRAINING 토글
# ③ TRACKING  : DETECT_EVERY 프레임마다 YOLO + 새 임베딩
# ④ LOCK      : TRACKING 중 db 유사도 ≥ THR → 트래커만
# ----------------------------------------------------

import time, pathlib, cv2, numpy as np, torch, torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
import platform

# ────────── 경로 & 모델 ──────────
ROOT       = pathlib.Path(__file__).resolve().parent
YOLO_W     = ROOT / "yolov11n-face.onnx"
EDGE_MODEL = "edgeface_xxs"
EDGE_CKPT  = ROOT / "edgeface_xxs.pt"          # 사전 다운로드 필요

if not EDGE_CKPT.exists():
    raise FileNotFoundError(
        "edgeface_xxs.pt 가 없습니다.\n"
        "https://huggingface.co/Idiap/EdgeFace-XXS/resolve/main/edgeface_xxs.pt"
        " 에서 다운로드해 이 스크립트와 같은 폴더에 넣어 주세요."
    )

yolo = YOLO(model=str(YOLO_W), task="detect")

# EdgeFace 구조만 로드 → 로컬 가중치 주입
edgeface = torch.hub.load(
    "otroshi/edgeface", EDGE_MODEL,
    pretrained=False, trust_repo=True, source="github"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
edgeface = edgeface.eval().to(DEVICE)
edgeface.load_state_dict(torch.load(EDGE_CKPT, map_location=DEVICE), strict=True)

# ────────── 전처리 & 임베딩 함수 ──────────
xfm = T.Compose([
    T.Resize((112, 112)), T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3)
])

@torch.no_grad()
def emb(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return edgeface(xfm(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                    ).squeeze(0).cpu().numpy()

def cos(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def create_tracker():
    if hasattr(cv2, "TrackerMOSSE_create"):            return cv2.TrackerMOSSE_create()
    if hasattr(cv2.legacy, "TrackerMOSSE_create"):     return cv2.legacy.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerKCF_create"):              return cv2.TrackerKCF_create()
    if hasattr(cv2.legacy, "TrackerKCF_create"):       return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("OpenCV MOSSE/KCF 트래커 미지원")

# ────────── 파라미터 ──────────
DETECT_EVERY = 20
THR          = 0.70
SAVE_DIR     = ROOT / "collected_images"
SAVE_DIR.mkdir(exist_ok=True)

# 트래커 + 캐시 임베딩 보관용
class FaceTrack:
    def __init__(self, tracker, vec):
        self.tracker = tracker
        self.vec     = vec

# ────────── 메인 루프 ──────────
def main():
    tracks: list[FaceTrack] = []
    db: dict[str, np.ndarray] = {}
    fid = saved = 0
    mode = "TRAINING"                 # TRAINING / TRACKING / LOCK
    prev, fps, ALPHA = time.time(), 0.0, 0.2

    # ── 카메라 초기화 (OS별 백엔드) ──
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 인덱스·권한·드라이버를 확인하세요.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  카메라 프레임을 읽을 수 없습니다."); break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fid  += 1

        now  = time.time()
        fps  = (1 - ALPHA) * fps + ALPHA / (now - prev)
        prev = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            mode = "TRAINING" if mode != "TRAINING" else "TRACKING"
            tracks.clear(); fid = 0
        elif key == 27:
            break

        faces = []
        need_det = (mode == "TRAINING") or (mode == "TRACKING" and fid % DETECT_EVERY == 0)

        # ── 트래커 업데이트 ──
        if mode != "TRAINING":
            for ft in tracks[:]:
                ok, bb = ft.tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, bb)
                    faces.append((x, y, x + w, y + h, ft))  # ft 포함
                else:
                    tracks.remove(ft)
                    mode = "TRACKING"

        # ── YOLO 탐지 & (한 번만) 임베딩 ──
        if need_det:
            res = yolo.predict(frame, conf=0.5, verbose=False)
            boxes = [tuple(map(int, b.xyxy[0])) for b in res[0].boxes]
            tracks.clear(); faces.clear()
            for (x1, y1, x2, y2) in boxes:
                tr = create_tracker()
                tr.init(frame, (x1, y1, x2 - x1, y2 - y1))
                vec = emb(frame[y1:y2, x1:x2])       # ← 이 시점 ‘한 번’ 계산
                ft  = FaceTrack(tr, vec)
                tracks.append(ft)
                faces.append((x1, y1, x2, y2, ft))

        # ── 얼굴별 처리 ──
        for (x1, y1, x2, y2, ft) in faces:
            crop = frame[y1:y2, x1:x2]
            if 0 in crop.shape:
                continue

            # SPACE 키로 샘플 저장
            if key == 32 and need_det:
                name = f"face_{saved}.jpg"
                cv2.imwrite(str(SAVE_DIR / name), crop)
                db[name] = emb(crop)
                saved += 1

            if db:
                best_name, sim = max(((n, cos(ft.vec, e)) for n, e in db.items()),
                                     key=lambda t: t[1])
                if mode != "TRAINING":
                    mode = "LOCK" if sim >= THR else "TRACKING"
                label, col = (f"TRACK {best_name} {sim:.2f}", (255, 0, 0)) if sim >= THR \
                             else (f"UNKNOWN {sim:.2f}", (0, 0, 255))
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"{mode}  FPS:{fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("EdgeFace USB Webcam (Gray, toggle+lock)", frame)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
