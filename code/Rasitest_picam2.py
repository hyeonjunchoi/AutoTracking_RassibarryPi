import sys
import pathlib
import os
import time

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
from picamera2 import Picamera2

# ── 경로/모델 로드 ──────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).resolve().parent
MNET_DIR = ROOT / 'MobileFaceNet-master'
sys.path.append(str(MNET_DIR))

from mobilefacenet import MobileFaceNet

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# MobileFaceNet 초기화
net = MobileFaceNet().to(DEVICE)
net.load_state_dict(
    torch.load(ROOT/'mobilefacenet.pt', map_location=DEVICE),
    strict=False
)
net.eval()

# YOLO 초기화 (파일명 확인)
yolo = YOLO(str(ROOT/'yolov11n-face.pt'))

# 임베딩 전처리 (MobileFaceNet expects RGB input)
transform = T.Compose([
    T.Resize((112,112)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

@torch.no_grad()
def emb_from_rgb(rgb):
    # rgb is H×W×3 uint8 in [0,255]
    x = transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    return net(x).squeeze(0).cpu().numpy()

def cos(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

def create_tracker():
    for ns, fn in [
        ("legacy","TrackerMOSSE_create"), (None,"TrackerMOSSE_create"),
        ("legacy","TrackerKCF_create"),   (None,"TrackerKCF_create"),
        ("legacy","TrackerCSRT_create"),  (None,"TrackerCSRT_create"),
        ("legacy","TrackerMIL_create"),   (None,"TrackerMIL_create")
    ]:
        mod = getattr(cv2, ns) if ns else cv2
        if hasattr(mod, fn):
            return getattr(mod, fn)()
    raise RuntimeError("지원되는 트래커가 없습니다.")

# ── 파라미터 ──────────────────────────────────────────────────────
detect_every, thr = 20, 0.8
save_dir = ROOT/'collected_images'
save_dir.mkdir(exist_ok=True)

def main():
    trackers, embeds = [], {}
    frame_id, count = 0, 0
    train = True
    prev_t, fps, ALPHA = time.time(), 0.0, 0.2

    # Picamera2 설정: 640×480, 3채널 RGB888
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # 워밍업

    window_name = "Picamera2 RGB / MobileFaceNet"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            rgb_frame = picam2.capture_array()  # (480,640,3) RGB
            frame_id += 1

            # FPS 계산
            now = time.time()
            fps = (1-ALPHA)*fps + ALPHA/(now-prev_t)
            prev_t = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                train = not train
                frame_id = 0
            if key == 27:  # ESC
                break

            # detection vs tracking 분기
            need_det = train or (frame_id % detect_every == 0)
            faces = []

            if not need_det:
                for tr in trackers:
                    ok, bb = tr.update(rgb_frame_bgr)
                    if ok:
                        x,y,w,h = map(int, bb)
                        faces.append((x, y, x+w, y+h))
                    else:
                        need_det = True
                        break

            if need_det:
                # YOLO expects BGR input
                rgb_frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                results = yolo.predict(rgb_frame_bgr, conf=0.5, verbose=False)[0]
                faces = [tuple(map(int, b.xyxy[0])) for b in results.boxes]

                # 트래커 재설정
                trackers.clear()
                for (x1,y1,x2,y2) in faces:
                    tr = create_tracker()
                    tr.init(rgb_frame_bgr, (x1, y1, x2-x1, y2-y1))
                    trackers.append(tr)

            # 얼굴별 처리
            for (x1,y1,x2,y2) in faces:
                crop_rgb = rgb_frame[y1:y2, x1:x2]
                if crop_rgb.size == 0:
                    continue

                # TRAIN 모드: 스페이스바로 저장
                if key == 32 and need_det:
                    fname = f"face_{count}.jpg"
                    cv2.imwrite(str(save_dir/fname), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                    embeds[fname] = emb_from_rgb(crop_rgb)
                    count += 1

                # 매칭
                if embeds:
                    e = emb_from_rgb(crop_rgb)
                    best, sim = max(
                        ((k, cos(e, v)) for k,v in embeds.items()),
                        key=lambda x: x[1]
                    )
                    if sim > thr:
                        txt, col = f"TRACK {best} {sim:.2f}", (255,0,0)
                    else:
                        txt, col = f"UNKNOWN {sim:.2f}", (0,0,255)
                    cv2.putText(rgb_frame, txt, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                cv2.rectangle(rgb_frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # 모드·FPS 표시
            mode = "TRAINING" if train else "TRACKING"
            cv2.putText(rgb_frame, f"{mode}  FPS:{fps:4.1f}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # 디스플레이: RGB→BGR로 변환
            display = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display)

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
