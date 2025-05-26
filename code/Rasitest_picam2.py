import sys, pathlib, os, time, cv2, numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO
from picamera2 import Picamera2

# ── 경로 설정 ───────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'MobileFaceNet-master'))

# ONNX 파일들
YOLO_ONNX   = str(ROOT / 'yolov11n-face.onnx')
MFN_ONNX    = str(ROOT / 'mobilefacenet.onnx')
SAVE_DIR    = ROOT / 'collected_images'
SAVE_DIR.mkdir(exist_ok=True)

# ── YOLOv11n-face 로드 ─────────────────────────────────────────────
yolo = YOLO(YOLO_ONNX)

# ── MobileFaceNet ONNX Runtime 세션 ─────────────────────────────────
mfn_sess = ort.InferenceSession(MFN_ONNX, providers=['CPUExecutionProvider'])
# 입력·출력 이름 가져오기
mfn_input  = mfn_sess.get_inputs()[0].name       # 보통 "input"
mfn_output = mfn_sess.get_outputs()[0].name      # 보통 "embeddings"

# ── 전처리 (112×112, RGB 정규화) ────────────────────────────────────
def emb_from_bgr_onnx(bgr):
    # BGR → RGB, resize, 정규화, 배치 차원 추가
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (112,112)).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    # H×W×C → 1×C×H×W
    tensor = img.transpose(2,0,1)[None, ...]
    # ONNX Runtime inference
    out = mfn_sess.run([mfn_output], {mfn_input: tensor})[0]
    return out[0]  # shape: (embedding_dim,)

# ── 기타 유틸 함수 ─────────────────────────────────────────────────
def cos(a, b):
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def create_tracker():
    for ns, fn in [("legacy","TrackerMOSSE_create"),(None,"TrackerMOSSE_create"),
                   ("legacy","TrackerKCF_create"),  (None,"TrackerKCF_create"),
                   ("legacy","TrackerCSRT_create"),(None,"TrackerCSRT_create")]:
        mod = getattr(cv2, ns) if ns else cv2
        if hasattr(mod, fn):
            return getattr(mod, fn)()
    raise RuntimeError("No tracker available")

# ── 메인 루프 ────────────────────────────────────────────────────
def main():
    trackers, embeds = [], {}
    frame_id, count = 0, 0
    train, prev_t, fps = True, time.time(), 0.0
    ALPHA = 0.2
    detect_every, thr = 20, 0.8

    # Picamera2 초기화
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={'format':'BGR888','size':(640,480)})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(2)

    try:
        while True:
            frame = picam2.capture_array()
            frame_id += 1
            now = time.time()
            fps = (1-ALPHA)*fps + ALPHA/(now-prev_t)
            prev_t = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                train = not train; frame_id = 0
            if key == 27:  # ESC
                break

            # 얼굴 박스 결정 (탐지 vs 트래킹)
            need = train or (frame_id % detect_every == 0)
            faces = []
            if not need:
                for tr in trackers:
                    ok, bb = tr.update(frame)
                    if ok:
                        x,y,w,h = map(int,bb)
                        faces.append((x,y,x+w,y+h))
                    else:
                        need = True
                        break

            if need:
                det = yolo.predict(frame, conf=0.5, verbose=False)[0].boxes
                faces = [tuple(map(int,b.xyxy[0])) for b in det]
                trackers.clear()
                for x1,y1,x2,y2 in faces:
                    tr = create_tracker()
                    tr.init(frame, (x1,y1,x2-x1,y2-y1))
                    trackers.append(tr)

            # 박스별 처리
            for x1,y1,x2,y2 in faces:
                crop = frame[y1:y2, x1:x2]
                if crop.size==0: continue

                # 학습 모드: Space 눌러 저장
                if key==32 and need:
                    fname = f"face_{count}.jpg"
                    cv2.imwrite(str(SAVE_DIR/fname), crop)
                    embeds[fname] = emb_from_bgr_onnx(crop)
                    count += 1

                # 임베딩 비교
                if embeds:
                    e = emb_from_bgr_onnx(crop)
                    best, sim = max(((k, cos(e,v)) for k,v in embeds.items()), key=lambda x:x[1])
                    label, col = (f"TRACK {best} {sim:.2f}", (255,0,0)) if sim>thr else (f"UNKNOWN {sim:.2f}", (0,0,255))
                    cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

            mode = "TRAINING" if train else "TRACKING"
            cv2.putText(frame, f"{mode} FPS:{fps:4.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.imshow("ONNX / Picamera2", frame)

    finally:
        picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
