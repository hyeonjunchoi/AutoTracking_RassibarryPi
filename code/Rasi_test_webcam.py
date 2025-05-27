import sys, pathlib, os, time, cv2, numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

# ── 경로 설정 ───────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\user\model")
sys.path.append(str(ROOT / 'MobileFaceNet-master'))

# ONNX 파일들
MFN_ONNX  = str(ROOT / 'mobilefacenet.onnx')
SAVE_DIR  = ROOT / 'collected_images'
SAVE_DIR.mkdir(exist_ok=True)

# ── YOLOv11n-face 로드 ─────────────────────────────────────────────
yolo = YOLO(model=str(ROOT/'yolov11n-face.onnx'), task="detect")

# ── MobileFaceNet ONNX Runtime 세션 ─────────────────────────────────
mfn_sess = ort.InferenceSession(MFN_ONNX, providers=['CPUExecutionProvider'])
mfn_input  = mfn_sess.get_inputs()[0].name
mfn_output = mfn_sess.get_outputs()[0].name

def emb_from_bgr_onnx(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (112,112)).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = img.transpose(2,0,1)[None, ...]
    out = mfn_sess.run([mfn_output], {mfn_input: tensor})[0]
    return out[0]

def cos(a, b):
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

def create_tracker():
    # legacy가 있는지 미리 확인
    has_legacy = hasattr(cv2, 'legacy')
    for ns, fn in [
        ("legacy","TrackerMOSSE_create"), (None,"TrackerMOSSE_create"),
        ("legacy","TrackerKCF_create"),   (None,"TrackerKCF_create"),
        ("legacy","TrackerCSRT_create"),  (None,"TrackerCSRT_create")
    ]:
        if ns == "legacy":
            if not has_legacy:
                continue
            mod = cv2.legacy
        else:
            mod = cv2

        if hasattr(mod, fn):
            return getattr(mod, fn)()
    raise RuntimeError("No tracker available")

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    trackers = []
    embeds   = {}
    faces    = []
    frame_id = 0
    count    = 0
    train    = True
    prev_t   = time.time()
    fps      = 0.0
    ALPHA    = 0.2

    detect_every  = 20
    emb_interval  = 30
    emb_threshold = 0.75

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            now = time.time()
            fps = (1-ALPHA)*fps + ALPHA/(now-prev_t)
            prev_t = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                print("Embeds:", list(embeds.keys()))
                train = not train
                frame_id = 0
                trackers.clear()
                faces.clear()
            if key == 27:
                break

            faces.clear()
            if train:
                det = yolo.predict(frame, conf=0.5, verbose=False)[0].boxes
                for b in det:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                    if key == 32:
                        crop = frame[y1:y2, x1:x2]
                        fname = f"face_{count}.jpg"
                        cv2.imwrite(str(SAVE_DIR/fname), crop)
                        embeds[fname] = emb_from_bgr_onnx(crop)
                        count += 1
            else:
                # 재탐지 조건
                if frame_id % detect_every == 0 or not trackers:
                    trackers.clear()
                    det = yolo.predict(frame, conf=0.5, verbose=False)[0].boxes
                    for b in det:
                        x1,y1,x2,y2 = map(int, b.xyxy[0])
                        tr = create_tracker()
                        tr.init(frame, (x1,y1,x2-x1,y2-y1))
                        trackers.append(tr)
                        faces.append((x1,y1,x2,y2))
                else:
                    ok, bb = trackers[0].update(frame)
                    if ok:
                        x,y,w,h = map(int, bb)
                        if frame_id % emb_interval == 0 and embeds:
                            crop = frame[y:y+h, x:x+w]
                            e = emb_from_bgr_onnx(crop)
                            best_sim = max(cos(e, v) for v in embeds.values())
                            if best_sim < emb_threshold:
                                trackers.clear()
                                det = yolo.predict(frame, conf=0.5, verbose=False)[0].boxes
                                for b in det:
                                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                                    tr = create_tracker()
                                    tr.init(frame, (x1,y1,x2-x1,y2-y1))
                                    trackers.append(tr)
                                    faces.append((x1,y1,x2,y2))
                            else:
                                faces.append((x, y, x+w, y+h))
                        else:
                            faces.append((x, y, x+w, y+h))
                    else:
                        trackers.clear()
                        det = yolo.predict(frame, conf=0.5, verbose=False)[0].boxes
                        for b in det:
                            x1,y1,x2,y2 = map(int, b.xyxy[0])
                            tr = create_tracker()
                            tr.init(frame, (x1,y1,x2-x1,y2-y1))
                            trackers.append(tr)
                            faces.append((x1,y1,x2,y2))

                for (x1,y1,x2,y2) in faces:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            mode = "TRAINING" if train else "TRACKING"
            cv2.putText(frame, f"{mode}  FPS:{fps:4.1f}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Webcam ONNX Tracking", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
