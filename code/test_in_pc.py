import cv2, os, numpy as np, torch
import torchvision.transforms as T, torchvision.models as models
from PIL import Image
from ultralytics import YOLO
from torchvision.models import ResNet50_Weights

# ── 모델 준비 ──────────────────────────────────────────────
yolo   = YOLO("yolov8n.pt")
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity();  resnet.eval()

transform = T.Compose([
    T.Resize((224,224)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
def get_emb(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad(): return resnet(x).squeeze(0).numpy()

def cos(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

# ── 설정 ───────────────────────────────────────────────────
save_dir = "collected_images"; os.makedirs(save_dir, exist_ok=True)
target   = "person"          # COCO 라벨
embeds   = {}                # {파일명: 벡터}
thr      = 0.8               # 코사인 유사도 임계값
count    = 0

print("[SPACE] 캡처 / [ESC] 종료")

# ── 카메라 한 번만 열고 전체 로직 처리 ─────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW로 열면 재오픈 문제 완화
while True:
    ok, frame = cap.read()
    if not ok: break

    # YOLO 추론
    results = yolo.predict(frame, conf=0.5)
    boxes   = results[0].boxes if len(results)>0 else []

    for box in boxes:
        cls = yolo.names[int(box.cls[0])]
        if cls != target: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        # ── 스페이스바 → 캡처 & 임베딩 저장 ──
        key = cv2.waitKey(1) & 0xFF
        if key == 32:                       # SPACE
            fname = f"{target}_{count}.jpg"
            cv2.imwrite(os.path.join(save_dir,fname), crop)
            embeds[fname] = get_emb(crop)
            count += 1
            print(f"[CAPTURE] {fname} 저장  (총 {count}장)")

        # ── 이미 임베딩이 있다면 실시간 트래킹 ──
        if embeds:
            emb  = get_emb(crop)
            best, sim = max(
                ((k, cos(emb,v)) for k,v in embeds.items()),
                key=lambda x: x[1]
            )
            if sim > thr:
                text, color = f"TRACK {best} {sim:.2f}", (255,0,0)
            else:
                text, color = f"UNKNOWN {sim:.2f}", (0,0,255)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("Capture & Tracking", frame)

    # ESC 종료
    if key == 27: break

cap.release(); cv2.destroyAllWindows()
