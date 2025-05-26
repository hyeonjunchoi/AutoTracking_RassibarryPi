# export_models_to_onnx.py

import torch
from ultralytics import YOLO
import sys
import pathlib

ROOT     = pathlib.Path(__file__).resolve().parent
MFN_DIR  = ROOT / "MobileFaceNet-master"
sys.path.append(str(MFN_DIR))

from mobilefacenet import MobileFaceNet


# 1. 설정
ROOT = pathlib.Path(__file__).resolve().parent

# PyTorch checkpoint 파일들
YOLOV11_PATH = ROOT / "yolov11n-face.pt"
YOLOV8_PATH  = ROOT / "yolov8n-face.pt"
MFN_PATH  = MFN_DIR / "mobilefacenet.pt" 

# 출력 ONNX 파일 이름
YOLOV11_ONNX = ROOT / "yolov11n-face.onnx"
YOLOV8_ONNX  = ROOT / "yolov8n-face.onnx"
MFN_ONNX     = ROOT / "mobilefacenet.onnx"

# 2. YOLO 모델 로드 & export
def export_yolo(pt_path, onnx_path, input_size=640):
    # ultralytics YOLO 객체
    yolo = YOLO(str(pt_path))
    # 내부 PyTorch nn.Module
    model = yolo.model

    model.eval()
    # dummy input: 배치1, 3채널, 정사각형
    dummy = torch.randn(1, 3, input_size, input_size)
    # Export
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        verbose=False,
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"}
        }
    )
    print(f"Exported {pt_path.name} → {onnx_path.name}")

# 3. MobileFaceNet 로드 & export
def export_mfn(pt_path, onnx_path):
    # MobileFaceNet 모듈 경로에 sys.path 추가가 필요할 수도 있습니다.
    mfn = MobileFaceNet().eval()
    # 가중치 로드
    mfn.load_state_dict(torch.load(pt_path, map_location="cpu"), strict=False)

    # 임베딩 입력은 112×112 이미지
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        mfn,
        dummy,
        str(onnx_path),
        verbose=False,
        opset_version=12,
        input_names=["input"],
        output_names=["embeddings"],
        dynamic_axes={"input": {0: "batch"}, "embeddings": {0: "batch"}}
    )
    print(f"Exported {pt_path.name} → {onnx_path.name}")

if __name__ == "__main__":
    # YOLOv11n-face
    export_yolo(YOLOV11_PATH, YOLOV11_ONNX, input_size=640)
    # YOLOv8n-face
    export_yolo(YOLOV8_PATH, YOLOV8_ONNX, input_size=640)
    # MobileFaceNet
    export_mfn(MFN_PATH, MFN_ONNX)
