# AutoTracking_RassibarryPi

## 📋 프로젝트 소개  
`PiFaceTrack`은 Raspberry Pi 카메라(PiCamera2)와 OpenCV, Ultralytics YOLO, MobileFaceNet 임베딩, 그리고 MOSSE/KCF 트래킹을 결합해 **실시간 얼굴 검출·학습·추적**을 수행하는 파이썬 스크립트입니다.  
- **TRAINING 모드**: 스페이스바 눌러 기준 얼굴 이미지를 수집 & 임베딩 저장  
- **TRACKING 모드**: YOLO 검출 후 저장된 임베딩과 코사인 유사도 비교, 가까운 얼굴을 “TRACK”  

---

## ⚙️ 주요 기능  
- **실시간 얼굴 검출**: Ultralytics YOLOv8n-face 모델 사용  
- **얼굴 임베딩**: MobileFaceNet을 활용해 112×112 RGB 이미지를 128차원 벡터로 변환  
- **코사인 유사도 매칭**: 저장한 얼굴 벡터와 실시간 얼굴 간 유사도 계산  
- **트래킹**: OpenCV Legacy/KCF/MOSSE 트래커로 검출된 얼굴 영역을 프레임 간 연속 추적  
- **Picamera2 연동**: libcamera 기반으로 3채널 `RGB888` 포맷 직접 캡처  
- **FPS 표시 & 모드 전환**: 화면 좌상단에 실시간 FPS 및 TRAIN/TRACK 모드 표시  

---

## 🛠️ 요구사항  
- **하드웨어**  
  - Raspberry Pi 4
  - Raspberry Pi picamer2 또는 호환 USB 웹캠  
- **소프트웨어**  
  - Raspberry Pi OS (Bullseye 이상 권장)  
  - Python 3.7 이상  
  - OpenCV (contrib 또는 apt 설치)  
  - PyTorch (ARMv7/ARM64 빌드)  
  - torchvision  
  - ultralytics (YOLO)  
  - picamera2  

---

## 🚀 설치 방법  

1. **시스템 업데이트 & 의존성 설치**  
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y python3-pip python3-venv \
       libopencv-dev python3-opencv \
       libatlas-base-dev libopenblas-dev \
       python3-picamera2 python3-libcamera
