# EdgeFace USB Webcam (pigpio Servo)

이 레포지토리는 USB 웹캠을 사용해 YOLO-v11 기반 얼굴 탐지와 EdgeFace 임베딩을 결합한 후, 싱글 얼굴을 트래킹하며 Raspberry Pi의 pigpio 라이브러리를 사용해 팬/틸트 서보를 제어하는 샘플 코드입니다.
핵심 기능은 다음과 같습니다:

* **TRAINING 모드**: 웹캠 영상에서 얼굴을 탐지하여, 사용자가 Space 키를 눌러 얼굴 이미지를 저장하고 EdgeFace 임베딩 벡터를 수집(데이터베이스화)
* **TRACKING 모드**: 수집된 DB(임베딩)와 실시간 프레임의 임베딩을 비교하여 유사도 기준으로 한 명의 얼굴만 트래킹
* **Servo 제어**: 트래킹된 얼굴의 화면 중앙 오차를 데드존+오차 제곱 제어 방식으로 계산하여, pigpio를 통해 팬(좌우)·틸트(상하) 서보를 실시간으로 이동

---

## 📂 디렉토리 구조

```
.
├── collected_images/            # TRAINING 모드에서 저장된 얼굴 이미지와 임베딩 DB  
│   ├── face_0.jpg  
│   ├── face_1.jpg  
│   └── ...  
├── edgeface_xxs.onnx            # EdgeFace 임베딩 ONNX 모델  
├── yolov11n-face-metadata.onnx  # YOLO-v11 얼굴 탐지 ONNX 모델  
└── run_edgeface_usb_gray_lock_toggle_v2_onnx_opt_iou_clear_servo_pigpio.py  
```

* `collected_images/`

  * TRAINING 모드(프레임 상의 얼굴에 Space를 누를 때)마다 `face_{번호}.jpg` 형태로 얼굴 이미지를 저장
  * 저장된 이미지는 EdgeFace 임베딩으로 변환 후 내부 메모리(DB)로 관리됨
* `edgeface_xxs.onnx`

  * Tiny/XXS 버전 MobileFaceNet 기반 EdgeFace 임베딩 모델 (학습된 ONNX 파일)
* `yolov11n-face-metadata.onnx`

  * YOLO-v11 Nano 버전 얼굴 탐지 ONNX 모델 (Metadata 포함)
* `run_edgeface_usb_gray_lock_toggle_v2_onnx_opt_iou_clear_servo_pigpio.py`

  * 메인 스크립트 파일. Python 3 환경에서 실행하며, TRAINING 모드와 TRACKING 모드를 토글하여 얼굴 수집 및 트래킹을 수행하고, Raspberry Pi에서 pigpio로 서보 제어

---

## 🎯 기능 요약

1. **TRAINING 모드 (기본 모드)**

   * 매 프레임 YOLO-v11 ONNX 모델로 얼굴을 탐지
   * 탐지된 얼굴 영역에 사각형을 그려 화면에 표시
   * 사용자가 Space('␣') 키를 누르면 해당 프레임의 얼굴 이미지를 `collected_images/face_{숫자}.jpg` 로 저장
   * 저장된 이미지는 EdgeFace ONNX 모델로 임베딩 벡터를 생성하여 내부 DB(`dict`)에 추가
   * 이 모드에서는 서보(PWM) 제어가 **비활성화**됨

2. **TRACKING 모드**

   * DETECT\_EVERY(기본 20) 프레임마다(예: 20프레임 주기) YOLO로 얼굴을 탐지
   * **기존 Tracker가 있으면** 먼저 KCF 트래커로 ROI 업데이트 시도

     * 업데이트 실패 시(Tracker 소실) Tracker 리스트를 초기화
     * 성공 시 Bounding Box만 갱신
   * 검출(Detection) 시마다 다음 절차 수행

     1. **IOU 검사**:

        * 기존 Tracker의 바운딩 박스와 최신 탐지 박스들 간 IOU를 계산
        * 모든 IOU가 IOU\_THR(0.6) 미만일 경우 Tracker를 클리어(삭제)
     2. **Tracker 클리어 상태**이면서 **임베딩 DB가 존재**하면:

        * 최신 탐지된 얼굴마다 EdgeFace 임베딩 계산
        * DB 내 저장된 임베딩 벡터와 Cosine 유사도(dot-product) 비교
        * 가장 높은 유사도를 보인 얼굴을 선택해 KCF 트래커 초기화
        * 트래킹 바운딩 박스에 “sim:{유사도}” 텍스트 출력
   * **서보 제어**

     * Tracker로부터 얻은 얼굴 바운딩 박스의 중심 좌표와 화면 중심 간 오차(픽셀) 계산
     * 데드존(DEAD\_ZONE = 8px) 이내면 무시, 바깥이면 오차 제곱 방식(Δ = K·err·|err|)으로 각도 보정
     * 보정된 각도를 0°–180°로 클램프 후, pigpio로 펄스 폭(500µs–2500µs) 전송 → 실제 팬/틸트 서보 이동
   * 이 모드에서는 Space 입력 무시, 대신 ‘m’ 키를 다시 누르면 TRAINING 모드로 돌아감

3. **키 입력**

   * `m` : TRAINING ↔ TRACKING 모드 토글
   * `SPACE` : TRAINING 모드에서만 얼굴 사진 저장 (및 임베딩 DB에 추가)
   * `ESC` : 프로그램 종료 (서보 PWM 종료 후 pigpio 데몬 연결 해제)

---

## ⚙️ 요구 사항 및 설치

### 1. 하드웨어

* **Raspberry Pi (권장: Pi 3 이상) + Raspbian OS**
* USB 웹캠 (리눅스에서 `/dev/video0` 또는 `/dev/video1` 형태로 인식)
* 팬/틸트용 서보 2개 (Pan용, Tilt용)
* 서보 전원 및 GPIO 연결 (BCM 5번 → Pan 서보, BCM 6번 → Tilt 서보)
* Raspberry Pi에 pigpio 데몬 구동

  ```bash
  sudo apt update
  sudo apt install pigpio python3-pigpio
  sudo systemctl enable pigpiod
  sudo systemctl start pigpiod
  ```

### 2. 소프트웨어

* Python 3.7 이상
* 필수 라이브러리 설치

  ```bash
  pip3 install opencv-python torch torchvision onnxruntime pillow ultralytics pigpio numpy
  ```
* ultralytics 패키지 버전은 YOLO v8을 지원하는 최신 버전이어야 합니다.

### 3. 모델 파일 준비

1. **YOLO-v11 얼굴 탐지 ONNX 모델**

   * `yolov11n-face-metadata.onnx` 파일을 프로젝트 루트에 위치시킵니다.
   * 예시 다운로드 링크(커스텀 모델이라면 직접 학습 후 Export):

     ```
       https://github.com/ultralytics/yolov5/releases  (YOLO v5/v8 공식 리포 참고)  
       ※ 만약 커스텀 얼굴 탐지 모델을 사용한다면, `export to ONNX` 작업 후 본 파일로 대체
     ```
2. **EdgeFace 임베딩 ONNX 모델**

   * `edgeface_xxs.onnx` 파일을 프로젝트 루트에 위치시킵니다.
   * GitHub 또는 PyPI에서 MobileFaceNet/EdgeFace ONNX 파일을 받거나, pretrained 모델을 export 권장

---

## 🚀 사용 방법

1. **레포지토리 클론**

   ```bash
   git clone https://YOUR_GIT_REPO_URL.git
   cd YOUR_REPO_DIRECTORY
   ```

2. **모델 파일 복사**

   * `yolov11n-face-metadata.onnx`
   * `edgeface_xxs.onnx`

3. **pigpio 데몬 실행 확인**

   ```bash
   sudo systemctl status pigpiod
   ```

   * 만약 비활성 상태라면:

     ```bash
     sudo systemctl start pigpiod
     ```
   * Raspberry Pi 외 다른 리눅스 환경에서는 pigpio가 제대로 동작하지 않을 수 있습니다.

4. **스크립트 실행**

   ```bash
   python3 run_edgeface_usb_gray_lock_toggle_v2_onnx_opt_iou_clear_servo_pigpio.py
   ```

   * Windows에서는 pigpio가 없으므로, `pigpio` 초기화가 실패하더라도 서보 제어 없이 얼굴 탐지/트래킹만 동작합니다.
   * Linux(RPi) 환경에서는 pigpio가 성공적으로 초기화되면, 서보 제어가 활성화됩니다.

5. **키 조작**

   * **화면 표시**: “TRAINING FPS:{fps}” 또는 “TRACKING FPS:{fps}” 가상창이 뜹니다.
   * **Space('␣')**: TRAINING 모드에서만 동작 (탐지된 얼굴을 `collected_images/face_{n}.jpg` 로 저장하고, 이를 임베딩 DB에 추가)
   * **‘m’키**: 모드 전환 (TRAINING ↔ TRACKING)

     * **TRACKING → TRAINING**: 트래킹 중이던 서보를 중립(90°, 5°) 위치로 초기화
     * **TRAINING → TRACKING**: 중립 위치 그대로 서보 동작 시작
   * **ESC**: 프로그램 종료 (서보 PWM 종료 후 pigpio 데몬 연결 해제)

---

## 📈 주요 매개변수

| 변수/상수명          | 의미                                             | 기본값     |
| --------------- | ---------------------------------------------- | ------- |
| `DETECT_EVERY`  | 몇 프레임마다 얼굴 탐지를 수행할지 주기 단위 (프레임 수)              | 20      |
| `THR`           | EdgeFace 유사도 임계값 (추후 비교용, 현재 코드에서는 최대 유사도만 사용) | 0.70    |
| `IOU_THR`       | 기존 트래커 박스와 새 탐지 박스 간 IOU 임계값                   | 0.60    |
| `DEAD_ZONE`     | 서보 제어 시 얼굴 오차(픽셀) 중 무시할 범위                     | 8 px    |
| `K_PAN`         | 팬(좌우) 각도 보정 상수 (오차 제곱 제어 계수)                   | 0.00002 |
| `K_TILT`        | 틸트(상하) 각도 보정 상수                                | 0.00002 |
| `START_ANGLE_X` | 팬 서보 중립 각도 (도)                                 | 90.0°   |
| `START_ANGLE_Y` | 틸트 서보 중립 각도 (도)                                | 5.0°    |
| `PWM_MIN_PW`    | 서보가 0°일 때 보내는 최소 펄스 폭(μs)                      | 500 μs  |
| `PWM_MAX_PW`    | 서보가 180°일 때 보내는 최대 펄스 폭(μs)                    | 2500 μs |

---

## 💡 실습 및 커스터마이징 팁

1. **DETECT\_EVERY 값 조절**

   * 낮게 설정하면(예: 10) 얼굴 탐지 빈도가 높아져 트래킹 정확도는 올라가지만 CPU 사용량이 증가
   * 높게 설정하면(예: 30–40) CPU 사용량은 줄어들지만, 얼굴 이동 속도가 빠를 때 트래킹이 흐트러질 수 있음

2. **K\_PAN / K\_TILT 계수 조정**

   * 서보 반응성(얼굴 오차에 대한 각도 보정 속도)을 조절함
   * 오차 제곱 제어 방식이므로, 계수가 크면(예: 0.00005) 얼굴이 화면 중앙으로 빠르게 이동하지만 서보가 진동할 수 있음
   * 계수가 작으면(예: 0.00001) 움직임이 부드럽지만 얼굴이 중앙으로 천천히 이동

3. **DEAD\_ZONE 크기 조정**

   * 얼굴이 화면 중심에서 |±DEAD\_ZONE| px 이내에 있으면 서보 이동을 멈춤
   * DEAD\_ZONE을 너무 작게 하면(예: 2) 작은 떨림에도 서보가 과도하게 움직일 수 있고, 너무 크게 하면(예: 20) 얼굴이 중심을 벗어난 상태로 머무를 수 있음

4. **모델 교체 또는 재학습**

   * YOLO-v11 Nano 모델(`yolov11n-face-metadata.onnx`) 대신 YOLO-v8n이나 YOLO-v5 딥페이크 얼굴 탐지 모델로 변경 가능
   * EdgeFace 대신 ArcFace, MobileFaceNet, FaceNet ONNX 모델 등으로 임베딩 성능 실험 가능
   * 모델을 교체할 때는

     ```
     yolo = YOLO(model="새로운_onnx파일", task="detect")
     ```

     형식으로 경로만 수정하면 됨

5. **다중 얼굴 트래킹(확장)**

   * 현재 코드는 DB에 저장된 임베딩과 가장 유사도가 높은 얼굴 한 명만 트래킹
   * 여러 얼굴을 동시에 트래킹하려면 `tracks` 리스트를 순회하면서 다수의 KCF 트래커를 생성하고, pigpio 제어를 별도 로직으로 확장 필요

---

## 🚩 주의사항

* **pigpio 연결 실패**

  * Raspberry Pi 환경이 아니거나 pigpio 데몬이 설치/실행되지 않은 경우 `PI_GPIO_OK = False`로 설정되어, 서보 제어 기능이 자동으로 비활성화됩니다.
  * 윈도우 등 다른 OS에서 테스트할 때는 얼굴 탐지/트래킹만 수행됩니다.

* **Webcam 인덱스**

  * Windows에서는 `cv2.VideoCapture(1, cv2.CAP_DSHOW)` (인덱스 1)
  * Linux(RPi)에서는 `cv2.VideoCapture(0, cv2.CAP_V4L2)` (인덱스 0)
  * 환경에 따라 `VideoCapture()` 인덱스가 달라질 수 있으므로, 필요 시 값을 변경하세요.

* **서보 전원**

  * Raspberry Pi의 5V 핀만으로 고전류 서보를 직접 구동하면 전원 과부하가 발생할 수 있습니다.
  * 별도의 5V 전원 어댑터를 사용하거나 서보 전용 전원 핀이 있는 경우, 안정적인 전원 공급을 권장합니다.

---

## 📜 라이선스

이 프로젝트는 특별한 라이선스를 지정하지 않습니다.
자유롭게 수정·배포하여 사용하실 수 있습니다.
다만, 본 레포지토리의 모델 파일(ONNX)은 원저작권자의 라이선스를 따릅니다.
