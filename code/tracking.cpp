#include <Servo.h>

// 서보 핀 설정
const int PAN_PIN  = 9;   // 팬(좌우) 서보 연결 핀
const int TILT_PIN = 10;  // 틸트(상하) 서보 연결 핀

Servo panServo;
Servo tiltServo;

// 제어 변수
float panAngle  = 90.0;  // 초기 중립(가운데) 각도
float tiltAngle = 90.0;

const float Kp_pan  = 0.05;  // 팬 제어 비례 상수 (적절히 조절)
const float Kp_tilt = 0.05;  // 틸트 제어 비례 상수

void setup() {
  Serial.begin(115200);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
}

void loop() {
  // 시리얼 한 줄 읽기: "face_cx,face_cy,screen_cx,screen_cy\n"
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  // 토큰 분리
  int vals[4];
  int idx = 0;
  char *buf = const_cast<char*>(line.c_str());
  char *tok = strtok(buf, ",");
  while (tok != NULL && idx < 4) {
    vals[idx++] = atoi(tok);
    tok = strtok(NULL, ",");
  }
  if (idx < 4) return;  // 파싱 실패 시 무시

  int face_cx   = vals[0];
  int face_cy   = vals[1];
  int screen_cx = vals[2];
  int screen_cy = vals[3];

  // 오차 계산
  int error_x = face_cx - screen_cx;  // 양수면 얼굴이 화면 오른쪽에 위치
  int error_y = face_cy - screen_cy;  // 양수면 얼굴이 화면 아래에 위치

  // P 제어: 오차 비례만큼 각도 조정
  panAngle  -= Kp_pan  * error_x;  // 팬: 얼굴이 오른쪽이면 왼쪽으로
  tiltAngle += Kp_tilt * error_y;  // 틸트: 얼굴이 아래면 시선 아래로

  // 안전 범위 제한
  panAngle  = constrain(panAngle,  0.0, 180.0);
  tiltAngle = constrain(tiltAngle, 0.0, 180.0);

  // 서보에 쓰기
  panServo.write((int)panAngle);
  tiltServo.write((int)tiltAngle);

  // 디버그용 모니터 출력
  Serial.print("ErrX="); Serial.print(error_x);
  Serial.print(" ErrY="); Serial.print(error_y);
  Serial.print(" Pan="); Serial.print(panAngle);
  Serial.print(" Tilt="); Serial.println(tiltAngle);
}
