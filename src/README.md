# 결재 레이더 — 통합 배포 가이드

## 구성

```
┌────────────── cam 기기 (예: 192.168.0.10) ──────────────┐
│  go_now_v2.py          카메라 표정 인식 + Flask 서버      │
│  radar_signal_processor.py   신호등 통합 판정 (싱글턴)    │
│  dashboard.html        대시보드 (Flask 가 서빙)           │
│                                                          │
│  · 0.0.0.0:5000 으로 대시보드 서빙                        │
│  · 표정은 in-process 로 add_expression()                 │
│  · 한숨/발화는 HTTP POST 로 수신                          │
└──────────────────────────────────────────────────────────┘
                         ▲  POST /api/radar/add-sigh
                         │  POST /api/radar/add-speech-emotion
┌────────────── mic 기기 ─┴────────────────────────────────┐
│  mic_agent.py   단일 마이크 → 한숨(YAMNet) + 발화(STT+KoBERT)│
│                 RADAR_URL 로 cam 기기에 전송               │
│  model/yamnet.tflite, model/yamnet_class_map.csv          │
└──────────────────────────────────────────────────────────┘
```

마이크는 mic_agent 가 **한 번만** 연다 → 한숨/발화를 한 프로세스에서 fan-out.
(두 프로세스가 같은 장치를 각각 open 하던 PortAudio/ALSA 경합이 사라짐)

## 파일 배치

| 기기 | 필요한 파일 |
|---|---|
| cam | `go_now_v2.py`, `radar_signal_processor.py`, `dashboard.html` |
| mic | `mic_agent.py`, `model/yamnet.tflite`, `model/yamnet_class_map.csv` |
| 둘 다 | `run_all.py` (같은 파일을 양쪽에 복사, `--role` 로 구분) |

## go_now_v2.py 수정 (한 곳)

`open_camera()` 함수를 `open_camera_patch.py` 의 내용으로 통째 교체.
이것 외에 go_now_v2.py 는 손댈 필요 없음 — `radar_signal_processor.py` 가
같은 폴더에 있으면 `_RADAR_AVAILABLE=True` 가 되어 `/api/radar/*` 가 살아남.

## 설치

```bash
# 공통(mic 기기)
pip install sounddevice webrtcvad numpy
sudo apt install libportaudio2

# 한숨 (mic 기기)
pip install tflite-runtime
# model/yamnet.tflite, model/yamnet_class_map.csv 준비

# 발화 (mic 기기)
pip install faster-whisper transformers torch kobert-transformers sentencepiece noisereduce

# cam 기기
pip install opencv-python mediapipe flask pillow numpy
# (선택) 표정 정확도 향상:
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## 실행

```bash
# ── cam 기기 ──
python3 run_all.py --role cam            # 카메라 0
python3 run_all.py --role cam 1          # 카메라 1 (인덱스는 v4l2-ctl --list-devices 로 확인)

# ── mic 기기 (cam 기기 IP 지정) ──
python3 run_all.py --role mic --radar-host 192.168.0.10
python3 run_all.py --role mic --radar-host 192.168.0.10 --no-calib   # 캘리브레이션 생략

# ── 단일 PC 테스트 ──
python3 run_all.py --role all
```

대시보드: `http://<cam_IP>:5000/`

## 신호등 판정 (구간별 독립, 우선순위)

각 구간(1분 / 5분 / 10분)에서:
1. 한숨 1회 이상 → **빨강** (최우선)
2. 표정 대표값 분노 → 빨강 / 행복 → **초록**
3. 표정 보통 + 발화 부정 개수 ≥ 임계값(1분 1 / 5분 2 / 10분 3) → 빨강
4. 그 외 → **노랑**

표정 대표값: 분노 비율 ≥ 0.30 → 분노, 행복 비율 ≥ 0.50 → 행복, 그 외 일반.
발화 극성: 긍정 = 기쁨, 부정 = 슬픔·분노·불안·당황·상처.

## 점검 명령

```bash
# cam 기기에서 신호 확인
curl http://localhost:5000/api/radar/signals
curl http://localhost:5000/api/radar/status
curl http://localhost:5000/api/device-status

# mic 기기에서 카메라/마이크 인덱스 확인
v4l2-ctl --list-devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

## 방화벽

cam 기기에서 5000 포트 인바운드 허용 필요:
```bash
sudo ufw allow 5000
```
