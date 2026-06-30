# 결재 레이더 — 실행·재현 문서 `RUN.md`

이 시스템은 두 역할로 나뉩니다 — **cam 기기**가 Flask 서버(`:5000`)를 띄우고,
**mic 기기**가 cam 기기 IP(`--radar-host`)로 한숨/발화 결과를 전송합니다.
단일 기기에서 둘 다 돌리려면 `--role all`을 씁니다.

---

## 1. 요구 환경

| 항목 | 사양 |
|---|---|
| 보드 | Raspberry Pi 5 (ARM64) |
| OS | Raspberry Pi OS (64-bit) |
| Python | 3.11+ (`python3 --version`) |
| cam 주변장치 | Pi 카메라 모듈 또는 USB 웹캠 |
| mic 주변장치 | USB 마이크 |
| 네트워크 | cam·mic 동일 LAN, **TCP 5000** 개방 |

---

## 2. 라이브러리 & 버전

설치 목록은 다음 3개 파일에 버전과 함께 명시되어 있습니다.

- **`requirements-cam.txt`** — `numpy`, `opencv-python`, `mediapipe`, `flask`, `Pillow`
- **`requirements-mic.txt`** — `numpy`, `sounddevice`, `webrtcvad`, `tflite-runtime`, `faster-whisper`, `torch`, `transformers`, `kobert-transformers`, `sentencepiece`
- **`requirements.txt`** — 위 둘을 `-r`로 합성(= `all` 한 번에 설치)

> **3개로 나눈 이유:** 분산 구조라 cam은 영상(opencv/mediapipe), mic은 음성·STT(torch/whisper/kobert)
> 스택으로 의존성이 거의 겹치지 않고 둘 다 무겁습니다. 역할별 기기에 불필요한 수 GB를 안 깔려는 분리입니다.

**버전 표기:** ARM/라즈베리파이 휠 호환을 위해 상·하한 범위(`>=x,<y`)로 지정했습니다. 정확한 `==` 고정은 ARM에서 휠 부재로 설치가 실패하기 쉬워 피했습니다.

---

## 3. 설치

```bash
# (선택) 가상환경
python3 -m venv .venv && source .venv/bin/activate

# 역할에 맞게 택1 — 실행권한 이슈를 피하려 bash로 호출
bash setup.sh all        # 단일 기기에서 cam+mic 모두
bash setup.sh cam        # cam 기기만
bash setup.sh mic        # mic 기기만
```

`setup.sh`가 하는 일: ① apt 시스템 패키지(`libportaudio2`,`ffmpeg`,`fonts-nanum`)
② 역할별 pip 설치 ③ 모델 처리(cam `face_landmarker.task` 자동 다운로드 / mic YAMNet 존재 확인).

> `./setup.sh`로 실행하려면 1회 `chmod +x setup.sh`. git에 실행권한 보존: `git update-index --chmod=+x setup.sh`.

**수동 설치(스크립트 미사용 시):**
```bash
sudo apt-get update && sudo apt-get install -y libportaudio2 ffmpeg fonts-nanum
pip install -r requirements.txt        # 또는 requirements-cam.txt / requirements-mic.txt
```

---

## 4. 설정 파일 & 실행 인자

> 이 프로젝트는 **환경 변수(`.env`)를 쓰지 않습니다.** 설정은 (A) 실행 인자와 (B) JSON 파일로 합니다.

### A. 실행 인자 (`run_all.py`)

| 인자 | 의미 | 예 |
|---|---|---|
| `--role` | `cam` / `mic` / `all` / `check` | `--role mic` |
| (위치) 카메라 인덱스 | cam/all에서 카메라 장치 번호 | `0` |
| `--device` | mic 입력 장치 인덱스(PortAudio) | `--device 1` |
| `--radar-host` | (mic) cam 기기 IP | `--radar-host 192.168.0.10` |

### B. 설정/데이터 파일 (모두 `src/` 아래, 실행도 `src/`에서)

> 앱은 `src/`를 작업 디렉터리로 보고 상대 경로로 설정·모델을 읽습니다. 그래서 실행 전 `cd src` 합니다.

| 파일 | 위치 | 역할 | 비고 |
|---|---|---|---|
| `weights_config.json` | `src/` | [cam] 위험점수 가중치·임계 | 없으면 내장 기본값. `setup/checker/emotion_tuner.py`로 생성 |
| `stt_config.json` | `src/` | [mic] STT 빔/VAD | `setup/checker/speech_tuner.py`로 생성 |
| `gate_config.json` | `src/` | [mic] 발화 폐기 게이트 | `setup/checker/gate_tuner.py`로 생성 |
| `outlook_config.json` | `src/` | [cam] (선택) Outlook 설정 | `outlook_sync.py` 사용 시 |
| `calendar_data.json` | `src/` | [cam] 팀장 일정 | UI/`/api/calendar`로 적재 |
| `approval_data.json` | `src/` | [cam] 결재 항목 | UI/`/api/approvals`로 적재 |
| `model/yamnet.tflite`, `yamnet_class_map.csv` | `src/model/` | [mic] 한숨 감지(YAMNet) | **필수** (없으면 `--no-sigh`) |
| `model/classifier.npz`, `threshold.txt` | `src/model/` | [mic] 한숨 분류기·임계 | **필수** |
| `face_landmarker.task` | `src/setup/` | [cam] 표정 정확도 향상 | **선택**, 없으면 자동 폴백 |

`approval_data.json` 예시 구조:
```json
[{ "id": "ap-0001", "date": "2026-06-30", "title": "마케팅 예산 집행", "note": "긴급" }]
```
> 판단/결재 탭은 `date` 기준으로 **그날 항목만** 표시합니다.

---

## 5. 외부 자원

- **외부 API 키 불필요** — 핵심 동작(표정/한숨/발화 → 신호등)은 인터넷 없이 로컬 수행. DB 없음.
- **모델** — cam `src/setup/face_landmarker.task`(선택, setup.sh 자동 다운로드),
  mic YAMNet·한숨 분류기 `src/model/`(필수, 직접 배치).
  수동 다운로드: `wget -O src/setup/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
  > setup.sh가 모델을 루트에 받는 구버전이면 위 경로(`src/setup/`, `src/model/`)로 옮기세요.
- **하드웨어** — 카메라(cam), USB 마이크(mic). **네트워크** — 동일 LAN, 포트 5000.
- **(선택) Outlook 동기화** — `outlook_sync.py` 있을 때만 활성. 없으면 해당 기능만 비활성, 본 시스템은 정상 동작.

---

## 6. 실행

실행 명령은 모두 `src/`에서 수행합니다(설정·모델 상대 경로 때문).

```bash
cd src

# 단일 기기 (cam+mic 함께)
python3 run_all.py --role all 0 --device 1

# 분산 구성
python3 run_all.py --role cam 0                                # ① cam 먼저 → 콘솔에 http://<cam_ip>:5000 출력
python3 run_all.py --role mic --radar-host <cam_ip> --device 1 # ② mic (cam IP 지정)
```

브라우저에서 **`http://<cam_ip>:5000`** 접속 → 대시보드.

> `--device 1`은 USB 마이크의 PortAudio 입력 인덱스 예시입니다. 환경마다 다르니 `python3 setup/checker/check_devices.py`로 확인하세요.

---

## 7. 설치·실행 검증

```bash
cd src
python3 setup/checker/check_devices.py     # 카메라/마이크/모델/패키지 점검
pip check                                  # 의존성 충돌 여부
python3 -c "import cv2, mediapipe, flask; print('cam deps OK')"               # cam
python3 -c "import sounddevice, faster_whisper, torch; print('mic deps OK')"  # mic
```

**기동 확인**
1. cam 기동 후 콘솔의 `http://<ip>:5000`로 접속 → 대시보드가 뜨면 성공.
2. 상단 **장치 상태 배너**의 📷 카메라 / 🎤 한숨 / 🗣 발화가 ✅로 바뀌는지 확인.
3. "판단" 탭 신호등이 측정중 → 🟢/🟡/🔴로 갱신되면 정상.

---

## 8. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `PortAudio -9998 (Invalid channel count)` | 실제 USB 마이크 인덱스로 `--device N` 지정 (`setup/checker/check_devices.py`로 확인) |
| 카메라가 안 열림 | 인덱스 변경: `run_all.py --role cam 1` / `v4l2-ctl --list-devices` |
| `face_landmarker.task` 없음 경고 | 선택 사항 — 기하학 분석으로 자동 폴백(정확도만 소폭 하락) |
| YAMNet 모델 누락 | `model/`에 두 파일 배치, 또는 `mic_agent`를 `--no-sigh`로 실행 |
| `./setup.sh: Permission denied` | `bash setup.sh all` 또는 `chmod +x setup.sh` |
| 카메라 영상의 한글 라벨이 네모로 깨져 보임 | `sudo apt-get install -y fonts-nanum` |
| 대시보드에 결재가 안 보임 | `approval_data.json` 항목의 `date`가 오늘과 같은지 확인 |

---

## 부록. 빠른 재현 (단일 기기)

```bash
git clone <repo> && cd go_now
python3 -m venv .venv && source .venv/bin/activate
bash setup.sh all                          # 루트에서 설치
cd src
python3 setup/checker/check_devices.py     # 점검
python3 run_all.py --role all 0 --device 1 # 기동
# 브라우저: http://<이 기기 IP>:5000
```
