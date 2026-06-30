# 지금 갈까요? — 결재 레이더

라즈베리파이 5 기반 **감정 신호등** 시스템. 팀장의 **표정**(카메라) + **한숨/발화**(마이크)를
종합해 "지금 결재받으러 가도 되는가"를 1·5·10분 구간의 신호등(🔴/🟡/🟢)으로 알려줍니다.

> 설치·실행·문제해결은 **[`RUN.md`](./RUN.md)** 참조.

---

## 목차

- [핵심 아이디어](#핵심-아이디어)
- [아키텍처](#아키텍처)
- [디렉터리 구조](#디렉터리-구조)
- [빠른 시작](#빠른-시작)
- [대시보드 화면](#대시보드-화면)
- [운영 점검](#운영-점검)
- [테스트](#테스트)
- [모델 출처](#모델-출처)

---

## 핵심 아이디어

표정·한숨·발화를 하나의 **위험점수 R**(−1 ~ +1)로 합산해 신호등을 결정합니다.

- **한숨 1회** 또는 **강한 분노 표정** → 점수와 무관하게 즉시 🔴 (하드 오버라이드)
- 그 외에는 **R = 가중(표정 위험) + 가중(발화 위험)** 으로 판정
  `R ≥ red_th → 🔴` · `R ≤ green_th → 🟢` · 그 외 → 🟡

판정은 1·5·10분 구간별로 독립 계산합니다. 가중치·임계는 `weights_config.json`으로
**재학습 없이** 조정할 수 있습니다(`emotion_tuner.py`).

---

## 아키텍처

```
┌────────── cam 기기 (예: 192.168.0.10) ──────────┐
│  go_now_v2.py ── Flask :5000 ── dashboard.html  │
│       │  표정 감정 (MediaPipe, in-process)      │
│       └─ radar_signal_processor.py (R 판정·싱글턴)│
│                     ▲  POST /api/radar/*          │
└─────────────────────┼─────────────────────────────┘
                      │  같은 LAN / HTTP
┌─────────────────────┴── mic 기기 ────────────────────────┐
│  mic_agent.py ── 한숨(YAMNet) + 발화(faster-whisper→KoBERT)│
│                 RADAR_URL(--radar-host) 로 cam 에 전송     │
└───────────────────────────────────────────────────────────┘

  진입점:  run_all.py  --role {cam | mic | all | check}
```

- **표정**은 cam 프로세스 내부에서 바로 반영(`add_expression`), **한숨/발화**는 HTTP POST로 수신합니다.
- 마이크는 `mic_agent`가 **한 번만 열어** 한숨·발화를 한 프로세스에서 분배(fan-out)합니다 —
  두 프로세스가 같은 장치를 각각 열며 생기던 PortAudio/ALSA 경합이 없습니다.

### 기기별 배치

양쪽 기기에 동일한 `src/`를 두고 `--role`로 구분합니다. 역할별로 실제 쓰이는 파일:

| 기기 | 핵심 파일 |
|---|---|
| cam | `go_now_v2.py`, `radar_signal_processor.py`, `dashboard.html` |
| mic | `mic_agent.py`, `model/` (YAMNet + 한숨 분류기) |

> `go_now_v2.py`의 카메라 오픈은 `open_camera_patch.py` 내용으로 교체해 사용합니다.
> `radar_signal_processor.py`가 같은 폴더에 있으면 `/api/radar/*` 엔드포인트가 활성화됩니다.

---

## 디렉터리 구조

```
go_now/
├── README.md                     # 프로젝트 소개 (이 문서)
├── RUN.md                        # 설치·실행·재현 가이드
├── setup.sh                      # 원샷 설치 (cam|mic|all)
├── requirements.txt              # 통합(= -r cam + -r mic)
├── requirements-cam.txt          # cam 전용 의존성
├── requirements-mic.txt          # mic 전용 의존성
│
├── src/                          # 애플리케이션 소스
│   ├── run_all.py                # 진입점 (역할별 기동)
│   ├── go_now_v2.py              # [cam] Flask 서버 + 표정 인식
│   ├── radar_signal_processor.py # [cam] 신호등 판정 (위험점수 R)
│   ├── dashboard.html            # [cam] 웹 UI
│   ├── mic_agent.py              # [mic] 한숨 + 발화 감정
│   ├── outlook_sync.py           # [cam] (선택) 일정 동기화
│   ├── open_camera_patch.py      # 카메라 오픈 보조 (go_now_v2 교체용)
│   │
│   ├── weights_config.json       # [cam] 위험점수 가중치·임계
│   ├── stt_config.json           # [mic] STT 빔/VAD
│   ├── gate_config.json          # [mic] 발화 폐기 게이트
│   ├── outlook_config.json       # [cam] (선택) Outlook 설정
│   ├── calendar_data.json        # [cam] 팀장 일정 데이터
│   ├── approval_data.json        # [cam] 결재 항목 데이터
│   ├── samples_example.json      # emotion_tuner 입력 예시
│   │
│   ├── model/                    # [mic] 한숨 감지 모델
│   │   ├── yamnet.tflite         #   YAMNet 임베딩 추출기
│   │   ├── yamnet_class_map.csv  #   YAMNet 클래스 매핑
│   │   ├── classifier.npz        #   한숨 분류기 가중치
│   │   └── threshold.txt         #   한숨 판정 임계값
│   │
│   └── setup/
│       ├── face_landmarker.task  # [cam] (선택) 표정 정확도 향상 모델
│       └── checker/              # 점검·튜닝 도구
│           ├── check_devices.py  #   설치/장치 점검
│           ├── emotion_tuner.py  #   위험점수 임계 튜닝
│           ├── speech_tuner.py   #   STT 파라미터 튜닝
│           └── gate_tuner.py     #   발화 게이트 튜닝
│
├── tests/                        # pytest 단위 테스트
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_mic_agent.py
│   ├── test_model.py
│   ├── test_outlook_sync.py
│   └── test_radar_signal_processor.py
│
├── test-results/                 # 테스트 산출물 (coverage·junit·report)
│   ├── coverage.xml
│   ├── junit.xml
│   ├── report.html
│   └── test_log.txt
│
└── output/                       # 산출 문서
    └── LG_부트캠프_13기_B반_요구사항_명세서(1팀).md
```

> `__pycache__/`, `*.pyc`, `.pytest_cache/` 등 실행 중 생성되는 캐시는 `.gitignore`로 제외합니다.

---

## 빠른 시작

```bash
# 설치 (저장소 루트에서)
bash setup.sh all

# 실행 (단일 기기에서 cam+mic 함께)
cd src && python3 run_all.py --role all 0 --device 1

# 브라우저:  http://<기기 IP>:5000
```

분산 구성·인자·문제해결은 [`RUN.md`](./RUN.md)를 참고하세요.

---

## 대시보드 화면

- **판단** — 현재 상태(감정 이모지) + 신호등 + 추천 타이밍 + 오늘 결재
- **캘린더** — 팀장 일정 (1시간 내 일정이면 주의 신호)
- **결재** — 날짜별 결재 항목 추가/수정 (`approval_data.json`에 저장)
- **분석** — 구간별 위험점수 R 게이지 + 표정/발화 기여도 분해

---

## 운영 점검

```bash
# cam 기기 — 신호/장치 상태 확인
curl http://localhost:5000/api/radar/signals
curl http://localhost:5000/api/device-status

# 인덱스 확인
v4l2-ctl --list-devices                                   # 카메라
python3 -c "import sounddevice as sd; print(sd.query_devices())"   # 마이크
```

cam 기기는 **5000 포트 인바운드**를 허용해야 mic 기기·브라우저가 접속할 수 있습니다:
```bash
sudo ufw allow 5000
```

---

## 테스트

`pytest` 기반 단위 테스트. 저장소 루트에서:

```bash
pytest                       # 전체 실행
```

- `tests/` — 설정·`mic_agent`·모델·`outlook_sync`·신호 판정(radar) 테스트
- `test-results/` — 커버리지(`coverage.xml`)·JUnit(`junit.xml`)·HTML 리포트(`report.html`)

---

## 모델 출처

- 표정: MediaPipe Face Landmarker (+ 기하학적 AU 분석 폴백)
- 한숨: YAMNet(TFLite) 임베딩 + 경량 분류기(`classifier.npz`)
- 발화: faster-whisper(STT) + KoBERT(6종 감정 분류)
