#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# 결재 레이더 — 원샷 설치 스크립트
#
#   ./setup.sh            # 전체(cam+mic, --role all 용)
#   ./setup.sh cam        # cam 기기만
#   ./setup.sh mic        # mic 기기만
#
# 하는 일:
#   1) 시스템 패키지(apt): PortAudio / 한글폰트 / ffmpeg 등
#   2) pip 패키지: 역할에 맞는 requirements 설치
#   3) 모델: face_landmarker.task(cam, 선택) 다운로드 + YAMNet 모델 확인(mic)
#
# 가상환경을 쓴다면 먼저 활성화하고 실행하세요:
#   source /path/to/venv/bin/activate && ./setup.sh all
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROLE="${1:-all}"
case "$ROLE" in
  cam|mic|all) ;;
  *) echo "사용법: ./setup.sh [cam|mic|all]"; exit 1 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

echo "=================================================="
echo "  결재 레이더 설치 — role=$ROLE"
echo "  python = $($PY --version 2>&1)"
echo "=================================================="

# ── 1) 시스템 패키지 (apt) ───────────────────────────────────
APT_PKGS=()
[[ "$ROLE" == "cam" || "$ROLE" == "all" ]] && APT_PKGS+=(fonts-nanum)
[[ "$ROLE" == "mic" || "$ROLE" == "all" ]] && APT_PKGS+=(libportaudio2 ffmpeg)

if command -v apt-get >/dev/null 2>&1 && ((${#APT_PKGS[@]})); then
  echo ">> apt 패키지 설치: ${APT_PKGS[*]}"
  sudo apt-get update
  sudo apt-get install -y "${APT_PKGS[@]}"
else
  echo ">> apt 생략(또는 apt 없음). 다음 패키지를 수동 확인하세요: ${APT_PKGS[*]:-없음}"
fi

# ── 2) pip 패키지 ────────────────────────────────────────────
echo ">> pip 업그레이드"
$PY -m pip install --upgrade pip

REQ="$HERE/requirements-$ROLE.txt"
[[ "$ROLE" == "all" ]] && REQ="$HERE/requirements.txt"
echo ">> pip 설치: $REQ"
$PY -m pip install -r "$REQ"

# tflite-runtime 설치 실패 시 tensorflow 폴백 (mic/all)
if [[ "$ROLE" == "mic" || "$ROLE" == "all" ]]; then
  if ! $PY -c "import tflite_runtime" >/dev/null 2>&1; then
    echo ">> tflite-runtime 미설치 — tensorflow 로 폴백 설치 시도"
    $PY -m pip install "tensorflow>=2.13" || \
      echo "   ⚠ tensorflow 설치 실패. 한숨 감지를 쓰려면 tflite-runtime 또는 tensorflow 를 직접 설치하세요."
  fi
fi

# ── 3) 모델 파일 ─────────────────────────────────────────────
# (cam) face_landmarker.task — 표정 정확도 향상(선택). 없으면 다운로드.
if [[ "$ROLE" == "cam" || "$ROLE" == "all" ]]; then
  TASK="$HERE/face_landmarker.task"
  if [[ -f "$TASK" ]]; then
    echo ">> face_landmarker.task 이미 있음 (skip)"
  else
    echo ">> face_landmarker.task 다운로드(선택, 표정 정확도 향상)"
    URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if command -v wget >/dev/null 2>&1; then
      wget -q -O "$TASK" "$URL" || echo "   ⚠ 다운로드 실패(선택 항목이라 무시 가능)"
    elif command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "$TASK" "$URL" || echo "   ⚠ 다운로드 실패(선택 항목이라 무시 가능)"
    else
      echo "   ⚠ wget/curl 없음 — 수동 다운로드: $URL"
    fi
  fi
fi

# (mic) YAMNet — 한숨 감지에 필수. 자동 다운로드 대신 존재 확인만.
if [[ "$ROLE" == "mic" || "$ROLE" == "all" ]]; then
  MISSING=()
  [[ -f "$HERE/model/yamnet.tflite" ]]        || MISSING+=("model/yamnet.tflite")
  [[ -f "$HERE/model/yamnet_class_map.csv" ]] || MISSING+=("model/yamnet_class_map.csv")
  if ((${#MISSING[@]})); then
    echo ">> ⚠ YAMNet 모델 누락: ${MISSING[*]}"
    echo "   한숨 감지에 필요합니다. 기존 sigh_detector 셋업(setup_mode_a.sh)에서 받은 파일을"
    echo "   model/ 폴더에 두거나, 없으면 mic_agent 를 --no-sigh 로 실행하세요."
  else
    echo ">> YAMNet 모델 확인 완료"
  fi
fi

echo "=================================================="
echo "  설치 완료 (role=$ROLE)"
echo "  실행 예:"
[[ "$ROLE" == "all" ]] && echo "    python3 run_all.py --role all 0 --device 1"
[[ "$ROLE" == "cam" ]] && echo "    python3 run_all.py --role cam 0"
[[ "$ROLE" == "mic" ]] && echo "    python3 run_all.py --role mic --radar-host <cam_IP> --device 1"
echo "  설치 점검:  python3 check_devices.py"
echo "=================================================="
