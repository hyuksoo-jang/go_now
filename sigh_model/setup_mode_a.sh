#!/bin/bash
# ============================================================
# 모드 A 설치 스크립트 — YAMNet TFLite 전용
# Python 3.11 + Raspberry Pi 5 (aarch64)
# ============================================================
set -e

echo "========================================"
echo "  모드 A 환경 설치 (YAMNet TFLite)"
echo "========================================"

echo ""
echo ">>> [1/4] 시스템 패키지 설치"
sudo apt-get update -qq
sudo apt-get install -y \
    portaudio19-dev \
    libatlas-base-dev \
    libopenblas-dev

echo ""
echo ">>> [2/4] Python 가상환경 생성"
python3 -m venv venv_mode_a --system-site-packages
source venv_mode_a/bin/activate

echo ""
echo ">>> [3/4] Python 패키지 설치"
pip install --upgrade pip -q
# tflite-runtime 2.14: Python 3.11 aarch64 휠 공식 지원
pip install tflite-runtime pyaudio numpy scipy -q
echo "✅ 패키지 설치 완료"

echo ""
echo ">>> [4/4] YAMNet 모델 + 레이블 다운로드"
mkdir -p model

echo "   YAMNet TFLite 모델 다운로드 중... (3.7MB)"
wget -q --show-progress \
    -O model/yamnet.tflite \
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_classification_tflite_1.tflite" \
  && echo "✅ 모델 다운로드 완료" \
  || { echo "❌ 모델 다운로드 실패. 인터넷 연결을 확인하세요."; exit 1; }

echo "   레이블 다운로드 중..."
wget -q --show-progress \
    -O model/yamnet_class_map.csv \
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv" \
  && echo "✅ 레이블 다운로드 완료" \
  || { echo "❌ 레이블 다운로드 실패."; exit 1; }

echo ""
echo "========================================"
echo "✅ 설치 완료! 실행 방법:"
echo ""
echo "   source venv_mode_a/bin/activate"
echo "   python sigh_detector_mode_a.py"
echo "========================================"
