#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
얼굴 감정 인식 시스템 (Flask 웹 스트리밍)
────────────────────────────────────────────────────────────────
라즈베리파이 5 + 카메라 모듈 / USB 웹캠

지원 감정:
  😐 일반 (Neutral)
  😠 분노 (Angry)
  😊 행복 (Happy)

감지 방식:
  MediaPipe FaceMesh → 468개 얼굴 랜드마크 추출 →
  기하학적 Action Unit 분석으로 감정 판별

실행 방법:
  python3 go_now.py            # 기본 카메라(0번)
  python3 go_now.py 1          # 카메라 인덱스 지정

접속 방법 (SSH 환경):
  브라우저에서 http://<라즈베리파이_IP>:5000 접속

종료: Ctrl+C
────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import sys
import os
import threading
from collections import deque, Counter
import json
import uuid
from datetime import datetime
from flask import Flask, Response, send_file, request, jsonify

# ─────────────────────────────────────────────────────────────
# PIL 한글 폰트 지원
# ─────────────────────────────────────────────────────────────
_USE_PIL = False
_KO_FONT = None
_KO_FONT_SM = None

try:
    from PIL import ImageFont, ImageDraw, Image as PILImage

    _FONT_CANDIDATES = [
        '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf',
        '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf',
    ]
    for _fp in _FONT_CANDIDATES:
        if os.path.exists(_fp):
            _KO_FONT    = ImageFont.truetype(_fp, 30)
            _KO_FONT_SM = ImageFont.truetype(_fp, 20)
            _USE_PIL = True
            print(f"[INFO] 한글 폰트 로드: {_fp}")
            break

    if not _USE_PIL:
        print("[WARNING] 한글 폰트 없음 → 영문 표시")
        print("          설치: sudo apt install fonts-nanum")

except ImportError:
    print("[WARNING] Pillow 없음 → 영문 표시")
    print("          설치: pip install Pillow")


# ─────────────────────────────────────────────────────────────
# 감정 설정
# ─────────────────────────────────────────────────────────────
EMOTION_EN    = {'일반': 'Neutral', '분노': 'Angry', '행복': 'Happy'}
EMOTION_COLOR = {           # BGR
    '일반': (  0, 200, 200),  # 노란색
    '분노': (  0,  50, 220),  # 빨간색
    '행복': (  0, 200,  80),  # 초록색
}

# ─────────────────────────────────────────────────────────────
# 감지 임계값
# ─────────────────────────────────────────────────────────────
SMILE_TH       = 0.012   # 베이스라인 대비 입꼬리 상승 최소치 (높을수록 행복 덜 예민)
SMILE_WIDTH_TH = 0.39    # 미소 시 입 가로 폭 기준 비율
EYE_SQUINT_TH  = 0.028   # 눈 찡그림 절대 임계값 (눈 개방도)
SMOOTH_WIN     = 75       # 시간축 스무딩 윈도우 (~2.5초 @ 30fps)
DETECT_CONF    = 0.5      # FaceMesh 감지/추적 신뢰도
DEBUG_OVERLAY  = True     # 화면에 수치 표시 (조정 완료 후 False로 변경)

# 적응형 베이스라인 버퍼 (최근 N프레임 기준)
#   각 사람의 무표정 기준값을 자동 추정
_BASELINE_BUF = 120       # 버퍼 크기 (~4초 @ 30fps)
_CALIB_MIN    = 40        # 최소 샘플 수 (이후부터 감정 판별 시작)

# ─────────────────────────────────────────────────────────────
# MediaPipe FaceMesh 주요 랜드마크 인덱스
#   좌/우는 카메라 기준(거울 방향)
# ─────────────────────────────────────────────────────────────
IDX = dict(
    L_MOUTH=61,  R_MOUTH=291,     # 입꼬리 (좌/우)
    MOUTH_T=13,  MOUTH_B=14,      # 입 위/아래 중심
    L_IBROW=107, R_IBROW=336,     # 내측 눈썹
    L_EYE_T=159, R_EYE_T=386,    # 눈 상단
    L_EYE_B=145, R_EYE_B=374,    # 눈 하단 (개방도 측정용)
)

# 적응형 베이스라인 버퍼 (전역 — 단일 얼굴 기준)
_ibrow_buf    = deque(maxlen=_BASELINE_BUF)
_brow_eye_buf = deque(maxlen=_BASELINE_BUF)
_smile_buf    = deque(maxlen=_BASELINE_BUF)
_width_buf    = deque(maxlen=_BASELINE_BUF)   # 입 가로 폭 베이스라인


# ─────────────────────────────────────────────────────────────
# 감정 분석 함수 (적응형 베이스라인 기반)
# ─────────────────────────────────────────────────────────────
def analyze_emotion(face_lms) -> tuple:
    """
    무표정 기준값을 자동 학습한 뒤 상대적 변화량으로 감정 판별.

    [행복] 입꼬리 상승 + 입 가로 폭 증가 (베이스라인 대비)
    [분노] 미간 눈썹 간격 축소 + 눈썹-눈 거리 축소 (베이스라인 대비)
    [일반] 위 조건 미해당 / 캘리브레이션 미완료

    Returns:
        (감정: str, 신뢰도: float 0~1, 캘리브레이션완료: bool)
    """
    global _ibrow_buf, _brow_eye_buf, _smile_buf, _width_buf

    lm = face_lms.landmark

    all_x = [l.x for l in lm]
    all_y = [l.y for l in lm]
    fw = max(all_x) - min(all_x)
    fh = max(all_y) - min(all_y)
    if fh < 0.01 or fw < 0.01:
        return '일반', 0.5, False

    gx = lambda i: lm[i].x
    gy = lambda i: lm[i].y

    # ── 원시 측정값 계산 ──────────────────────────────────────
    mouth_cy    = (gy(IDX['MOUTH_T']) + gy(IDX['MOUTH_B'])) / 2
    l_lift      = (mouth_cy - gy(IDX['L_MOUTH'])) / fh
    r_lift      = (mouth_cy - gy(IDX['R_MOUTH'])) / fh
    smile       = (l_lift + r_lift) / 2
    mouth_width = abs(gx(IDX['R_MOUTH']) - gx(IDX['L_MOUTH'])) / fw

    ibrow_dist  = abs(gx(IDX['L_IBROW']) - gx(IDX['R_IBROW'])) / fw
    l_be        = abs(gy(IDX['L_IBROW']) - gy(IDX['L_EYE_T'])) / fh
    r_be        = abs(gy(IDX['R_IBROW']) - gy(IDX['R_EYE_T'])) / fh
    brow_eye    = (l_be + r_be) / 2
    l_eye_h     = abs(gy(IDX['L_EYE_T']) - gy(IDX['L_EYE_B'])) / fh
    r_eye_h     = abs(gy(IDX['R_EYE_T']) - gy(IDX['R_EYE_B'])) / fh
    eye_open    = (l_eye_h + r_eye_h) / 2

    # ── 버퍼 업데이트 ─────────────────────────────────────────
    _ibrow_buf.append(ibrow_dist)
    _brow_eye_buf.append(brow_eye)
    _smile_buf.append(smile)
    _width_buf.append(mouth_width)

    if len(_ibrow_buf) < _CALIB_MIN:
        return '일반', 0.5, False   # 캘리브레이션 중

    # 버퍼 중 하나라도 비어있으면 안전하게 리턴
    if not _ibrow_buf or not _brow_eye_buf or not _smile_buf or not _width_buf:
        return '일반', 0.5, False

    # ── 적응형 베이스라인 ──────────────────────────────────────
    # 눈썹 간격/거리: 중립 상태 = 값이 큰 편 → 85th percentile 사용
    # 미소/입 폭: 중립 상태 = 낮은 편 → 35th percentile 사용
    base_ibrow    = float(np.percentile(_ibrow_buf,    85))
    base_brow_eye = float(np.percentile(_brow_eye_buf, 85))
    base_smile    = float(np.percentile(_smile_buf,    35))
    base_width    = float(np.percentile(_width_buf,    35))

    # ── 분노 점수 (베이스라인 대비 상대 변화율) ───────────────
    # 눈썹이 좁아질수록, 눈썹이 눈에 가까워질수록 양수
    ibrow_chg    = max(0.0, (base_ibrow    - ibrow_dist) / (base_ibrow    + 1e-6))
    brow_eye_chg = max(0.0, (base_brow_eye - brow_eye  ) / (base_brow_eye + 1e-6))
    eye_squint   = max(0.0, (EYE_SQUINT_TH - eye_open  ) / (EYE_SQUINT_TH + 1e-6))

    # 입이 넓게 벌어져 있으면(웃음) lip_depress 억제
    width_ratio  = (mouth_width - base_width) / (base_width + 1e-6)
    mouth_wide   = width_ratio > 0.06   # 베이스라인 대비 6% 이상 넓으면 웃음 상태
    lip_depress  = 0.0 if mouth_wide else \
                   max(0.0, (base_smile - smile - SMILE_TH) / (abs(base_smile) + SMILE_TH + 1e-6))

    anger_score  = (ibrow_chg    * 0.48
                  + brow_eye_chg * 0.37
                  + eye_squint   * 0.10
                  + lip_depress  * 0.05)

    # ── 행복 점수 (베이스라인 대비 상대 변화율) ───────────────
    smile_lift  = max(0.0, (smile - base_smile - SMILE_TH) / (SMILE_TH * 3 + 1e-6))
    width_score = max(0.0, width_ratio / 0.20)   # 베이스라인 대비 상대 폭 증가
    happy_score = smile_lift * 0.65 + width_score * 0.35

    # ── 감정 결정 ─────────────────────────────────────────────
    if happy_score > 0.10 and happy_score >= anger_score:
        conf = float(np.clip(0.55 + happy_score * 0.44, 0, 0.99))
        return '행복', conf, True, ibrow_chg, brow_eye_chg, anger_score, happy_score
    elif anger_score > 0.04 and ibrow_chg > 0.015:
        conf = float(np.clip(0.55 + anger_score * 0.44, 0, 0.99))
        return '분노', conf, True, ibrow_chg, brow_eye_chg, anger_score, happy_score
    else:
        return '일반', 0.70, True, ibrow_chg, brow_eye_chg, anger_score, happy_score


# ─────────────────────────────────────────────────────────────
# 그리기 유틸
# ─────────────────────────────────────────────────────────────
def draw_bar(img, color, pct, x, y, w, h=8):
    """신뢰도 막대(Progress bar) 그리기"""
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    bw = int(w * float(np.clip(pct, 0, 1)))
    if bw > 0:
        cv2.rectangle(img, (x, y), (x + bw, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)


def apply_text_overlays(img, overlays):
    """
    PIL을 사용해 한글 텍스트를 한 번에 오버레이 (효율 최적화).

    overlays: list of (text, (x, y), (B, G, R), font_size)
    """
    if not overlays:
        return img

    if _USE_PIL:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        draw    = ImageDraw.Draw(pil_img)
        for text, pos, bgr, fsize in overlays:
            font      = _KO_FONT if fsize >= 26 else _KO_FONT_SM
            rgb_color = (bgr[2], bgr[1], bgr[0])
            draw.text(pos, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        # Pillow 없을 때 영문 대체
        for text, pos, bgr, _ in overlays:
            eng = EMOTION_EN.get(text, text)
            cv2.putText(img, eng, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, bgr, 2, cv2.LINE_AA)
        return img


def open_camera(preferred_idx: int) -> cv2.VideoCapture:
    """카메라를 열고 실제 프레임이 들어오는지 확인."""
    indices = [preferred_idx] + [i for i in range(4) if i != preferred_idx]
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] 카메라 인덱스 {idx} 사용")
                return cap
            cap.release()
    raise RuntimeError("사용 가능한 카메라를 찾을 수 없습니다.")


# ─────────────────────────────────────────────────────────────
# Flask 앱 + 전역 상태
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
_frame_lock      = threading.Lock()
_latest_jpeg     = b''        # 최신 JPEG 프레임
_current_emotion = '일반'     # 신호등용 현재 감정 상태

_CALENDAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calendar_data.json')
_APPROVAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'approval_data.json')
_data_lock = threading.Lock()

# 감정 이력 (최대 10분 / 600개 @ 1초 간격)
_EMOTION_HISTORY = deque(maxlen=600)  # (timestamp_float, emotion_str)
_emotion_hist_lock = threading.Lock()
_last_record_ts = 0.0   # 마지막 기록 시각


def _load_json(path):
    with _data_lock:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
    return []


def _save_json(path, data):
    with _data_lock:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


HTML_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>팀장님 모니터링 대시보드</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}html,body{height:100%}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',Arial,sans-serif;
     display:flex;flex-direction:column;overflow:hidden}
header{background:#161b22;border-bottom:1px solid #30363d;padding:0 18px;
       height:46px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
header h1{font-size:.92rem;letter-spacing:1px}
.clk{font-size:.8rem;color:#8b949e}
.db{flex:1;display:flex;gap:10px;padding:10px;min-height:0;overflow:hidden}
.col-l,.col-r{display:flex;flex-direction:column;gap:10px;min-height:0}
.col-l{flex:0 0 42%}.col-r{flex:1;min-width:0}
.pn{background:#161b22;border:1px solid #30363d;border-radius:10px;
    padding:11px;display:flex;flex-direction:column;overflow:hidden}
.pt{font-size:.73rem;color:#8b949e;font-weight:700;letter-spacing:.5px;
    text-transform:uppercase;margin-bottom:8px;display:flex;
    justify-content:space-between;align-items:center}
.pn-cam{flex:1;min-height:0}
.pn-cam img{width:100%;border-radius:6px;display:block;background:#1c2128;min-height:60px}
.pn-sig{flex:0 0 auto}
.tl{display:flex;gap:12px;justify-content:center;padding:4px 0 5px}
.tlc{width:40px;height:40px;border-radius:50%;opacity:.15;transition:opacity .3s,box-shadow .3s}
.tlc.on{opacity:1}
.tlc.r{background:#ff3b30}.tlc.r.on{box-shadow:0 0 16px #ff3b30}
.tlc.o{background:#ff9500}.tlc.o.on{box-shadow:0 0 16px #ff9500}
.tlc.g{background:#34c759}.tlc.g.on{box-shadow:0 0 16px #34c759}
.sigtxt{text-align:center;font-size:.86rem;font-weight:600}
.emtxt{text-align:center;font-size:.75rem;color:#8b949e;margin-top:2px}
.pn-cal{flex:1;min-height:0}
.pn-ap{flex:0 0 170px}
.pn-go{flex:0 0 76px;align-items:center;justify-content:center}
.ilist{flex:1;overflow-y:auto}
.irow{display:flex;align-items:center;gap:6px;padding:6px 9px;
      border-radius:6px;border:1px solid #30363d;margin-bottom:5px;
      background:#1c2128;font-size:.78rem}
.ii{flex:1;min-width:0}
.it{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.im{color:#8b949e;font-size:.71rem;margin-top:2px}
.ia{display:flex;gap:3px;flex-shrink:0}
.drow{display:flex;gap:6px;align-items:center;margin-bottom:7px}
.drow input[type=date]{background:#1c2128;color:#e6edf3;border:1px solid #30363d;
  border-radius:5px;padding:3px 7px;font-size:.76rem}
.btn{padding:4px 10px;border:none;border-radius:5px;cursor:pointer;font-size:.74rem;transition:.15s}
.ba{background:#238636;color:#fff}.ba:hover{background:#2ea043}
.be{background:#1f6feb;color:#fff}.be:hover{background:#388bfd}
.bd{background:#da3633;color:#fff}.bd:hover{background:#f85149}
.bc{background:#21262d;color:#e6edf3}
.btn-calib{background:#6e40c9;color:#fff;border:none;border-radius:7px;
           margin-top:8px;width:100%;padding:7px;font-size:.83rem;cursor:pointer}
.btn-calib:hover{background:#8957e5}
.cmsg{text-align:center;font-size:.76rem;min-height:1em;margin-top:4px}
.btn-go{background:#238636;color:#fff;border:none;border-radius:10px;
        font-size:1.05rem;padding:16px;cursor:pointer;width:100%;
        letter-spacing:1px;transition:.2s}
.btn-go:hover{background:#2ea043}
.empty{color:#8b949e;text-align:center;padding:12px;font-size:.76rem}
.mo{display:none;position:fixed;inset:0;background:rgba(0,0,0,.78);
    z-index:300;align-items:center;justify-content:center}
.mo.open{display:flex}
.mc{background:#161b22;border:1px solid #30363d;border-radius:10px;
    padding:20px;min-width:300px;max-width:450px;width:92%}
.mc h2{margin-bottom:13px;font-size:.92rem}
.mc label{display:block;margin-bottom:9px}
.mc label span{display:block;color:#8b949e;font-size:.72rem;margin-bottom:3px}
.mc input,.mc textarea{width:100%;background:#0d1117;color:#e6edf3;
  border:1px solid #30363d;border-radius:5px;padding:6px;font-size:.8rem}
.mc textarea{resize:vertical;min-height:50px}
.mc-acts{display:flex;gap:7px;justify-content:flex-end;margin-top:12px}
.gstatus{text-align:center;padding:12px;border-radius:8px;font-size:.9rem;
         font-weight:600;margin-bottom:12px}
.gsr{background:rgba(255,59,48,.15);color:#ff3b30}
.gso{background:rgba(255,149,0,.15);color:#ff9500}
.gsg{background:rgba(52,199,89,.15);color:#34c759}
.aism{padding:6px 9px;background:#1c2128;border-radius:5px;margin-bottom:4px;font-size:.78rem}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:2px}
</style>
</head>
<body>
<header>
  <h1>🏢 팀장님 모니터링 대시보드</h1>
  <span class="clk" id="clk"></span>
</header>

<div class="db">
  <!-- ① 왼쪽: 카메라 + 결재 -->
  <div class="col-l">
    <div class="pn pn-cam">
      <div class="pt">
        <span>📷 얼굴 감정 인식</span>
        <span id="embdg" style="font-size:.83rem;color:#e6edf3"></span>
      </div>
      <img id="feed" src="/video_feed" alt="camera"
           onerror="setTimeout(()=>{this.src='/video_feed?t='+Date.now()},2000)">
      <button class="btn-calib" onclick="doCalib()">🔄 캘리브레이션 다시하기</button>
      <div class="cmsg" id="cmsg"></div>
    </div>
    <div class="pn pn-ap">
      <div class="pt">
        <span>📋 날짜별 결재 리스트</span>
        <button class="btn ba" onclick="openApMo()">+ 추가</button>
      </div>
      <div class="drow"><input type="date" id="ap-date" onchange="rendAp()"></div>
      <div class="ilist" id="ap-list"></div>
    </div>
  </div>

  <!-- ② 오른쪽: 신호등 + 캘린더 + 버튼 -->
  <div class="col-r">
    <div class="pn pn-sig">
      <div class="pt">🚦 현재 방문 가능 상태</div>
      <div class="tl">
        <div class="tlc r" id="tl-r"></div>
        <div class="tlc o" id="tl-o"></div>
        <div class="tlc g" id="tl-g"></div>
      </div>
      <div class="sigtxt" id="sigtxt">–</div>
      <div class="emtxt"  id="emtxt"></div>
    </div>
    <div class="pn pn-cal">
      <div class="pt">
        <span>📅 팀장님 캘린더</span>
        <button class="btn ba" onclick="openCalMo()">+ 추가</button>
      </div>
      <div class="drow"><input type="date" id="cal-date" onchange="rendCal()"></div>
      <div class="ilist" id="cal-list"></div>
    </div>
    <div class="pn pn-go">
      <button class="btn-go" onclick="openGoMo()">🚀 지금 갈까요?</button>
    </div>
  </div>
</div>

<!-- 캘린더 모달 -->
<div class="mo" id="cal-mo">
  <div class="mc">
    <h2 id="cal-mo-h">일정 추가</h2>
    <input type="hidden" id="cal-eid">
    <label><span>날짜</span><input type="date" id="cal-fd"></label>
    <label><span>시간</span><input type="time" id="cal-ft"></label>
    <label><span>제목 *</span><input type="text" id="cal-ftt" placeholder="일정 제목"></label>
    <label><span>메모</span><textarea id="cal-fn" placeholder="메모 (선택)"></textarea></label>
    <div class="mc-acts">
      <button class="btn bc" onclick="closeMo('cal-mo')">취소</button>
      <button class="btn ba" onclick="saveCal()">저장</button>
    </div>
  </div>
</div>

<!-- 결재 모달 -->
<div class="mo" id="ap-mo">
  <div class="mc">
    <h2 id="ap-mo-h">결재 추가</h2>
    <input type="hidden" id="ap-eid">
    <label><span>날짜</span><input type="date" id="ap-fd"></label>
    <label><span>제목 *</span><input type="text" id="ap-ftt" placeholder="결재 제목"></label>
    <label><span>메모</span><textarea id="ap-fn" placeholder="메모 (선택)"></textarea></label>
    <div class="mc-acts">
      <button class="btn bc" onclick="closeMo('ap-mo')">취소</button>
      <button class="btn ba" onclick="saveAp()">저장</button>
    </div>
  </div>
</div>

<!-- 지금갈까요 모달 -->
<div class="mo" id="go-mo">
  <div class="mc">
    <h2>🚀 지금 갈까요?</h2>
    <div id="go-sdiv"></div>
    <div style="font-size:.8rem;font-weight:700;margin-bottom:6px">📋 오늘의 결재 리스트</div>
    <div id="go-alist" style="max-height:210px;overflow-y:auto"></div>
    <div class="mc-acts">
      <button class="btn bc" onclick="closeMo('go-mo')">닫기</button>
    </div>
  </div>
</div>

<script>
const _td=()=>new Date().toISOString().split('T')[0];
document.getElementById('cal-date').value=_td();
document.getElementById('ap-date').value=_td();

// 시계
(function tk(){
  document.getElementById('clk').textContent=new Date().toLocaleString('ko-KR',
    {year:'numeric',month:'2-digit',day:'2-digit',
     hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  setTimeout(tk,1000);
})();

// 신호등
const SMAP={
  red:   ['r','🔴 분노 상태 — 방문 자제'],
  orange:['o','🟠 주의 — 1시간 내 일정 있음'],
  green: ['g','🟢 방문 가능'],
};
const EMKO={'일반':'😐 일반','행복':'😊 행복','분노':'😠 분노'};
let curSig='green';
function setSig(s){
  curSig=s;
  ['r','o','g'].forEach(c=>document.getElementById('tl-'+c).classList.remove('on'));
  const [c,t]=SMAP[s]||SMAP.green;
  document.getElementById('tl-'+c).classList.add('on');
  document.getElementById('sigtxt').textContent=t;
}
async function pollSig(){
  try{
    const d=await fetch('/api/status').then(r=>r.json());
    setSig(d.signal);
    const et=EMKO[d.emotion]||d.emotion;
    document.getElementById('embdg').textContent=et;
    document.getElementById('emtxt').textContent='현재 감정: '+et;
  }catch(e){}
}
pollSig();setInterval(pollSig,2000);

// 캘린더
let calD=[];
async function loadCal(){calD=await fetch('/api/calendar').then(r=>r.json());rendCal();}
function rendCal(){
  const dt=document.getElementById('cal-date').value;
  const ls=calD.filter(e=>e.date===dt).sort((a,b)=>(a.time||'').localeCompare(b.time||''));
  const el=document.getElementById('cal-list');
  if(!ls.length){el.innerHTML='<div class=empty>일정 없음</div>';return;}
  el.innerHTML=ls.map(e=>
    '<div class=irow>'+
    '<div class=ii>'+
    '<div class=it>'+esc(e.title)+'</div>'+
    '<div class=im>'+(e.time||'시간 미정')+(e.note?' · '+esc(e.note):'')+'</div>'+
    '</div>'+
    '<div class=ia>'+
    '<button class="btn be" onclick="editCal(\''+e.id+'\')">수정</button>'+
    '<button class="btn bd" onclick="delCal(\''+e.id+'\')">삭제</button>'+
    '</div></div>'
  ).join('');
}
function openCalMo(id){
  document.getElementById('cal-mo-h').textContent=id?'일정 수정':'일정 추가';
  document.getElementById('cal-eid').value=id||'';
  const ev=id&&calD.find(e=>e.id===id);
  document.getElementById('cal-fd').value =ev?ev.date:document.getElementById('cal-date').value;
  document.getElementById('cal-ft').value =ev?ev.time||'':'';
  document.getElementById('cal-ftt').value=ev?ev.title:'';
  document.getElementById('cal-fn').value =ev?ev.note||'':'';
  openMo('cal-mo');
}
function editCal(id){openCalMo(id);}
async function saveCal(){
  const id=document.getElementById('cal-eid').value;
  const b={date:document.getElementById('cal-fd').value,
            time:document.getElementById('cal-ft').value,
            title:document.getElementById('cal-ftt').value.trim(),
            note:document.getElementById('cal-fn').value.trim()};
  if(!b.title||!b.date){alert('날짜와 제목을 입력하세요.');return;}
  await fetch(id?'/api/calendar/'+id:'/api/calendar',
    {method:id?'PUT':'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  closeMo('cal-mo');await loadCal();
}
async function delCal(id){
  if(!confirm('일정을 삭제하시겠습니까?'))return;
  await fetch('/api/calendar/'+id,{method:'DELETE'});await loadCal();
}

// 결재
let apD=[];
async function loadAp(){apD=await fetch('/api/approvals').then(r=>r.json());rendAp();}
function rendAp(){
  const dt=document.getElementById('ap-date').value;
  const ls=apD.filter(a=>a.date===dt);
  const el=document.getElementById('ap-list');
  if(!ls.length){el.innerHTML='<div class=empty>결재 항목 없음</div>';return;}
  el.innerHTML=ls.map(a=>
    '<div class=irow>'+
    '<div class=ii>'+
    '<div class=it>'+esc(a.title)+'</div>'+
    (a.note?'<div class=im>'+esc(a.note)+'</div>':'')+
    '</div>'+
    '<div class=ia>'+
    '<button class="btn be" onclick="editAp(\''+a.id+'\')">수정</button>'+
    '<button class="btn bd" onclick="delAp(\''+a.id+'\')">삭제</button>'+
    '</div></div>'
  ).join('');
}
function openApMo(id){
  document.getElementById('ap-mo-h').textContent=id?'결재 수정':'결재 추가';
  document.getElementById('ap-eid').value=id||'';
  const ap=id&&apD.find(a=>a.id===id);
  document.getElementById('ap-fd').value =ap?ap.date:document.getElementById('ap-date').value;
  document.getElementById('ap-ftt').value=ap?ap.title:'';
  document.getElementById('ap-fn').value =ap?ap.note||'':'';
  openMo('ap-mo');
}
function editAp(id){openApMo(id);}
async function saveAp(){
  const id=document.getElementById('ap-eid').value;
  const b={date:document.getElementById('ap-fd').value,
            title:document.getElementById('ap-ftt').value.trim(),
            note:document.getElementById('ap-fn').value.trim()};
  if(!b.title||!b.date){alert('날짜와 제목을 입력하세요.');return;}
  await fetch(id?'/api/approvals/'+id:'/api/approvals',
    {method:id?'PUT':'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  closeMo('ap-mo');await loadAp();
}
async function delAp(id){
  if(!confirm('결재 항목을 삭제하시겠습니까?'))return;
  await fetch('/api/approvals/'+id,{method:'DELETE'});await loadAp();
}

// 지금 갈까요
async function openGoMo(){
  const d=await fetch('/api/status').then(r=>r.json());
  const gm={
    red:   ['gsr','🔴 위험! 팀장님이 분노 상태입니다. 방문을 자제하세요.'],
    orange:['gso','🟠 주의! 1시간 내 팀장님 일정이 있습니다. 신중하게 방문하세요.'],
    green: ['gsg','🟢 방문하기 좋은 상태입니다!'],
  };
  const [c,m]=gm[d.signal]||gm.green;
  document.getElementById('go-sdiv').innerHTML='<div class="gstatus '+c+'">'+m+'</div>';
  const tod=_td();
  const ls=apD.filter(a=>a.date===tod);
  document.getElementById('go-alist').innerHTML=ls.length?
    ls.map(a=>'<div class=aism><strong>'+esc(a.title)+'</strong>'+
      (a.note?'<br><span style="color:#8b949e;font-size:.74rem">'+esc(a.note)+'</span>':'')+
      '</div>').join(''):
    '<div class=empty>오늘 결재 항목 없음</div>';
  openMo('go-mo');
}

// 캘리브레이션
async function doCalib(){
  const msg=document.getElementById('cmsg');
  msg.style.color='#ffcc00';msg.textContent='캘리브레이션 초기화 중...';
  try{
    const d=await fetch('/recalibrate',{method:'POST'}).then(r=>r.json());
    msg.style.color='#66ff88';msg.textContent='✅ '+d.message;
    setTimeout(()=>{msg.textContent='';},3000);
  }catch{msg.style.color='#ff6666';msg.textContent='❌ 요청 실패';}
}

// 유틸
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function openMo(id){document.getElementById(id).classList.add('open');}
function closeMo(id){document.getElementById(id).classList.remove('open');}
document.querySelectorAll('.mo').forEach(el=>{
  el.addEventListener('click',e=>{if(e.target===el)el.classList.remove('open');});
});
loadCal();loadAp();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html'))


def _mjpeg_generator():
    global _latest_jpeg
    while True:
        with _frame_lock:
            frame_data = _latest_jpeg
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.03)   # ~30fps 상한


@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    global _ibrow_buf, _brow_eye_buf, _smile_buf, _width_buf
    _ibrow_buf.clear()
    _brow_eye_buf.clear()
    _smile_buf.clear()
    _width_buf.clear()
    print('[INFO] 캘리브레이션 초기화')
    return {'message': '무표정으로 1~2초 유지하세요'}


@app.route('/video_feed')
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ─────────────────────────────────────────────────────────────
# REST API 엔드포인트
# ─────────────────────────────────────────────────────────────
@app.route('/api/emotion_history')
def api_emotion_history():
    """최근 감정 이력 반환 (최대 600초치)"""
    minutes = request.args.get('minutes', 5, type=int)
    minutes = max(1, min(minutes, 10))
    cutoff  = time.time() - minutes * 60
    with _emotion_hist_lock:
        data = [(ts, em) for ts, em in _EMOTION_HISTORY if ts >= cutoff]
    return jsonify([{'t': round(ts, 2), 'e': em} for ts, em in data])


@app.route('/api/status')
def api_status():
    global _current_emotion
    em = _current_emotion
    if em == '분노':
        signal = 'red'
    elif em == '행복':
        signal = 'green'
    else:
        events = _load_json(_CALENDAR_FILE)
        now    = datetime.now()
        today  = now.strftime('%Y-%m-%d')
        signal = 'green'
        for ev in events:
            if ev.get('date') != today:
                continue
            try:
                ev_dt = datetime.strptime(f"{today} {ev.get('time','00:00')}", '%Y-%m-%d %H:%M')
                diff  = (ev_dt - now).total_seconds()
                if 0 <= diff <= 3600:
                    signal = 'orange'
                    break
            except Exception:
                pass
    return jsonify({'emotion': em, 'signal': signal})


@app.route('/api/calendar', methods=['GET', 'POST'])
def api_calendar():
    if request.method == 'GET':
        return jsonify(_load_json(_CALENDAR_FILE))
    data   = request.get_json(force=True)
    events = _load_json(_CALENDAR_FILE)
    new_ev = {
        'id':    str(uuid.uuid4()),
        'date':  data.get('date',  ''),
        'time':  data.get('time',  ''),
        'title': data.get('title', ''),
        'note':  data.get('note',  ''),
    }
    events.append(new_ev)
    _save_json(_CALENDAR_FILE, events)
    return jsonify(new_ev), 201


@app.route('/api/calendar/<eid>', methods=['PUT', 'DELETE'])
def api_calendar_item(eid):
    events = _load_json(_CALENDAR_FILE)
    if request.method == 'DELETE':
        events = [e for e in events if e.get('id') != eid]
        _save_json(_CALENDAR_FILE, events)
        return jsonify({'ok': True})
    data = request.get_json(force=True)
    for ev in events:
        if ev.get('id') == eid:
            for k in ('date', 'time', 'title', 'note'):
                if k in data:
                    ev[k] = data[k]
            break
    _save_json(_CALENDAR_FILE, events)
    return jsonify({'ok': True})


@app.route('/api/approvals', methods=['GET', 'POST'])
def api_approvals():
    if request.method == 'GET':
        return jsonify(_load_json(_APPROVAL_FILE))
    data  = request.get_json(force=True)
    items = _load_json(_APPROVAL_FILE)
    new_ap = {
        'id':    str(uuid.uuid4()),
        'date':  data.get('date',  ''),
        'title': data.get('title', ''),
        'note':  data.get('note',  ''),
    }
    items.append(new_ap)
    _save_json(_APPROVAL_FILE, items)
    return jsonify(new_ap), 201


@app.route('/api/approvals/<aid>', methods=['PUT', 'DELETE'])
def api_approvals_item(aid):
    items = _load_json(_APPROVAL_FILE)
    if request.method == 'DELETE':
        items = [a for a in items if a.get('id') != aid]
        _save_json(_APPROVAL_FILE, items)
        return jsonify({'ok': True})
    data = request.get_json(force=True)
    for ap in items:
        if ap.get('id') == aid:
            for k in ('date', 'title', 'note'):
                if k in data:
                    ap[k] = data[k]
            break
    _save_json(_APPROVAL_FILE, items)
    return jsonify({'ok': True})


# ─────────────────────────────────────────────────────────────
# 카메라 처리 루프 (별도 스레드)
# ─────────────────────────────────────────────────────────────
def camera_loop(cam_idx: int):
    global _latest_jpeg, _ibrow_buf, _brow_eye_buf, _smile_buf, _width_buf, _current_emotion

    try:
        cap = open_camera(cam_idx)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("  • 카메라 연결 상태 확인")
        print("  • python3 go_now.py 1  (다른 인덱스 시도)")
        print("  • v4l2-ctl --list-devices  (장치 목록 확인)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 해상도: {W}×{H}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=DETECT_CONF,
        min_tracking_confidence=DETECT_CONF,
    )

    em_hist = [deque(maxlen=SMOOTH_WIN) for _ in range(3)]
    prev_t  = time.time()
    fps_buf = deque(maxlen=30)
    _frame_skip   = 0
    _SKIP_EVERY   = 2          # 매 N프레임마다 MediaPipe 실행
    _last_result  = None       # 이전 FaceMesh 결과 재사용

    print("\n[INFO] 감정 인식 시작!")
    print("─" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임 읽기 실패")
            break

        frame = cv2.flip(frame, 1)
        fH, fW = frame.shape[:2]

        _frame_skip += 1
        if _frame_skip >= _SKIP_EVERY:
            _frame_skip = 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            _last_result = face_mesh.process(rgb)
            rgb.flags.writeable = True
        result = _last_result

        ko_overlays = []

        if result is not None and result.multi_face_landmarks:
            for fi, face_lms in enumerate(result.multi_face_landmarks):

                xs = [int(l.x * fW) for l in face_lms.landmark]
                ys = [int(l.y * fH) for l in face_lms.landmark]
                bx1 = max(0,   min(xs) - 15)
                by1 = max(0,   min(ys) - 20)
                bx2 = min(fW,  max(xs) + 15)
                by2 = min(fH,  max(ys) + 10)
                bw  = bx2 - bx1

                res = analyze_emotion(face_lms)
                raw_em, raw_cf, calibrated = res[0], res[1], res[2]
                dbg = res[3:] if len(res) > 3 else (0, 0, 0, 0)

                if not calibrated:
                    # 캘리브레이션 중 — 진행률 표시
                    pct = len(_ibrow_buf) / _CALIB_MIN
                    bar_w = bx2 - bx1
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (180,180,0), 2)
                    draw_bar(frame, (0, 200, 255), pct, bx1, by2 + 5, bar_w)
                    ko_overlays.append(('캘리브레이션 중...', (bx1, by2 + 18), (0, 200, 255), 20))
                    continue

                hi = min(fi, 2)
                em_hist[hi].append(raw_em)
                cnt = Counter(em_hist[hi])
                em  = cnt.most_common(1)[0][0]
                cf  = cnt[em] / len(em_hist[hi]) * 0.5 + raw_cf * 0.5
                _current_emotion = em   # 신호등 상태 업데이트

                # 1초마다 감정 이력 기록
                global _last_record_ts
                now_ts = time.time()
                if now_ts - _last_record_ts >= 1.0:
                    with _emotion_hist_lock:
                        _EMOTION_HISTORY.append((now_ts, em))
                    _last_record_ts = now_ts

                col = EMOTION_COLOR[em]

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)

                label = f"{EMOTION_EN[em]}  {cf * 100:.0f}%"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)

                lbg_x2 = min(bx1 + tw + 12, fW)
                lbg_y1 = max(by1 - th - 14, 0)
                cv2.rectangle(frame,
                              (bx1, lbg_y1), (lbg_x2, by1), col, -1)
                cv2.putText(frame, label, (bx1 + 5, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2, cv2.LINE_AA)

                ko_y = lbg_y1 - 36 if lbg_y1 > 36 else by2 + 14
                ko_overlays.append((em, (bx1, ko_y), col, 30))

                draw_bar(frame, col, cf, bx1, by2 + 5, bw)

                # ── 디버그 수치 오버레이 ──────────────────────
                # if DEBUG_OVERLAY and len(dbg) == 4:
                #     ibrow_v, brow_v, ang_v, hap_v = dbg
                #     lines = [
                #         f"ibrow_chg:  {ibrow_v:.3f}",
                #         f"brow_eye:   {brow_v:.3f}",
                #         f"anger_scr:  {ang_v:.3f}",
                #         f"happy_scr:  {hap_v:.3f}",
                #     ]
                #     dy = by1 + 18
                #     for ln in lines:
                #         cv2.putText(frame, ln, (bx2 + 6, dy),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                #                     (200, 200, 200), 1, cv2.LINE_AA)
                #         dy += 16

        frame = apply_text_overlays(frame, ko_overlays)

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_t, 1e-9))
        prev_t = now
        fps = float(np.mean(fps_buf))

        cv2.putText(frame, f"FPS {fps:.1f}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        ly = fH - 82
        for em_name, c in EMOTION_COLOR.items():
            cv2.rectangle(frame,
                          (fW - 130, ly), (fW - 115, ly + 14), c, -1)
            cv2.putText(frame, EMOTION_EN[em_name], (fW - 110, ly + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            ly += 22

        # JPEG 인코딩 후 공유 버퍼에 저장
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _frame_lock:
                _latest_jpeg = buf.tobytes()

    cap.release()
    face_mesh.close()


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # 카메라 루프를 백그라운드 스레드로 실행
    t = threading.Thread(target=camera_loop, args=(cam_idx,), daemon=True)
    t.start()

    # 라즈베리파이 IP 안내
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "라즈베리파이_IP"

    print(f"\n[INFO] 웹 브라우저에서 접속:")
    print(f"       http://{ip}:5000")
    print(f"       종료: Ctrl+C")
    print("─" * 40)

    # Flask 서버 시작 (외부 접속 허용)
    app.run(host='0.0.0.0', port=5000, threaded=True)


if __name__ == "__main__":
    main()
