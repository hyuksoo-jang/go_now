#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
얼굴 감정 인식 시스템 v2 (Flask 웹 스트리밍) — 인식률 개선판
────────────────────────────────────────────────────────────────
라즈베리파이 5 + 카메라 모듈 / USB 웹캠

v2 주요 개선사항:
  ① Tier-1: MediaPipe Face Landmarker (Tasks API)
            52개 blendshape 계수를 직접 활용 — 딥러닝 모델 기반 (정확도 최고)
            모델 다운로드:
              wget -O face_landmarker.task \\
                https://storage.googleapis.com/mediapipe-models/\\
face_landmarker/face_landmarker/float16/1/face_landmarker.task
  ② Tier-2: FaceMesh + 확장 AU 기하학적 분석 (5개→12개+ AU)
            Tier-1 없을 때 단독 사용 / 있을 때 앙상블 보조
  ③ Duchenne 미소 마커 — 눈 찡그림의 맥락(행복 vs 분노) 구분
  ④ AU별 독립 적응형 베이스라인으로 개인화 정확도 향상
  ⑤ 시간축 스무딩 + 투표 앙상블로 안정적 감정 출력

실행: python3 go_now_v2.py [카메라_인덱스]
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
# Tier-1: MediaPipe Face Landmarker (Tasks API + 52 blendshapes)
# ─────────────────────────────────────────────────────────────
_USE_BLENDSHAPES = False   # face_landmarker.task 있을 때 True로 전환
_face_landmarker = None    # FaceLandmarker 인스턴스


def _init_face_landmarker():
    """face_landmarker.task 파일이 있으면 Tasks API로 초기화."""
    global _USE_BLENDSHAPES, _face_landmarker
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
    if not os.path.exists(model_path):
        return   # 조용히 스킵 — Tier-2 폴백
    try:
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _mp_vision
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        _face_landmarker  = _mp_vision.FaceLandmarker.create_from_options(options)
        _USE_BLENDSHAPES  = True
        print("[INFO] ✅ Tier-1 Face Landmarker (52 blendshapes) 활성화")
    except Exception as _exc:
        print(f"[WARNING] Face Landmarker 초기화 실패 ({_exc}) → Tier-2 사용")


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

_init_face_landmarker()
if not _USE_BLENDSHAPES:
    print("[INFO] face_landmarker.task 없음 → Tier-2(AU 기하학적) 사용")
    print("       정확도 향상 방법:")
    print("       wget -O face_landmarker.task \\")
    print("         https://storage.googleapis.com/mediapipe-models/"
          "face_landmarker/face_landmarker/float16/1/face_landmarker.task")


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
# 공통 설정
# ─────────────────────────────────────────────────────────────
FRONTAL_YAW_TH   = 0.25   # 좌우 고개 편차 허용 비율
FRONTAL_PITCH_TH = 0.22   # 상하 고개 편차 허용 비율
SMOOTH_WIN       = 60     # 시간축 스무딩 윈도우 (~2초 @ 30fps)
DETECT_CONF      = 0.45   # FaceMesh 감지/추적 신뢰도


# ═════════════════════════════════════════════════════════════
# ■ Tier-1: 52 Blendshape 기반 감정 분석 (높은 정확도)
# ═════════════════════════════════════════════════════════════
_BS_BUFS      = {}   # {blendshape_name: deque}
_BS_BASELINE  = {}   # 개인화 베이스라인
_BS_CALIB_MIN = 35
_BS_LOCK      = __import__('threading').Lock()

_ALL_TRACKED_BS = [
    # 행복 신호
    'mouthSmileLeft', 'mouthSmileRight',
    'cheekSquintLeft', 'cheekSquintRight',
    'eyeSquintLeft', 'eyeSquintRight',
    # 분노 신호 (FACS: AU4 + AU5 + AU7 + AU23)
    'browDownLeft', 'browDownRight',        # AU4: 눈썹 내림  ★ 핵심
    'eyeWideLeft', 'eyeWideRight',          # AU5: 눈 크게 뜸 (분노 응시)
    'mouthPressLeft', 'mouthPressRight',    # AU23: 입술 조임
    'noseSneerLeft', 'noseSneerRight',
    'mouthFrownLeft', 'mouthFrownRight',
]


def _bs_update_baseline(bs_dict):
    with _BS_LOCK:
        for k in _ALL_TRACKED_BS:
            v = bs_dict.get(k, 0.0)
            if k not in _BS_BUFS:
                _BS_BUFS[k] = deque(maxlen=180)
            _BS_BUFS[k].append(v)
        if len(_BS_BUFS.get('mouthSmileLeft', [])) >= _BS_CALIB_MIN:
            for k, buf in list(_BS_BUFS.items()):
                _BS_BASELINE[k] = float(np.percentile(buf, 25))


def analyze_blendshapes(bs_dict):
    """
    52개 blendshape 계수로 감정 판별 (Tier-1).
    MediaPipe 딥러닝 모델이 추출한 계수이므로 기하학적 계산보다 정확.
    Returns: (emotion_ko, confidence, calibrated, debug_dict)
    """
    _bs_update_baseline(bs_dict)
    if len(_BS_BUFS.get('mouthSmileLeft', [])) < _BS_CALIB_MIN:
        pct = len(_BS_BUFS.get('mouthSmileLeft', [])) / _BS_CALIB_MIN
        return '일반', 0.5, False, {'calib_pct': pct}

    def adj(name):
        """개인 베이스라인 제거 후 값 (음수→0 클램프)."""
        return max(0.0, bs_dict.get(name, 0.0) - _BS_BASELINE.get(name, 0.0))

    # ── 분노/행복 원시값 ───────────────────────────────────────
    smile      = (adj('mouthSmileLeft')  + adj('mouthSmileRight'))  / 2
    cheek      = (adj('cheekSquintLeft') + adj('cheekSquintRight')) / 2
    eye_sq     = (adj('eyeSquintLeft')   + adj('eyeSquintRight'))   / 2
    brow_down  = (adj('browDownLeft')    + adj('browDownRight'))    / 2  # AU4 ★
    eye_wide   = (adj('eyeWideLeft')     + adj('eyeWideRight'))     / 2  # AU5
    lip_press  = (adj('mouthPressLeft')  + adj('mouthPressRight'))  / 2  # AU23
    nose_sneer = (adj('noseSneerLeft')   + adj('noseSneerRight'))   / 2

    # ── 행복 신호 ─────────────────────────────────────────────
    smile_gate  = float(min(1.0, smile * 8.0))
    happy_raw   = smile * 0.70 + (cheek * smile_gate) * 0.20 + (eye_sq * smile_gate) * 0.10
    brow_veto   = float(min(1.0, brow_down * 12.0))
    happy_score = happy_raw * (1.0 - brow_veto)

    # ── 분노 신호 (FACS: AU4+AU5+AU23) ──────────────────────
    anger_gate  = float(max(0.0, 1.0 - smile * 8.0))
    eye_sq_ang  = eye_sq * anger_gate    # AU7: smile 없을 때만
    # eye_wide(AU5 응시)는 smile 없을 때 분노 신호, 있을 때는 무시
    eye_wide_ang = eye_wide * anger_gate
    angry_score = (brow_down   * 0.50    # AU4: 눈썹 내림  ★ 최우선
                 + eye_wide_ang * 0.20   # AU5: 분노 응시
                 + lip_press   * 0.15    # AU23: 입술 조임
                 + eye_sq_ang  * 0.10    # AU7: 눈꺼풀 조임
                 + nose_sneer  * 0.05)   # AU9: 코 찡그림 (보조)

    dbg = {
        'smile': round(smile, 3), 'brow_down': round(brow_down, 3),
        'eye_wide': round(eye_wide, 3), 'lip_press': round(lip_press, 3),
        'brow_veto': round(brow_veto, 3),
        'happy': round(happy_score, 3), 'angry': round(angry_score, 3),
    }

    if happy_score >= 0.07 and smile >= 0.03 and happy_score >= angry_score + 0.03:
        return '행복', float(np.clip(0.55 + happy_score * 0.44, 0, 0.99)), True, dbg
    elif angry_score >= 0.04 and brow_down > 0.006:
        return '분노', float(np.clip(0.55 + angry_score * 0.44, 0, 0.99)), True, dbg
    else:
        return '일반', 0.50, True, dbg


# ═════════════════════════════════════════════════════════════
# ■ Tier-2: FaceMesh 기하학적 AU 확장 분석 (12개+ 특징)
# ═════════════════════════════════════════════════════════════
# 주요 랜드마크 인덱스 (입/눈썹/눈/볼/코)
_LM = dict(
    L_MOUTH=61,  R_MOUTH=291,        # 입꼬리
    MOUTH_T=13,  MOUTH_B=14,         # 입 상/하 중심
    L_IBROW=107, R_IBROW=336,        # 내측 눈썹
    L_MBROW=66,  R_MBROW=296,        # 중간 눈썹
    L_OBROW=46,  R_OBROW=276,        # 외측 눈썹
    L_EYE_T=159, R_EYE_T=386,       # 눈 상단
    L_EYE_B=145, R_EYE_B=374,       # 눈 하단
    L_EYE_IN=133, R_EYE_IN=362,     # 눈 내측 모서리
    L_EYE_OUT=33, R_EYE_OUT=263,    # 눈 외측 모서리
    L_CHEEK=234, R_CHEEK=454,        # 볼 폭 기준
    FOREHEAD=10, CHIN=152,           # 높이 기준
    NOSE_TIP=1,
    L_NOSTRIL=49, R_NOSTRIL=279,     # 콧구멍 (AU9 근사)
    L_SMILE_CHEEK=116, R_SMILE_CHEEK=345,  # 볼 미소 영역
)

_AU_BUFS      = {}   # {feature_name: deque}
_AU_BASELINE  = {}   # {feature_name: float}
_AU_CALIB_MIN = 45
_AU_BUF_SIZE  = 150  # ~5초 @ 30fps
_AU_LOCK      = __import__('threading').Lock()


def _compute_au_features(face_lms):
    """FaceMesh 468 랜드마크 → 12개+ AU 특징값 추출 (정규화)."""
    lm = face_lms.landmark
    gx = lambda i: lm[i].x
    gy = lambda i: lm[i].y
    xs = [l.x for l in lm]; ys = [l.y for l in lm]
    fw = max(xs) - min(xs);  fh = max(ys) - min(ys)
    if fw < 0.01 or fh < 0.01:
        return None

    f = {}
    L = _LM

    # AU12: 입꼬리 당김 (Lip Corner Puller) → 행복
    mc_y         = (gy(L['MOUTH_T']) + gy(L['MOUTH_B'])) / 2
    f['au12_L']  = (mc_y - gy(L['L_MOUTH'])) / fh
    f['au12_R']  = (mc_y - gy(L['R_MOUTH'])) / fh
    f['au12']    = (f['au12_L'] + f['au12_R']) / 2

    # 입 가로 폭 (웃음 시 증가)
    f['mouth_w'] = abs(gx(L['R_MOUTH']) - gx(L['L_MOUTH'])) / fw

    # AU25: 입 열림
    f['au25']    = abs(gy(L['MOUTH_T']) - gy(L['MOUTH_B'])) / fh

    # AU4: 미간 내측 눈썹 간격 (Brow Lowerer) → 분노
    f['ibrow_dist']  = abs(gx(L['L_IBROW']) - gx(L['R_IBROW'])) / fw

    # AU4: 눈썹-눈 거리 (눈썹이 눈으로 내려올수록 감소)
    f['brow_eye_L']  = abs(gy(L['L_IBROW']) - gy(L['L_EYE_T'])) / fh
    f['brow_eye_R']  = abs(gy(L['R_IBROW']) - gy(L['R_EYE_T'])) / fh
    f['brow_eye']    = (f['brow_eye_L'] + f['brow_eye_R']) / 2

    # AU5/AU7: 눈 개방도 (Lid Tightener)
    f['eye_h_L']     = abs(gy(L['L_EYE_T']) - gy(L['L_EYE_B'])) / fh
    f['eye_h_R']     = abs(gy(L['R_EYE_T']) - gy(L['R_EYE_B'])) / fh
    f['eye_h']       = (f['eye_h_L'] + f['eye_h_R']) / 2

    # AU1/AU2: 눈썹 높이 (이마 기준)
    fhd = gy(L['FOREHEAD'])
    f['ibrow_h_L']   = (fhd - gy(L['L_IBROW'])) / fh
    f['ibrow_h_R']   = (fhd - gy(L['R_IBROW'])) / fh
    f['ibrow_h']     = (f['ibrow_h_L'] + f['ibrow_h_R']) / 2
    f['obrow_h_L']   = (fhd - gy(L['L_OBROW'])) / fh
    f['obrow_h_R']   = (fhd - gy(L['R_OBROW'])) / fh
    f['obrow_h']     = (f['obrow_h_L'] + f['obrow_h_R']) / 2

    # AU6: 볼 부풀음 근사 (눈 하단 ~ 볼 중간 거리 축소)
    f['cheek_L']     = abs(gy(L['L_EYE_B']) - gy(L['L_SMILE_CHEEK'])) / fh
    f['cheek_R']     = abs(gy(L['R_EYE_B']) - gy(L['R_SMILE_CHEEK'])) / fh
    f['cheek']       = (f['cheek_L'] + f['cheek_R']) / 2

    # AU9: 코 찡그림 근사 (콧구멍 폭) → 분노
    f['nostril_w']   = abs(gx(L['R_NOSTRIL']) - gx(L['L_NOSTRIL'])) / fw

    # 좌우 비대칭 지표
    f['au12_asym']   = abs(f['au12_L'] - f['au12_R']) / (abs(f['au12']) + 1e-6)
    f['brow_asym']   = abs(f['brow_eye_L'] - f['brow_eye_R']) / (f['brow_eye'] + 1e-6)

    return f


def _au_update_baseline(feat):
    """각 AU 특징의 적응형 베이스라인 갱신."""
    with _AU_LOCK:
        for k, v in feat.items():
            if k not in _AU_BUFS:
                _AU_BUFS[k] = deque(maxlen=_AU_BUF_SIZE)
            _AU_BUFS[k].append(v)
        if len(_AU_BUFS.get('au12', [])) >= _AU_CALIB_MIN:
            for k, buf in list(_AU_BUFS.items()):
                # 입꼬리/입폭 계열 → 낮은 쪽이 중립(35th)
                # 눈썹/눈 거리 계열 → 높은 쪽이 중립(80th)
                pct = 35 if k in ('au12', 'au12_L', 'au12_R',
                                   'mouth_w', 'au25', 'nostril_w') else 80
                _AU_BASELINE[k] = float(np.percentile(buf, pct))


def analyze_emotion_geometric(face_lms):
    """
    FaceMesh 기하학적 AU 분석 (Tier-2).
    Returns: (emotion_ko, confidence, calibrated, debug_dict)
    """
    feat = _compute_au_features(face_lms)
    if feat is None:
        return '일반', 0.5, False, {}

    _au_update_baseline(feat)
    if len(_AU_BUFS.get('au12', [])) < _AU_CALIB_MIN:
        pct = len(_AU_BUFS.get('au12', [])) / _AU_CALIB_MIN
        return '일반', 0.5, False, {'calib_pct': pct}

    def rel(key, hi_neutral, sign=1.0):
        """베이스라인 대비 상대 변화율 (0~1.5 클램프)."""
        base = _AU_BASELINE.get(key, feat[key])
        raw  = (feat[key] - base) * sign / (abs(base) + 1e-5)
        return float(np.clip(raw, 0.0, 1.5))

    # ── 공통 특징 계산 ────────────────────────────────────────
    au12_up    = rel('au12',       False, +1.0)   # 입꼬리 올라감 (AU12)
    mouth_wide = rel('mouth_w',    False, +1.0)   # 입 폭 증가
    cheek_up   = rel('cheek',      True,  -1.0)   # 볼 올라감 (AU6)
    eye_sq_raw = rel('eye_h',      True,  -1.0)   # 눈 찡그림 (AU7, 눈 작아짐)
    eye_wide   = rel('eye_h',      True,  +1.0)   # 눈 크게 뜸 (AU5, 분노 응시)
    ibrow_close = rel('ibrow_dist', True, -1.0)   # 미간 좁아짐 (AU4) ★
    brow_press  = rel('brow_eye',  True,  -1.0)   # 눈썹 눈에 가까워짐 (AU4) ★

    # ── 행복 신호 ─────────────────────────────────────────────
    au12_gate   = float(min(1.0, au12_up * 8.0))
    happy_raw   = au12_up * 0.70 + mouth_wide * 0.20 + (cheek_up * au12_gate) * 0.10
    frown_veto  = float(min(1.0, max(ibrow_close, brow_press) * 8.0))
    happy_score = happy_raw * (1.0 - frown_veto)

    # ── 분노 신호 (FACS: AU4+AU5+AU7) ───────────────────────
    anger_gate   = float(max(0.0, 1.0 - au12_up * 6.0))
    eye_sq_anger = eye_sq_raw * anger_gate   # AU7: smile 없을 때만
    eye_wide_ang = eye_wide   * anger_gate   # AU5: 분노 응시 (smile 없을 때)
    mw_ratio     = (feat['mouth_w'] - _AU_BASELINE.get('mouth_w', feat['mouth_w'])) \
                   / (abs(_AU_BASELINE.get('mouth_w', feat['mouth_w'])) + 1e-5)
    lip_depress  = 0.0 if mw_ratio > 0.06 else rel('au12', False, -1.0)
    angry_score  = (ibrow_close  * 0.45    # AU4 ★ 최우선
                  + brow_press   * 0.30    # AU4 ★
                  + eye_wide_ang * 0.12    # AU5: 응시
                  + eye_sq_anger * 0.08    # AU7: 눈꺼풀 조임
                  + lip_depress  * 0.05)   # 입꼬리 내려감

    dbg = {
        'au12_up':     round(au12_up,     3),
        'ibrow_close': round(ibrow_close, 3),
        'brow_press':  round(brow_press,  3),
        'eye_wide_ang':round(eye_wide_ang,3),
        'frown_veto':  round(frown_veto,  3),
        'happy':       round(happy_score, 3),
        'angry':       round(angry_score, 3),
    }

    if happy_score >= 0.08 and au12_up >= 0.05 and happy_score >= angry_score + 0.03:
        return '행복', float(np.clip(0.55 + happy_score * 0.44, 0, 0.99)), True, dbg
    elif angry_score >= 0.04 and ibrow_close > 0.006:
        return '분노', float(np.clip(0.55 + angry_score * 0.44, 0, 0.99)), True, dbg
    else:
        return '일반', 0.50, True, dbg


# ═════════════════════════════════════════════════════════════
# ■ 앙상블: Tier-1 + Tier-2 신뢰도 가중 합산
# ═════════════════════════════════════════════════════════════
def ensemble_emotions(res1, res2):
    """
    Tier-1(blendshape)과 Tier-2(geometric) 결과를 신뢰도 가중 앙상블.
    res1, res2: (emotion_ko, confidence, calibrated, dbg_dict)
    """
    em1, cf1, cal1, _ = res1
    em2, cf2, cal2, _ = res2
    if not cal1 and not cal2:
        return '일반', 0.5, False
    if not cal1:
        return em2, cf2, cal2
    if not cal2:
        return em1, cf1, cal1
    # 두 결과 모두 유효 ─────────────────────────────────────
    # 핵심 원칙: 한쪽만 비중립(분노/행복) 감지 → 그쪽을 신뢰
    # (중립은 기본값이므로 비중립 신호를 우선 존중)
    if em1 == '일반' and em2 != '일반':
        return em2, cf2, True
    if em1 != '일반' and em2 == '일반':
        return em1, cf1, True
    # 둘 다 동일한 비중립 → 합의 보너스
    if em1 == em2:
        return em1, float(min(0.99, (cf1 * 0.55 + cf2 * 0.45) * 1.10)), True
    # 둘 다 비중립이지만 불일치 → 신뢰도 높은 쪽
    return (em1, cf1, True) if cf1 >= cf2 else (em2, cf2, True)



def is_frontal_face(face_lms) -> bool:
    """
    얼굴이 정면을 바라보고 있는지 확인.

    Yaw  (좌우 회전): 코 끝이 좌우 볼 중심에서 벗어난 정도
    Pitch(상하 회전): 코 끝이 이마-턱 중심에서 벗어난 정도
    """
    lm = face_lms.landmark

    # ── Yaw (좌우 고개 돌림) ──────────────────────────────────
    nose_x    = lm[1].x
    l_cheek_x = lm[234].x
    r_cheek_x = lm[454].x
    face_half_w = abs(r_cheek_x - l_cheek_x) / 2
    if face_half_w < 0.01:
        return False
    yaw_dev = abs(nose_x - (l_cheek_x + r_cheek_x) / 2) / face_half_w

    # ── Pitch (상하 고개 기울임) ──────────────────────────────
    nose_y     = lm[4].y
    forehead_y = lm[10].y
    chin_y     = lm[152].y
    face_half_h = abs(chin_y - forehead_y) / 2
    if face_half_h < 0.01:
        return False
    pitch_dev = abs(nose_y - (forehead_y + chin_y) / 2) / face_half_h

    return yaw_dev < FRONTAL_YAW_TH and pitch_dev < FRONTAL_PITCH_TH


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


# ─────────────────────────────────────────────────────────────
# 평균 감정 계산 (신호등 구간 선택: 실시간 / 1분 / 5분 / 10분)
# ─────────────────────────────────────────────────────────────
# 감정 → 정서가(valence) 매핑: 분노 -1, 일반 0, 행복 +1
_EMO_VALENCE = {'분노': -1.0, '일반': 0.0, '행복': 1.0}


# 구간이 '충분히 찼다'고 볼 최소 커버리지 (0.0~1.0)
# 예: 0.9 → 5분 선택 시 데이터가 약 4.5분 이상 모여야 평균을 산출(그 전엔 '측정중')
_WINDOW_MIN_COVERAGE = 0.9


def _window_emotion(minutes):
    """최근 N분 감정 이력의 '평균 감정'을 카테고리로 반환.
    - 구간에 샘플이 전혀 없거나, 모인 데이터가 구간의 _WINDOW_MIN_COVERAGE
      비율만큼 채워지지 않았으면 None(→ 호출부에서 '측정중' 처리)을 반환한다.
    - 충분히 찼으면 valence(분노 -1 / 일반 0 / 행복 +1) 평균을 ±0.33 임계로 분류한다."""
    now    = time.time()
    cutoff = now - minutes * 60
    with _emotion_hist_lock:
        samples = [(ts, em) for ts, em in _EMOTION_HISTORY if ts >= cutoff]
    if not samples:
        return None
    # 가장 오래된 샘플이 구간의 일정 비율 이상을 커버해야 '측정 완료'로 간주
    span = now - samples[0][0]
    if span < minutes * 60 * _WINDOW_MIN_COVERAGE:
        return None
    vals = [_EMO_VALENCE.get(em, 0.0) for _, em in samples]
    avg = sum(vals) / len(vals)
    if avg <= -0.33:
        return '분노'
    if avg >= 0.33:
        return '행복'
    return '일반'


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
    """모든 베이스라인 버퍼 초기화 (Tier-1 + Tier-2 동시)."""
    with _BS_LOCK:
        _BS_BUFS.clear(); _BS_BASELINE.clear()
    with _AU_LOCK:
        _AU_BUFS.clear(); _AU_BASELINE.clear()
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
    """현재 방문 가능 신호 반환.

    쿼리 파라미터 window:
      'realtime'(기본) → 실시간 현재 감정(_current_emotion)
      '1' / '5' / '10' → 최근 N분 평균 감정(_window_emotion)
    평균 감정이 분노/행복이면 곧바로 빨강/초록, 그 외(일반)는
    캘린더상 1시간 내 임박 일정이 있으면 주황으로 처리한다.
    """
    global _current_emotion

    window = request.args.get('window', 'realtime')
    if window in ('1', '5', '10'):
        em = _window_emotion(int(window))
        if em is None:
            # 해당 구간에 누적된 감정 데이터가 아직 없음 → 측정중
            return jsonify({'emotion': None, 'signal': 'measuring', 'window': window})
    else:
        window = 'realtime'
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
    return jsonify({'emotion': em, 'signal': signal, 'window': window})


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
    global _latest_jpeg, _current_emotion, _last_record_ts

    try:
        cap = open_camera(cam_idx)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("  • 카메라 연결 상태 확인")
        print("  • python3 go_now_v2.py 1  (다른 인덱스 시도)")
        print("  • v4l2-ctl --list-devices  (장치 목록 확인)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 해상도: {W}×{H}")
    mode_tag = "Tier-1(Blendshape)+Tier-2(AU)" if _USE_BLENDSHAPES else "Tier-2(AU 기하학적)"
    print(f"[INFO] 감정 분석 모드: {mode_tag}")

    # FaceMesh (Tier-2 + 공통 바운딩박스/정면감지)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=DETECT_CONF,
        min_tracking_confidence=DETECT_CONF,
    )

    em_hist   = deque(maxlen=SMOOTH_WIN)
    conf_hist = deque(maxlen=SMOOTH_WIN)
    prev_t    = time.time()
    fps_buf   = deque(maxlen=30)
    _frame_skip  = 0
    _SKIP_EVERY  = 2
    _last_mesh   = None
    _tier1_skip  = 0
    _T1_EVERY    = 2
    _last_t1_res = ('일반', 0.5, False, {})

    print("\n[INFO] 감정 인식 시작!")
    print("─" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임 읽기 실패")
            break

        frame = cv2.flip(frame, 1)
        fH, fW = frame.shape[:2]

        # FaceMesh 처리 (N 프레임마다 실행)
        _frame_skip += 1
        if _frame_skip >= _SKIP_EVERY:
            _frame_skip = 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            _last_mesh = face_mesh.process(rgb)
            rgb.flags.writeable = True

        ko_overlays   = []
        face_detected = (_last_mesh is not None
                         and _last_mesh.multi_face_landmarks)

        if not face_detected:
            pass
        else:
            face_lms = _last_mesh.multi_face_landmarks[0]

            # 바운딩박스 계산
            xs_px = [int(l.x * fW) for l in face_lms.landmark]
            ys_px = [int(l.y * fH) for l in face_lms.landmark]
            bx1 = max(0,  min(xs_px) - 15)
            by1 = max(0,  min(ys_px) - 20)
            bx2 = min(fW, max(xs_px) + 15)
            by2 = min(fH, max(ys_px) + 10)
            bw  = bx2 - bx1

            # 정면 감지
            if not is_frontal_face(face_lms):
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (120, 120, 120), 2)
                ko_overlays.append(
                    ('얼굴이 정면을 바라보고 있지 않습니다.', (bx1, by2 + 14), (120, 120, 120), 20))
                frame = apply_text_overlays(frame, ko_overlays)
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with _frame_lock:
                        _latest_jpeg = buf.tobytes()
                prev_t = time.time(); fps_buf.append(30.0)
                continue

            # ── Tier-1: Face Landmarker blendshapes ──────────────
            if _USE_BLENDSHAPES:
                _tier1_skip += 1
                if _tier1_skip >= _T1_EVERY:
                    _tier1_skip = 0
                    try:
                        mp_img = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        t1_r = _face_landmarker.detect(mp_img)
                        if t1_r.face_blendshapes:
                            bs = {b.category_name: b.score
                                  for b in t1_r.face_blendshapes[0]}
                            _last_t1_res = analyze_blendshapes(bs)
                    except Exception as _exc:
                        print(f"[WARNING] Tier-1 오류: {_exc}")

                t2_res = analyze_emotion_geometric(face_lms)
                raw_em, raw_cf, calibrated = ensemble_emotions(_last_t1_res, t2_res)
                calib_pct = _last_t1_res[3].get('calib_pct',
                            t2_res[3].get('calib_pct', 0))
            else:
                # Tier-2만 사용
                t2_res = analyze_emotion_geometric(face_lms)
                raw_em, raw_cf, calibrated = t2_res[0], t2_res[1], t2_res[2]
                calib_pct = t2_res[3].get('calib_pct', 0)

            # 캘리브레이션 중 표시
            if not calibrated:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (180, 180, 0), 2)
                draw_bar(frame, (0, 200, 255), calib_pct, bx1, by2 + 5, bw)
                ko_overlays.append(('캘리브레이션 중...', (bx1, by2 + 18),
                                    (0, 200, 255), 20))
                frame = apply_text_overlays(frame, ko_overlays)
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with _frame_lock:
                        _latest_jpeg = buf.tobytes()
                prev_t = time.time(); fps_buf.append(30.0)
                continue

            # 시간축 스무딩 (투표)
            em_hist.append(raw_em)
            conf_hist.append(raw_cf)
            cnt = Counter(em_hist)
            em  = cnt.most_common(1)[0][0]
            cf  = cnt[em] / len(em_hist) * 0.5 + float(np.mean(conf_hist)) * 0.5
            _current_emotion = em

            # 1초마다 감정 이력 기록
            now_ts = time.time()
            if now_ts - _last_record_ts >= 1.0:
                with _emotion_hist_lock:
                    _EMOTION_HISTORY.append((now_ts, em))
                _last_record_ts = now_ts

            col = EMOTION_COLOR[em]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)

            label = f"{EMOTION_EN[em]}  {cf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            lbg_x2 = min(bx1 + tw + 12, fW)
            lbg_y1 = max(by1 - th - 14, 0)
            cv2.rectangle(frame, (bx1, lbg_y1), (lbg_x2, by1), col, -1)
            cv2.putText(frame, label, (bx1 + 5, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)

            ko_y = lbg_y1 - 36 if lbg_y1 > 36 else by2 + 14
            ko_overlays.append((em, (bx1, ko_y), col, 30))
            draw_bar(frame, col, cf, bx1, by2 + 5, bw)

        frame = apply_text_overlays(frame, ko_overlays)

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_t, 1e-9))
        prev_t = now
        fps = float(np.mean(fps_buf))

        cv2.putText(frame, f"FPS {fps:.1f}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        disp_tag = "BS+GEO" if _USE_BLENDSHAPES else "GEO-v2"
        cv2.putText(frame, disp_tag, (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        ly = fH - 82
        for em_name, c in EMOTION_COLOR.items():
            cv2.rectangle(frame, (fW - 130, ly), (fW - 115, ly + 14), c, -1)
            cv2.putText(frame, EMOTION_EN[em_name], (fW - 110, ly + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            ly += 22

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _frame_lock:
                _latest_jpeg = buf.tobytes()

    cap.release()
    face_mesh.close()
    if _face_landmarker is not None:
        try:
            _face_landmarker.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    t = threading.Thread(target=camera_loop, args=(cam_idx,), daemon=True)
    t.start()

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "라즈베리파이_IP"

    print(f"\n[INFO] 웹 브라우저에서 접속: http://{ip}:5000")
    print(f"       종료: Ctrl+C")
    print("─" * 40)

    app.run(host='0.0.0.0', port=5000, threaded=True)


if __name__ == "__main__":
    main()
