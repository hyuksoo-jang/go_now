#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
얼굴 감정 인식 시스템 v2 (Flask 웹 스트리밍) — 인식률 개선판
────────────────────────────────────────────────────────────────
라즈베리파이 5 + 카메라 모듈 / USB 웹캠

실행: python3 go_now_v2.py [카메라_인덱스]
────────────────────────────────────────────────────────────────
"""

import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")  # OpenCV 내부 WARN/ERROR 숨김

import cv2
import numpy as np
import mediapipe as mp
import time
import sys
import threading
from collections import deque, Counter
import json
import uuid
from datetime import datetime
from flask import Flask, Response, send_file, request, jsonify

# 신호등 통합 판정 모듈
try:
    from radar_signal_processor import get_processor
    _RADAR_AVAILABLE = True
except ImportError:
    _RADAR_AVAILABLE = False
    print("[WARNING] radar_signal_processor 모듈을 찾을 수 없습니다.")

# ─────────────────────────────────────────────────────────────
# Tier-1: MediaPipe Face Landmarker (Tasks API + 52 blendshapes)
# ─────────────────────────────────────────────────────────────
_USE_BLENDSHAPES = False
_face_landmarker = None


def _init_face_landmarker():
    """face_landmarker.task 파일이 있으면 Tasks API로 초기화."""
    global _USE_BLENDSHAPES, _face_landmarker
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
    if not os.path.exists(model_path):
        return
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
        print("[INFO] Tier-1 Face Landmarker (52 blendshapes) 활성화")
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

except ImportError:
    print("[WARNING] Pillow 없음 → 영문 표시")

_init_face_landmarker()
if not _USE_BLENDSHAPES:
    print("[INFO] face_landmarker.task 없음 → Tier-2(AU 기하학적) 사용")


# ─────────────────────────────────────────────────────────────
# 감정 설정
# ─────────────────────────────────────────────────────────────
EMOTION_EN    = {'일반': 'Neutral', '분노': 'Angry', '행복': 'Happy'}
EMOTION_COLOR = {
    '일반': (  0, 200, 200),
    '분노': (  0,  50, 220),
    '행복': (  0, 200,  80),
}

FRONTAL_YAW_TH   = 0.25
FRONTAL_PITCH_TH = 0.09
SMOOTH_WIN       = 60
DETECT_CONF      = 0.45


# ═════════════════════════════════════════════════════════════
# ■ Tier-1: 52 Blendshape 기반 감정 분석
# ═════════════════════════════════════════════════════════════
_BS_BUFS      = {}
_BS_BASELINE  = {}
_BS_CALIB_MIN = 35
_BS_LOCK      = threading.Lock()
_bs_in_anger  = False

_ALL_TRACKED_BS = [
    'mouthSmileLeft', 'mouthSmileRight',
    'cheekSquintLeft', 'cheekSquintRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'browDownLeft', 'browDownRight',
    'eyeWideLeft', 'eyeWideRight',
    'mouthPressLeft', 'mouthPressRight',
    'noseSneerLeft', 'noseSneerRight',
    'mouthFrownLeft', 'mouthFrownRight',
]


def _bs_update_baseline(bs_dict):
    with _BS_LOCK:
        if not _bs_in_anger:
            for k in _ALL_TRACKED_BS:
                v = bs_dict.get(k, 0.0)
                if k not in _BS_BUFS:
                    _BS_BUFS[k] = deque(maxlen=180)
                _BS_BUFS[k].append(v)
            if len(_BS_BUFS.get('mouthSmileLeft', [])) >= _BS_CALIB_MIN:
                for k, buf in list(_BS_BUFS.items()):
                    _BS_BASELINE[k] = float(np.percentile(buf, 25))


def analyze_blendshapes(bs_dict):
    _bs_update_baseline(bs_dict)
    if len(_BS_BUFS.get('mouthSmileLeft', [])) < _BS_CALIB_MIN:
        pct = len(_BS_BUFS.get('mouthSmileLeft', [])) / _BS_CALIB_MIN
        return '일반', 0.5, False, {'calib_pct': pct}

    def adj(name):
        return max(0.0, bs_dict.get(name, 0.0) - _BS_BASELINE.get(name, 0.0))

    smile      = (adj('mouthSmileLeft')  + adj('mouthSmileRight'))  / 2
    cheek      = (adj('cheekSquintLeft') + adj('cheekSquintRight')) / 2
    eye_sq     = (adj('eyeSquintLeft')   + adj('eyeSquintRight'))   / 2
    brow_down  = (adj('browDownLeft')    + adj('browDownRight'))    / 2
    eye_wide   = (adj('eyeWideLeft')     + adj('eyeWideRight'))     / 2
    lip_press  = (adj('mouthPressLeft')  + adj('mouthPressRight'))  / 2
    nose_sneer = (adj('noseSneerLeft')   + adj('noseSneerRight'))   / 2

    smile_gate  = float(min(1.0, smile * 8.0))
    happy_raw   = smile * 0.70 + (cheek * smile_gate) * 0.20 + (eye_sq * smile_gate) * 0.10
    brow_veto   = float(min(1.0, brow_down * 12.0))
    happy_score = happy_raw * (1.0 - brow_veto)

    anger_gate  = float(max(0.0, 1.0 - smile * 8.0))
    eye_sq_ang  = eye_sq * anger_gate
    eye_wide_ang = eye_wide * anger_gate
    angry_score = (brow_down   * 0.50
                 + eye_wide_ang * 0.20
                 + lip_press   * 0.15
                 + eye_sq_ang  * 0.10
                 + nose_sneer  * 0.05)

    dbg = {
        'smile': round(smile, 3), 'brow_down': round(brow_down, 3),
        'eye_wide': round(eye_wide, 3), 'lip_press': round(lip_press, 3),
        'brow_veto': round(brow_veto, 3),
        'happy': round(happy_score, 3), 'angry': round(angry_score, 3),
    }

    global _bs_in_anger
    if happy_score >= 0.14 and smile >= 0.07 and happy_score >= angry_score + 0.05:
        _bs_in_anger = False
        return '행복', float(np.clip(0.55 + happy_score * 0.44, 0, 0.99)), True, dbg
    elif angry_score >= 0.04 and brow_down > 0.006:
        _bs_in_anger = True
        return '분노', float(np.clip(0.55 + angry_score * 0.44, 0, 0.99)), True, dbg
    else:
        _bs_in_anger = False
        return '일반', 0.50, True, dbg


# ═════════════════════════════════════════════════════════════
# ■ Tier-2: FaceMesh 기하학적 AU 확장 분석
# ═════════════════════════════════════════════════════════════
_LM = dict(
    L_MOUTH=61,  R_MOUTH=291,
    MOUTH_T=13,  MOUTH_B=14,
    L_IBROW=107, R_IBROW=336,
    L_MBROW=66,  R_MBROW=296,
    L_OBROW=46,  R_OBROW=276,
    L_EYE_T=159, R_EYE_T=386,
    L_EYE_B=145, R_EYE_B=374,
    L_EYE_IN=133, R_EYE_IN=362,
    L_EYE_OUT=33, R_EYE_OUT=263,
    L_CHEEK=234, R_CHEEK=454,
    FOREHEAD=10, CHIN=152,
    NOSE_TIP=1,
    L_NOSTRIL=49, R_NOSTRIL=279,
    L_SMILE_CHEEK=116, R_SMILE_CHEEK=345,
)

_AU_BUFS      = {}
_AU_BASELINE  = {}
_AU_CALIB_MIN = 45
_AU_BUF_SIZE  = 150
_AU_LOCK      = threading.Lock()
_au_in_anger  = False


def _compute_au_features(face_lms):
    lm = face_lms.landmark
    gx = lambda i: lm[i].x
    gy = lambda i: lm[i].y
    xs = [l.x for l in lm]; ys = [l.y for l in lm]
    fw = max(xs) - min(xs);  fh = max(ys) - min(ys)
    if fw < 0.01 or fh < 0.01:
        return None

    f = {}
    L = _LM

    mc_y         = (gy(L['MOUTH_T']) + gy(L['MOUTH_B'])) / 2
    f['au12_L']  = (mc_y - gy(L['L_MOUTH'])) / fh
    f['au12_R']  = (mc_y - gy(L['R_MOUTH'])) / fh
    f['au12']    = (f['au12_L'] + f['au12_R']) / 2

    f['mouth_w'] = abs(gx(L['R_MOUTH']) - gx(L['L_MOUTH'])) / fw
    f['au25']    = abs(gy(L['MOUTH_T']) - gy(L['MOUTH_B'])) / fh
    f['ibrow_dist']  = abs(gx(L['L_IBROW']) - gx(L['R_IBROW'])) / fw

    f['brow_eye_L']  = abs(gy(L['L_IBROW']) - gy(L['L_EYE_T'])) / fh
    f['brow_eye_R']  = abs(gy(L['R_IBROW']) - gy(L['R_EYE_T'])) / fh
    f['brow_eye']    = (f['brow_eye_L'] + f['brow_eye_R']) / 2

    f['eye_h_L']     = abs(gy(L['L_EYE_T']) - gy(L['L_EYE_B'])) / fh
    f['eye_h_R']     = abs(gy(L['R_EYE_T']) - gy(L['R_EYE_B'])) / fh
    f['eye_h']       = (f['eye_h_L'] + f['eye_h_R']) / 2

    fhd = gy(L['FOREHEAD'])
    f['ibrow_h_L']   = (fhd - gy(L['L_IBROW'])) / fh
    f['ibrow_h_R']   = (fhd - gy(L['R_IBROW'])) / fh
    f['ibrow_h']     = (f['ibrow_h_L'] + f['ibrow_h_R']) / 2
    f['obrow_h_L']   = (fhd - gy(L['L_OBROW'])) / fh
    f['obrow_h_R']   = (fhd - gy(L['R_OBROW'])) / fh
    f['obrow_h']     = (f['obrow_h_L'] + f['obrow_h_R']) / 2

    f['cheek_L']     = abs(gy(L['L_EYE_B']) - gy(L['L_SMILE_CHEEK'])) / fh
    f['cheek_R']     = abs(gy(L['R_EYE_B']) - gy(L['R_SMILE_CHEEK'])) / fh
    f['cheek']       = (f['cheek_L'] + f['cheek_R']) / 2

    f['nostril_w']   = abs(gx(L['R_NOSTRIL']) - gx(L['L_NOSTRIL'])) / fw

    f['au12_asym']   = abs(f['au12_L'] - f['au12_R']) / (abs(f['au12']) + 1e-6)
    f['brow_asym']   = abs(f['brow_eye_L'] - f['brow_eye_R']) / (f['brow_eye'] + 1e-6)

    return f


def _au_update_baseline(feat):
    with _AU_LOCK:
        if not _au_in_anger:
            for k, v in feat.items():
                if k not in _AU_BUFS:
                    _AU_BUFS[k] = deque(maxlen=_AU_BUF_SIZE)
                _AU_BUFS[k].append(v)
            if len(_AU_BUFS.get('au12', [])) >= _AU_CALIB_MIN:
                for k, buf in list(_AU_BUFS.items()):
                    pct = 35 if k in ('au12', 'au12_L', 'au12_R',
                                       'mouth_w', 'au25', 'nostril_w') else 80
                    _AU_BASELINE[k] = float(np.percentile(buf, pct))


def analyze_emotion_geometric(face_lms):
    feat = _compute_au_features(face_lms)
    if feat is None:
        return '일반', 0.5, False, {}

    _au_update_baseline(feat)
    if len(_AU_BUFS.get('au12', [])) < _AU_CALIB_MIN:
        pct = len(_AU_BUFS.get('au12', [])) / _AU_CALIB_MIN
        return '일반', 0.5, False, {'calib_pct': pct}

    def rel(key, hi_neutral, sign=1.0):
        base = _AU_BASELINE.get(key, feat[key])
        raw  = (feat[key] - base) * sign / (abs(base) + 1e-5)
        return float(np.clip(raw, 0.0, 1.5))

    au12_up    = rel('au12',       False, +1.0)
    mouth_wide = rel('mouth_w',    False, +1.0)
    cheek_up   = rel('cheek',      True,  -1.0)
    eye_sq_raw = rel('eye_h',      True,  -1.0)
    eye_wide   = rel('eye_h',      True,  +1.0)
    ibrow_close = rel('ibrow_dist', True, -1.0)
    brow_press  = rel('brow_eye',  True,  -1.0)

    au12_gate   = float(min(1.0, au12_up * 8.0))
    happy_raw   = au12_up * 0.80 + (mouth_wide * au12_gate) * 0.10 + (cheek_up * au12_gate) * 0.10
    frown_veto  = float(min(1.0, max(ibrow_close, brow_press) * 8.0))
    happy_score = happy_raw * (1.0 - frown_veto)

    anger_gate   = float(max(0.0, 1.0 - au12_up * 6.0))
    eye_sq_anger = eye_sq_raw * anger_gate
    eye_wide_ang = eye_wide   * anger_gate
    mw_ratio     = (feat['mouth_w'] - _AU_BASELINE.get('mouth_w', feat['mouth_w'])) \
                   / (abs(_AU_BASELINE.get('mouth_w', feat['mouth_w'])) + 1e-5)
    lip_depress  = 0.0 if mw_ratio > 0.06 else rel('au12', False, -1.0)
    angry_score  = (ibrow_close  * 0.45
                  + brow_press   * 0.30
                  + eye_wide_ang * 0.12
                  + eye_sq_anger * 0.08
                  + lip_depress  * 0.05)

    dbg = {
        'au12_up':     round(au12_up,     3),
        'ibrow_close': round(ibrow_close, 3),
        'brow_press':  round(brow_press,  3),
        'eye_wide_ang':round(eye_wide_ang,3),
        'frown_veto':  round(frown_veto,  3),
        'happy':       round(happy_score, 3),
        'angry':       round(angry_score, 3),
    }

    global _au_in_anger
    if happy_score >= 0.14 and au12_up >= 0.09 and happy_score >= angry_score + 0.05:
        _au_in_anger = False
        return '행복', float(np.clip(0.55 + happy_score * 0.44, 0, 0.99)), True, dbg
    elif angry_score >= 0.04 and ibrow_close > 0.006:
        _au_in_anger = True
        return '분노', float(np.clip(0.55 + angry_score * 0.44, 0, 0.99)), True, dbg
    else:
        _au_in_anger = False
        return '일반', 0.50, True, dbg


def ensemble_emotions(res1, res2):
    em1, cf1, cal1, _ = res1
    em2, cf2, cal2, _ = res2
    if not cal1 and not cal2:
        return '일반', 0.5, False
    if not cal1:
        return em2, cf2, cal2
    if not cal2:
        return em1, cf1, cal1
    if em1 == '일반' and em2 != '일반':
        return em2, cf2, True
    if em1 != '일반' and em2 == '일반':
        return em1, cf1, True
    if em1 == em2:
        return em1, float(min(0.99, (cf1 * 0.55 + cf2 * 0.45) * 1.10)), True
    return (em1, cf1, True) if cf1 >= cf2 else (em2, cf2, True)


def is_frontal_face(face_lms) -> bool:
    lm = face_lms.landmark
    nose_x    = lm[1].x
    l_cheek_x = lm[234].x
    r_cheek_x = lm[454].x
    face_half_w = abs(r_cheek_x - l_cheek_x) / 2
    if face_half_w < 0.01:
        return False
    yaw_dev = abs(nose_x - (l_cheek_x + r_cheek_x) / 2) / face_half_w

    nose_y     = lm[4].y
    forehead_y = lm[10].y
    chin_y     = lm[152].y
    face_half_h = abs(chin_y - forehead_y) / 2
    if face_half_h < 0.01:
        return False
    pitch_dev = abs(nose_y - (forehead_y + chin_y) / 2) / face_half_h

    return yaw_dev < FRONTAL_YAW_TH and pitch_dev < FRONTAL_PITCH_TH


def draw_bar(img, color, pct, x, y, w, h=8):
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    bw = int(w * float(np.clip(pct, 0, 1)))
    if bw > 0:
        cv2.rectangle(img, (x, y), (x + bw, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)


def apply_text_overlays(img, overlays):
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
        for text, pos, bgr, _ in overlays:
            eng = EMOTION_EN.get(text, text)
            cv2.putText(img, eng, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, bgr, 2, cv2.LINE_AA)
        return img


def open_camera(preferred_idx: int) -> cv2.VideoCapture:
    """카메라를 열고 실제 프레임이 들어오는지 확인 (워밍업/백엔드 강화)."""
    backends  = [cv2.CAP_V4L2, cv2.CAP_ANY]
    idx_order = [preferred_idx] + [i for i in range(8) if i != preferred_idx]

    for backend in backends:
        for idx in idx_order:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    bname = "V4L2" if backend == cv2.CAP_V4L2 else "ANY"
                    print(f"[INFO] 카메라 인덱스 {idx} 사용 (backend={bname})")
                    return cap
                time.sleep(0.1)
            cap.release()

    gst = ("libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 "
           "! videoconvert ! appsink")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                print("[INFO] libcamera GStreamer 파이프라인 사용")
                return cap
            time.sleep(0.1)
        cap.release()

    raise RuntimeError("사용 가능한 카메라를 찾을 수 없습니다.")


# ─────────────────────────────────────────────────────────────
# Flask 앱 + 전역 상태
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
_frame_lock      = threading.Lock()
_latest_jpeg     = b''
_camera_ok       = False
_current_emotion = '일반'

_CALENDAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calendar_data.json')
_APPROVAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'approval_data.json')
_data_lock = threading.Lock()
_speech_recalib_flag = threading.Event()
_speech_calib_status = {"phase": "idle", "since": 0.0, "text": "",
                        "noise_sec": 0.0, "enroll_sec": 0.0, "level": 0.0, "seq": 0}
_speech_calib_lock = threading.Lock()

_EMOTION_HISTORY = deque(maxlen=600)
_emotion_hist_lock = threading.Lock()
_last_record_ts = 0.0


def _window_emotion(minutes):
    now        = time.time()
    cutoff     = now - minutes * 60
    target_sec = minutes * 60

    with _emotion_hist_lock:
        window_items = [(ts, em) for ts, em in _EMOTION_HISTORY if ts >= cutoff]
        first_ts = _EMOTION_HISTORY[0][0] if _EMOTION_HISTORY else None

    if not window_items or first_ts is None:
        return None

    elapsed = now - first_ts
    if elapsed < target_sec:
        return None

    samples = [em for _, em in window_items]
    total       = len(samples)
    anger_ratio = samples.count('분노') / total
    if anger_ratio >= 0.30:
        return '분노'
    happy_ratio = samples.count('행복') / total
    if happy_ratio >= 0.50:
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


@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html'))


@app.route('/dashboard-integrated')
def dashboard_integrated():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html'))


def _mjpeg_generator():
    global _latest_jpeg

    _ph = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(_ph, 'Initializing camera...', (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)
    _, _buf = cv2.imencode('.jpg', _ph, [cv2.IMWRITE_JPEG_QUALITY, 60])
    _BLANK_JPEG = _buf.tobytes()

    def _pack(data):
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

    yield _pack(_BLANK_JPEG)

    while True:
        with _frame_lock:
            frame_data = _latest_jpeg
        yield _pack(frame_data if frame_data else _BLANK_JPEG)
        time.sleep(0.033)


@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    global _bs_in_anger, _au_in_anger
    with _BS_LOCK:
        _BS_BUFS.clear(); _BS_BASELINE.clear()
    with _AU_LOCK:
        _AU_BUFS.clear(); _AU_BASELINE.clear()
    _bs_in_anger = False
    _au_in_anger = False
    print('[INFO] 캘리브레이션 초기화')
    return {'message': '무표정으로 1~2초 유지하세요'}


@app.route('/video_feed')
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/emotion_history')
def api_emotion_history():
    minutes = request.args.get('minutes', 5, type=int)
    minutes = max(1, min(minutes, 10))
    cutoff  = time.time() - minutes * 60
    with _emotion_hist_lock:
        data = [(ts, em) for ts, em in _EMOTION_HISTORY if ts >= cutoff]
    return jsonify([{'t': round(ts, 2), 'e': em} for ts, em in data])


@app.route('/api/status')
def api_status():
    global _current_emotion

    window = request.args.get('window', 'realtime')
    if window in ('1', '5', '10'):
        em = _window_emotion(int(window))
        if em is None:
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


@app.route('/api/radar/signals')
def api_radar_signals():
    if not _RADAR_AVAILABLE:
        return jsonify({"error": "radar processor not available"}), 503
    try:
        radar = get_processor()
        signals = radar.get_all_signals()
        return jsonify(signals)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/radar/add-sigh', methods=['POST'])
def api_radar_add_sigh():
    if not _RADAR_AVAILABLE:
        return jsonify({"error": "radar processor not available"}), 503
    try:
        data = request.get_json(force=True)
        detected = data.get("detected", False)
        timestamp = data.get("timestamp", None)
        radar = get_processor()
        radar.add_sigh(bool(detected), timestamp)
        return jsonify({"ok": True, "detected": detected})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/radar/add-speech-emotion', methods=['POST'])
def api_radar_add_speech_emotion():
    if not _RADAR_AVAILABLE:
        return jsonify({"error": "radar processor not available"}), 503
    try:
        data = request.get_json(force=True)
        emotion = data.get("emotion", "")
        timestamp = data.get("timestamp", None)
        if not emotion:
            return jsonify({"error": "emotion required"}), 400
        radar = get_processor()
        radar.add_speech_emotion(emotion, timestamp)
        return jsonify({"ok": True, "emotion": emotion})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/device-status')
def api_device_status():
    sigh_ok = False
    speech_ok = False
    if _RADAR_AVAILABLE:
        try:
            radar = get_processor()
            alive = radar.device_alive(timeout=20.0)
            sigh_ok = alive['sigh']
            speech_ok = alive['speech']
        except Exception:
            pass
    return jsonify({'camera': _camera_ok, 'sigh': sigh_ok, 'speech': speech_ok})


@app.route('/api/radar/speech-heartbeat', methods=['POST'])
def api_radar_speech_heartbeat():
    """발화 파이프라인 alive 신호 (감정 데이터 없이 장치 상태만 갱신)."""
    if not _RADAR_AVAILABLE:
        return jsonify({"error": "radar processor not available"}), 503
    try:
        get_processor().mark_speech_alive()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/speech-recalibrate', methods=['POST'])
def api_speech_recalibrate():
    """대시보드 → speech_agent에 음성 캘리브레이션 재실행 요청"""
    _speech_recalib_flag.set()
    with _speech_calib_lock:
        _speech_calib_status.update(phase="requested", since=time.time(),
                                    seq=_speech_calib_status["seq"] + 1)
    return jsonify({'ok': True})


@app.route('/api/speech-calib-status', methods=['GET', 'POST'])
def api_speech_calib_status():
    """음성 캘리브레이션 단계 (mic_agent가 POST 갱신, 대시보드가 GET 폴링)."""
    if request.method == 'POST':
        data = request.get_json(force=True)
        with _speech_calib_lock:
            for k in ('phase', 'text', 'noise_sec', 'enroll_sec', 'level'):
                if k in data:
                    _speech_calib_status[k] = data[k]
            _speech_calib_status['since'] = time.time()
        return jsonify({'ok': True})
    with _speech_calib_lock:
        return jsonify(dict(_speech_calib_status))


@app.route('/api/speech-recalibrate-flag')
def api_speech_recalib_flag():
    requested = _speech_recalib_flag.is_set()
    if requested:
        _speech_recalib_flag.clear()
    return jsonify({'requested': requested})


@app.route('/api/radar/status')
def api_radar_status():
    if not _RADAR_AVAILABLE:
        return jsonify({"error": "radar processor not available"}), 503
    try:
        radar = get_processor()
        status = radar.get_status()
        recent = radar.get_recent_data(10)
        return jsonify({**status, "recent_data": recent})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

            # 시간축 스무딩 (투표) — 분노 히스테리시스: 다수결보다 분노 비율이 높을 때만 유지
            em_hist.append(raw_em)
            conf_hist.append(raw_cf)
            cnt = Counter(em_hist)
            recent = list(em_hist)
            top_em = cnt.most_common(1)[0][0]
            anger_ratio = recent.count('분노') / len(recent)
            # 분노가 다수결 감정이거나, 다수결과 동률에 가깝고(50% 이상) 실제 지배적일 때만 분노 유지
            if anger_ratio >= 0.50 and anger_ratio >= cnt.get(top_em, 0) / len(recent) * 0.85:
                em = '분노'
            else:
                em = top_em
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

    app.run(host='0.0.0.0', port=5000, threaded=True)


if __name__ == "__main__":
    main()