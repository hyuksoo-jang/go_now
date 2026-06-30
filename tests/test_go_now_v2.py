#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
go_now_v2.py 단위 테스트
────────────────────────────────────────────────────────────────
얼굴 감정 인식 + Flask 대시보드 서버.

카메라/MediaPipe 같은 하드웨어 의존부(camera_loop 의 본 루프)는
단위 테스트 대상에서 제외하고,
  · 순수 감정 분석 로직 (blendshape / 기하학 / 앙상블 / 정면감지)
  · 전역 상태 헬퍼 (자리비움, 감정 히스토리, JSON IO)
  · Flask 라우트 (test client + radar/outlook mock)
를 검증한다.
"""
import sys, os, json, time
from collections import deque
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import go_now_v2 as gn


# ─────────────────────────────────────────────────────────────
# 공통 헬퍼 / fixture
# ─────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def reset_state():
    """각 테스트 전 전역 상태 초기화 (테스트 간 누수 방지)."""
    with gn._BS_LOCK:
        gn._BS_BUFS.clear()
        gn._BS_BASELINE.clear()
        gn._BS_BASELINE_LOCKED = False
    with gn._AU_LOCK:
        gn._AU_BUFS.clear()
        gn._AU_BASELINE.clear()
        gn._AU_BASELINE_LOCKED = False
    gn._bs_in_anger = False
    gn._au_in_anger = False
    gn._DISABLE_BASELINE_LEARNING = False
    gn._USE_BLENDSHAPES = False
    gn._face_landmarker = None
    gn._current_emotion = '일반'
    gn._face_absent = False
    gn._last_face_seen_ts = time.time()
    with gn._emotion_hist_lock:
        gn._EMOTION_HISTORY.clear()
    gn._last_record_ts = 0.0
    yield


def _prime_bs_baseline(baseline=None):
    """blendshape 베이스라인을 0(또는 지정값)으로 고정하고 캘리브레이션 게이트 통과시킴."""
    if baseline is None:
        baseline = {k: 0.0 for k in gn._ALL_TRACKED_BS}
    with gn._BS_LOCK:
        gn._BS_BASELINE.clear()
        gn._BS_BASELINE.update(baseline)
        gn._BS_BASELINE_LOCKED = True            # update_baseline 가 즉시 반환 → 값 보존
        gn._BS_BUFS['mouthSmileLeft'] = deque([0.0] * 40, maxlen=180)  # 게이트 통과용


class _Pt:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z=0.0):
        self.x = x; self.y = y; self.z = z


class _FakeFace:
    """MediaPipe FaceMesh 결과의 .landmark 리스트 흉내."""
    def __init__(self, points):
        self.landmark = points


def make_landmarks(overrides=None, n=478, spread=True):
    """기본 478개 점을 만들고 overrides(dict idx->(x,y))로 특정 점 지정."""
    pts = []
    for i in range(n):
        if spread:
            pts.append(_Pt((i % 50) / 50.0, (i // 50) / 50.0))
        else:
            pts.append(_Pt(0.5, 0.5))
    if overrides:
        for idx, (x, y) in overrides.items():
            pts[idx] = _Pt(x, y)
    return _FakeFace(pts)


# ═════════════════════════════════════════════════════════════
# G01: analyze_blendshapes
# ═════════════════════════════════════════════════════════════
class TestAnalyzeBlendshapes:
    """G01: 52 blendshape 기반 감정 분석"""

    def test_G01_calibrating_returns_neutral_uncalibrated(self):
        """버퍼 부족 시 캘리브레이션 중(일반, calibrated=False)"""
        em, cf, calibrated, dbg = gn.analyze_blendshapes({})
        assert em == '일반'
        assert calibrated is False
        assert 'calib_pct' in dbg

    def test_G01_happy_detected(self):
        """미소가 크면 '행복' 판정"""
        _prime_bs_baseline()
        bs = {'mouthSmileLeft': 0.5, 'mouthSmileRight': 0.5}
        em, cf, calibrated, dbg = gn.analyze_blendshapes(bs)
        assert em == '행복'
        assert calibrated is True
        assert 0 < cf <= 0.99

    def test_G01_angry_detected(self):
        """눈썹 내림 + 눈 크게 뜸 + 입술 압박 → '분노'"""
        _prime_bs_baseline()
        bs = {
            'browDownLeft': 1.0, 'browDownRight': 1.0,
            'eyeWideLeft': 0.5, 'eyeWideRight': 0.5,
            'mouthPressLeft': 1.0, 'mouthPressRight': 1.0,
        }
        em, cf, calibrated, dbg = gn.analyze_blendshapes(bs)
        assert em == '분노'
        assert calibrated is True
        assert gn._bs_in_anger is True

    def test_G01_neutral_when_no_signal(self):
        """특징 없음 → '일반'"""
        _prime_bs_baseline()
        em, cf, calibrated, dbg = gn.analyze_blendshapes({})
        assert em == '일반'
        assert calibrated is True

    def test_G01_blink_suppresses_anger(self):
        """눈 감김(blink>0.5)이면 분노 특징이 있어도 '일반'으로 억제"""
        _prime_bs_baseline()
        bs = {
            'browDownLeft': 1.0, 'browDownRight': 1.0,
            'mouthPressLeft': 1.0, 'mouthPressRight': 1.0,
            'eyeBlinkLeft': 0.9, 'eyeBlinkRight': 0.9,
        }
        em, cf, calibrated, dbg = gn.analyze_blendshapes(bs)
        assert em == '일반'
        assert calibrated is True

    def test_G01_disable_learning_locks_immediately(self):
        """_DISABLE_BASELINE_LEARNING=True 면 1회 캡처 후 즉시 잠금"""
        gn._DISABLE_BASELINE_LEARNING = True
        gn.analyze_blendshapes({'mouthSmileLeft': 0.3})
        assert gn._BS_BASELINE_LOCKED is True


# ═════════════════════════════════════════════════════════════
# G02: _bs_update_baseline
# ═════════════════════════════════════════════════════════════
class TestBsUpdateBaseline:
    """G02: blendshape 베이스라인 학습"""

    def test_G02_locked_baseline_not_updated(self):
        """잠금 상태면 갱신하지 않음"""
        with gn._BS_LOCK:
            gn._BS_BASELINE_LOCKED = True
        gn._bs_update_baseline({'mouthSmileLeft': 0.9})
        assert 'mouthSmileLeft' not in gn._BS_BUFS

    def test_G02_appends_until_calib_min(self):
        """충분히 쌓이면 baseline 계산"""
        for _ in range(gn._BS_CALIB_MIN + 1):
            gn._bs_update_baseline({k: 0.2 for k in gn._ALL_TRACKED_BS})
        assert 'mouthSmileLeft' in gn._BS_BASELINE

    def test_G02_locks_when_buffer_full(self):
        """버퍼 180개 가득 차면 baseline 고정"""
        for _ in range(181):
            gn._bs_update_baseline({k: 0.1 for k in gn._ALL_TRACKED_BS})
        assert gn._BS_BASELINE_LOCKED is True

    def test_G02_anger_freezes_learning(self):
        """분노 상태(_bs_in_anger)면 버퍼에 쌓지 않음"""
        gn._bs_in_anger = True
        gn._bs_update_baseline({'mouthSmileLeft': 0.5})
        assert 'mouthSmileLeft' not in gn._BS_BUFS


# ═════════════════════════════════════════════════════════════
# G03: _compute_au_features / analyze_emotion_geometric
# ═════════════════════════════════════════════════════════════
class TestGeometricAU:
    """G03: FaceMesh 기하학 AU 분석"""

    def test_G03_features_none_for_degenerate_face(self):
        """얼굴 폭/높이가 0에 가까우면 None"""
        face = make_landmarks(spread=False)  # 모든 점 동일 → fw,fh≈0
        assert gn._compute_au_features(face) is None

    def test_G03_features_dict_for_valid_face(self):
        """정상 얼굴이면 주요 특징 키 포함 dict 반환"""
        face = make_landmarks()
        feat = gn._compute_au_features(face)
        assert feat is not None
        for key in ('au12', 'mouth_w', 'eye_h', 'brow_eye', 'cheek'):
            assert key in feat

    def test_G03_geometric_returns_neutral_for_degenerate(self):
        """특징 None → '일반', calibrated False"""
        face = make_landmarks(spread=False)
        em, cf, calibrated, dbg = gn.analyze_emotion_geometric(face)
        assert em == '일반'
        assert calibrated is False

    def test_G03_geometric_calibrating(self):
        """버퍼 부족하면 캘리브레이션 중"""
        face = make_landmarks()
        em, cf, calibrated, dbg = gn.analyze_emotion_geometric(face)
        assert em == '일반'
        assert calibrated is False
        assert 'calib_pct' in dbg

    def test_G03_geometric_calibrated_after_enough_frames(self):
        """충분한 프레임 후 calibrated=True"""
        face = make_landmarks()
        last = None
        for _ in range(gn._AU_CALIB_MIN + 2):
            last = gn.analyze_emotion_geometric(face)
        assert last[2] is True


# ═════════════════════════════════════════════════════════════
# G04: ensemble_emotions
# ═════════════════════════════════════════════════════════════
class TestEnsembleEmotions:
    """G04: Tier-1/Tier-2 앙상블"""

    def test_G04_both_uncalibrated(self):
        r = gn.ensemble_emotions(('분노', 0.9, False, {}), ('행복', 0.9, False, {}))
        assert r == ('일반', 0.5, False)

    def test_G04_first_uncalibrated_uses_second(self):
        r = gn.ensemble_emotions(('분노', 0.9, False, {}), ('행복', 0.8, True, {}))
        assert r == ('행복', 0.8, True)

    def test_G04_second_uncalibrated_uses_first(self):
        r = gn.ensemble_emotions(('분노', 0.9, True, {}), ('행복', 0.8, False, {}))
        assert r == ('분노', 0.9, True)

    def test_G04_neutral_yields_to_emotion(self):
        r = gn.ensemble_emotions(('일반', 0.5, True, {}), ('분노', 0.8, True, {}))
        assert r[0] == '분노'
        r2 = gn.ensemble_emotions(('행복', 0.8, True, {}), ('일반', 0.5, True, {}))
        assert r2[0] == '행복'

    def test_G04_same_emotion_boosts_confidence(self):
        r = gn.ensemble_emotions(('분노', 0.8, True, {}), ('분노', 0.6, True, {}))
        assert r[0] == '분노'
        assert r[2] is True
        assert r[1] <= 0.99

    def test_G04_conflict_takes_higher_confidence(self):
        r = gn.ensemble_emotions(('분노', 0.9, True, {}), ('행복', 0.6, True, {}))
        assert r[0] == '분노'
        r2 = gn.ensemble_emotions(('분노', 0.5, True, {}), ('행복', 0.95, True, {}))
        assert r2[0] == '행복'


# ═════════════════════════════════════════════════════════════
# G05: is_frontal_face
# ═════════════════════════════════════════════════════════════
class TestIsFrontalFace:
    """G05: 정면 얼굴 판정"""

    def _frontal_overrides(self):
        # 코(1,4)가 양 볼(234,454) 중앙, 이마(10)/턱(152) 중앙
        return {
            1: (0.50, 0.50),
            4: (0.50, 0.50),
            234: (0.30, 0.50),
            454: (0.70, 0.50),
            10: (0.50, 0.30),
            152: (0.50, 0.70),
        }

    def test_G05_frontal_true(self):
        face = make_landmarks(self._frontal_overrides())
        assert gn.is_frontal_face(face) is True

    def test_G05_yaw_off_returns_false(self):
        ov = self._frontal_overrides()
        ov[1] = (0.69, 0.50)   # 코가 오른쪽으로 치우침
        ov[4] = (0.69, 0.50)
        face = make_landmarks(ov)
        assert gn.is_frontal_face(face) is False

    def test_G05_pitch_off_returns_false(self):
        ov = self._frontal_overrides()
        ov[4] = (0.50, 0.69)   # 코 y가 아래로 치우침
        face = make_landmarks(ov)
        assert gn.is_frontal_face(face) is False

    def test_G05_degenerate_width_false(self):
        ov = self._frontal_overrides()
        ov[234] = (0.50, 0.50)
        ov[454] = (0.50, 0.50)  # 볼 폭 0
        face = make_landmarks(ov)
        assert gn.is_frontal_face(face) is False

    def test_G05_degenerate_height_false(self):
        ov = self._frontal_overrides()
        ov[10] = (0.50, 0.50)
        ov[152] = (0.50, 0.50)  # 이마-턱 높이 0
        face = make_landmarks(ov)
        assert gn.is_frontal_face(face) is False


# ═════════════════════════════════════════════════════════════
# G06: draw_bar / apply_text_overlays (cv2 렌더링)
# ═════════════════════════════════════════════════════════════
class TestRendering:
    """G06: OpenCV 기반 렌더링 헬퍼"""

    def _img(self):
        import numpy as np
        return np.zeros((100, 200, 3), dtype='uint8')

    def test_G06_draw_bar_modifies_image(self):
        img = self._img()
        before = img.copy()
        gn.draw_bar(img, (0, 255, 0), 0.5, 10, 10, 100)
        assert (img != before).any()

    def test_G06_draw_bar_zero_pct(self):
        """pct=0이어도 외곽선은 그려져 크래시 없음"""
        img = self._img()
        gn.draw_bar(img, (0, 255, 0), 0.0, 10, 10, 100)

    def test_G06_apply_overlays_empty_returns_same(self):
        img = self._img()
        out = gn.apply_text_overlays(img, [])
        assert out is img

    def test_G06_apply_overlays_text(self):
        """오버레이가 있으면 이미지 반환 (영문 putText 경로)"""
        img = self._img()
        out = gn.apply_text_overlays(img, [('분노', (10, 50), (0, 0, 255), 30)])
        assert out is not None
        assert out.shape == img.shape

    def test_G06_apply_overlays_pil_path(self, monkeypatch):
        """_USE_PIL=True 면 PIL 한글 렌더링 경로 사용"""
        from PIL import ImageFont
        font = ImageFont.load_default()
        monkeypatch.setattr(gn, '_USE_PIL', True)
        monkeypatch.setattr(gn, '_KO_FONT', font)
        monkeypatch.setattr(gn, '_KO_FONT_SM', font)
        img = self._img()
        out = gn.apply_text_overlays(img, [('분노', (10, 50), (0, 0, 255), 30),
                                           ('일반', (10, 80), (0, 200, 200), 20)])
        assert out.shape == img.shape


# ═════════════════════════════════════════════════════════════
# G03b: _au_update_baseline 분기
# ═════════════════════════════════════════════════════════════
class TestAuUpdateBaseline:
    """G03b: AU 베이스라인 학습"""

    def _feat(self, v=0.2):
        return {'au12': v, 'mouth_w': v, 'eye_h': v, 'brow_eye': v, 'cheek': v}

    def test_G03b_locked_not_updated(self):
        with gn._AU_LOCK:
            gn._AU_BASELINE_LOCKED = True
        gn._au_update_baseline(self._feat())
        assert gn._AU_BUFS == {}

    def test_G03b_disable_learning_locks_immediately(self):
        gn._DISABLE_BASELINE_LEARNING = True
        gn._au_update_baseline(self._feat(0.3))
        assert gn._AU_BASELINE_LOCKED is True
        assert gn._AU_BASELINE['au12'] == 0.3

    def test_G03b_anger_freezes_learning(self):
        gn._au_in_anger = True
        gn._au_update_baseline(self._feat())
        assert gn._AU_BUFS == {}

    def test_G03b_locks_when_buffer_full(self):
        for _ in range(gn._AU_BUF_SIZE + 1):
            gn._au_update_baseline(self._feat())
        assert gn._AU_BASELINE_LOCKED is True


# ═════════════════════════════════════════════════════════════
# G03c: analyze_emotion_geometric 감정 분기 (feat mock)
# ═════════════════════════════════════════════════════════════
class TestGeometricEmotionBranches:
    """G03c: 기하학 분석의 행복/분노/눈감김 분기 (특징값 주입)"""

    _KEYS = ['au12', 'mouth_w', 'eye_h', 'brow_eye', 'cheek', 'ibrow_dist']

    def _prime(self, baseline):
        with gn._AU_LOCK:
            gn._AU_BASELINE.clear()
            gn._AU_BASELINE.update(baseline)
            gn._AU_BASELINE_LOCKED = True
            gn._AU_BUFS['au12'] = deque([0.0] * 60, maxlen=gn._AU_BUF_SIZE)

    def _run(self, monkeypatch, feat):
        monkeypatch.setattr(gn, '_compute_au_features', lambda lms: feat)
        return gn.analyze_emotion_geometric(object())

    def test_G03c_happy(self, monkeypatch):
        base = {k: 0.1 for k in self._KEYS}
        self._prime(base)
        feat = dict(base)
        feat['au12'] = 0.2       # au12 상승 → 미소
        feat['mouth_w'] = 0.2
        feat['eye_h'] = 0.1      # 눈 정상
        em, cf, calibrated, dbg = self._run(monkeypatch, feat)
        assert em == '행복'
        assert calibrated is True

    def test_G03c_angry(self, monkeypatch):
        base = {k: 0.1 for k in self._KEYS}
        self._prime(base)
        feat = dict(base)
        feat['ibrow_dist'] = 0.05   # 미간 좁아짐
        feat['brow_eye'] = 0.05     # 눈썹 내려옴
        feat['eye_h'] = 0.1
        em, cf, calibrated, dbg = self._run(monkeypatch, feat)
        assert em == '분노'
        assert gn._au_in_anger is True

    def test_G03c_eye_closed_suppresses(self, monkeypatch):
        base = {k: 0.1 for k in self._KEYS}
        self._prime(base)
        feat = dict(base)
        feat['ibrow_dist'] = 0.05
        feat['brow_eye'] = 0.05
        feat['eye_h'] = 0.02        # 베이스라인의 50% 미만 → 눈 감김
        em, cf, calibrated, dbg = self._run(monkeypatch, feat)
        assert em == '일반'

    def test_G03c_neutral(self, monkeypatch):
        base = {k: 0.1 for k in self._KEYS}
        self._prime(base)
        feat = dict(base)          # 변화 없음
        em, cf, calibrated, dbg = self._run(monkeypatch, feat)
        assert em == '일반'
        assert calibrated is True


# ═════════════════════════════════════════════════════════════
# G07: 자리비움(face absence) 상태
# ═════════════════════════════════════════════════════════════
class TestFaceAbsence:
    """G07: 얼굴 부재 타임아웃"""

    def test_G07_face_detected_resets(self):
        absent = gn._update_face_absence(True)
        assert absent is False
        assert gn._is_face_absent() is False

    def test_G07_absent_after_timeout(self):
        t0 = 1000.0
        gn._update_face_absence(True, t0)
        absent = gn._update_face_absence(False, t0 + gn._FACE_ABSENT_TIMEOUT_SEC + 1)
        assert absent is True
        assert gn._is_face_absent() is True

    def test_G07_not_absent_before_timeout(self):
        t0 = 2000.0
        gn._update_face_absence(True, t0)
        absent = gn._update_face_absence(False, t0 + 1.0)
        assert absent is False


# ═════════════════════════════════════════════════════════════
# G08: _window_emotion
# ═════════════════════════════════════════════════════════════
class TestWindowEmotion:
    """G08: 시간창 감정 집계"""

    def test_G08_none_when_empty(self):
        assert gn._window_emotion(1) is None

    def test_G08_none_when_not_enough_elapsed(self):
        now = time.time()
        with gn._emotion_hist_lock:
            gn._EMOTION_HISTORY.append((now - 5, '일반'))
        assert gn._window_emotion(1) is None  # elapsed<60s

    def test_G08_anger_ratio(self):
        now = time.time()
        with gn._emotion_hist_lock:
            gn._EMOTION_HISTORY.append((now - 120, '일반'))  # 오래된 첫 항목
            for _ in range(7):
                gn._EMOTION_HISTORY.append((now - 5, '분노'))
            for _ in range(3):
                gn._EMOTION_HISTORY.append((now - 5, '일반'))
        assert gn._window_emotion(1) == '분노'

    def test_G08_happy_ratio(self):
        now = time.time()
        with gn._emotion_hist_lock:
            gn._EMOTION_HISTORY.append((now - 120, '일반'))
            for _ in range(8):
                gn._EMOTION_HISTORY.append((now - 5, '행복'))
            for _ in range(2):
                gn._EMOTION_HISTORY.append((now - 5, '일반'))
        assert gn._window_emotion(1) == '행복'

    def test_G08_neutral_default(self):
        now = time.time()
        with gn._emotion_hist_lock:
            gn._EMOTION_HISTORY.append((now - 120, '일반'))
            for _ in range(10):
                gn._EMOTION_HISTORY.append((now - 5, '일반'))
        assert gn._window_emotion(1) == '일반'


# ═════════════════════════════════════════════════════════════
# G09: _load_json / _save_json / _next_event_within_1h
# ═════════════════════════════════════════════════════════════
class TestJsonAndEvents:
    """G09: JSON IO 및 1시간 내 일정 탐색"""

    def test_G09_load_missing_returns_empty(self, tmp_path):
        assert gn._load_json(str(tmp_path / 'nope.json')) == []

    def test_G09_save_then_load(self, tmp_path):
        p = str(tmp_path / 'd.json')
        gn._save_json(p, [{'a': 1}])
        assert gn._load_json(p) == [{'a': 1}]

    def test_G09_load_corrupt_returns_empty(self, tmp_path):
        p = tmp_path / 'bad.json'
        p.write_text('{not json', encoding='utf-8')
        assert gn._load_json(str(p)) == []

    def test_G09_next_event_within_1h(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta
        now = datetime.now()
        ev_time = (now + timedelta(minutes=30)).strftime('%H:%M')
        p = tmp_path / 'cal.json'
        p.write_text(json.dumps([{
            'id': 'e1', 'date': now.strftime('%Y-%m-%d'),
            'time': ev_time, 'title': '회의', 'note': ''
        }]), encoding='utf-8')
        monkeypatch.setattr(gn, '_CALENDAR_FILE', str(p))
        ev = gn._next_event_within_1h()
        assert ev is not None
        assert ev['title'] == '회의'

    def test_G09_next_event_none_when_far(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta
        now = datetime.now()
        ev_time = (now + timedelta(hours=3)).strftime('%H:%M')
        p = tmp_path / 'cal.json'
        p.write_text(json.dumps([{
            'id': 'e1', 'date': now.strftime('%Y-%m-%d'),
            'time': ev_time, 'title': '먼회의', 'note': ''
        }]), encoding='utf-8')
        monkeypatch.setattr(gn, '_CALENDAR_FILE', str(p))
        assert gn._next_event_within_1h() is None

    def test_G09_next_event_skips_other_date_and_bad_time(self, tmp_path, monkeypatch):
        """다른 날짜 이벤트는 건너뛰고, 잘못된 시간 포맷은 예외 무시"""
        from datetime import datetime, timedelta
        now = datetime.now()
        p = tmp_path / 'cal.json'
        p.write_text(json.dumps([
            {'id': 'a', 'date': '2000-01-01', 'time': '10:00', 'title': '과거'},
            {'id': 'b', 'date': now.strftime('%Y-%m-%d'),
             'time': '99:99', 'title': '잘못된시간'},
        ]), encoding='utf-8')
        monkeypatch.setattr(gn, '_CALENDAR_FILE', str(p))
        assert gn._next_event_within_1h() is None


class TestOpenCamera:
    """G09b: go_now_v2.open_camera (cv2 mock)"""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(gn.time, 'sleep', lambda s: None)

    def _cap(self, opened=True, frames=None):
        cap = MagicMock()
        cap.isOpened.return_value = opened
        cap.read.return_value = (False, None) if frames is None else None
        if frames is not None:
            cap.read.side_effect = frames
        return cap

    def test_G09b_first_index_success(self, monkeypatch):
        good = self._cap(True, [(True, MagicMock())])
        monkeypatch.setattr(gn.cv2, 'VideoCapture', lambda *a, **k: good)
        assert gn.open_camera(0) is good

    def test_G09b_gstreamer_fallback(self, monkeypatch):
        dead = self._cap(False)
        gst = self._cap(True, [(True, MagicMock())])
        monkeypatch.setattr(gn.cv2, 'VideoCapture',
                            lambda arg, backend=None: gst if isinstance(arg, str) else dead)
        assert gn.open_camera(0) is gst

    def test_G09b_all_fail_raises(self, monkeypatch):
        dead = self._cap(False)
        monkeypatch.setattr(gn.cv2, 'VideoCapture', lambda *a, **k: dead)
        with pytest.raises(RuntimeError):
            gn.open_camera(0)

    def test_G09b_warmup_exhausted_then_gstreamer_fails(self, monkeypatch):
        """열리지만 프레임이 안 들어오는 카메라 → 워밍업 소진 후 release,
        GStreamer 폴백도 프레임 실패 → RuntimeError (485-486, 497-498 경로)"""
        opened_no_frame = self._cap(True)  # isOpened True, read 항상 실패
        monkeypatch.setattr(gn.cv2, 'VideoCapture', lambda *a, **k: opened_no_frame)
        with pytest.raises(RuntimeError):
            gn.open_camera(0)
        assert opened_no_frame.release.called


# ═════════════════════════════════════════════════════════════
# G10: _mjpeg_generator
# ═════════════════════════════════════════════════════════════
class TestMjpegGenerator:
    """G10: MJPEG 스트림 제너레이터 (무한 루프 → 앞 2개만 검증)"""

    def test_G10_first_chunks_are_multipart(self, monkeypatch):
        monkeypatch.setattr(gn.time, 'sleep', lambda s: None)
        gen = gn._mjpeg_generator()
        first = next(gen)
        second = next(gen)
        third = next(gen)  # 루프 2회차(sleep 경유)까지 진입
        assert first.startswith(b'--frame')
        assert b'Content-Type: image/jpeg' in first
        assert second.startswith(b'--frame')
        assert third.startswith(b'--frame')


# ═════════════════════════════════════════════════════════════
# G11: Flask 라우트
# ═════════════════════════════════════════════════════════════
@pytest.fixture
def client():
    gn.app.config['TESTING'] = True
    return gn.app.test_client()


class TestStaticRoutes:
    """G11: 정적/대시보드 라우트"""

    def test_G11_index_serves_dashboard(self, client):
        r = client.get('/')
        assert r.status_code == 200

    def test_G11_dashboard_integrated(self, client):
        r = client.get('/dashboard-integrated')
        assert r.status_code == 200

    def test_G11_video_feed_response(self):
        """스트림 소비 없이 라우트 본문(Response 생성)만 검증"""
        with gn.app.test_request_context('/video_feed'):
            resp = gn.video_feed()
        assert 'multipart/x-mixed-replace' in resp.mimetype


class TestRecalibrate:
    """G12: /recalibrate"""

    def test_G12_recalibrate_clears_state(self, client):
        gn._BS_BASELINE['x'] = 1.0
        gn._BS_BASELINE_LOCKED = True
        r = client.post('/recalibrate')
        assert r.status_code == 200
        assert 'message' in r.get_json()
        assert gn._BS_BASELINE_LOCKED is False
        assert gn._BS_BASELINE == {}


class TestStatusRoute:
    """G13: /api/status"""

    def test_G13_face_absent(self, client):
        gn._face_absent = True
        r = client.get('/api/status')
        data = r.get_json()
        assert data['emotion'] == '자리비움'
        assert data['face_absent'] is True

    def test_G13_realtime_anger_red(self, client, monkeypatch):
        gn._face_absent = False
        gn._current_emotion = '분노'
        r = client.get('/api/status?window=realtime')
        data = r.get_json()
        assert data['emotion'] == '분노'
        assert data['signal'] == 'red'

    def test_G13_realtime_green_when_no_event(self, client, monkeypatch):
        gn._face_absent = False
        gn._current_emotion = '일반'
        monkeypatch.setattr(gn, '_next_event_within_1h', lambda: None)
        r = client.get('/api/status')
        data = r.get_json()
        assert data['signal'] == 'green'

    def test_G13_realtime_orange_with_event(self, client, monkeypatch):
        gn._face_absent = False
        gn._current_emotion = '일반'
        monkeypatch.setattr(gn, '_next_event_within_1h',
                            lambda: {'id': 'e', 'title': 't'})
        r = client.get('/api/status')
        data = r.get_json()
        assert data['signal'] == 'orange'
        assert data['has_upcoming_1h'] is True

    def test_G13_window_measuring(self, client, monkeypatch):
        gn._face_absent = False
        monkeypatch.setattr(gn, '_window_emotion', lambda m: None)
        r = client.get('/api/status?window=5')
        data = r.get_json()
        assert data['signal'] == 'measuring'

    def test_G13_window_emotion(self, client, monkeypatch):
        gn._face_absent = False
        monkeypatch.setattr(gn, '_window_emotion', lambda m: '일반')
        monkeypatch.setattr(gn, '_next_event_within_1h', lambda: None)
        r = client.get('/api/status?window=10')
        data = r.get_json()
        assert data['emotion'] == '일반'
        assert data['window'] == '10'


class TestEmotionHistoryRoute:
    """G14: /api/emotion_history"""

    def test_G14_returns_recent_only(self, client):
        now = time.time()
        with gn._emotion_hist_lock:
            gn._EMOTION_HISTORY.append((now - 10000, '분노'))  # 너무 오래됨
            gn._EMOTION_HISTORY.append((now - 5, '행복'))
        r = client.get('/api/emotion_history?minutes=5')
        data = r.get_json()
        emotions = [d['e'] for d in data]
        assert '행복' in emotions
        assert '분노' not in emotions

    def test_G14_minutes_clamped(self, client):
        r = client.get('/api/emotion_history?minutes=999')
        assert r.status_code == 200  # max 10으로 클램프, 크래시 없음


class TestRadarRoutes:
    """G15: /api/radar/* (get_processor mock)"""

    def test_G15_signals_ok(self, client, monkeypatch):
        proc = MagicMock()
        proc.get_all_signals.return_value = {'sigh': 'green'}
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        monkeypatch.setattr(gn, '_next_event_within_1h', lambda: None)
        gn._face_absent = False
        r = client.get('/api/radar/signals')
        data = r.get_json()
        assert data['sigh'] == 'green'
        assert 'has_upcoming_1h' in data

    def test_G15_signals_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', False)
        r = client.get('/api/radar/signals')
        assert r.status_code == 503

    def test_G15_add_sigh_ok(self, client, monkeypatch):
        proc = MagicMock()
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.post('/api/radar/add-sigh', json={'detected': True})
        assert r.status_code == 200
        assert r.get_json()['ok'] is True
        proc.add_sigh.assert_called_once()

    def test_G15_add_sigh_blocked_when_absent(self, client, monkeypatch):
        monkeypatch.setattr(gn, 'get_processor', lambda: MagicMock())
        gn._face_absent = True
        r = client.post('/api/radar/add-sigh', json={'detected': True})
        assert r.status_code == 423
        assert r.get_json()['blocked'] is True

    def test_G15_add_speech_emotion_ok(self, client, monkeypatch):
        proc = MagicMock()
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.post('/api/radar/add-speech-emotion', json={'emotion': '분노'})
        assert r.status_code == 200
        proc.add_speech_emotion.assert_called_once()

    def test_G15_add_speech_emotion_requires_emotion(self, client, monkeypatch):
        monkeypatch.setattr(gn, 'get_processor', lambda: MagicMock())
        gn._face_absent = False
        r = client.post('/api/radar/add-speech-emotion', json={'emotion': ''})
        assert r.status_code == 400

    def test_G15_speech_heartbeat(self, client, monkeypatch):
        proc = MagicMock()
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.post('/api/radar/speech-heartbeat')
        assert r.status_code == 200
        proc.mark_speech_alive.assert_called_once()

    def test_G15_radar_status(self, client, monkeypatch):
        proc = MagicMock()
        proc.get_status.return_value = {'state': 'ok'}
        proc.get_recent_data.return_value = []
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        r = client.get('/api/radar/status')
        data = r.get_json()
        assert data['state'] == 'ok'
        assert 'recent_data' in data


class TestDeviceStatus:
    """G16: /api/device-status"""

    def test_G16_device_status(self, client, monkeypatch):
        proc = MagicMock()
        proc.device_alive.return_value = {'sigh': True, 'speech': False}
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._camera_ok = True
        r = client.get('/api/device-status')
        data = r.get_json()
        assert data['sigh'] is True
        assert data['speech'] is False
        assert data['camera'] is True

    def test_G16_device_status_radar_unavailable(self, client, monkeypatch):
        """radar 미가용 → sigh/speech False 로만 응답 (787->795 분기)"""
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', False)
        r = client.get('/api/device-status')
        data = r.get_json()
        assert data['sigh'] is False
        assert data['speech'] is False


class TestSpeechCalibRoutes:
    """G17: 음성 캘리브레이션 라우트"""

    def test_G17_recalibrate_sets_flag(self, client):
        r = client.post('/api/speech-recalibrate')
        assert r.get_json()['ok'] is True
        assert gn._speech_recalib_flag.is_set()
        gn._speech_recalib_flag.clear()

    def test_G17_recalib_flag_get_and_clear(self, client):
        gn._speech_recalib_flag.set()
        r = client.get('/api/speech-recalibrate-flag')
        assert r.get_json()['requested'] is True
        assert not gn._speech_recalib_flag.is_set()  # 조회 후 클리어

    def test_G17_calib_status_post_then_get(self, client):
        client.post('/api/speech-calib-status', json={'phase': 'noise', 'level': 0.3})
        r = client.get('/api/speech-calib-status')
        data = r.get_json()
        assert data['phase'] == 'noise'
        assert data['level'] == 0.3


class TestOutlookRoutes:
    """G18: Outlook 동기화 라우트"""

    def test_G18_sync_ok(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_outlook_sync_now', lambda: (True, '완료'))
        r = client.post('/api/outlook/sync')
        assert r.status_code == 200
        assert r.get_json()['ok'] is True

    def test_G18_sync_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_OUTLOOK_AVAILABLE', False)
        r = client.post('/api/outlook/sync')
        assert r.status_code == 503

    def test_G18_sync_status(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_outlook_get_status',
                            lambda: {'ok': True, 'count': 3, 'ts': 1, 'message': 'x'})
        r = client.get('/api/outlook/sync-status')
        assert r.get_json()['count'] == 3

    def test_G18_sync_status_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_OUTLOOK_AVAILABLE', False)
        r = client.get('/api/outlook/sync-status')
        assert r.get_json()['ok'] is None


class TestCalendarRoutes:
    """G19: /api/calendar CRUD"""

    @pytest.fixture
    def cal_file(self, tmp_path, monkeypatch):
        p = tmp_path / 'cal.json'
        monkeypatch.setattr(gn, '_CALENDAR_FILE', str(p))
        return p

    def test_G19_get_empty(self, client, cal_file):
        r = client.get('/api/calendar')
        assert r.get_json() == []

    def test_G19_post_creates(self, client, cal_file):
        r = client.post('/api/calendar',
                        json={'date': '2026-06-30', 'time': '10:00', 'title': '회의'})
        assert r.status_code == 201
        new = r.get_json()
        assert new['title'] == '회의'
        assert 'id' in new
        assert len(client.get('/api/calendar').get_json()) == 1

    def test_G19_put_updates(self, client, cal_file):
        new = client.post('/api/calendar', json={'title': '원본'}).get_json()
        client.put(f"/api/calendar/{new['id']}", json={'title': '수정됨'})
        items = client.get('/api/calendar').get_json()
        assert items[0]['title'] == '수정됨'

    def test_G19_delete(self, client, cal_file):
        new = client.post('/api/calendar', json={'title': '삭제대상'}).get_json()
        r = client.delete(f"/api/calendar/{new['id']}")
        assert r.get_json()['ok'] is True
        assert client.get('/api/calendar').get_json() == []

    def test_G19_put_second_item_skips_first(self, client, cal_file):
        """첫 항목 id 불일치 → 건너뛰고 두 번째 항목 수정 (913->912 분기)"""
        a = client.post('/api/calendar', json={'title': 'A'}).get_json()
        b = client.post('/api/calendar', json={'title': 'B'}).get_json()
        client.put(f"/api/calendar/{b['id']}", json={'title': 'B수정'})
        items = {e['title'] for e in client.get('/api/calendar').get_json()}
        assert 'A' in items and 'B수정' in items

    def test_G19_put_nonexistent_id_no_match(self, client, cal_file):
        """존재하지 않는 id → 루프가 매칭 없이 완료 (912->918 분기)"""
        client.post('/api/calendar', json={'title': 'A'})
        r = client.put('/api/calendar/does-not-exist', json={'title': 'X'})
        assert r.get_json()['ok'] is True


class TestApprovalRoutes:
    """G20: /api/approvals CRUD"""

    @pytest.fixture
    def ap_file(self, tmp_path, monkeypatch):
        p = tmp_path / 'ap.json'
        monkeypatch.setattr(gn, '_APPROVAL_FILE', str(p))
        return p

    def test_G20_get_empty(self, client, ap_file):
        assert client.get('/api/approvals').get_json() == []

    def test_G20_post_creates(self, client, ap_file):
        r = client.post('/api/approvals', json={'title': '결재1'})
        assert r.status_code == 201
        assert r.get_json()['title'] == '결재1'

    def test_G20_put_updates(self, client, ap_file):
        new = client.post('/api/approvals', json={'title': 'A'}).get_json()
        client.put(f"/api/approvals/{new['id']}", json={'title': 'B', 'note': 'n'})
        items = client.get('/api/approvals').get_json()
        assert items[0]['title'] == 'B'
        assert items[0]['note'] == 'n'

    def test_G20_delete(self, client, ap_file):
        new = client.post('/api/approvals', json={'title': 'X'}).get_json()
        client.delete(f"/api/approvals/{new['id']}")
        assert client.get('/api/approvals').get_json() == []

    def test_G20_put_second_item_skips_first(self, client, ap_file):
        """첫 항목 불일치 후 두 번째 수정 (948->947 분기)"""
        client.post('/api/approvals', json={'title': 'A'})
        b = client.post('/api/approvals', json={'title': 'B'}).get_json()
        client.put(f"/api/approvals/{b['id']}", json={'title': 'B수정'})
        titles = {a['title'] for a in client.get('/api/approvals').get_json()}
        assert 'A' in titles and 'B수정' in titles

    def test_G20_put_nonexistent_id_no_match(self, client, ap_file):
        """존재하지 않는 id → 매칭 없이 루프 완료 (947->953 분기)"""
        client.post('/api/approvals', json={'title': 'A'})
        r = client.put('/api/approvals/nope', json={'title': 'X'})
        assert r.get_json()['ok'] is True


# ═════════════════════════════════════════════════════════════
# G22: 라우트 예외/엣지 경로
# ═════════════════════════════════════════════════════════════
class TestRouteErrorPaths:
    """G22: 라우트의 503/500/400/202 등 예외 분기"""

    def test_G22_signals_processor_error_500(self, client, monkeypatch):
        proc = MagicMock()
        proc.get_all_signals.side_effect = RuntimeError("boom")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.get('/api/radar/signals')
        assert r.status_code == 500
        assert 'error' in r.get_json()

    def test_G22_add_sigh_unavailable_503(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', False)
        r = client.post('/api/radar/add-sigh', json={'detected': True})
        assert r.status_code == 503

    def test_G22_add_sigh_processor_error_400(self, client, monkeypatch):
        proc = MagicMock()
        proc.add_sigh.side_effect = RuntimeError("x")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.post('/api/radar/add-sigh', json={'detected': True})
        assert r.status_code == 400

    def test_G22_add_speech_unavailable_503(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', False)
        r = client.post('/api/radar/add-speech-emotion', json={'emotion': '분노'})
        assert r.status_code == 503

    def test_G22_add_speech_blocked_423(self, client, monkeypatch):
        monkeypatch.setattr(gn, 'get_processor', lambda: MagicMock())
        gn._face_absent = True
        r = client.post('/api/radar/add-speech-emotion', json={'emotion': '분노'})
        assert r.status_code == 423

    def test_G22_add_speech_value_error_400(self, client, monkeypatch):
        proc = MagicMock()
        proc.add_speech_emotion.side_effect = ValueError("bad")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.post('/api/radar/add-speech-emotion', json={'emotion': '분노'})
        assert r.status_code == 400

    def test_G22_add_speech_generic_error_500(self, client, monkeypatch):
        proc = MagicMock()
        proc.add_speech_emotion.side_effect = RuntimeError("boom")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.post('/api/radar/add-speech-emotion', json={'emotion': '분노'})
        assert r.status_code == 500

    def test_G22_speech_heartbeat_unavailable_503(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', False)
        r = client.post('/api/radar/speech-heartbeat')
        assert r.status_code == 503

    def test_G22_speech_heartbeat_blocked_423(self, client, monkeypatch):
        monkeypatch.setattr(gn, 'get_processor', lambda: MagicMock())
        gn._face_absent = True
        r = client.post('/api/radar/speech-heartbeat')
        assert r.status_code == 423

    def test_G22_speech_heartbeat_error_500(self, client, monkeypatch):
        proc = MagicMock()
        proc.mark_speech_alive.side_effect = RuntimeError("x")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        gn._face_absent = False
        r = client.post('/api/radar/speech-heartbeat')
        assert r.status_code == 500

    def test_G22_radar_status_unavailable_503(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', False)
        r = client.get('/api/radar/status')
        assert r.status_code == 503

    def test_G22_radar_status_error_500(self, client, monkeypatch):
        proc = MagicMock()
        proc.get_status.side_effect = RuntimeError("x")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        r = client.get('/api/radar/status')
        assert r.status_code == 500

    def test_G22_device_status_swallows_error(self, client, monkeypatch):
        proc = MagicMock()
        proc.device_alive.side_effect = RuntimeError("x")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        r = client.get('/api/device-status')
        # 예외를 삼키고 sigh/speech=False 로 응답
        assert r.status_code == 200
        assert r.get_json()['sigh'] is False

    def test_G22_outlook_sync_timeout_202(self, client, monkeypatch):
        """동기화 스레드가 30초 내 끝나지 않으면 202"""
        monkeypatch.setattr(gn, '_outlook_sync_now',
                            lambda: (_ for _ in ()).throw(AssertionError("호출 안 됨")))

        # join 이 즉시 반환되도록 Thread mock → result 비어있음 → 202
        class _FakeThread:
            def __init__(self, target=None, daemon=None):
                pass
            def start(self):
                pass
            def join(self, timeout=None):
                pass
        monkeypatch.setattr(gn.threading, 'Thread', _FakeThread)
        r = client.post('/api/outlook/sync')
        assert r.status_code == 202

    def test_G22_outlook_sync_failure_500(self, client, monkeypatch):
        monkeypatch.setattr(gn, '_outlook_sync_now', lambda: (False, '실패'))
        r = client.post('/api/outlook/sync')
        assert r.status_code == 500
        assert r.get_json()['ok'] is False

    def test_G22_recalib_flag_not_set(self, client):
        gn._speech_recalib_flag.clear()
        r = client.get('/api/speech-recalibrate-flag')
        assert r.get_json()['requested'] is False


# ═════════════════════════════════════════════════════════════
# G21: camera_loop / main (하드웨어/서버 — 진입·예외 경로만)
# ═════════════════════════════════════════════════════════════
class _Result:
    """FaceMesh.process() 결과 흉내."""
    def __init__(self, faces):
        self.multi_face_landmarks = faces


class _FakeCap:
    """cv2.VideoCapture 흉내: good 프레임 N개 후 (False, None) 으로 루프 종료."""
    def __init__(self, n_good):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._seq = [(True, frame.copy()) for _ in range(n_good)] + [(False, None)]
        self._i = 0
        self.released = False

    def set(self, *a):
        return True

    def get(self, prop):
        return 640.0

    def read(self):
        r = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return r

    def release(self):
        self.released = True


class _FakeMesh:
    def __init__(self, result):
        self._result = result
        self.closed = False

    def process(self, rgb):
        return self._result

    def close(self):
        self.closed = True


def _run_camera_loop(monkeypatch, result, n_good=5):
    """camera_loop 를 mock 카메라/FaceMesh 로 구동."""
    cap = _FakeCap(n_good)
    monkeypatch.setattr(gn, 'open_camera', lambda idx: cap)
    monkeypatch.setattr(gn.mp.solutions.face_mesh, 'FaceMesh',
                        lambda **kw: _FakeMesh(result))
    gn.camera_loop(0)
    return cap


class TestCameraLoop:
    """G21: 카메라 처리 루프 (mock 카메라/FaceMesh)"""

    def test_G21_open_failure_returns(self, monkeypatch):
        """open_camera 가 RuntimeError 면 루프 진입 없이 반환"""
        def _raise(idx):
            raise RuntimeError("no camera")
        monkeypatch.setattr(gn, 'open_camera', _raise)
        gn.camera_loop(0)  # 예외 없이 반환되면 성공

    def test_G21_no_face_absent_overlay(self, monkeypatch):
        """얼굴 미검출 + 자리비움 오버레이 경로"""
        gn._last_face_seen_ts = 0.0  # 즉시 자리비움 판정
        cap = _run_camera_loop(monkeypatch, _Result(None))
        assert cap.released is True
        assert gn._current_emotion == '자리비움'

    def test_G21_face_not_frontal(self, monkeypatch):
        """얼굴 검출되었으나 비정면 → 안내 오버레이 후 continue"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: False)
        cap = _run_camera_loop(monkeypatch, _Result([face]))
        assert cap.released is True

    def test_G21_calibrating_overlay(self, monkeypatch):
        """정면 + 캘리브레이션 중(calibrated=False) → 진행 표시 후 continue"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.5, False, {'calib_pct': 0.4}))
        cap = _run_camera_loop(monkeypatch, _Result([face]))
        assert cap.released is True

    def test_G21_calibrated_anger_records_history(self, monkeypatch):
        """정면 + calibrated 분노 → 스무딩/이력기록/레이더 전송/그리기"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('분노', 0.9, True, {}))
        proc = MagicMock()
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', True)
        gn._last_record_ts = 0.0
        cap = _run_camera_loop(monkeypatch, _Result([face]), n_good=6)
        assert cap.released is True
        assert gn._current_emotion == '분노'
        assert len(gn._EMOTION_HISTORY) >= 1
        proc.add_expression.assert_called()

    def test_G21_calibrated_neutral(self, monkeypatch):
        """정면 + calibrated 일반 → 다수결 분기(else)"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.6, True, {}))
        gn._last_record_ts = 0.0
        cap = _run_camera_loop(monkeypatch, _Result([face]), n_good=5)
        assert gn._current_emotion == '일반'

    def test_G21_blendshape_tier1_path(self, monkeypatch):
        """Tier-1(blendshape) 활성 경로: _face_landmarker.detect 사용"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', True)

        class _BS:
            def __init__(self, name, score):
                self.category_name = name; self.score = score
        t1_result = MagicMock()
        t1_result.face_blendshapes = [[_BS('mouthSmileLeft', 0.5)]]
        landmarker = MagicMock()
        landmarker.detect.return_value = t1_result
        monkeypatch.setattr(gn, '_face_landmarker', landmarker)
        monkeypatch.setattr(gn.mp, 'Image', lambda **kw: object())
        monkeypatch.setattr(gn, 'analyze_blendshapes',
                            lambda bs: ('행복', 0.8, True, {}))
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('행복', 0.8, True, {}))
        gn._last_record_ts = 0.0
        cap = _run_camera_loop(monkeypatch, _Result([face]), n_good=6)
        assert cap.released is True
        landmarker.detect.assert_called()

    def test_G21_tier1_detect_exception_swallowed(self, monkeypatch):
        """Tier-1 detect 예외 → 경고 후 Tier-2 로 진행"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', True)
        landmarker = MagicMock()
        landmarker.detect.side_effect = RuntimeError("detect boom")
        monkeypatch.setattr(gn, '_face_landmarker', landmarker)
        monkeypatch.setattr(gn.mp, 'Image', lambda **kw: object())
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.5, True, {}))
        cap = _run_camera_loop(monkeypatch, _Result([face]), n_good=6)
        assert cap.released is True

    def test_G21_radar_add_expression_exception_swallowed(self, monkeypatch):
        """이력 기록 시 radar add_expression 예외는 무시"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.6, True, {}))
        proc = MagicMock()
        proc.add_expression.side_effect = RuntimeError("x")
        monkeypatch.setattr(gn, 'get_processor', lambda: proc)
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', True)
        gn._last_record_ts = 0.0
        cap = _run_camera_loop(monkeypatch, _Result([face]), n_good=5)
        assert cap.released is True

    def test_G21_recovers_from_absent(self, monkeypatch):
        """직전이 '자리비움' 인데 얼굴 복귀(미부재) → '일반' 으로 복원 (1032)"""
        gn._current_emotion = '자리비움'
        gn._last_face_seen_ts = time.time()  # 부재 아님
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.5, False, {'calib_pct': 0.3}))
        _run_camera_loop(monkeypatch, _Result([make_landmarks()]), n_good=4)
        assert gn._current_emotion != '자리비움'

    def test_G21_record_without_radar(self, monkeypatch):
        """이력 기록 시 radar 미가용 → add_expression 미호출 (1123->1128)"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.6, True, {}))
        monkeypatch.setattr(gn, '_RADAR_AVAILABLE', False)
        gn._last_record_ts = 0.0
        _run_camera_loop(monkeypatch, _Result([face]), n_good=5)
        assert len(gn._EMOTION_HISTORY) >= 1

    def test_G21_blendshape_empty_result(self, monkeypatch):
        """Tier-1 detect 가 빈 blendshapes → bs 분석 건너뜀 (1072->1079)"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', True)
        t1_result = MagicMock()
        t1_result.face_blendshapes = []   # 비어있음
        landmarker = MagicMock()
        landmarker.detect.return_value = t1_result
        monkeypatch.setattr(gn, '_face_landmarker', landmarker)
        monkeypatch.setattr(gn.mp, 'Image', lambda **kw: object())
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.5, True, {}))
        cap = _run_camera_loop(monkeypatch, _Result([face]), n_good=6)
        assert cap.released is True

    def test_G21_landmarker_close_exception(self, monkeypatch):
        """정리 단계에서 _face_landmarker.close() 예외 무시 (1176-1177)"""
        face = make_landmarks()
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', True)
        landmarker = MagicMock()
        landmarker.detect.return_value = MagicMock(face_blendshapes=[])
        landmarker.close.side_effect = RuntimeError("close boom")
        monkeypatch.setattr(gn, '_face_landmarker', landmarker)
        monkeypatch.setattr(gn.mp, 'Image', lambda **kw: object())
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.5, True, {}))
        gn._last_record_ts = 0.0
        cap = _run_camera_loop(monkeypatch, _Result([face]), n_good=5)
        assert cap.released is True
        landmarker.close.assert_called()

    def test_G21_imencode_failure_branches(self, monkeypatch):
        """imencode 실패 시 저장 스킵 분기들 (1056->1059 / 1097->1100 / 1167->1003)"""
        monkeypatch.setattr(gn.cv2, 'imencode', lambda *a, **k: (False, None))
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        # calibrated 경로(하단 imencode) + 비정면/캘리브 경로를 한 번씩 태우기 위해
        # 정면+calibrated 로 하단 분기(1167->1003) 커버
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.6, True, {}))
        gn._last_record_ts = 0.0
        cap = _run_camera_loop(monkeypatch, _Result([make_landmarks()]), n_good=4)
        assert cap.released is True

    def test_G21_imencode_failure_not_frontal(self, monkeypatch):
        """비정면 경로의 imencode 실패 분기 (1056->1059)"""
        monkeypatch.setattr(gn.cv2, 'imencode', lambda *a, **k: (False, None))
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: False)
        cap = _run_camera_loop(monkeypatch, _Result([make_landmarks()]), n_good=4)
        assert cap.released is True

    def test_G21_imencode_failure_calibrating(self, monkeypatch):
        """캘리브레이션 경로의 imencode 실패 분기 (1097->1100)"""
        monkeypatch.setattr(gn.cv2, 'imencode', lambda *a, **k: (False, None))
        monkeypatch.setattr(gn, 'is_frontal_face', lambda lms: True)
        monkeypatch.setattr(gn, '_USE_BLENDSHAPES', False)
        monkeypatch.setattr(gn, 'analyze_emotion_geometric',
                            lambda lms: ('일반', 0.5, False, {'calib_pct': 0.4}))
        cap = _run_camera_loop(monkeypatch, _Result([make_landmarks()]), n_good=4)
        assert cap.released is True


class TestMain:
    """G23: main 부트스트랩"""

    def test_G23_main_bootstraps_without_blocking(self, monkeypatch):
        monkeypatch.setattr(gn, 'camera_loop', lambda idx: None)
        run_called = {}
        monkeypatch.setattr(gn.app, 'run', lambda **kw: run_called.update(kw))
        monkeypatch.setattr(gn, '_outlook_start_auto_sync', lambda **kw: None)
        monkeypatch.setattr(sys, 'argv', ['go_now_v2.py', '0'])
        gn.main()
        assert run_called.get('port') == 5000

    def test_G23_main_socket_failure_fallback(self, monkeypatch):
        """IP 조회 socket 예외 → fallback IP 로 계속 진행"""
        monkeypatch.setattr(gn, 'camera_loop', lambda idx: None)
        monkeypatch.setattr(gn.app, 'run', lambda **kw: None)
        monkeypatch.setattr(gn, '_outlook_start_auto_sync', lambda **kw: None)
        import socket as _sock
        def _boom(*a, **k):
            raise OSError("no network")
        monkeypatch.setattr(_sock, 'socket', _boom)
        monkeypatch.setattr(sys, 'argv', ['go_now_v2.py'])
        gn.main()  # 예외 없이 반환되면 성공

    def test_G23_main_without_outlook(self, monkeypatch):
        """_OUTLOOK_AVAILABLE=False → 자동 동기화 미시작"""
        monkeypatch.setattr(gn, 'camera_loop', lambda idx: None)
        monkeypatch.setattr(gn.app, 'run', lambda **kw: None)
        monkeypatch.setattr(gn, '_OUTLOOK_AVAILABLE', False)
        called = {'n': 0}
        monkeypatch.setattr(gn, '_outlook_start_auto_sync',
                            lambda **kw: called.__setitem__('n', called['n'] + 1))
        monkeypatch.setattr(sys, 'argv', ['go_now_v2.py'])
        gn.main()
        assert called['n'] == 0


# ═════════════════════════════════════════════════════════════
# G24: _init_face_landmarker
# ═════════════════════════════════════════════════════════════
class TestInitFaceLandmarker:
    """G24: Tier-1 Face Landmarker 초기화"""

    _MODEL = os.path.join(os.path.dirname(os.path.abspath(gn.__file__)),
                          'face_landmarker.task')

    def test_G24_no_model_file_noop(self, monkeypatch):
        """모델 파일 없으면 아무것도 안 함"""
        monkeypatch.setattr(gn.os.path, 'exists', lambda p: False)
        gn._init_face_landmarker()
        assert gn._USE_BLENDSHAPES is False

    def test_G24_success_activates_blendshapes(self, monkeypatch):
        """모델 로드 성공 → _USE_BLENDSHAPES=True (57-70 경로)"""
        from mediapipe.tasks.python import vision as mpv
        monkeypatch.setattr(gn.os.path, 'exists',
                            lambda p: p == self._MODEL)
        monkeypatch.setattr(mpv.FaceLandmarker, 'create_from_options',
                            staticmethod(lambda opts: MagicMock()))
        gn._init_face_landmarker()
        assert gn._USE_BLENDSHAPES is True
        assert gn._face_landmarker is not None

    def test_G24_init_failure_falls_back(self, monkeypatch):
        """모델 로드 실패(예외) → 경고 후 Tier-2 유지 (71-72 경로)"""
        from mediapipe.tasks.python import vision as mpv
        def _boom(opts):
            raise RuntimeError("bad model")
        monkeypatch.setattr(gn.os.path, 'exists',
                            lambda p: p == self._MODEL)
        monkeypatch.setattr(mpv.FaceLandmarker, 'create_from_options',
                            staticmethod(_boom))
        gn._init_face_landmarker()
        assert gn._USE_BLENDSHAPES is False


# ═════════════════════════════════════════════════════════════
# G25: import 시점 의존성 폴백 (모듈 재실행)
# ═════════════════════════════════════════════════════════════
class TestImportTimeFallbacks:
    """G25: radar/outlook ImportError, PIL 폰트 로드 분기 — 모듈을 격리 재실행"""

    _SRC = os.path.join(os.path.dirname(os.path.abspath(gn.__file__)), 'go_now_v2.py')

    def _reexec(self, monkeypatch, *, hide_radar=False, hide_outlook=False,
                fonts=False, hide_pil=False, model=False):
        """go_now_v2.py 소스를 격리 네임스페이스에서 재실행."""
        with open(self._SRC, 'r', encoding='utf-8') as f:
            src = f.read()
        code = compile(src, self._SRC, 'exec')

        saved = {}
        def _hide(name):
            saved[name] = sys.modules.get(name, '__absent__')
            sys.modules[name] = None  # import 시 ImportError 유발

        if hide_radar:
            _hide('radar_signal_processor')
        if hide_outlook:
            _hide('outlook_sync')
        if hide_pil:
            _hide('PIL')

        if fonts:
            from PIL import ImageFont
            real_exists = os.path.exists
            font_paths = [
                '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf',
                '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf',
            ]
            monkeypatch.setattr(os.path, 'exists',
                                lambda p: True if p in font_paths else real_exists(p))
            monkeypatch.setattr(ImageFont, 'truetype',
                                lambda fp, size: MagicMock())

        if model:
            from mediapipe.tasks.python import vision as mpv
            real_exists = os.path.exists
            monkeypatch.setattr(os.path, 'exists',
                                lambda p: True if p.endswith('face_landmarker.task')
                                else real_exists(p))
            monkeypatch.setattr(mpv.FaceLandmarker, 'create_from_options',
                                staticmethod(lambda opts: MagicMock()))

        ns = {'__name__': 'go_now_v2_reexec', '__file__': self._SRC}
        try:
            exec(code, ns)
        finally:
            for name, val in saved.items():
                if val == '__absent__':
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = val
        return ns

    def test_G25_radar_import_fallback(self, monkeypatch):
        """radar 모듈 ImportError → _RADAR_AVAILABLE=False (31-33)"""
        ns = self._reexec(monkeypatch, hide_radar=True)
        assert ns['_RADAR_AVAILABLE'] is False

    def test_G25_outlook_import_fallback(self, monkeypatch):
        """outlook 모듈 ImportError → _OUTLOOK_AVAILABLE=False (39-41)"""
        ns = self._reexec(monkeypatch, hide_outlook=True)
        assert ns['_OUTLOOK_AVAILABLE'] is False

    def test_G25_font_loaded_path(self, monkeypatch):
        """폰트 후보가 존재 → _USE_PIL=True 분기 (93-99)"""
        ns = self._reexec(monkeypatch, fonts=True)
        assert ns['_USE_PIL'] is True
        assert ns['_KO_FONT'] is not None

    def test_G25_pil_import_fallback(self, monkeypatch):
        """PIL ImportError → 영문 표시 폴백 (104-105)"""
        ns = self._reexec(monkeypatch, hide_pil=True)
        assert ns['_USE_PIL'] is False

    def test_G25_blendshapes_active_skips_notice(self, monkeypatch):
        """모델 로드 성공 → _USE_BLENDSHAPES=True (108->115 분기)"""
        ns = self._reexec(monkeypatch, model=True)
        assert ns['_USE_BLENDSHAPES'] is True
