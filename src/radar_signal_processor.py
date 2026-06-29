#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radar_signal_processor.py
────────────────────────────────────────────────────────────────
결재 레이더 신호등 통합 판정기.

세 가지 입력을 타임스탬프와 함께 롤링 버퍼에 누적하고,
최근 1분 / 5분 / 10분 구간을 각각 독립적으로 평가해 신호등을 출력한다.

  · 표정정보   (카메라)  : go_now_v2.py 에서 add_expression()
  · 한숨정보   (음성)    : mic_agent.py 에서 POST → add_sigh()
  · 발화 감정  (발화)    : mic_agent.py 에서 POST → add_speech_emotion()

판정 알고리즘 (우선순위, 먼저 만족하는 규칙에서 확정):
  1) 구간에 '한숨 발생' 1회 이상            → 빨강 (최우선 override)
  2) 표정 대표값 == 분노                    → 빨강
     표정 대표값 == 행복                    → 초록
  3) 표정 대표값 == 보통(일반) → 발화 부정 개수 >= 임계값(구간) → 빨강
       임계값: 1분→1, 5분→2, 10분→3
  4) 그 외                                  → 노랑

표정 대표값(평균) 계산: go_now_v2.py 의 _window_emotion 과 동일한 비율 기준.
  분노 비율 >= 0.30 → 분노 / 행복 비율 >= 0.50 → 행복 / 그 외 → 일반(보통)

이 모듈은 표준 라이브러리만 사용한다 (numpy 등 외부 의존성 없음).
────────────────────────────────────────────────────────────────
"""

import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

# 발화 감정 6종 → 극성 매핑 (긍정 = 기쁨, 그 외 5종 = 부정)
POSITIVE_EMOTIONS = {"기쁨"}
NEGATIVE_EMOTIONS = {"슬픔", "분노", "불안", "당황", "상처"}
VALID_SPEECH_EMOTIONS = POSITIVE_EMOTIONS | NEGATIVE_EMOTIONS

# 구간 정의 (대시보드 키 ↔ 초)
WINDOWS = {"1m": 60, "5m": 300, "10m": 600}

# 구간별 발화 부정 임계값 (이 개수 이상이면 빨강)
NEG_THRESHOLD = {"1m": 1, "5m": 2, "10m": 3}

# 표정 대표값 비율 임계 (go_now_v2._window_emotion 과 일치)
ANGER_RATIO_TH = 0.30
HAPPY_RATIO_TH = 0.50

# 표정 라벨 정규화 — go_now 는 '일반', 스펙은 '보통' 을 쓰므로 '일반' 으로 통일
NEUTRAL_LABEL = "일반"

# 버퍼 최대 길이 (10분 + 여유. 표정 1Hz, 한숨/발화는 더 드묾)
_BUF_MAXLEN = 5000


# ─────────────────────────────────────────────────────────────
# 타임스탬프 버퍼
# ─────────────────────────────────────────────────────────────
class TimedBuffer:
    """(timestamp, value) 튜플을 시간순으로 보관하는 스레드 안전 버퍼."""

    def __init__(self, maxlen: int = _BUF_MAXLEN):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, value, ts: float = None):
        if ts is None:
            ts = time.time()
        with self._lock:
            self._buf.append((float(ts), value))

    def get_window(self, seconds: float):
        """최근 `seconds` 초 이내의 [(ts, value), ...] 반환."""
        cutoff = time.time() - seconds
        with self._lock:
            return [(ts, v) for ts, v in self._buf if ts >= cutoff]

    def tail(self, n: int):
        with self._lock:
            return list(self._buf)[-n:]

    def first_ts(self):
        with self._lock:
            return self._buf[0][0] if self._buf else None

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ─────────────────────────────────────────────────────────────
# 신호등 프로세서
# ─────────────────────────────────────────────────────────────
class RadarSignalProcessor:
    def __init__(self):
        self.expression_buffer = TimedBuffer()
        self.sigh_buffer = TimedBuffer()
        self.speech_buffer = TimedBuffer()
        self._start_ts = time.time()
        self._sigh_alive_ts = 0.0     # 마지막 한숨 입력/heartbeat 시각
        self._speech_alive_ts = 0.0   # 마지막 발화 입력/heartbeat 시각

    # ── 입력 ─────────────────────────────────────────────────
    def add_expression(self, emotion: str, timestamp: float = None):
        """표정 결과 추가. 입력 라벨('분노'/'행복'/'일반'/'보통')을 정규화."""
        if emotion in ("보통", "중립", "neutral", "Neutral"):
            emotion = NEUTRAL_LABEL
        self.expression_buffer.add(emotion, timestamp)

    def add_sigh(self, detected: bool, timestamp: float = None):
        """한숨 감지 결과 추가. detected=True 만 판정에 영향(heartbeat용 False 허용)."""
        self.sigh_buffer.add(bool(detected), timestamp)
        self._sigh_alive_ts = timestamp if timestamp else time.time()

    def add_speech_emotion(self, emotion: str, timestamp: float = None):
        """발화 감정 추가. 6종 외 입력은 ValueError."""
        if emotion not in VALID_SPEECH_EMOTIONS:
            raise ValueError(
                f"알 수 없는 발화 감정: {emotion} "
                f"(허용: {sorted(VALID_SPEECH_EMOTIONS)})"
            )
        self.speech_buffer.add(emotion, timestamp)
        self._speech_alive_ts = timestamp if timestamp else time.time()

    def mark_sigh_alive(self, timestamp: float = None):
        self._sigh_alive_ts = timestamp if timestamp else time.time()

    def mark_speech_alive(self, timestamp: float = None):
        """발화 파이프라인 alive 표시 (감정 데이터 없이 장치 상태만 갱신)."""
        self._speech_alive_ts = timestamp if timestamp else time.time()

    def device_alive(self, timeout: float = 20.0) -> dict:
        """최근 timeout 초 내 입력/heartbeat 가 있으면 해당 장치 alive."""
        now = time.time()
        return {
            "sigh": (now - self._sigh_alive_ts) < timeout,
            "speech": (now - self._speech_alive_ts) < timeout,
        }

    # ── 표정 대표값 ──────────────────────────────────────────
    def _representative_expr(self, samples) -> str:
        """구간 표정 샘플 → 대표값('분노'/'행복'/'일반', 없으면 '불명')."""
        total = len(samples)
        if total == 0:
            return "불명"
        anger = sum(1 for s in samples if s == "분노") / total
        if anger >= ANGER_RATIO_TH:
            return "분노"
        happy = sum(1 for s in samples if s == "행복") / total
        if happy >= HAPPY_RATIO_TH:
            return "행복"
        return NEUTRAL_LABEL

    # ── 단일 구간 판정 ───────────────────────────────────────
    def _decide_window(self, win_key: str) -> dict:
        seconds = WINDOWS[win_key]

        expr_items = self.expression_buffer.get_window(seconds)
        sigh_items = self.sigh_buffer.get_window(seconds)
        speech_items = self.speech_buffer.get_window(seconds)

        expr_samples = [v for _, v in expr_items]
        sigh_detected = any(v for _, v in sigh_items)
        neg_count = sum(1 for _, v in speech_items if v in NEGATIVE_EMOTIONS)
        expr = self._representative_expr(expr_samples)

        # 구간이 실제로 찼는지 — 첫 표정 샘플 이후 경과 시간이 구간 길이 미만이면
        # 아직 '측정중'(ready=False). go_now_v2._window_emotion 의 게이팅과 동일.
        first_expr = self.expression_buffer.first_ts()
        ready = (first_expr is not None) and (time.time() - first_expr >= seconds)

        # ── 우선순위 판정 ──
        if sigh_detected:                       # 1) 한숨 최우선
            signal = "red"
        elif expr == "분노":                    # 2) 표정 분노
            signal = "red"
        elif expr == "행복":                    #    표정 행복
            signal = "green"
        elif neg_count >= NEG_THRESHOLD[win_key]:  # 3) 발화 부정 누적
            signal = "red"
        else:                                   # 4) 기본값
            signal = "yellow"

        return {
            "signal": signal,
            "expr": expr,
            "sigh": sigh_detected,
            "neg_count": neg_count,
            "ready": ready,
            "data_points": {
                "expression": len(expr_samples),
                "sigh": len(sigh_items),
                "speech": len(speech_items),
            },
        }

    # ── 전체 신호 (1m/5m/10m) ────────────────────────────────
    def get_all_signals(self) -> dict:
        out = {key: self._decide_window(key) for key in WINDOWS}
        out["updated_at"] = datetime.now(KST).isoformat()
        return out

    # ── 디버깅용 상태 ────────────────────────────────────────
    def get_status(self) -> dict:
        return {
            "buffer_sizes": {
                "expression": len(self.expression_buffer),
                "sigh": len(self.sigh_buffer),
                "speech": len(self.speech_buffer),
            },
            "is_ready": len(self.expression_buffer) > 0,
            "uptime_sec": round(time.time() - self._start_ts, 1),
        }

    def get_recent_data(self, n: int = 10) -> dict:
        def fmt(buf):
            return [{"t": round(ts, 2), "v": v} for ts, v in buf.tail(n)]
        return {
            "expression": fmt(self.expression_buffer),
            "sigh": fmt(self.sigh_buffer),
            "speech": fmt(self.speech_buffer),
        }


# ─────────────────────────────────────────────────────────────
# 싱글턴 — go_now_v2.py 의 get_processor() 가 이걸 호출
# ─────────────────────────────────────────────────────────────
_processor = None
_proc_lock = threading.Lock()


def get_processor() -> RadarSignalProcessor:
    global _processor
    if _processor is None:
        with _proc_lock:
            if _processor is None:
                _processor = RadarSignalProcessor()
    return _processor


# ─────────────────────────────────────────────────────────────
# 자체 테스트
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    r = get_processor()
    now = time.time()

    # 표정: 최근 10분간 대부분 일반, 일부 분노
    for i in range(120):
        r.add_expression("일반", now - 600 + i * 5)
    for i in range(20):
        r.add_expression("분노", now - 60 + i * 2)   # 최근 1분에 분노 집중

    # 한숨: 5분 구간에만 1회
    r.add_sigh(True, now - 200)

    # 발화: 부정 감정 산포
    r.add_speech_emotion("불안", now - 30)
    r.add_speech_emotion("슬픔", now - 120)
    r.add_speech_emotion("분노", now - 400)
    r.add_speech_emotion("기쁨", now - 10)

    print(json.dumps(r.get_all_signals(), ensure_ascii=False, indent=2))
    print(json.dumps(r.get_status(), ensure_ascii=False, indent=2))