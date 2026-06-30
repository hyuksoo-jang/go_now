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
  · 발화 감정  (발화)    : mic_agent.py 에서 POST(probs 포함) → add_speech_emotion()

판정 알고리즘 (가중 위험점수 + 하드 오버라이드, 구간마다 독립):
  0) 구간 준비 안됨(첫 표정 이후 구간시간 미경과) → ready=False (대시보드 '측정중')
  1) 구간 내 한숨 >= 1                         -> 빨강  (하드 오버라이드)
  2) 표정 분노비율 a >= anger_hard_th          -> 빨강  (하드 오버라이드)
  3) 위험점수 R 계산 후 임계 비교
        face_risk   = clamp(a - h, -1, +1)               (a=분노비율, h=행복비율)
        speech_risk = 구간 발화 위험 r_i 평균 (없으면 0)
        wf, ws      = 발화 있으면 Wf/(Wf+Ws), Ws/(Wf+Ws) / 없으면 1, 0
        R = wf*face_risk + ws*speech_risk
        R >= red_th -> 빨강 / R <= green_th -> 초록 / 그 외 -> 노랑

  발화 1건 위험:  r_i = sum_e w_e*p_e  (확신도 게이트 + 기쁨 게이트)
  가중치/임계는 weights_config.json 으로 조정(emotion_tuner.py 가 생성). 없으면 기본값.

이 모듈은 표준 라이브러리만 사용한다 (numpy 등 외부 의존성 없음).
────────────────────────────────────────────────────────────────
"""

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

# 발화 감정 6종 -> 극성 매핑 (긍정 = 기쁨, 그 외 5종 = 부정). neg_count 표시용으로 유지.
POSITIVE_EMOTIONS = {"기쁨"}
NEGATIVE_EMOTIONS = {"슬픔", "분노", "불안", "당황", "상처"}
VALID_SPEECH_EMOTIONS = POSITIVE_EMOTIONS | NEGATIVE_EMOTIONS

# 구간 정의 (대시보드 키 <-> 초)
WINDOWS = {"1m": 60, "5m": 300, "10m": 600}

# 표정 대표값(표시용 라벨) 비율 임계 — go_now_v2._window_emotion 과 일치
ANGER_RATIO_TH = 0.30
HAPPY_RATIO_TH = 0.50

# 표정 라벨 정규화 — go_now 는 '일반'
NEUTRAL_LABEL = "일반"

# 버퍼 최대 길이 (10분 + 여유. 표정 1Hz, 한숨/발화는 더 드묾)
_BUF_MAXLEN = 5000

# ── 위험점수 기본 설정 (weights_config.json 으로 덮어쓰기) ──
DEFAULT_WEIGHTS_CONFIG = {
    "emotion_weights": {"분노": 1.0, "불안": 0.7, "당황": 0.5,
                        "상처": 0.3, "슬픔": 0.2, "기쁨": -0.6},
    "min_conf": 0.45,
    "happy_gate": 0.70,
    "w_face": 0.6,
    "w_speech": 0.4,
    "anger_hard_th": 0.6,
    "red_th": 0.5,
    "green_th": -0.3,
}

_SCALAR_KEYS = ("min_conf", "happy_gate", "w_face", "w_speech",
                "anger_hard_th", "red_th", "green_th")


def load_weights_config(explicit=None):
    """weights_config.json 자동 탐색/로드 -> (병합 dict, 사용경로|None).
    DEFAULT_WEIGHTS_CONFIG 위에 파일 값을 덮어쓴다(부분 지정 허용).
    탐색 순서: 명시경로 -> 현재폴더 -> 이 파일 폴더."""
    cfg = dict(DEFAULT_WEIGHTS_CONFIG)
    cfg["emotion_weights"] = dict(DEFAULT_WEIGHTS_CONFIG["emotion_weights"])
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(os.path.join(os.getcwd(), "weights_config.json"))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "weights_config.json"))
    seen = set()
    for p in cands:
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data.get("emotion_weights"), dict):
                    cfg["emotion_weights"].update(data["emotion_weights"])
                for k in _SCALAR_KEYS:
                    if data.get(k) is not None:
                        cfg[k] = data[k]
                return cfg, p
            except Exception as e:
                print(f"[weights] config 읽기 실패({p}): {e}", flush=True)
    return cfg, None


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def speech_utterance_risk(probs, top_prob=None, weights=None,
                          min_conf=0.45, happy_gate=0.70):
    """발화 1건 위험  r_i = sum_e w_e * p_e  (확신도/기쁨 게이트 적용).
      - 확신도 게이트: max_e p_e < min_conf 이면 r_i *= (max_e p_e / min_conf)
      - 기쁨 게이트  : p_기쁨 < happy_gate 이면 그 발화의 기쁨 가중치 = 0
    probs 가 없으면(라벨만 온 과도기 입력) 호출측에서 one-hot 을 넘긴다."""
    if weights is None:
        weights = DEFAULT_WEIGHTS_CONFIG["emotion_weights"]
    probs = probs or {}
    if top_prob is None:
        top_prob = max(probs.values()) if probs else 0.0
    r = 0.0
    for e, w in weights.items():
        p = float(probs.get(e, 0.0))
        if e == "기쁨" and p < happy_gate:     # 기쁨은 확실할 때만 호신호로 인정
            continue
        r += w * p
    if min_conf and top_prob < min_conf:        # 어정쩡한 발화는 영향력 감쇠
        r *= (top_prob / min_conf)
    return r


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
        """최근 seconds 초 이내의 [(ts, value), ...] 반환."""
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
    def __init__(self, weights_config=None):
        self.expression_buffer = TimedBuffer()
        self.sigh_buffer = TimedBuffer()
        self.speech_buffer = TimedBuffer()
        self._start_ts = time.time()
        self._sigh_alive_ts = 0.0     # 마지막 한숨 입력/heartbeat 시각
        self._speech_alive_ts = 0.0   # 마지막 발화 입력/heartbeat 시각

        self.cfg, self._cfg_path = load_weights_config(weights_config)
        self.weights = self.cfg["emotion_weights"]
        if self._cfg_path:
            print(f"[radar] 위험 가중치 설정 적용: {self._cfg_path}", flush=True)
        else:
            print("[radar] weights_config.json 없음 -> 기본 가중치 사용 "
                  "(emotion_tuner.py 로 생성 가능)", flush=True)

    # ── 설정 재로딩(선택) ────────────────────────────────────
    def reload_weights(self, weights_config=None):
        self.cfg, self._cfg_path = load_weights_config(weights_config)
        self.weights = self.cfg["emotion_weights"]
        return self._cfg_path

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

    def add_speech_emotion(self, emotion: str, timestamp: float = None,
                           probs: dict = None, top_prob: float = None,
                           risk: float = None):
        """발화 감정 추가. 6종 외 입력은 ValueError.
        probs 가 오면 위험점수 r_i 를 계산해 저장(권장 경로).
        probs 가 없으면(라벨만, 과도기/테스트) 라벨 one-hot 으로 폴백 계산."""
        if emotion not in VALID_SPEECH_EMOTIONS:
            raise ValueError(
                f"알 수 없는 발화 감정: {emotion} "
                f"(허용: {sorted(VALID_SPEECH_EMOTIONS)})"
            )
        if risk is None:
            if probs:
                risk = speech_utterance_risk(
                    probs, top_prob, self.weights,
                    self.cfg["min_conf"], self.cfg["happy_gate"])
            else:
                risk = speech_utterance_risk(
                    {emotion: 1.0}, 1.0, self.weights,
                    self.cfg["min_conf"], self.cfg["happy_gate"])
        self.speech_buffer.add(
            {"emotion": emotion, "risk": float(risk),
             "top_prob": (round(float(top_prob), 4) if top_prob is not None else None)},
            timestamp)
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

    # ── 표정 대표값(표시용) ──────────────────────────────────
    def _representative_expr(self, samples) -> str:
        """구간 표정 샘플 -> 대표값('분노'/'행복'/'일반', 없으면 '불명')."""
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

        # ── 표정: 비율 -> face_risk ──
        expr_samples = [v for _, v in expr_items]
        total = len(expr_samples)
        a = (sum(1 for s in expr_samples if s == "분노") / total) if total else 0.0
        h = (sum(1 for s in expr_samples if s == "행복") / total) if total else 0.0
        face_risk = _clamp(a - h, -1.0, 1.0)
        expr = self._representative_expr(expr_samples)

        # ── 한숨 ──
        sigh_detected = any(v for _, v in sigh_items)
        sigh_count = sum(1 for _, v in sigh_items if v)   # 실제 감지(True)만 집계

        # ── 발화: r_i 평균 -> speech_risk (dict 신규 / str 과도기 모두 허용) ──
        risks, neg_count = [], 0
        for _, v in speech_items:
            if isinstance(v, dict):
                risks.append(float(v.get("risk", 0.0)))
                if v.get("emotion") in NEGATIVE_EMOTIONS:
                    neg_count += 1
            else:  # 과도기: 라벨 문자열만 저장된 경우
                risks.append(speech_utterance_risk(
                    {v: 1.0}, 1.0, self.weights,
                    self.cfg["min_conf"], self.cfg["happy_gate"]))
                if v in NEGATIVE_EMOTIONS:
                    neg_count += 1
        M = len(risks)
        speech_risk = (sum(risks) / M) if M else 0.0

        # ── 동적 가중치 -> R ──
        Wf, Ws = self.cfg["w_face"], self.cfg["w_speech"]
        if M > 0 and (Wf + Ws) > 0:
            wf, ws = Wf / (Wf + Ws), Ws / (Wf + Ws)
        else:
            wf, ws = 1.0, 0.0
        R = wf * face_risk + ws * speech_risk

        # 구간이 실제로 찼는지 — 첫 표정 샘플 이후 경과가 구간 길이 미만이면 측정중
        first_expr = self.expression_buffer.first_ts()
        ready = (first_expr is not None) and (time.time() - first_expr >= seconds)

        # ── 순서도대로 판정 ──
        if sigh_detected:                              # 1) 한숨 하드 오버라이드
            signal, reason = "red", "sigh"
        elif a >= self.cfg["anger_hard_th"]:           # 2) 강한 분노 하드 오버라이드
            signal, reason = "red", "anger_hard"
        elif R >= self.cfg["red_th"]:                  # 3) 점수 빨강
            signal, reason = "red", "score"
        elif R <= self.cfg["green_th"]:                #    점수 초록
            signal, reason = "green", "score"
        else:                                          #    그 외 노랑
            signal, reason = "yellow", "score"

        return {
            "signal": signal,
            "expr": expr,
            "sigh": sigh_detected,
            "neg_count": neg_count,          # 표시용 유지(부정 발화 개수)
            "risk": round(R, 3),
            "face_risk": round(face_risk, 3),
            "speech_risk": round(speech_risk, 3),
            "reason": reason,
            "ready": ready,
            "data_points": {
                "expression": len(expr_samples),
                "sigh": sigh_count,
                "speech": M,
            },
        }

    # ── 전체 신호 (1m/5m/10m) ────────────────────────────────
    def get_all_signals(self) -> dict:
        out = {key: self._decide_window(key) for key in WINDOWS}
        out["updated_at"] = datetime.now(KST).isoformat()
        out["thresholds"] = {            # 대시보드 게이지 눈금/기여도 계산용
            "red_th": self.cfg["red_th"],
            "green_th": self.cfg["green_th"],
            "anger_hard_th": self.cfg["anger_hard_th"],
            "w_face": self.cfg["w_face"],
            "w_speech": self.cfg["w_speech"],
        }
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
            "weights_config": self._cfg_path,
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
    r = get_processor()
    now = time.time()

    for i in range(120):
        r.add_expression("일반", now - 600 + i * 5)
    for i in range(20):
        r.add_expression("분노", now - 60 + i * 2)

    r.add_sigh(True, now - 200)

    r.add_speech_emotion("불안", now - 30,
                         probs={"기쁨": 0.05, "슬픔": 0.1, "분노": 0.1,
                                "불안": 0.6, "당황": 0.1, "상처": 0.05},
                         top_prob=0.6)
    r.add_speech_emotion("기쁨", now - 10,
                         probs={"기쁨": 0.8, "슬픔": 0.05, "분노": 0.03,
                                "불안": 0.05, "당황": 0.04, "상처": 0.03},
                         top_prob=0.8)
    r.add_speech_emotion("분노", now - 400)  # 라벨만 -> one-hot 폴백

    print(json.dumps(r.get_all_signals(), ensure_ascii=False, indent=2))
    print(json.dumps(r.get_status(), ensure_ascii=False, indent=2))