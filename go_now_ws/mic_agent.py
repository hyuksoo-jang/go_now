#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mic_agent.py
────────────────────────────────────────────────────────────────
결재 레이더 통합 음성 에이전트 (한숨 + 발화 감정)

핵심 설계 — 마이크는 한 번만 연다:
  하나의 sounddevice 입력 스트림(16kHz/30ms 프레임)을 열고,
  읽은 프레임을 두 파이프라인으로 fan-out 한다.

    [마이크] ── 30ms 프레임 ──┬──▶ sigh_q  ──▶ YAMNet 한숨 감지 ──▶ POST /api/radar/add-sigh
                              └──▶ speech_q ──▶ VAD 발화조립 ──▶ STT ──▶ KoBERT ──▶ POST /api/radar/add-speech-emotion

  → 두 프로세스가 같은 장치를 각각 open 하던 PortAudio/ALSA 경합이 사라진다.

전송 대상(cam 기기 Flask)은 환경변수 RADAR_URL 로 받는다 (기본 localhost:5000).
  run_all.py 가 RADAR_URL 을 넣어주거나, 직접:
    RADAR_URL=http://192.168.0.10:5000 python3 mic_agent.py

실행:
  python3 mic_agent.py
  python3 mic_agent.py --no-calibrate
  python3 mic_agent.py --device 1 --gate-margin 1.6
  python3 mic_agent.py --no-speech          # 한숨만
  python3 mic_agent.py --no-sigh            # 발화만

의존성:
  pip install sounddevice webrtcvad numpy
  # 한숨:   pip install tflite-runtime   (+ model/yamnet.tflite, model/yamnet_class_map.csv)
  # 발화:   pip install faster-whisper transformers torch kobert-transformers sentencepiece noisereduce
  # sudo apt install libportaudio2
────────────────────────────────────────────────────────────────
"""

import argparse
import collections
import csv
import json
import os
import json
import queue
import sys
import threading
import time
import unicodedata
import urllib.request

import numpy as np
import sounddevice as sd
import webrtcvad

# ── 전송 대상 ────────────────────────────────────────────────
RADAR_URL = os.environ.get("RADAR_URL", "http://localhost:5000").rstrip("/")

# 캘리브레이션 실시간 레벨 (캡처 루프가 갱신, poster 스레드가 전송)
_calib_live = {"rms": 0.0, "active": False}

# 실행 중 STT 파라미터 (CLI 초기값 → 캘리브레이션 자동튜닝이 갱신)
STT_CFG  = {"beam": 1, "vad_filter": False}
STT_LOCK = threading.Lock()
_tune_q  = queue.Queue()          # 캘리브레이션 발화 → 자동튜닝 요청 전달


def _norm_ko(s: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFC", s):
        if ch.isspace():
            continue
        if unicodedata.category(ch)[0] in ("P", "S"):   # 구두점/기호 제거
            continue
        out.append(ch.lower())
    return "".join(out)


def _cer(ref: str, hyp: str) -> float:
    """문자 오류율 (공백·부호 무시)."""
    r, h = _norm_ko(ref), _norm_ko(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, ca in enumerate(r, 1):
        cur = [i]
        for j, cb in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(r)


def load_stt_config(explicit=None):
    """STT 설정(stt_config.json) 자동 탐색·로드. 반환: (dict, 사용경로|None).
    탐색 순서: 명시경로 → 현재폴더 → mic_agent.py 폴더."""
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(os.path.join(os.getcwd(), "stt_config.json"))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "stt_config.json"))
    seen = set()
    for p in cands:
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f), p
            except Exception as e:
                print(f"[stt] ⚠ 설정 읽기 실패({p}): {e}", flush=True)
    return {}, None

# ── 오디오 공통 ──────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_MS      = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000          # 480
FRAME_BYTES   = FRAME_SAMPLES * 2                        # int16

# ── 한숨(YAMNet) 설정 ────────────────────────────────────────
SIGH_MODEL_PATH = "model/yamnet.tflite"
SIGH_LABEL_PATH = "model/yamnet_class_map.csv"
SIGH_KEYWORDS   = ["sigh", "breathing", "exhale"]
SIGH_THRESHOLD  = 0.3
SIGH_COOLDOWN   = 5.0
SIGH_INFER_EVERY = 8          # 30ms × 8 ≈ 240ms 마다 추론
SIGH_HEARTBEAT  = 10.0        # 감지 없어도 N초마다 detected:false 전송(장치 alive 표시용)

# ── 발화(STT+감정) 설정 ──────────────────────────────────────
SILENCE_TAIL_MS = 700
MAX_UTTER_MS    = 8000
MAX_UTTER_QUEUE = 10
KOBERT_MODEL    = "jeongyoonhuh/kobert-emotion-6class"
EMOTIONS        = ["기쁨", "슬픔", "분노", "불안", "당황", "상처"]


# ═════════════════════════════════════════════════════════════
# POST 헬퍼 (requests 의존 없이 urllib)
# ═════════════════════════════════════════════════════════════
_post_fail_logged = {"sigh": 0.0, "speech": 0.0}


def _post(path: str, payload: dict, kind: str = "") -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RADAR_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return 200 <= r.status < 300
    except Exception as e:
        now = time.time()
        if now - _post_fail_logged.get(kind, 0.0) > 10.0:   # 10초에 한 번만 로그
            _post_fail_logged[kind] = now
            print(f"[mic] ⚠ 전송 실패 {path} ({e}) — RADAR_URL={RADAR_URL}", flush=True)
        return False


# ═════════════════════════════════════════════════════════════
# 큐 적재 (drop-oldest)
# ═════════════════════════════════════════════════════════════
def put_drop_oldest(q: queue.Queue, item, maxsize: int):
    while q.qsize() >= maxsize:
        try:
            q.get_nowait()
        except queue.Empty:
            break
    q.put(item)


# ═════════════════════════════════════════════════════════════
# 1) 한숨 파이프라인 (YAMNet TFLite)
# ═════════════════════════════════════════════════════════════
def _load_yamnet(model_path, label_path):
    try:
        import tflite_runtime.interpreter as tflite
        src = "tflite_runtime"
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite
        src = "tensorflow"

    labels = {}
    with open(label_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["index"])] = row["display_name"].lower()
    sigh_idx = [i for i, n in labels.items()
                if any(k in n for k in SIGH_KEYWORDS)]

    interp = tflite.Interpreter(model_path=model_path, num_threads=2)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_shape = in_det[0]["shape"]
    window_len = int(in_shape[0]) if len(in_shape) == 1 else int(in_shape[1])

    print(f"[sigh] YAMNet 로딩 완료 ({src}) · window {window_len/SAMPLE_RATE*1000:.0f}ms "
          f"· 감지클래스 {len(sigh_idx)}개", flush=True)
    return interp, in_det, out_det, window_len, sigh_idx, labels


def sigh_loop(sigh_q, stop_evt, model_path, label_path):
    try:
        interp, in_det, out_det, window_len, sigh_idx, labels = \
            _load_yamnet(model_path, label_path)
    except Exception as e:
        print(f"[sigh] ❌ 비활성화 — 모델 로딩 실패: {e}", flush=True)
        # 큐를 비워 speech 쪽 메모리 누수 방지
        while not stop_evt.is_set():
            try:
                sigh_q.get(timeout=0.5)
            except queue.Empty:
                pass
        return

    if not sigh_idx:
        print("[sigh] ⚠ 한숨 관련 클래스 없음 — 키워드를 확인하세요", flush=True)

    buf = np.zeros(window_len, dtype=np.float32)
    frame_cnt = 0
    last_detect = 0.0
    last_beat = time.time()
    in_shape = in_det[0]["shape"]

    while not stop_evt.is_set():
        try:
            frame = sigh_q.get(timeout=0.5)
        except queue.Empty:
            frame = None

        if frame is not None:
            chunk = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            n = len(chunk)
            buf = np.roll(buf, -n)
            buf[-n:] = chunk
            frame_cnt += 1

        now = time.time()

        # 추론 (일정 프레임마다)
        if frame_cnt >= SIGH_INFER_EVERY and sigh_idx:
            frame_cnt = 0
            inp = buf if len(in_shape) == 1 else buf.reshape(1, window_len)
            interp.set_tensor(in_det[0]["index"], inp.astype(np.float32))
            interp.invoke()
            scores = interp.get_tensor(out_det[0]["index"])[0]
            max_prob = float(np.max(scores[sigh_idx]))

            if max_prob >= SIGH_THRESHOLD and (now - last_detect) >= SIGH_COOLDOWN:
                last_detect = now
                ok = _post("/api/radar/add-sigh",
                           {"detected": True, "timestamp": now}, kind="sigh")
                best = sigh_idx[int(np.argmax(scores[sigh_idx]))]
                print(f"[sigh] 💨 한숨 감지 {labels.get(best,'?')} "
                      f"(p={max_prob:.3f}) → {'OK' if ok else 'FAIL'}", flush=True)
                last_beat = now

        # heartbeat (장치 alive 표시 — detected:false 는 판정에 영향 없음)
        if now - last_beat >= SIGH_HEARTBEAT:
            last_beat = now
            _post("/api/radar/add-sigh",
                  {"detected": False, "timestamp": now}, kind="sigh")


# ═════════════════════════════════════════════════════════════
# 2) 발화 파이프라인 (VAD 조립 → STT → KoBERT)
# ═════════════════════════════════════════════════════════════
class EmotionClassifier:
    def __init__(self, model_name=KOBERT_MODEL, threads=4):
        import torch
        from transformers import AutoModelForSequenceClassification
        from kobert_transformers import get_tokenizer
        torch.set_num_threads(threads)
        self._torch = torch
        self.tokenizer = get_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, text: str) -> dict:
        with self._torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=128)
            probs = self._torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        p = {e: probs[i].item() for i, e in enumerate(EMOTIONS)}
        top = max(p, key=p.get)
        return {"top": top, "top_prob": p[top], "probs": p}


def _measure_frames(seconds, vad=None, collect_audio=False):
    """캘리브레이션용 임시 측정 (메인 캡처 스트림이 닫혀 있을 때만 호출)."""
    n = int(SAMPLE_RATE * seconds)
    got, out = 0, []
    raw = [] if collect_audio else None
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                           dtype="int16", channels=1) as s:
        while got < n:
            data, _ = s.read(FRAME_SAMPLES)
            frame = bytes(data)
            i16 = np.frombuffer(frame, dtype=np.int16)
            if raw is not None:
                raw.append(i16.copy())
            samples = i16.astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
            _calib_live["rms"] = rms          # 실시간 레벨 공유 (poster가 전송)
            voiced = bool(vad.is_speech(frame, SAMPLE_RATE)) if vad else True
            out.append((rms, voiced))
            got += len(samples)
    if collect_audio:
        audio = np.concatenate(raw) if raw else np.zeros(0, dtype=np.int16)
        return out, audio
    return out


def _report_calib(phase, text="", noise_sec=0.0, enroll_sec=0.0):
    """캘리브레이션 진행 단계를 cam 기기로 보고 (대시보드 팝업 안내용)."""
    _post("/api/speech-calib-status",
          {"phase": phase, "text": text,
           "noise_sec": noise_sec, "enroll_sec": enroll_sec}, kind="calib")


def calibrate(noise_sec=1.5, enroll_sec=4.0, margin=1.6, vad_aggr=1,
              enroll_text="결재 부탁드립니다", autotune=True):
    _calib_live["active"] = True
    try:
        _report_calib("noise", noise_sec=noise_sec)
        print(f"[speech] ① 주변 소음 측정 — {noise_sec:.1f}초 조용히...", flush=True)
        noise = _measure_frames(noise_sec)
        noise_rms = float(np.median([r for r, _ in noise])) or 1.0

        vad = webrtcvad.Vad(vad_aggr)
        _report_calib("speak", text=enroll_text, enroll_sec=enroll_sec)
        print(f'[speech] ② 테스트 발화 — "{enroll_text}" ({enroll_sec:.0f}초)...', flush=True)
        sp, enroll_audio = _measure_frames(enroll_sec, vad=vad, collect_audio=True)
        voiced = [r for r, v in sp if v and r > noise_rms] or [r for r, _ in sp]
        speech_rms = float(np.median(voiced))
        upper = speech_rms * margin
        snr = 20.0 * np.log10(max(speech_rms, 1e-6) / max(noise_rms, 1e-6))
        print(f"[speech] 게이트 캘리브레이션 완료 · 소음 {noise_rms:.0f} · 발화 {speech_rms:.0f} "
              f"· 상한 {upper:.0f} · SNR {snr:+.1f}dB", flush=True)
        if snr < 6.0:
            print("[speech] ⚠ SNR<6dB — 마이크를 더 가까이(~20cm) 권장", flush=True)

        # 게이트 완료 → 팝업에 '완료' 즉시 표시 (튜닝은 백그라운드, 대기 없음)
        _report_calib("done")
        if autotune and enroll_audio.size > int(SAMPLE_RATE * 0.5):
            audio_f32 = enroll_audio.astype(np.float32) / 32768.0
            _tune_q.put((audio_f32, enroll_text))
            print("[speech] ③ 발화 인식 파라미터 백그라운드 튜닝 예약...", flush=True)
        return upper
    finally:
        _calib_live["active"] = False


def calib_level_poster(stop_evt):
    """캘리브레이션 중 실시간 입력 레벨을 cam 기기로 전송 (대시보드 막대)."""
    while not stop_evt.is_set():
        if _calib_live["active"]:
            lvl = min(1.0, _calib_live["rms"] / 3000.0)   # int16 RMS → 0~1 근사
            _post("/api/speech-calib-status", {"level": round(lvl, 3)}, kind="calib")
        time.sleep(0.15)


def speech_assemble_loop(speech_q, utter_q, stop_evt, gate, vad_aggr):
    """프레임 큐 → VAD 발화 조립 → utter_q (drop-oldest)."""
    vad = webrtcvad.Vad(vad_aggr)
    silence_frames = SILENCE_TAIL_MS // FRAME_MS
    triggered = False
    ring = collections.deque(maxlen=8)
    voiced, voiced_rms = [], []
    num_silence = 0
    over_upper = False

    def finish():
        nonlocal triggered, voiced, voiced_rms, num_silence, over_upper
        if voiced:
            stats = {"min": float(np.min(voiced_rms)),
                     "mean": float(np.mean(voiced_rms)),
                     "max": float(np.max(voiced_rms))}
            if over_upper:
                put_drop_oldest(utter_q, (None, stats, "too_loud"), MAX_UTTER_QUEUE)
            else:
                pcm = b"".join(voiced)
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                put_drop_oldest(utter_q, (audio, stats, "ok"), MAX_UTTER_QUEUE)
        triggered = False
        voiced, voiced_rms = [], []
        num_silence = 0
        over_upper = False
        ring.clear()

    while not stop_evt.is_set():
        try:
            frame = speech_q.get(timeout=0.5)
        except queue.Empty:
            continue
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2)) if samples.size else 0.0
        upper = gate["upper"]
        is_speech = (rms <= upper) and vad.is_speech(frame, SAMPLE_RATE)

        if not triggered:
            ring.append(frame)
            if is_speech:
                triggered = True
                voiced.extend(ring)
                voiced_rms.append(rms)
                ring.clear()
                num_silence = 0
                over_upper = False
        else:
            if rms > upper:
                over_upper = True
            voiced.append(frame)
            voiced_rms.append(rms)
            num_silence = 0 if is_speech else num_silence + 1
            if num_silence > silence_frames or len(voiced) * FRAME_MS > MAX_UTTER_MS:
                finish()


def _run_autotune(stt, audio, ref):
    """캘리브레이션 발화로 beam 빠른 선택(백그라운드). vad_filter 는 설정값 유지.
    (VAD 토글은 onnxruntime 로드가 무거워 인라인 제외 — 정밀비교는 speech_tuner)"""
    with STT_LOCK:
        vadf = STT_CFG["vad_filter"]
    results = []
    for beam in (1, 5):
        segs, _ = stt.transcribe(audio, language="ko", beam_size=beam,
                                 vad_filter=vadf, temperature=0.0,
                                 condition_on_previous_text=False)
        hyp = "".join(s.text for s in segs).strip()
        results.append((_cer(ref, hyp), beam, hyp))
    results.sort(key=lambda x: (x[0], x[1]))   # CER 낮은 순 → 동률이면 beam 작은(빠른) 쪽
    best = results[0]
    with STT_LOCK:
        STT_CFG["beam"] = best[1]
    msg = f"beam {best[1]} · VAD {'on' if vadf else 'off'} (CER {best[0]*100:.0f}%)"
    print(f'[speech] 🔧 자동 튜닝(백그라운드) 적용 → {msg}  인식="{best[2]}"', flush=True)


def speech_process_loop(utter_q, stop_evt, whisper_size, threads, min_chars,
                        compute_type="int8"):
    """utter_q → 최신 발화만 STT → KoBERT → POST. STT_CFG(beam/vad)는 라이브 반영."""
    from faster_whisper import WhisperModel
    try:
        import onnxruntime
        onnxruntime.set_default_logger_severity(3)  # VAD(onnxruntime) GPU 탐지 경고 억제
    except Exception:
        pass
    with STT_LOCK:
        _b, _v = STT_CFG["beam"], STT_CFG["vad_filter"]
    print(f"[speech] STT(faster-whisper) 로딩... size={whisper_size} "
          f"compute={compute_type} beam={_b} vad_filter={_v}", flush=True)
    stt = WhisperModel(whisper_size, device="cpu", compute_type=compute_type,
                       cpu_threads=threads)
    print("[speech] KoBERT 감정 모델 로딩...", flush=True)
    clf = EmotionClassifier(threads=threads)
    print("[speech] 발화 분석 준비 완료", flush=True)

    last_beat = 0.0
    while not stop_evt.is_set():
        now = time.time()
        if now - last_beat >= 5.0:        # 발화 파이프라인 alive heartbeat
            last_beat = now
            _post("/api/radar/speech-heartbeat", {}, kind="speech")

        # 캘리브레이션 자동 튜닝 요청 처리
        try:
            tune_audio, tune_ref = _tune_q.get_nowait()
            _run_autotune(stt, tune_audio, tune_ref)
            continue
        except queue.Empty:
            pass

        try:
            item = utter_q.get(timeout=0.5)
        except queue.Empty:
            continue
        # 밀린 발화는 버리고 가장 최근만 처리
        while not utter_q.empty():
            try:
                item = utter_q.get_nowait()
            except queue.Empty:
                break

        audio, stats, reason = item
        if reason == "too_loud":
            print("[speech] 🚫 폐기(상한 초과 소음 혼입)", flush=True)
            continue

        with STT_LOCK:
            beam, vadf = STT_CFG["beam"], STT_CFG["vad_filter"]
        segments, _ = stt.transcribe(audio, language="ko", beam_size=beam,
                                     condition_on_previous_text=False,
                                     vad_filter=vadf)
        text = "".join(seg.text for seg in segments).strip()
        if len(text) < min_chars:
            continue

        res = clf.predict(text)
        emo = res["top"]
        ok = _post("/api/radar/add-speech-emotion",
                   {"emotion": emo, "timestamp": time.time()}, kind="speech")
        polarity = "긍정" if emo == "기쁨" else "부정"
        print(f'[speech] 🗣 "{text}" → {emo} ({res["top_prob"]*100:.0f}%, {polarity}) '
              f"→ {'OK' if ok else 'FAIL'}", flush=True)


# ═════════════════════════════════════════════════════════════
# 재캘리브레이션 폴러 (대시보드 버튼 → go_now 플래그 → 여기서 감지)
# ═════════════════════════════════════════════════════════════
def recalib_poller(stop_evt, recalib_req):
    url = RADAR_URL + "/api/speech-recalibrate-flag"
    while not stop_evt.is_set():
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                d = json.loads(r.read().decode("utf-8"))
                if d.get("requested"):
                    print("[speech] 🔄 재캘리브레이션 요청 수신", flush=True)
                    recalib_req.set()
        except Exception:
            pass
        time.sleep(0.5)


# ═════════════════════════════════════════════════════════════
# 오디오 슈퍼바이저 — 스트림을 소유하고 fan-out / 재캘리브레이션 관리
# ═════════════════════════════════════════════════════════════
def audio_supervisor(stop_evt, recalib_req, gate, args,
                     sigh_q, speech_q, do_initial_calib):
    need_calib = do_initial_calib
    while not stop_evt.is_set():
        if need_calib and not args.no_speech:
            need_calib = False
            try:
                gate["upper"] = calibrate(
                    enroll_sec=args.enroll_sec, margin=args.gate_margin,
                    vad_aggr=args.vad, enroll_text=args.enroll_text,
                    autotune=not args.no_autotune)
            except Exception as e:
                print(f"[speech] ⚠ 캘리브레이션 실패({e}) — 게이트 off", flush=True)
                gate["upper"] = float("inf")

        dev = args.device
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                                   dtype="int16", channels=1,
                                   device=dev) as stream:
                print(f"[mic] 🎧 캡처 시작 (device={dev if dev is not None else 'default'}, "
                      f"RADAR_URL={RADAR_URL})", flush=True)
                while not stop_evt.is_set() and not recalib_req.is_set():
                    data, _ = stream.read(FRAME_SAMPLES)
                    frame = bytes(data)
                    if not args.no_sigh:
                        put_drop_oldest(sigh_q, frame, 200)
                    if not args.no_speech:
                        put_drop_oldest(speech_q, frame, 400)
        except Exception as e:
            print(f"[mic] ❌ 스트림 오류({e}) — 2초 후 재시도", flush=True)
            time.sleep(2.0)

        if recalib_req.is_set():
            recalib_req.clear()
            need_calib = True   # 스트림 닫힌 상태에서 재측정


# ═════════════════════════════════════════════════════════════
# 메인
# ═════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="결재 레이더 통합 음성 에이전트 (단일 마이크)")
    ap.add_argument("--device", type=int, default=None,
                    help="입력 장치 인덱스 (기본: 시스템 기본 입력)")
    ap.add_argument("--vad", type=int, default=1, help="VAD aggressiveness 0~3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--whisper-size", default=None,
                    help="tiny/base/small (미지정 시 stt_config→base)")
    ap.add_argument("--beam-size", type=int, default=None,
                    help="STT beam size (미지정 시 stt_config→1, 캘리브레이션이 갱신)")
    ap.add_argument("--vad-filter", dest="vad_filter", action="store_const",
                    const=True, default=None,
                    help="STT 내부 VAD 필터 사용 (미지정 시 stt_config→off)")
    ap.add_argument("--compute-type", default=None,
                    help="faster-whisper compute_type (미지정 시 stt_config→int8)")
    ap.add_argument("--stt-config", default=None,
                    help="STT 설정 JSON 경로 (기본: stt_config.json 자동 탐색)")
    ap.add_argument("--no-autotune", action="store_true",
                    help="캘리브레이션 시 beam/VAD 자동 튜닝 생략")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--gate-margin", type=float, default=1.6,
                    help="발화 상한 = 테스트발화 RMS × margin")
    ap.add_argument("--enroll-text", default="결재 부탁드립니다")
    ap.add_argument("--enroll-sec", type=float, default=4.0)
    ap.add_argument("--no-calibrate", action="store_true", help="발화 게이트 캘리브레이션 생략")
    ap.add_argument("--no-sigh", action="store_true", help="한숨 파이프라인 비활성")
    ap.add_argument("--no-speech", action="store_true", help="발화 파이프라인 비활성")
    ap.add_argument("--sigh-model", default=SIGH_MODEL_PATH)
    ap.add_argument("--sigh-labels", default=SIGH_LABEL_PATH)
    args = ap.parse_args()

    if args.no_sigh and args.no_speech:
        sys.exit("한숨·발화를 모두 끄면 할 일이 없습니다.")

    print("=" * 58)
    print("  🎙️  결재 레이더 통합 음성 에이전트 (단일 마이크 fan-out)")
    print(f"  전송 대상 RADAR_URL = {RADAR_URL}")
    print(f"  한숨={'off' if args.no_sigh else 'on'} · "
          f"발화={'off' if args.no_speech else 'on'}")
    print("=" * 58, flush=True)

    # ── STT 설정 해석: CLI > stt_config.json > 기본값 ──
    _cfg, _cfg_path = load_stt_config(args.stt_config)

    def _pick(cli_val, key, default):
        if cli_val is not None:
            return cli_val
        if isinstance(_cfg, dict) and _cfg.get(key) is not None:
            return _cfg[key]
        return default

    whisper_size = _pick(args.whisper_size, "whisper_size", "base")
    compute_type = _pick(args.compute_type, "compute_type", "int8")
    beam_size    = int(_pick(args.beam_size, "beam_size", 1))
    vad_filter   = bool(_pick(args.vad_filter, "vad_filter", False))

    if not args.no_speech:
        if _cfg_path:
            print(f"[stt] 설정 파일 적용: {_cfg_path}", flush=True)
        else:
            print("[stt] stt_config.json 없음 → 기본값/CLI 사용 "
                  "(speech_tuner.py 로 생성 가능)", flush=True)
        print(f"[stt] size={whisper_size} compute={compute_type} "
              f"beam={beam_size} vad_filter={vad_filter}  (CLI>config>기본)",
              flush=True)

    stop_evt = threading.Event()
    recalib_req = threading.Event()
    gate = {"upper": float("inf")}
    sigh_q = queue.Queue()
    speech_q = queue.Queue()
    utter_q = queue.Queue()

    # CLI/설정 초기 STT 값 (캘리브레이션 자동튜닝이 이후 beam/VAD 갱신)
    with STT_LOCK:
        STT_CFG["beam"] = beam_size
        STT_CFG["vad_filter"] = vad_filter

    threads = []

    if not args.no_sigh:
        threads.append(threading.Thread(
            target=sigh_loop, args=(sigh_q, stop_evt, args.sigh_model, args.sigh_labels),
            daemon=True))

    if not args.no_speech:
        threads.append(threading.Thread(
            target=speech_assemble_loop,
            args=(speech_q, utter_q, stop_evt, gate, args.vad), daemon=True))
        threads.append(threading.Thread(
            target=speech_process_loop,
            args=(utter_q, stop_evt, whisper_size, args.threads, args.min_chars),
            kwargs=dict(compute_type=compute_type),
            daemon=True))
        threads.append(threading.Thread(
            target=recalib_poller, args=(stop_evt, recalib_req), daemon=True))
        threads.append(threading.Thread(
            target=calib_level_poster, args=(stop_evt,), daemon=True))

    for t in threads:
        t.start()

    do_initial_calib = (not args.no_speech) and (not args.no_calibrate)

    try:
        audio_supervisor(stop_evt, recalib_req, gate, args,
                         sigh_q, speech_q, do_initial_calib)
    except KeyboardInterrupt:
        print("\n[mic] 종료합니다.", flush=True)
        stop_evt.set()
        time.sleep(0.5)


if __name__ == "__main__":
    main()