#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gate_tuner.py
────────────────────────────────────────────────────────────────
mic_agent.py 의 "🚫 폐기(상한 초과 소음 혼입)" 기준을 데이터로 측정·결정한다.

[현재 로직의 문제]
  upper = median(발화 RMS) × margin(=1.6)  →  중앙값의 1.6배밖에 안 되고,
  "프레임 1개라도 upper 초과 → 발화 전체 폐기" 라서
  강세·파열음 등 순간 피크 하나에 멀쩡한 발화가 통째로 버려진다.

[이 튜너가 정하는 것]
  ① gate_margin : upper = median(발화) × margin  의 margin (상한 높이)
  ② over_frac   : "초과 프레임 비율 > over_frac" 일 때만 폐기 (단일 프레임 X)
  두 값을 실제 사용자 발화 분포(고 퍼센타일 피크 포함)와
  실제 소음 이벤트로부터 측정하여, 멀쩡한 발화는 통과 / 진짜 소음은 폐기
  되도록 동시에 최적화한다. 결과는 gate_config.json 으로 저장.

사용:
  # 실측(권장): 조용히(소음) → 평소 말투로 여러 문장 → (선택)소음 이벤트
  python3 gate_tuner.py
  python3 gate_tuner.py --device 1 --clean-sec 40 --loud --loud-sec 15
  python3 gate_tuner.py --no-loud            # 소음 이벤트 녹음 생략

  # 마이크 없이 로직 점검(합성 데이터)
  python3 gate_tuner.py --simulate

  # 튜닝 후: 그 설정이 실제 발화를 잘 통과시키는지 마이크로 즉석 확인
  #   통과한 발화는 STT+감정까지 해석해 제대로 인식됐는지 보여준다
  python3 gate_tuner.py --test
  python3 gate_tuner.py --test --no-stt                            # 해석 생략(통과/폐기만)
  python3 gate_tuner.py --test --gate-margin 2.6 --over-frac 0.15  # 값 직접 지정

  # mic_agent.py 에 적용할 패치 출력
  python3 gate_tuner.py --print-patch

의존성(실측 시): pip install sounddevice webrtcvad numpy
────────────────────────────────────────────────────────────────
"""

import argparse
import collections
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# ── mic_agent 와 동일한 오디오 규격 ──────────────────────────
SAMPLE_RATE   = 16000
FRAME_MS      = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000          # 480
SILENCE_TAIL_MS = 700                                   # 발화 종료 무음(=mic_agent)
MAX_UTTER_MS    = 8000                                  # 최대 발화 길이(=mic_agent)
MIN_UTTER_FRAMES = 8                                    # 너무 짧은 조각 제외
OUT_DEFAULT   = "../../gate_config.json"

# ── 해석(STT+감정) 설정 — mic_agent 와 동일 ─────────────────
KOBERT_MODEL = "jeongyoonhuh/kobert-emotion-6class"
EMOTIONS     = ["기쁨", "슬픔", "분노", "불안", "당황", "상처"]


# ═════════════════════════════════════════════════════════════
# RMS — mic_agent 와 동일하게 int16 스케일에서 계산
# ═════════════════════════════════════════════════════════════
def frame_rms_int16(i16: np.ndarray) -> float:
    if i16.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(i16.astype(np.float32) ** 2)))


# ═════════════════════════════════════════════════════════════
# 녹음 (sounddevice + webrtcvad) — 실측 모드에서만 import
# ═════════════════════════════════════════════════════════════
def _open_stream(device):
    import sounddevice as sd
    return sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                             dtype="int16", channels=1, device=device)


def record_seconds(seconds, device, vad=None):
    """seconds 동안 프레임 단위 (rms, voiced) 와 원음을 수집."""
    n_target = int(SAMPLE_RATE * seconds)
    got, rows, raw = 0, [], []
    with _open_stream(device) as s:
        while got < n_target:
            data, _ = s.read(FRAME_SAMPLES)
            frame = bytes(data)
            i16 = np.frombuffer(frame, dtype=np.int16)
            raw.append(i16)
            rms = frame_rms_int16(i16)
            voiced = bool(vad.is_speech(frame, SAMPLE_RATE)) if vad else True
            rows.append((rms, voiced, frame))
            got += i16.size
    return rows, np.concatenate(raw) if raw else np.zeros(0, np.int16)


def segment_utterances(rows, noise_rms, vad):
    """프레임 (rms, voiced, frame) 목록 → VAD 무음으로 끊어 발화별 voiced-RMS 배열 리스트."""
    silence_frames = SILENCE_TAIL_MS // FRAME_MS
    utts, cur = [], []
    num_sil = 0
    triggered = False
    for rms, voiced, _ in rows:
        speechy = voiced and rms > noise_rms          # 소음floor 위의 유성 프레임만
        if not triggered:
            if speechy:
                triggered = True
                cur = [rms]
                num_sil = 0
        else:
            cur.append(rms)
            num_sil = 0 if speechy else num_sil + 1
            if num_sil > silence_frames:
                body = cur[:-num_sil] if num_sil else cur
                if len(body) >= MIN_UTTER_FRAMES:
                    utts.append(np.asarray(body, dtype=np.float32))
                triggered = False
                cur = []
                num_sil = 0
    if triggered and len(cur) >= MIN_UTTER_FRAMES:
        utts.append(np.asarray(cur, dtype=np.float32))
    return utts


# ═════════════════════════════════════════════════════════════
# 핵심 분석 (순수 함수 — 합성/실측 공통)
# ═════════════════════════════════════════════════════════════
def analyze(clean_utts, loud_rms, noise_rms,
            target_accept=0.98, frac_floor=0.10, frac_cap=0.50,
            margin_grid=None):
    """
    clean_utts : list[np.ndarray]  발화별 voiced-frame RMS
    loud_rms   : np.ndarray|None   소음 이벤트 프레임 RMS (없으면 None)
    반환: dict(stats..., table=[...], recommend={...}, current={...})
    """
    pooled = np.concatenate(clean_utts)
    m   = float(np.median(pooled))
    p95 = float(np.percentile(pooled, 95))
    p99 = float(np.percentile(pooled, 99))
    mx  = float(pooled.max())
    snr = 20.0 * np.log10(max(m, 1e-6) / max(noise_rms, 1e-6))

    if margin_grid is None:
        margin_grid = np.round(np.arange(1.6, 4.01, 0.1), 2)

    def single_frame_accept(upper):
        # 현재(=단일 프레임 초과 폐기) 규칙으로 살아남는 발화 비율
        return float(np.mean([bool(np.all(u <= upper)) for u in clean_utts]))

    table = []
    for mg in margin_grid:
        upper = m * mg
        fracs = np.array([float(np.mean(u > upper)) for u in clean_utts])
        # over_frac 후보: 멀쩡한 발화의 초과비율 98%ile 보다 살짝 위 → clean 거의 통과
        T = float(np.clip(np.percentile(fracs, 98) + 0.04, frac_floor, frac_cap))
        clean_accept = float(np.mean(fracs <= T))
        if loud_rms is not None and len(loud_rms):
            loud_over = float(np.mean(loud_rms > upper))   # 소음 프레임이 상한 넘는 비율
            loud_reject = bool(loud_over > T)              # 소음 지속 발화면 폐기되는가
        else:
            loud_over, loud_reject = None, None
        table.append(dict(margin=round(float(mg), 2), upper=round(upper, 1),
                          over_frac=round(T, 3), clean_accept=round(clean_accept, 3),
                          single_frame_accept=round(single_frame_accept(upper), 3),
                          loud_over=(round(loud_over, 3) if loud_over is not None else None),
                          loud_reject=loud_reject))

    # ── 선택: clean 통과 ≥ target & (소음 폐기 가능) 중에서
    #        over_frac 최소(부분 소음 혼입까지 잘 잡힘) → 동률이면 margin 작은 쪽 ──
    ok = [r for r in table if r["clean_accept"] >= target_accept
          and (loud_rms is None or len(loud_rms) == 0 or r["loud_reject"])]
    if not ok:                                   # 소음/발화가 안 갈리면 clean 우선
        ok = [r for r in table if r["clean_accept"] >= target_accept] or \
             sorted(table, key=lambda r: -r["clean_accept"])[:3]
    choice = min(ok, key=lambda r: (r["over_frac"], r["margin"]))

    # 부분 소음 혼입 민감도: 추천값에서 발화 일부에 소음을 섞었을 때 잡히는 최소 비율
    burst_sens = _burst_sensitivity(clean_utts, loud_rms, choice["upper"],
                                    choice["over_frac"], p99)

    return dict(speech_rms_median=round(m, 1), speech_rms_p95=round(p95, 1),
                speech_rms_p99=round(p99, 1), speech_rms_max=round(mx, 1),
                noise_rms=round(float(noise_rms), 1), snr_db=round(float(snr), 1),
                n_utts=len(clean_utts), table=table, recommend=choice,
                current=dict(margin=1.6, rule="single_frame",
                             upper=round(m * 1.6, 1),
                             clean_accept=round(single_frame_accept(m * 1.6), 3)),
                burst_sensitivity=burst_sens)


def _burst_sensitivity(clean_utts, loud_rms, upper, over_frac, p99):
    """추천 (upper, over_frac) 로, 발화에 소음 burst 를 f 비율 섞었을 때
    폐기되기 시작하는 최소 f 를 추정 (작을수록 민감). 동시에 clean 오폐기율."""
    loud_level = float(np.median(loud_rms)) if (loud_rms is not None and len(loud_rms)) \
        else max(upper * 1.5, p99 * 1.5)
    rng = np.random.default_rng(1)
    false_discard = float(np.mean([float(np.mean(u > upper)) > over_frac for u in clean_utts]))
    min_catch = None
    for f in np.arange(0.05, 0.61, 0.05):
        caught = []
        for u in clean_utts:
            k = max(1, int(round(len(u) * f)))
            v = u.copy()
            idx = rng.choice(len(u), size=min(k, len(u)), replace=False)
            v[idx] = loud_level
            caught.append(float(np.mean(v > upper)) > over_frac)
        if np.mean(caught) >= 0.9:               # 90% 이상 발화에서 잡히면 그 f 채택
            min_catch = round(float(f), 2)
            break
    return dict(false_discard_rate=round(false_discard, 3),
                min_burst_fraction_caught=min_catch, loud_level=round(loud_level, 1))


# ═════════════════════════════════════════════════════════════
# 합성 데이터 (마이크 없이 로직 점검용)
# ═════════════════════════════════════════════════════════════
def make_synth(seed=0):
    rng = np.random.default_rng(seed)
    def utt():
        n = rng.integers(30, 80)
        base = np.exp(rng.normal(np.log(800), 0.45, n))      # median ~800, 동적
        for _ in range(rng.integers(1, 4)):                  # 파열음/강세 피크
            base[rng.integers(0, n)] *= rng.uniform(2.2, 3.8)
        return base.astype(np.float32)
    clean = [utt() for _ in range(40)]
    loud = np.concatenate([rng.uniform(3800, 9000, rng.integers(10, 40))
                           for _ in range(8)]).astype(np.float32)
    return clean, loud, 150.0


# ═════════════════════════════════════════════════════════════
# 라이브 테스트 — 튜닝된 설정이 실제 발화를 잘 통과시키는지 확인
# ═════════════════════════════════════════════════════════════
def record_utterances_live(device, vad, noise_rms, min_utts=8, max_sec=25.0):
    """문장을 읽는 동안 실시간으로 발화를 잘라 모은다.
    발화가 min_utts 개 모이거나 max_sec 가 지나면 자동 종료 → 길게 읽을 필요 없음."""
    silence_frames = SILENCE_TAIL_MS // FRAME_MS
    utts, cur = [], []
    triggered, num_sil = False, 0
    t0 = time.time()
    with _open_stream(device) as s:
        while len(utts) < min_utts and (time.time() - t0) < max_sec:
            data, _ = s.read(FRAME_SAMPLES)
            frame = bytes(data)
            rms = frame_rms_int16(np.frombuffer(frame, dtype=np.int16))
            speechy = vad.is_speech(frame, SAMPLE_RATE) and rms > noise_rms
            if not triggered:
                if speechy:
                    triggered, cur, num_sil = True, [rms], 0
            else:
                cur.append(rms)
                num_sil = 0 if speechy else num_sil + 1
                if num_sil > silence_frames:
                    body = cur[:-num_sil] if num_sil else cur
                    if len(body) >= MIN_UTTER_FRAMES:
                        utts.append(np.asarray(body, dtype=np.float32))
                        print(f"\r   발화 수집 {len(utts)}/{min_utts} …", end="", flush=True)
                    triggered, cur, num_sil = False, [], 0
        if triggered and len(cur) >= MIN_UTTER_FRAMES:
            utts.append(np.asarray(cur, dtype=np.float32))
    print()
    return utts


# ═════════════════════════════════════════════════════════════
# STT + 감정 해석 (통과 발화가 제대로 인식되는지 확인용) — mic_agent 와 동일
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

    def predict(self, text):
        with self._torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            probs = self._torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        p = {e: probs[i].item() for i, e in enumerate(EMOTIONS)}
        top = max(p, key=p.get)
        return top, p[top]


def load_interpreters(whisper_size="base", compute_type="int8", threads=4):
    """(stt, clf) 반환. 의존성이 없으면 (None, None) 또는 (stt, None) 으로 우아하게 강등."""
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(f"   [해석 비활성] faster-whisper 미설치({e}) → 통과/폐기만 표시")
        return None, None
    try:
        import onnxruntime
        onnxruntime.set_default_logger_severity(3)
    except Exception:
        pass
    print(f"   STT(faster-whisper {whisper_size}/{compute_type}) 로딩…", flush=True)
    stt = WhisperModel(whisper_size, device="cpu", compute_type=compute_type, cpu_threads=threads)
    clf = None
    try:
        clf = EmotionClassifier(threads=threads)
        print("   KoBERT 감정 모델 로딩 완료", flush=True)
    except Exception as e:
        print(f"   [감정 비활성] KoBERT 로딩 실패({e}) → 인식 텍스트만 표시")
    return stt, clf


def interpret(stt, clf, audio, beam=1, min_chars=2):
    """audio(float32) → (text, (emo,prob)|None, polarity|None)."""
    segs, _ = stt.transcribe(audio, language="ko", beam_size=beam,
                             condition_on_previous_text=False)
    text = "".join(s.text for s in segs).strip()
    if len(text) < min_chars:
        return text, None, None
    if clf is not None:
        emo, prob = clf.predict(text)
        pol = "긍정" if emo == "기쁨" else "부정"
        return text, (emo, prob), pol
    return text, None, None


def quick_speech_rms(device, vad, noise_rms, seconds=4.0,
                     enroll_text="결재 부탁드립니다"):
    """짧은 발화로 이번 세션의 발화 RMS 중앙값을 잡는다 (mic_agent calibrate 와 동일 개념)."""
    print(f'   "{enroll_text}" 를 평소 톤으로 말해주세요 ({seconds:.0f}초)...')
    rows, _ = record_seconds(seconds, device, vad=vad)
    voiced = [r for r, v, _ in rows if v and r > noise_rms]
    if not voiced:
        voiced = [r for r, _, _ in rows]
    return float(np.median(voiced)) if voiced else noise_rms


def run_live_test(args, cfg):
    """gate_config 의 margin·over_frac 으로 upper 를 잡고, 마이크로 들어오는 발화를
    mic_agent 와 동일하게 조립해 통과/폐기를 표시하고, 통과 발화는 STT+감정까지 해석한다."""
    import webrtcvad
    vad = webrtcvad.Vad(args.vad)
    margin = args.gate_margin if args.gate_margin is not None else float(cfg.get("gate_margin", 2.6))
    over_frac = args.over_frac if args.over_frac is not None else float(cfg.get("over_frac", 0.15))

    print("=" * 64)
    print("  라이브 테스트 — 튜닝 설정이 실제 발화를 잡는지 확인")
    print(f"      margin={margin}  over_frac={over_frac}")
    print("=" * 64)

    # 해석기(STT+감정) 로딩 — 통과 발화가 제대로 인식되는지 보기 위함
    stt = clf = None
    if not args.no_stt:
        print("\n[0] 해석 모델 로딩 (통과 발화 인식 확인용)…")
        stt, clf = load_interpreters(args.whisper_size, args.compute_type, args.threads)

    input(f"\n[1] 주변 소음 측정 ({args.noise_sec:.0f}초). 조용히 한 뒤 Enter…")
    nrows, _ = record_seconds(args.noise_sec, args.device)
    noise_rms = float(np.median([r for r, _, _ in nrows])) or 1.0

    input("\n[2] 게이트 기준 발화 측정. 준비되면 Enter…")
    speech_rms = quick_speech_rms(args.device, vad, noise_rms)
    upper = speech_rms * margin
    print(f"\n   발화 RMS={speech_rms:.0f} · 소음={noise_rms:.0f} -> 상한 upper={upper:.0f} "
          f"(초과 프레임 {over_frac*100:.0f}% 넘으면 폐기)")

    print("\n[3] 이제 자유롭게 테스트하세요. 발화가 끝날 때마다 결과를 표시합니다.")
    print("   - 평소 말투 문장        -> [통과] + 인식 텍스트/감정")
    print("   - 소리치거나 책상 두드리기 -> [폐기]")
    print("   (종료: Ctrl+C)\n")

    silence_frames = SILENCE_TAIL_MS // FRAME_MS
    stats = {"pass": 0, "drop": 0}
    triggered = False
    voiced_frames, voiced_rms = [], []
    num_sil, over_cnt = 0, 0
    ring = collections.deque(maxlen=8)              # (frame, rms)

    def verdict():
        nonlocal triggered, voiced_frames, voiced_rms, num_sil, over_cnt
        keep = len(voiced_rms) - num_sil if num_sil else len(voiced_rms)
        body_rms = voiced_rms[:keep]
        body_frames = voiced_frames[:keep]
        if len(body_rms) >= MIN_UTTER_FRAMES:
            of = over_cnt / len(body_rms)
            dur = len(body_rms) * FRAME_MS / 1000.0
            meta = (f"{dur:4.1f}s · 평균 {np.mean(body_rms):5.0f} · 최대 {np.max(body_rms):5.0f} "
                    f"· 초과 {of*100:3.0f}%")
            if of > over_frac:
                stats["drop"] += 1
                print(f"   [폐기] {meta} (상한 {upper:.0f}) "
                      f"· 누적 통과 {stats['pass']}/폐기 {stats['drop']}")
            else:
                stats["pass"] += 1
                line = f"   [통과] {meta} · 누적 통과 {stats['pass']}/폐기 {stats['drop']}"
                if stt is not None:
                    audio = np.frombuffer(b"".join(body_frames), dtype=np.int16).astype(np.float32) / 32768.0
                    try:
                        text, emo, pol = interpret(stt, clf, audio, min_chars=args.min_chars)
                    except Exception as e:
                        text, emo, pol = None, None, None
                        line += f"\n          [해석 오류] {e}"
                    if text:
                        if emo:
                            line += f'\n          🗣 "{text}"  →  {emo[0]} {emo[1]*100:.0f}% ({pol})'
                        else:
                            line += f'\n          🗣 "{text}"'
                    elif text == "":
                        line += "\n          🗣 (인식 텍스트 없음/너무 짧음)"
                print(line)
        triggered = False
        voiced_frames, voiced_rms = [], []
        num_sil, over_cnt = 0, 0
        ring.clear()

    try:
        with _open_stream(args.device) as s:
            while True:
                data, _ = s.read(FRAME_SAMPLES)
                frame = bytes(data)
                rms = frame_rms_int16(np.frombuffer(frame, dtype=np.int16))
                is_speech = (rms <= upper) and vad.is_speech(frame, SAMPLE_RATE)
                if not triggered:
                    ring.append((frame, rms))
                    if is_speech:
                        triggered = True
                        voiced_frames = [f for f, _ in ring]
                        voiced_rms = [r for _, r in ring]
                        num_sil, over_cnt = 0, 0
                        ring.clear()
                else:
                    if rms > upper:
                        over_cnt += 1
                    voiced_frames.append(frame)
                    voiced_rms.append(rms)
                    num_sil = 0 if is_speech else num_sil + 1
                    if num_sil > silence_frames or len(voiced_rms) * FRAME_MS > MAX_UTTER_MS:
                        verdict()
    except KeyboardInterrupt:
        tot = stats["pass"] + stats["drop"]
        print(f"\n\n[요약] 발화 {tot}개 중 통과 {stats['pass']} · 폐기 {stats['drop']}")
        if tot:
            print("       평소 말투가 대부분 통과(+정상 인식)하고, 일부러 낸 소음만 폐기됐다면 설정이 적절합니다.")


# ═════════════════════════════════════════════════════════════
# 리포트 / 저장 / 패치 출력
# ═════════════════════════════════════════════════════════════
def print_report(res):
    print("\n" + "=" * 64)
    print("  게이트 분석 결과")
    print("=" * 64)
    print(f"  소음 RMS        : {res['noise_rms']:.0f}")
    print(f"  발화 RMS 중앙값  : {res['speech_rms_median']:.0f}   "
          f"p95 {res['speech_rms_p95']:.0f}  p99 {res['speech_rms_p99']:.0f}  "
          f"max {res['speech_rms_max']:.0f}")
    print(f"  SNR             : {res['snr_db']:+.1f} dB   (발화 {res['n_utts']}개 분석)")
    print("-" * 64)
    print("  margin  upper   over_frac | 비율규칙통과  단일프레임통과  소음폐기")
    rec = res["recommend"]
    for r in res["table"]:
        if abs((round(r["margin"] * 10)) % 2) > 0:   # 0.2 간격만 출력
            continue
        star = " ◀ 추천" if r is rec else ""
        lo = "  -  " if r["loud_reject"] is None else (" 예  " if r["loud_reject"] else " 아니오")
        print(f"  {r['margin']:5.1f}  {r['upper']:6.0f}   {r['over_frac']:6.2f}   |"
              f"   {r['clean_accept']*100:4.0f}%        {r['single_frame_accept']*100:4.0f}%      "
              f"{lo}{star}")
    cur = res["current"]
    bs = res["burst_sensitivity"]
    print("-" * 64)
    print(f"  [현재 설정] margin 1.6 · 단일프레임 규칙 → 멀쩡한 발화 통과율 "
          f"{cur['clean_accept']*100:.0f}%  ← 과도 폐기")
    print(f"  [추 천]     margin {rec['margin']:.1f} · over_frac {rec['over_frac']:.2f}  "
          f"→ 발화 통과율 {rec['clean_accept']*100:.0f}%"
          + (f" · 소음 폐기 {'O' if rec['loud_reject'] else 'X'}" if rec['loud_reject'] is not None else ""))
    print(f"              upper = 발화중앙값 × {rec['margin']:.1f} ≈ {rec['upper']:.0f}")
    if bs["min_burst_fraction_caught"] is not None:
        print(f"              부분 소음 혼입은 약 {bs['min_burst_fraction_caught']*100:.0f}% "
              f"이상부터 폐기 · clean 오폐기율 {bs['false_discard_rate']*100:.0f}%")
    print("=" * 64)


def save_config(res, path, vad_aggr):
    rec = res["recommend"]
    cfg = {
        "gate_margin": rec["margin"],
        "over_frac": rec["over_frac"],
        "_upper_rms_at_tuning": rec["upper"],
        "_speech_rms_median": res["speech_rms_median"],
        "_speech_rms_p95": res["speech_rms_p95"],
        "_speech_rms_p99": res["speech_rms_p99"],
        "_noise_rms": res["noise_rms"],
        "_snr_db": res["snr_db"],
        "_clean_accept": rec["clean_accept"],
        "_frame_ms": FRAME_MS,
        "_vad_aggr": vad_aggr,
        "_generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {path}\n" + json.dumps(cfg, ensure_ascii=False, indent=2))
    return cfg


PATCH = r'''
────────────────────────── mic_agent.py 적용 패치 ──────────────────────────
gate_config.json 의 gate_margin / over_frac 을 읽어, 폐기 규칙을
"단일 프레임 초과" → "초과 프레임 비율 > over_frac" 으로 바꾼다.

(1) load_stt_config 근처에 추가:

    def load_gate_config(explicit=None):
        cands = [explicit] if explicit else []
        cands += [os.path.join(os.getcwd(), "gate_config.json"),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_config.json")]
        for p in cands:
            if p and os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f), p
                except Exception as e:
                    print(f"[gate] ⚠ 설정 읽기 실패({p}): {e}", flush=True)
        return {}, None

(2) argparse:
    - --gate-margin 의 default 를 1.6 → None 으로
    - 추가:
        ap.add_argument("--over-frac", type=float, default=None,
                        help="발화 중 상한 초과 프레임 비율이 이 값을 넘으면 폐기 (기본 gate_config→0.15)")
        ap.add_argument("--gate-config", default=None, help="gate_config.json 경로")

(3) main() 에서 값 해석 + gate 딕셔너리:
        _gcfg, _gpath = load_gate_config(args.gate_config)
        if args.gate_margin is None:
            args.gate_margin = float(_gcfg.get("gate_margin", 2.6))   # 안전한 새 기본
        over_frac = args.over_frac if args.over_frac is not None else float(_gcfg.get("over_frac", 0.15))
        if not args.no_speech:
            print(f"[gate] margin={args.gate_margin} over_frac={over_frac} "
                  f"(source={'gate_config' if _gpath else 'default/CLI'})", flush=True)
        ...
        gate = {"upper": float("inf"), "over_frac": over_frac}   # 기존 gate 정의 교체

(4) speech_assemble_loop : over_upper(bool) → over_cnt(int) 비율 판정으로 교체
    - finish() 안:
        n = len(voiced_rms)
        of = (over_cnt / n) if n else 0.0
        stats = {"min": float(np.min(voiced_rms)), "mean": float(np.mean(voiced_rms)),
                 "max": float(np.max(voiced_rms)), "over_frac": round(of, 3)}
        if of > gate.get("over_frac", 0.15):
            put_drop_oldest(utter_q, (None, stats, "too_loud"), MAX_UTTER_QUEUE)
        else:
            pcm = b"".join(voiced)
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            put_drop_oldest(utter_q, (audio, stats, "ok"), MAX_UTTER_QUEUE)
      그리고 초기화에서 over_upper=False → over_cnt=0
    - 루프 본문:
        else:
            if rms > upper:
                over_cnt += 1            # (기존 over_upper=True 대신 카운트)
            voiced.append(frame); voiced_rms.append(rms)
            ...

(5) speech_process_loop 폐기 로그를 비율 표시로(선택):
        if reason == "too_loud":
            print(f"[speech] 🚫 폐기(소음 {stats.get('over_frac',0)*100:.0f}% "
                  f"> {gate.get('over_frac',0.15)*100:.0f}%)", flush=True)
            continue
────────────────────────────────────────────────────────────────────────────
'''


def main():
    ap = argparse.ArgumentParser(description="too_loud 폐기 기준(gate) 데이터 튜너")
    ap.add_argument("--device", type=int, default=None, help="입력 장치 인덱스")
    ap.add_argument("--vad", type=int, default=1, help="webrtcvad aggressiveness 0~3 (mic_agent 기본 1)")
    ap.add_argument("--noise-sec", type=float, default=2.0)
    ap.add_argument("--clean-sec", type=float, default=25.0,
                    help="문장 읽기 최대 시간(초) — 발화가 충분히 모이면 더 일찍 종료")
    ap.add_argument("--min-utts", type=int, default=8,
                    help="이 개수만큼 발화가 모이면 문장 읽기를 자동 종료")
    ap.add_argument("--loud", dest="loud", action="store_true", default=True,
                    help="소음 이벤트도 녹음 (기본 on)")
    ap.add_argument("--no-loud", dest="loud", action="store_false")
    ap.add_argument("--loud-sec", type=float, default=12.0)
    ap.add_argument("--target-accept", type=float, default=0.98, help="멀쩡한 발화 목표 통과율")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--simulate", action="store_true", help="마이크 없이 합성 데이터로 로직 점검")
    ap.add_argument("--test", action="store_true",
                    help="튜닝된 gate_config 설정이 실제 발화를 잘 통과시키는지 마이크로 즉석 확인")
    ap.add_argument("--gate-config", default=OUT_DEFAULT, help="테스트에 쓸 설정 파일 (기본 gate_config.json)")
    ap.add_argument("--gate-margin", type=float, default=None, help="(테스트) margin 직접 지정 — 미지정 시 설정파일 값")
    ap.add_argument("--over-frac", type=float, default=None, help="(테스트) over_frac 직접 지정 — 미지정 시 설정파일 값")
    # 통과 발화 해석(STT+감정) 옵션
    ap.add_argument("--no-stt", action="store_true",
                    help="(테스트) 통과 발화의 STT+감정 해석 생략 — 통과/폐기만 표시")
    ap.add_argument("--whisper-size", default="base", help="(테스트) faster-whisper 모델 크기")
    ap.add_argument("--compute-type", default="int8", help="(테스트) faster-whisper compute_type")
    ap.add_argument("--threads", type=int, default=4, help="(테스트) STT/감정 CPU 스레드")
    ap.add_argument("--min-chars", type=int, default=2, help="(테스트) 이보다 짧게 인식되면 텍스트 미표시")
    ap.add_argument("--print-patch", action="store_true", help="mic_agent.py 패치만 출력")
    args = ap.parse_args()

    if args.test:
        cfg = {}
        if os.path.exists(args.gate_config):
            with open(args.gate_config, encoding="utf-8") as f:
                cfg = json.load(f)
            print(f"[설정] {args.gate_config} 로드: margin={cfg.get('gate_margin')} over_frac={cfg.get('over_frac')}")
        elif args.gate_margin is None or args.over_frac is None:
            sys.exit(f"{args.gate_config} 없음 — 먼저 튜닝하거나 --gate-margin/--over-frac 으로 직접 지정하세요.")
        try:
            import webrtcvad  # noqa
        except Exception:
            sys.exit("sounddevice/webrtcvad 필요: pip install sounddevice webrtcvad numpy")
        run_live_test(args, cfg)
        return

    if args.print_patch:
        print(PATCH)
        return

    if args.simulate:
        print("[모드] 합성 데이터 (마이크 미사용)")
        clean, loud, noise = make_synth()
        res = analyze(clean, loud, noise, target_accept=args.target_accept)
        print_report(res)
        save_config(res, args.out, args.vad)
        print(PATCH)
        return

    # ── 실측 ──
    try:
        import webrtcvad  # noqa
    except Exception:
        sys.exit("webrtcvad/sounddevice 필요: pip install sounddevice webrtcvad numpy "
                 "(마이크 없이 점검은 --simulate)")
    import webrtcvad
    vad = webrtcvad.Vad(args.vad)

    print("=" * 58)
    print("  🎙  gate 튜너 — too_loud 폐기 기준 측정")
    print("=" * 58)

    input(f"\n① 주변 소음 측정 ({args.noise_sec:.0f}초). 조용히 한 뒤 Enter…")
    nrows, _ = record_seconds(args.noise_sec, args.device)
    noise_rms = float(np.median([r for r, _, _ in nrows])) or 1.0
    print(f"   소음 RMS ≈ {noise_rms:.0f}")

    input(f"\n② 평소 말투로 문장을 몇 개 읽어주세요. "
          f"발화 {args.min_utts}개가 모이면 자동 종료(최대 {args.clean_sec:.0f}초). "
          f"준비되면 Enter…\n   (예: 결재 부탁드립니다 / 오늘 일정 확인해 주세요 / 회의 자료 공유드립니다 …)")
    clean = record_utterances_live(args.device, vad, noise_rms,
                                   min_utts=args.min_utts, max_sec=args.clean_sec)
    if len(clean) < 3:
        sys.exit(f"발화 조각이 너무 적습니다({len(clean)}개). 또렷하게 다시 시도하세요.")
    print(f"   발화 {len(clean)}개 추출")

    loud_rms = None
    if args.loud:
        input(f"\n③ (선택) 평소 작업 중 나는 큰 소리를 내주세요 ({args.loud_sec:.0f}초): "
              f"키보드 타건·문 여닫기·헛기침·목청 높이기 등. Enter…")
        lrows, _ = record_seconds(args.loud_sec, args.device)
        loud_rms = np.array([r for r, _, _ in lrows if r > noise_rms * 2.0], dtype=np.float32)
        print(f"   소음 이벤트 프레임 {len(loud_rms)}개 수집")
        if len(loud_rms) < 20:
            print("   ⚠ 소음 샘플이 적어 소음-폐기 검증은 생략될 수 있습니다.")
            if len(loud_rms) == 0:
                loud_rms = None

    res = analyze(clean, loud_rms, noise_rms, target_accept=args.target_accept)
    print_report(res)
    save_config(res, args.out, args.vad)
    print("\n저장된 설정이 실제 발화를 잘 통과시키는지 바로 확인:  python3 gate_tuner.py --test")
    print("적용 패치:  python3 gate_tuner.py --print-patch")


if __name__ == "__main__":
    main()