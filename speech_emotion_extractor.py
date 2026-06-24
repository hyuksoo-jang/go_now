#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
결재 레이더 (Approval Radar) v7 - 음성 감정 판단 (발화 기준 게이트 + 큐 적체 관리)
================================================================================
마이크(캡처 스레드) -> [상한 게이트] -> 큐(drop-oldest) -> 처리 워커
  -> (선택)디노이징 -> STT(faster-whisper) -> KoBERT 감정 -> 신호등

게이트 기준이 '소음'이 아니라 '내 발화'다:
  시작 시 ① 소음 측정 ② 테스트 문장 발화 측정 -> 상한 = 발화RMS × margin.
  내 발화보다 margin배 이상 큰 소리(외부 대화/큰 소음)가 섞인 발화는 폐기.
  SNR(발화/소음)도 함께 표시해 환경이 인식에 적합한지 알려줌.

타깃 시나리오: 큰 소음이 끼면 인식이 어려우므로, 그런 발화는 버린다.
  - 상한 초과 = 발화보다 훨씬 큰 소리 섞임 -> 발화 통째로 버림
  - 그 외      = VAD가 음성으로 잡은 것       -> 분석

v6 -> v7 변경점(큐 적체 관리):
  * 큐 상한 MAX_QUEUE(=10): 캡처 입구에서 가득 차면 오래된 발화부터 버림(drop-oldest)
  * 처리 워커는 매 턴 큐를 비우고 '가장 최근' 발화만 처리 -> 지연 누적 방지
  * 적체 발생 시 실시간 로그: 🟥 캡처측 드롭 / ⏩ 처리측 스킵 / 각 줄 끝 '큐 N'
  ※ "지금 가도 되나"가 중요하므로 밀린 옛 발화보다 최신 발화 우선이 용도에 맞음

설치:
  pip install faster-whisper webrtcvad sounddevice numpy noisereduce
  pip install transformers torch kobert-transformers sentencepiece
  # sudo apt install libportaudio2

실행:
  python approval_radar_v7.py
  python approval_radar_v7.py --gate-margin 1.6 --vad 1
  python approval_radar_v7.py --calib-only                 # 게이트 튜닝만(감정/STT 생략)
  python approval_radar_v7.py --enroll-text "확인 부탁드립니다"
"""

import argparse
import collections
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad
import noisereduce as nr
import torch
from transformers import AutoModelForSequenceClassification
from faster_whisper import WhisperModel


# ============================================================
# 설정
# ============================================================
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
SILENCE_TAIL_MS = 700
MAX_UTTER_MS = 8000
MAX_QUEUE = 10               # 큐가 이보다 많이 쌓이면 오래된 발화부터 버림(drop-oldest)
MODEL_NAME = "jeongyoonhuh/kobert-emotion-6class"
EMOTIONS = ['기쁨', '슬픔', '분노', '불안', '당황', '상처']


def rms_to_db(rms, ref):
    """기준 ref(예: 소음 RMS) 대비 상대 dB. 양수면 기준보다 큼."""
    return 20.0 * np.log10(max(rms, 1e-6) / max(ref, 1e-6))


# ============================================================
# KoBERT 감정 분류기
# ============================================================
class EmotionClassifier:
    def __init__(self, model_name=MODEL_NAME, threads=4):
        torch.set_num_threads(threads)
        from kobert_transformers import get_tokenizer
        self.tokenizer = get_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=128)
        probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        p = {e: probs[i].item() for i, e in enumerate(EMOTIONS)}
        top = max(p, key=p.get)
        return {"top": top, "top_prob": p[top], "probs": p}


# ============================================================
# 신호등 판정
# ============================================================
def traffic_light(p: dict, face_anger_prob: float = 0.0,
                  w_text: float = 0.4, w_face: float = 0.6) -> dict:
    text_risk = (p['분노'] + 0.5 * (p['불안'] + p['당황'] + p['상처'] + p['슬픔'])
                 - p['기쁨'])
    text_risk = max(0.0, min(1.0, text_risk))
    risk = w_face * face_anger_prob + w_text * text_risk
    if risk > 0.5:
        light, msg = "🔴 빨강", "지금은 피하세요"
    elif risk > 0.2:
        light, msg = "🟠 주황", "조심스럽게 접근"
    else:
        light, msg = "🟢 초록", "가도 좋습니다"
    return {"risk": risk, "light": light, "message": msg}


# ============================================================
# 디노이징 (기본 off)
# ============================================================
class Denoiser:
    def __init__(self, strength=0.0, noise_clip=None):
        self.strength = strength
        self.noise_clip = noise_clip
        self.enabled = strength > 0.0

    def __call__(self, audio_f32: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio_f32
        if self.noise_clip is not None:
            out = nr.reduce_noise(y=audio_f32, sr=SAMPLE_RATE, y_noise=self.noise_clip,
                                  prop_decrease=self.strength, stationary=True)
        else:
            out = nr.reduce_noise(y=audio_f32, sr=SAMPLE_RATE,
                                  prop_decrease=self.strength, stationary=False)
        return out.astype(np.float32)


# ============================================================
# 2단계 캘리브레이션: ① 소음 측정 -> ② 테스트 발화 측정
#   상한 = 테스트 발화 RMS(중앙값) × margin  (발화 기준, 소음 기준 아님)
#   발화 RMS는 VAD가 음성으로 잡은 프레임들의 중앙값(median)으로 산정 -> 안정적
# ============================================================
def _measure_frames(seconds, vad=None):
    """주어진 시간 동안 프레임별 (rms, is_voiced)를 수집해 반환."""
    n = int(SAMPLE_RATE * seconds)
    got = 0
    out = []
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                           dtype='int16', channels=1) as s:
        while got < n:
            data, _ = s.read(FRAME_SAMPLES)
            frame = bytes(data)
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
            voiced = bool(vad.is_speech(frame, SAMPLE_RATE)) if vad is not None else True
            out.append((rms, voiced))
            got += len(samples)
    return out


def calibrate(noise_sec=1.5, enroll_sec=4.0, margin=1.6, vad_aggr=1,
              enroll_text="결재 부탁드립니다"):
    # ① 주변 소음
    print(f"① 주변 소음 측정 — {noise_sec:.1f}초간 조용히 해주세요...")
    noise_frames = _measure_frames(noise_sec)
    noise_rms = float(np.median([r for r, _ in noise_frames])) or 1.0

    # ② 테스트 발화
    vad = webrtcvad.Vad(vad_aggr)
    print(f'② 테스트 발화 — 평소 톤으로 말해주세요: "{enroll_text}" ({enroll_sec:.0f}초)...')
    sp_frames = _measure_frames(enroll_sec, vad=vad)
    voiced_rms = [r for r, v in sp_frames if v and r > noise_rms]  # 음성+소음 이상만
    if not voiced_rms:
        voiced_rms = [r for r, _ in sp_frames]  # 발화 검출 실패 시 폴백
        print("  ⚠ 발화를 또렷이 못 잡았습니다. 마이크를 가까이 두고 다시 시도 권장.")
    speech_rms = float(np.median(voiced_rms))

    upper = speech_rms * margin
    snr_db = 20.0 * np.log10(max(speech_rms, 1e-6) / max(noise_rms, 1e-6))

    print(f"\n측정 완료 · 소음 {noise_rms:.0f} · 발화 {speech_rms:.0f} · "
          f"상한 {upper:.0f} (발화×{margin}) · SNR {snr_db:+.1f}dB")
    if snr_db < 6.0:
        print("  ⚠ SNR이 낮습니다(<6dB). 이 환경은 인식이 불안정할 수 있어요 — "
              "마이크를 더 가까이(약 20cm) 두거나 조용한 곳을 권장합니다.")
    print("— 상한 초과(=발화보다 훨씬 큰 소음)가 섞인 발화는 폐기합니다.\n")
    return noise_rms, speech_rms, upper, snr_db


# ============================================================
# 큐 적재 헬퍼: 가득 차면 가장 오래된 항목을 버리고 새 항목을 넣음(drop-oldest)
# ============================================================
def put_capped(audio_q, item, drop_counter, maxsize=MAX_QUEUE):
    dropped = 0
    while audio_q.qsize() >= maxsize:
        try:
            audio_q.get_nowait()        # 가장 오래된 것 제거
            dropped += 1
        except queue.Empty:
            break
    audio_q.put(item)
    if dropped:
        drop_counter[0] += dropped       # 캡처 입구에서 버린 누적 개수
        print(f"🟥 캡처 적체: 큐가 가득 차 오래된 발화 {dropped}개 버림 "
              f"(누적 {drop_counter[0]}개)", flush=True)
    # 발화가 큐에 적재될 때마다 현재 큐 길이 출력
    reason = item[2]
    tag = "폐기발화" if reason == "too_loud" else "발화"
    print(f"➕ 큐 적재({tag}) · 현재 큐 {audio_q.qsize()}/{maxsize}", flush=True)


# ============================================================
# 캡처 스레드: 마이크 -> 상한 게이트 -> VAD -> 큐 적재
#   상한 초과가 섞인 발화는 폐기. 폐기된 발화도 진단용으로 큐에 넣어 로그 표시.
# ============================================================
def capture_loop(audio_q, stop_evt, drop_counter, aggressiveness=1, upper=float('inf')):
    vad = webrtcvad.Vad(aggressiveness)
    silence_frames = SILENCE_TAIL_MS // FRAME_MS
    triggered = False
    ring = collections.deque(maxlen=8)
    voiced = []
    voiced_rms = []
    num_silence = 0
    over_upper = False        # 상한 초과가 섞였는지

    def finish():
        nonlocal triggered, voiced, voiced_rms, num_silence, over_upper
        if voiced:
            stats = {
                "min": float(np.min(voiced_rms)),
                "mean": float(np.mean(voiced_rms)),
                "max": float(np.max(voiced_rms)),
            }
            if over_upper:
                # 상한 초과로 폐기 — 오디오는 버리되 진단은 넘김
                put_capped(audio_q, (None, stats, "too_loud"), drop_counter)
            else:
                pcm = b''.join(voiced)
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                put_capped(audio_q, (audio, stats, "ok"), drop_counter)
        triggered = False
        voiced = []
        voiced_rms = []
        num_silence = 0
        over_upper = False
        ring.clear()

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                           dtype='int16', channels=1) as stream:
        while not stop_evt.is_set():
            data, _ = stream.read(FRAME_SAMPLES)
            frame = bytes(data)
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(samples ** 2)) if samples.size else 0.0

            # 상한 이하 + VAD가 음성으로 볼 때만 '발화 프레임'
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


# ============================================================
# STT 헬퍼
# ============================================================
def transcribe(stt, audio_f32):
    segments, _ = stt.transcribe(audio_f32, language="ko", beam_size=1,
                                 condition_on_previous_text=False, vad_filter=False)
    return "".join(seg.text for seg in segments).strip()


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="결재 레이더 v7 (발화 기준 게이트 + 큐 적체 관리)")
    ap.add_argument("--whisper-size", default="base", help="tiny/base/small")
    ap.add_argument("--vad", type=int, default=1,
                    help="VAD aggressiveness 0~3 (조용한 발화 타깃이면 0~1 권장)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--gate-margin", type=float, default=1.6,
                    help="상한 = 테스트발화 RMS × margin. 이보다 큰 소리는 외부소음으로 폐기(1.5~2.0 권장)")
    ap.add_argument("--enroll-text", default="결재 부탁드립니다",
                    help="시작 시 따라 말할 테스트 문장")
    ap.add_argument("--enroll-sec", type=float, default=4.0, help="테스트 발화 측정 시간(초)")
    ap.add_argument("--denoise-strength", type=float, default=0.0, help="0=off")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--calib-only", action="store_true",
                    help="게이트 튜닝 전용: 음량만 찍고 STT/감정은 생략(빠름)")
    args = ap.parse_args()

    stt = clf = None
    if not args.calib_only:
        print("[1/3] STT 모델 로딩...", flush=True)
        stt = WhisperModel(args.whisper_size, device="cpu", compute_type="int8",
                           cpu_threads=args.threads)
        print("[2/3] KoBERT 감정 모델 로딩...", flush=True)
        clf = EmotionClassifier(threads=args.threads)

    noise_rms, speech_rms, upper = (1.0, 0.0, float('inf'))
    calibrated = not args.no_calibrate
    if calibrated:
        noise_rms, speech_rms, upper, _snr = calibrate(
            enroll_sec=args.enroll_sec, margin=args.gate_margin,
            vad_aggr=args.vad, enroll_text=args.enroll_text)
    denoise = Denoiser(strength=args.denoise_strength, noise_clip=None)

    band = f"상한 {upper:.0f}" if calibrated else "off"
    head = "음량 진단 전용(STT/감정 생략)" if args.calib_only else "감정 분석"
    print(f"마이크 대기 중 · VAD {args.vad} · {band} · {head}")
    print("말씀하세요 (Ctrl+C 종료)\n", flush=True)

    audio_q = queue.Queue()
    stop_evt = threading.Event()
    drop_counter = [0]          # 캡처 입구에서 버린 누적 발화 수(스레드 공유)
    cap = threading.Thread(target=capture_loop,
                           args=(audio_q, stop_evt, drop_counter, args.vad, upper),
                           daemon=True)
    cap.start()

    try:
        while True:
            item = audio_q.get()                 # 최소 1개 대기(블로킹)
            # 처리가 밀려 큐에 쌓여 있으면 전부 비우고 '가장 최근' 발화만 채택
            skipped = 0
            while not audio_q.empty():
                item = audio_q.get_nowait()
                skipped += 1
            backlog = audio_q.qsize()            # 비운 뒤라 보통 0
            if skipped:
                print(f"⏩ 처리 적체: 밀린 발화 {skipped}개 건너뛰고 최신만 처리 "
                      f"(현재 큐 {backlog})", flush=True)

            audio, stats, reason = item

            # --- 발화 음량 진단 (소음 대비 상대 dB) ---
            mean_db = rms_to_db(stats["mean"], noise_rms)
            max_db = rms_to_db(stats["max"], noise_rms)
            loud = (f'음량 RMS 평균 {stats["mean"]:.0f}({mean_db:+.1f}dB) '
                    f'최대 {stats["max"]:.0f}({max_db:+.1f}dB) · '
                    f'소음 {noise_rms:.0f} · 상한 {upper:.0f}')

            # 게이트로 폐기된 발화: 사유와 음량만 찍고 분석은 건너뜀
            if reason == "too_loud":
                print(f'🚫 폐기(너무 큼: 상한 초과) · {loud} · 큐 {backlog}\n', flush=True)
                continue

            if args.calib_only:
                # 튜닝 모드: 통과한 발화의 음량만 확인
                print(f'✅ 통과 · {loud} · 큐 {backlog}\n', flush=True)
                continue

            # --- 정상 처리: 디노이즈 -> STT -> 감정 -> 신호등 ---
            audio_dn = denoise(audio)
            t1 = time.time(); text = transcribe(stt, audio_dn); stt_ms = (time.time()-t1)*1000
            if len(text) < args.min_chars:
                print(f'⚠  인식 실패(빈 텍스트) · {loud}\n', flush=True)
                continue
            t2 = time.time(); res = clf.predict(text); emo_ms = (time.time()-t2)*1000
            light = traffic_light(res["probs"])
            ranked = sorted(res["probs"].items(), key=lambda x: -x[1])

            print(f'🗣  "{text}"')
            print(f'   {loud}')
            print(f'   감정: {res["top"]} ({res["top_prob"]*100:.1f}%)'
                  f'  |  위험도 {light["risk"]:.2f}  {light["light"]} {light["message"]}')
            print("   분포: " + "  ".join(f"{e} {v*100:.0f}%" for e, v in ranked))
            print(f"   (STT {stt_ms:.0f}ms · 감정 {emo_ms:.0f}ms · 큐 {backlog})\n",
                  flush=True)
    except KeyboardInterrupt:
        print("\n종료합니다.")
        stop_evt.set()


if __name__ == "__main__":
    main()