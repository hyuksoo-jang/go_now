#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speech_tuner.py — 발화 인식률(STT) 최적 파라미터 탐색
────────────────────────────────────────────────────────────────
결재 레이더의 발화 파이프라인(faster-whisper)에서, 실제 내 목소리·마이크
환경에 대해 가장 인식률이 좋은 설정을 찾아준다.

동작:
  1) 기준 문장들을 화면에 띄우고 차례로 녹음(또는 기존 녹음 재사용)
  2) 모델크기 × beam × VAD필터(× compute) 격자로 STT 실행
  3) CER(문자 오류율) + 지연시간 측정 → 순위표
  4) "최고 정확도" / "정확도-속도 균형" 설정 추천 + mic_agent 적용값 출력

사용:
  python3 speech_tuner.py                      # 녹음 후 기본 격자 벤치마크
  python3 speech_tuner.py --quick              # 빠른 격자(base/small, beam1/5)
  python3 speech_tuner.py --no-record          # 기존 녹음만으로 재벤치마크
  python3 speech_tuner.py --phrases my.txt     # 내 문장 목록(줄당 1문장)
  python3 speech_tuner.py --sizes base,small --beams 1,5 --vad-filter 0,1
  python3 speech_tuner.py --device 1 --secs 4

필요: faster-whisper, sounddevice, numpy  (pip install -r requirements-mic.txt)
────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import json
import time
import wave
import argparse
import unicodedata

SAMPLE_RATE = 16000
SAMPLES_DIR = "tuner_samples"
MANIFEST    = "manifest.json"

# 도메인 기준 문장(팀장 결재 상황에서 실제로 쓰는 말투)
DEFAULT_PHRASES = [
    "결재 부탁드립니다",
    "지금 잠깐 시간 괜찮으세요",
    "이 건 오늘까지 처리해야 합니다",
    "검토 부탁드려도 될까요",
    "급한 결재가 하나 있습니다",
    "확인하고 다시 말씀드리겠습니다",
]


# ─────────────────────────────────────────────────────────────
# CER (문자 오류율)
# ─────────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    """공백·문장부호 제거 + NFC 정규화 (한글 STT 비교용)."""
    s = unicodedata.normalize("NFC", s)
    out = []
    for ch in s:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):  # 구두점/기호 제거
            continue
        out.append(ch.lower())
    return "".join(out)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """문자 오류율 = 편집거리 / 기준 길이 (공백·부호 무시)."""
    r, h = _normalize(ref), _normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


# ─────────────────────────────────────────────────────────────
# 녹음 / 로드
# ─────────────────────────────────────────────────────────────
def record_samples(phrases, secs, device, outdir):
    import numpy as np
    import sounddevice as sd

    os.makedirs(outdir, exist_ok=True)
    manifest = []
    print(f"\n총 {len(phrases)}문장을 녹음합니다. 각 문장당 {secs:.1f}초.\n")
    for i, phrase in enumerate(phrases):
        slug = "".join(c for c in phrase if c.isalnum())[:12] or f"p{i}"
        wav_path = os.path.join(outdir, f"{i:03d}_{slug}.wav")
        print(f"[{i+1}/{len(phrases)}] 또렷하게 읽어주세요:")
        print(f'    "{phrase}"')
        try:
            input("    준비되면 Enter (건너뛰기: s+Enter) ▶ ")
        except EOFError:
            pass
        if sys.stdin and not sys.stdin.closed:
            pass
        for c in (3, 2, 1):
            print(f"    {c}...", end=" ", flush=True)
            time.sleep(0.6)
        print("🔴 녹음", flush=True)

        kw = {"samplerate": SAMPLE_RATE, "channels": 1, "dtype": "int16"}
        if device is not None:
            kw["device"] = device
        audio = sd.rec(int(secs * SAMPLE_RATE), **kw)
        sd.wait()
        audio = audio.reshape(-1)

        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())

        peak = int(np.max(np.abs(audio))) if audio.size else 0
        flag = "⚠ 너무 작음" if peak < 500 else "✅"
        print(f"    저장: {wav_path}  (피크 {peak} {flag})\n")
        manifest.append({"wav": wav_path, "ref": phrase})

    with open(os.path.join(outdir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_manifest(outdir):
    path = os.path.join(outdir, MANIFEST)
    if not os.path.exists(path):
        sys.exit(f"❌ 기존 녹음이 없습니다: {path}\n   --no-record 없이 먼저 녹음하세요.")
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    return [m for m in man if os.path.exists(m["wav"])]


def load_wav_float32(path):
    import numpy as np
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a


# ─────────────────────────────────────────────────────────────
# 벤치마크 (주입형 — 테스트 가능)
# ─────────────────────────────────────────────────────────────
def run_benchmark(samples, sizes, computes, beams, vadfs, load_model, transcribe,
                  progress=True):
    """
    samples:   [{"ref":str, "audio":<obj>}]
    load_model(size, compute) -> model
    transcribe(model, audio, beam, vadf) -> hyp_text
    반환: rows[ {size,compute,beam,vad_filter,cer_mean,cer_med,lat_mean,details[]} ]
    """
    rows = []
    total = len(sizes) * len(computes) * len(beams) * len(vadfs)
    idx = 0
    for size in sizes:
        for compute in computes:
            model = load_model(size, compute)
            for beam in beams:
                for vadf in vadfs:
                    idx += 1
                    cers, lats, details = [], [], []
                    for smp in samples:
                        t0 = time.time()
                        hyp = transcribe(model, smp["audio"], beam, vadf)
                        lat = time.time() - t0
                        c = cer(smp["ref"], hyp)
                        cers.append(c); lats.append(lat)
                        details.append({"ref": smp["ref"], "hyp": hyp,
                                        "cer": round(c, 4), "lat": round(lat, 3)})
                    cers_sorted = sorted(cers)
                    med = cers_sorted[len(cers_sorted) // 2] if cers_sorted else 1.0
                    row = {
                        "size": size, "compute": compute, "beam": beam,
                        "vad_filter": vadf,
                        "cer_mean": sum(cers) / len(cers) if cers else 1.0,
                        "cer_med": med,
                        "lat_mean": sum(lats) / len(lats) if lats else 0.0,
                        "details": details,
                    }
                    rows.append(row)
                    if progress:
                        print(f"  [{idx}/{total}] size={size} compute={compute} "
                              f"beam={beam} vad={int(vadf)} → "
                              f"CER {row['cer_mean']*100:5.1f}%  "
                              f"{row['lat_mean']:.2f}s/문장")
    return rows


def rank(rows, tol=0.02):
    """cer_mean 오름차순. 추천 = best CER 와 tol 이내 중 가장 빠른 것."""
    by_cer = sorted(rows, key=lambda r: (r["cer_mean"], r["lat_mean"]))
    best = by_cer[0]
    near = [r for r in by_cer if r["cer_mean"] <= best["cer_mean"] + tol]
    rec = min(near, key=lambda r: r["lat_mean"])
    return by_cer, best, rec


# ─────────────────────────────────────────────────────────────
# 실제 faster-whisper 연결
# ─────────────────────────────────────────────────────────────
def make_real_callables(threads):
    cache = {}

    def load_model(size, compute):
        key = (size, compute)
        if key not in cache:
            from faster_whisper import WhisperModel
            print(f"  · 모델 로드: {size} ({compute}) ...", flush=True)
            cache[key] = WhisperModel(size, device="cpu",
                                      compute_type=compute, cpu_threads=threads)
        return cache[key]

    def transcribe(model, audio, beam, vadf):
        segments, _ = model.transcribe(
            audio, language="ko", beam_size=beam, vad_filter=vadf,
            temperature=0.0, condition_on_previous_text=False)
        return "".join(s.text for s in segments).strip()

    return load_model, transcribe


# ─────────────────────────────────────────────────────────────
# 리포트
# ─────────────────────────────────────────────────────────────
def print_report(by_cer, best, rec):
    print("\n" + "=" * 64)
    print("  순위표 (CER 낮을수록 좋음)")
    print("=" * 64)
    print(f"  {'순위':>2} {'size':<7}{'comp':<6}{'beam':<5}{'vad':<4}"
          f"{'CER%':>7}{'지연(s)':>9}")
    for i, r in enumerate(by_cer, 1):
        star = "  ★" if r is best else ("  ◎" if r is rec else "")
        print(f"  {i:>2} {r['size']:<7}{r['compute']:<6}{r['beam']:<5}"
              f"{int(r['vad_filter']):<4}{r['cer_mean']*100:>6.1f} "
              f"{r['lat_mean']:>8.2f}{star}")
    print("  ★ 최고 정확도   ◎ 추천(정확도-속도 균형)")

    print("\n" + "-" * 64)
    print("  추천 설정 (정확도-속도 균형)")
    print("-" * 64)
    print(f"    whisper_size = {rec['size']}")
    print(f"    compute_type = {rec['compute']}")
    print(f"    beam_size    = {rec['beam']}")
    print(f"    vad_filter   = {bool(rec['vad_filter'])}")
    print(f"    → 평균 CER {rec['cer_mean']*100:.1f}% · {rec['lat_mean']:.2f}s/문장")

    print("\n  mic_agent.py 적용:")
    print(f"    실행 플래그:  --whisper-size {rec['size']}")
    print("    transcribe() 인자 (speech_process_loop 안):")
    print(f"      model.transcribe(audio, language=\"ko\",")
    print(f"          beam_size={rec['beam']}, vad_filter={bool(rec['vad_filter'])},")
    print(f"          temperature=0.0, condition_on_previous_text=False)")
    if rec["compute"] != "int8":
        print(f"    WhisperModel(..., compute_type=\"{rec['compute']}\")")
    print("=" * 64)


def write_config(rec, path):
    import datetime
    cfg = {
        "whisper_size": rec["size"],
        "compute_type": rec["compute"],
        "beam_size": rec["beam"],
        "vad_filter": bool(rec["vad_filter"]),
        "_source": "speech_tuner",
        "_cer_mean_pct": round(rec["cer_mean"] * 100, 2),
        "_updated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\n✅ STT 설정 저장: {os.path.abspath(path)}")
    print("   → mic_agent 가 시작 시 이 파일을 자동으로 읽어 size/compute 를 적용합니다.")
    print("   → beam/VAD 는 매 세션 대시보드 캘리브레이션이 자동으로 재조정합니다.")
    print("   (소스 수정 불필요. mic_agent 와 같은 폴더에 두거나 --stt-config 로 지정)")


def save_outputs(rows, outdir):
    # 설정별 요약 CSV
    with open(os.path.join(outdir, "tuner_results.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        wtr = csv.writer(f)
        wtr.writerow(["size", "compute", "beam", "vad_filter",
                      "cer_mean_%", "cer_median_%", "lat_mean_s"])
        for r in sorted(rows, key=lambda x: x["cer_mean"]):
            wtr.writerow([r["size"], r["compute"], r["beam"], int(r["vad_filter"]),
                          round(r["cer_mean"] * 100, 2), round(r["cer_med"] * 100, 2),
                          round(r["lat_mean"], 3)])
    # 문장별 상세 JSON
    with open(os.path.join(outdir, "tuner_details.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {outdir}/tuner_results.csv , {outdir}/tuner_details.json")


# ─────────────────────────────────────────────────────────────
def _csv_list(s, conv=str):
    return [conv(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="발화 STT 인식률 최적 파라미터 탐색")
    ap.add_argument("--phrases", help="기준 문장 파일(줄당 1문장). 없으면 기본 문장")
    ap.add_argument("--secs", type=float, default=4.0, help="문장당 녹음 길이(초)")
    ap.add_argument("--device", type=int, default=None, help="입력 장치 인덱스")
    ap.add_argument("--samples-dir", default=SAMPLES_DIR, help="녹음 저장 폴더")
    ap.add_argument("--no-record", action="store_true", help="기존 녹음으로만 벤치마크")
    ap.add_argument("--sizes", default="tiny,base,small", help="모델 크기 목록")
    ap.add_argument("--computes", default="int8", help="compute_type 목록")
    ap.add_argument("--beams", default="1,5", help="beam_size 목록")
    ap.add_argument("--vad-filter", default="0,1", help="VAD 필터 0/1 목록")
    ap.add_argument("--threads", type=int, default=4, help="CPU 스레드")
    ap.add_argument("--quick", action="store_true",
                    help="빠른 격자(base,small × beam1,5 × vad0)")
    ap.add_argument("--config", default="../../stt_config.json",
                    help="추천 size/compute 를 저장할 STT 설정 파일 (mic_agent 가 읽음)")
    ap.add_argument("--no-write-config", action="store_true",
                    help="추천값을 stt_config.json 에 저장하지 않음")
    args = ap.parse_args()

    if args.quick:
        args.sizes, args.beams, args.vad_filter = "base,small", "1,5", "0"

    # 문장
    if args.phrases:
        with open(args.phrases, encoding="utf-8") as f:
            phrases = [ln.strip() for ln in f if ln.strip()]
    else:
        phrases = DEFAULT_PHRASES

    # 녹음 or 로드
    if args.no_record:
        manifest = load_manifest(args.samples_dir)
        print(f"기존 녹음 {len(manifest)}개 사용")
    else:
        manifest = record_samples(phrases, args.secs, args.device, args.samples_dir)

    if not manifest:
        sys.exit("❌ 녹음 샘플이 없습니다.")

    samples = [{"ref": m["ref"], "audio": load_wav_float32(m["wav"])}
               for m in manifest]

    sizes   = _csv_list(args.sizes)
    computes = _csv_list(args.computes)
    beams   = _csv_list(args.beams, int)
    vadfs   = [bool(int(x)) for x in _csv_list(args.vad_filter)]

    print(f"\n격자: sizes={sizes} computes={computes} beams={beams} "
          f"vad={[int(v) for v in vadfs]}  "
          f"→ {len(sizes)*len(computes)*len(beams)*len(vadfs)}개 설정 × "
          f"{len(samples)}문장\n")

    load_model, transcribe = make_real_callables(args.threads)
    rows = run_benchmark(samples, sizes, computes, beams, vadfs,
                         load_model, transcribe)
    by_cer, best, rec = rank(rows)
    print_report(by_cer, best, rec)
    save_outputs(rows, args.samples_dir)
    if not args.no_write_config:
        write_config(rec, args.config)


if __name__ == "__main__":
    main()