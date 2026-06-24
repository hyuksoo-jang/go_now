#!/usr/bin/env python3
"""
sigh_detector_mode_a.py
────────────────────────────────────────────────────────────────
모드 A — YAMNet TFLite 기반 실시간 한숨 감지기
하드웨어: Raspberry Pi 5 + NC-150 (USB, 장치 인덱스 1)

확인된 모델 스펙 (lite-model_yamnet_classification_tflite_1):
  입력: [15600]  float32  1D (975ms @ 16kHz)
  출력: [1, 521] float32  521개 AudioSet 클래스 확률
────────────────────────────────────────────────────────────────
"""

import sys
import time
import queue
import csv
import wave
import os
import numpy as np

# ── 의존성 확인 ──────────────────────────────────────────────
try:
    import pyaudio
except ImportError:
    sys.exit("❌ pyaudio 없음.  pip install pyaudio")

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_SRC = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_SRC = "tensorflow"
    except ImportError:
        sys.exit("❌ tflite-runtime 없음.  pip install tflite-runtime")


# ══════════════════════════════════════════════════════════════
# ▶ 설정값 — 여기만 수정하면 됩니다
# ══════════════════════════════════════════════════════════════

DEVICE_INDEX     = 1
MODEL_PATH       = "model/yamnet.tflite"
LABEL_PATH       = "model/yamnet_class_map.csv"

SAMPLE_RATE      = 16000
CHUNK_MS         = 250

THRESHOLD        = 0.15
COOLDOWN_SEC     = 2.0

SIGH_KEYWORDS    = ["sigh", "breathing", "exhale"]

DEBUG            = False
SHOW_DETECT_DATA = True

# ── 감지 시 저장 설정 ─────────────────────────────────────────
SAVE_AUDIO       = True          # False 로 바꾸면 저장 안 함
SAVE_DIR         = "detected"    # 저장 폴더 (없으면 자동 생성)


# ══════════════════════════════════════════════════════════════
# 1. 레이블 로딩
# ══════════════════════════════════════════════════════════════
def load_labels(path: str) -> dict[int, str]:
    labels = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[int(row["index"])] = row["display_name"].lower()
    except FileNotFoundError:
        sys.exit(f"❌ 레이블 파일 없음: {path}\n   setup_mode_a.sh 를 먼저 실행하세요.")
    return labels


def find_sigh_indices(labels: dict[int, str]) -> list[int]:
    indices = [
        idx for idx, name in labels.items()
        if any(kw in name for kw in SIGH_KEYWORDS)
    ]
    if not indices:
        print(f"⚠️  SIGH_KEYWORDS {SIGH_KEYWORDS} 에 해당하는 클래스 없음")
    else:
        print(f"📋 감지 대상 클래스 ({len(indices)}개):")
        for i in indices:
            print(f"   [{i:3d}] {labels[i]}")
    return indices


# ══════════════════════════════════════════════════════════════
# 2. YAMNet TFLite 모델 로딩
# ══════════════════════════════════════════════════════════════
def load_model(path: str) -> tuple:
    try:
        interp = tflite.Interpreter(model_path=path, num_threads=4)
        interp.allocate_tensors()
    except FileNotFoundError:
        sys.exit(f"❌ 모델 파일 없음: {path}\n   setup_mode_a.sh 를 먼저 실행하세요.")
    except Exception as e:
        sys.exit(f"❌ 모델 로딩 실패: {e}")

    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    in_shape  = in_det[0]["shape"]
    out_shape = out_det[0]["shape"]
    window_len = int(in_shape[0]) if len(in_shape) == 1 else int(in_shape[1])

    print(f"\n🧠 YAMNet 로딩 완료 ({TFLITE_SRC})")
    print(f"   입력 shape : {list(in_shape)}  ({window_len / SAMPLE_RATE * 1000:.0f}ms @ {SAMPLE_RATE}Hz)")
    print(f"   출력 shape : {list(out_shape)}  (클래스 수: {out_shape[-1]})")

    return interp, in_det, out_det, window_len


# ══════════════════════════════════════════════════════════════
# 3. 추론
# ══════════════════════════════════════════════════════════════
def run_inference(
    interp,
    in_det,
    out_det,
    audio: np.ndarray,
    window_len: int,
    sigh_indices: list[int],
    all_labels: dict[int, str],
) -> tuple[bool, float, str, np.ndarray]:
    in_shape = in_det[0]["shape"]
    clipped  = audio[:window_len].astype(np.float32)
    inp      = clipped if len(in_shape) == 1 else clipped.reshape(1, window_len)

    interp.set_tensor(in_det[0]["index"], inp)
    interp.invoke()

    scores      = interp.get_tensor(out_det[0]["index"])
    mean_scores = scores[0]

    if not sigh_indices:
        return False, 0.0, "unknown", mean_scores

    sigh_probs = mean_scores[sigh_indices]
    best_local = int(np.argmax(sigh_probs))
    max_prob   = float(sigh_probs[best_local])
    best_idx   = sigh_indices[best_local]
    top_class  = all_labels.get(best_idx, f"class_{best_idx}")

    if DEBUG:
        top5_idx  = np.argsort(mean_scores)[::-1][:5]
        top5_info = [(all_labels.get(i, f"c{i}"), f"{mean_scores[i]:.3f}")
                     for i in top5_idx]
        print(f"  [DEBUG] top5: {top5_info}")

    return max_prob >= THRESHOLD, max_prob, top_class, mean_scores


# ══════════════════════════════════════════════════════════════
# 4. 오디오 저장
# ══════════════════════════════════════════════════════════════
def save_detected_audio(audio: np.ndarray, prob: float):
    """
    감지된 975ms 오디오를 WAV 파일로 저장
    파일명: detected/sigh_20240623_142301_p0.312.wav
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = f"sigh_{ts}_p{prob:.3f}.wav"
    path     = os.path.join(SAVE_DIR, filename)

    # float32 [-1.0, 1.0] → int16 변환
    pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return path


# ══════════════════════════════════════════════════════════════
# 5. 마이크 장치 탐색
# ══════════════════════════════════════════════════════════════
def find_mic(pa: pyaudio.PyAudio) -> int:
    if DEVICE_INDEX is not None:
        info = pa.get_device_info_by_index(DEVICE_INDEX)
        print(f"✅ 마이크 고정 지정 [{DEVICE_INDEX}]: {info['name']}")
        return DEVICE_INDEX

    print("\n🔍 오디오 입력 장치 목록:")
    usb_idx = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            name   = info["name"]
            is_usb = any(k in name.lower() for k in ["usb", "nc-150", "nc150"])
            print(f"   [{i}] {name}" + (" ← USB 후보" if is_usb else ""))
            if usb_idx is None and is_usb:
                usb_idx = i

    if usb_idx is None:
        usb_idx = pa.get_default_input_device_info()["index"]
        print(f"\n⚠️  USB 자동 탐색 실패 → 기본 장치 [{usb_idx}] 사용")
    else:
        print(f"\n✅ 자동 선택 [{usb_idx}]: {pa.get_device_info_by_index(usb_idx)['name']}")
    return usb_idx


# ══════════════════════════════════════════════════════════════
# 6. 메인 루프
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 58)
    print("  🎙️  실시간 한숨 감지기 — 모드 A (YAMNet TFLite)")
    print("=" * 58)

    labels       = load_labels(LABEL_PATH)
    sigh_indices = find_sigh_indices(labels)
    interp, in_det, out_det, window_len = load_model(MODEL_PATH)

    pa      = pyaudio.PyAudio()
    dev_idx = find_mic(pa)
    chunk_n = int(SAMPLE_RATE * CHUNK_MS / 1000)

    if SAVE_AUDIO:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"💾 감지 시 저장 경로: {os.path.abspath(SAVE_DIR)}/")

    audio_q: queue.Queue = queue.Queue(maxsize=40)

    def callback(in_data, frame_count, time_info, status):
        if status:
            print(f"⚠️  오디오 상태: {status}", end="")
        audio_q.put(in_data)
        return (None, pyaudio.paContinue)

    try:
        stream = pa.open(
            format             = pyaudio.paInt16,
            channels           = 1,
            rate               = SAMPLE_RATE,
            input              = True,
            input_device_index = dev_idx,
            frames_per_buffer  = chunk_n,
            stream_callback    = callback,
        )
    except Exception as e:
        pa.terminate()
        sys.exit(f"❌ 마이크 스트림 오류: {e}")

    stream.start_stream()

    print(f"\n🎧 감지 시작 (Ctrl+C 로 종료)")
    print(f"   THRESHOLD={THRESHOLD}  COOLDOWN={COOLDOWN_SEC}s  "
          f"WINDOW={window_len/SAMPLE_RATE*1000:.0f}ms  DEBUG={DEBUG}")
    print("-" * 58)
    print("  [시각]  True   ← 클래스명  (확률)")
    print("-" * 58)

    buffer      = np.zeros(window_len, dtype=np.float32)
    last_detect = 0.0

    try:
        while stream.is_active():
            try:
                raw = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            chunk  = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk

            is_sigh, prob, cls_name, all_scores = run_inference(
                interp, in_det, out_det, buffer, window_len, sigh_indices, labels
            )

            now = time.time()
            ts  = time.strftime("%H:%M:%S")

            if is_sigh and (now - last_detect) >= COOLDOWN_SEC:
                last_detect = now
                print(f"[{ts}]  True   ← {cls_name:<20s}  (확률: {prob:.3f})")

                if SHOW_DETECT_DATA:
                    top5_idx = np.argsort(all_scores)[::-1][:5]
                    top5_str = "  ".join(
                        f"{labels.get(i, f'c{i}')}({all_scores[i]:.3f})"
                        for i in top5_idx
                    )
                    sigh_str = "  ".join(
                        f"{labels.get(i, f'c{i}')}({all_scores[i]:.3f})"
                        for i in sigh_indices
                    )
                    rms  = float(np.sqrt(np.mean(buffer ** 2)))
                    peak = float(np.max(np.abs(buffer)))
                    print(f"  ├ top5    : {top5_str}")
                    print(f"  ├ 한숨류  : {sigh_str}")
                    print(f"  └ 에너지  : RMS={rms:.4f}  peak={peak:.4f}  "
                          f"window={window_len/SAMPLE_RATE*1000:.0f}ms")

                # ── 감지된 오디오 저장 ──────────────────────
                if SAVE_AUDIO:
                    saved_path = save_detected_audio(buffer.copy(), prob)
                    print(f"  💾 저장: {saved_path}")

            elif DEBUG:
                print(f"[{ts}]  False  ({cls_name}: {prob:.3f})")

    except KeyboardInterrupt:
        print("\n\n👋 종료합니다.")
        if SAVE_AUDIO:
            saved = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".wav")])
            print(f"   총 저장된 파일: {saved}개 → {os.path.abspath(SAVE_DIR)}/")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("🔌 마이크 스트림 닫힘.")


if __name__ == "__main__":
    main()