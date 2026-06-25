#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_devices.py
────────────────────────────────────────────────────────────────
결재 레이더 — 카메라/마이크 연결 사전 점검 모드

본 실행(go_now_v2 / mic_agent) 전에, 어느 인덱스의 카메라가 실제로
프레임을 주는지, 어느 마이크가 소리를 잡는지 먼저 확인한다.
점검 결과의 카메라 탐색 로직은 go_now_v2 의 open_camera 와 동일하므로,
여기서 OK 로 나온 인덱스를 그대로 쓰면 된다.

실행:
  python3 check_devices.py                # 카메라 + 마이크 모두
  python3 check_devices.py --cam-only     # 카메라만 (cam 기기)
  python3 check_devices.py --mic-only     # 마이크만 (mic 기기)
  python3 check_devices.py 1              # 카메라 선호 인덱스 1 부터
  python3 check_devices.py --device 1     # 마이크 device 1 집중 점검
  python3 check_devices.py --mic-secs 3   # 마이크 측정 시간(초)
────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import time


# ─────────────────────────────────────────────────────────────
# 카메라 점검 (open_camera 와 동일한 탐색 로직)
# ─────────────────────────────────────────────────────────────
def check_cameras(preferred_idx=0, max_idx=8):
    print("─" * 58)
    print("📷  카메라 점검")
    print("─" * 58)
    try:
        import cv2
    except ImportError:
        print("  ❌ opencv-python 미설치 → 카메라 점검 불가")
        print("     설치: pip install opencv-python")
        return []

    backends = [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]
    idx_order = [preferred_idx] + [i for i in range(max_idx) if i != preferred_idx]

    working = []
    seen = set()
    for bname, backend in backends:
        for idx in idx_order:
            key = (idx, bname)
            if key in seen:
                continue
            seen.add(key)
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            got = False
            for _ in range(10):           # 워밍업 재시도
                ret, frame = cap.read()
                if ret and frame is not None:
                    got = True
                    break
                time.sleep(0.1)
            if got:
                h, w = frame.shape[:2]
                print(f"  ✅ index {idx:2d} [{bname:4s}]  {w}×{h}  프레임 OK")
                working.append((idx, bname))
            else:
                print(f"  ⚠  index {idx:2d} [{bname:4s}]  열림 but 프레임 없음")
            cap.release()

    # libcamera GStreamer 폴백 점검
    gst = ("libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 "
           "! videoconvert ! appsink")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                print("  ✅ libcamera GStreamer 파이프라인  프레임 OK")
                working.append(("gstreamer", "GST"))
                break
            time.sleep(0.1)
        cap.release()

    print()
    if working:
        idx0 = working[0][0]
        print(f"  → 사용 가능: {working}")
        if isinstance(idx0, int):
            print(f"  → 추천 실행: python3 run_all.py --role cam {idx0}")
    else:
        print("  ❌ 프레임을 주는 카메라 없음.")
        print("     · v4l2-ctl --list-devices 로 노드 확인")
        print("     · CSI 모듈이면 OpenCV V4L2 로 안 잡힐 수 있음 → Picamera2 경로 필요")
    return working


# ─────────────────────────────────────────────────────────────
# 마이크 점검 (장치 목록 + 실제 캡처 레벨)
# ─────────────────────────────────────────────────────────────
def check_microphones(device=None, seconds=1.5):
    print("─" * 58)
    print("🎙️   마이크 점검")
    print("─" * 58)
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("  ❌ sounddevice/numpy 미설치 → 마이크 점검 불가")
        print("     설치: pip install sounddevice numpy  (+ sudo apt install libportaudio2)")
        return []
    except OSError as e:
        # sounddevice 는 PortAudio 미설치 시 OSError 를 던진다
        print(f"  ❌ 오디오 백엔드 사용 불가: {e}")
        print("     설치: sudo apt install libportaudio2")
        return []

    # 입력 장치 목록
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"  ❌ 오디오 시스템 조회 실패: {e}")
        return []

    inputs = [(i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    if not inputs:
        print("  ❌ 입력(마이크) 장치가 없습니다.")
        return []

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    print("  입력 장치 목록:")
    for i, d in inputs:
        mark = " ← 기본" if i == default_in else ""
        usb = " (USB)" if any(k in d["name"].lower() for k in ("usb", "nc-150", "nc150")) else ""
        print(f"    [{i:2d}] {d['name']}{usb}{mark}")
    print()

    # 실제 캡처 레벨 측정
    targets = [device] if device is not None else (
        [default_in] if default_in is not None else [inputs[0][0]])
    SR, FRAME = 16000, 480
    working = []
    for dev in targets:
        try:
            name = sd.query_devices(dev)["name"]
        except Exception:
            name = "?"
        print(f"  ▶ device {dev} ({name}) 캡처 테스트 — {seconds:.1f}초간 말해보세요...")
        rms_vals = []
        try:
            with sd.RawInputStream(samplerate=SR, blocksize=FRAME, dtype="int16",
                                   channels=1, device=dev) as s:
                t_end = time.time() + seconds
                peak = 0.0
                while time.time() < t_end:
                    data, _ = s.read(FRAME)
                    x = np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
                    rms_vals.append(rms)
                    peak = max(peak, float(np.max(np.abs(x))) if x.size else 0.0)
                    bar = "█" * min(40, int(rms / 50))
                    print(f"\r     레벨 |{bar:<40}| RMS {rms:6.0f}", end="", flush=True)
            print()
        except Exception as e:
            print(f"\n  ❌ device {dev} 캡처 실패: {e}")
            continue

        mean_rms = sum(rms_vals) / len(rms_vals) if rms_vals else 0.0
        if peak < 50:
            print(f"  ⚠  device {dev}: 신호가 거의 없음 (peak {peak:.0f}) — "
                  "음소거/연결/권한 확인")
        else:
            print(f"  ✅ device {dev}: 캡처 OK (평균 RMS {mean_rms:.0f}, peak {peak:.0f})")
            working.append(dev)
        print()

    if working:
        print(f"  → 사용 가능: device {working}")
        print(f"  → 추천 실행: python3 run_all.py --role mic "
              f"--radar-host <cam_IP> --device {working[0]}")
    else:
        print("  ❌ 캡처되는 마이크 없음. 연결/기본장치/권한을 확인하세요.")
    return working


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="결재 레이더 카메라/마이크 사전 점검")
    ap.add_argument("cam_idx", nargs="?", type=int, default=0,
                    help="카메라 선호 인덱스 (기본 0)")
    ap.add_argument("--device", type=int, default=None, help="마이크 device 인덱스")
    ap.add_argument("--cam-only", action="store_true")
    ap.add_argument("--mic-only", action="store_true")
    ap.add_argument("--mic-secs", type=float, default=1.5, help="마이크 측정 시간(초)")
    args = ap.parse_args()

    print("=" * 58)
    print("  🔍  결재 레이더 — 장치 연결 사전 점검")
    print("=" * 58)

    cams, mics = [], []
    if not args.mic_only:
        cams = check_cameras(args.cam_idx)
    if not args.cam_only:
        mics = check_microphones(args.device, args.mic_secs)

    print("=" * 58)
    print("  요약")
    if not args.mic_only:
        print(f"    카메라: {'OK ' + str(cams) if cams else '없음 ❌'}")
    if not args.cam_only:
        print(f"    마이크: {'OK device ' + str(mics) if mics else '없음 ❌'}")
    print("=" * 58)

    # 종료 코드: 점검 대상이 하나라도 실패하면 1
    failed = ((not args.mic_only and not cams) or
              (not args.cam_only and not mics))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
