#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — 결재 레이더 통합 런처 (cam 기기 / mic 기기 분산)
────────────────────────────────────────────────────────────────
구성:
  · cam 기기  : go_now_v2.py (Flask + 카메라 표정) + radar_signal_processor.py
                → 대시보드 서빙, 한숨·발화 POST 수신, 신호등 판정
  · mic 기기  : mic_agent.py (단일 마이크 → 한숨 + 발화 fan-out)
                → cam 기기로 POST 전송

실행:
  # ── cam 기기 ──
  python3 run_all.py --role cam            # 카메라 0
  python3 run_all.py --role cam 1          # 카메라 1

  # ── mic 기기 (cam 기기 IP 지정) ──
  python3 run_all.py --role mic --radar-host 192.168.0.10
  python3 run_all.py --role mic --radar-host 192.168.0.10 --no-calib

  # ── 단일 기기에서 전부 (cam+mic 같은 PC, 테스트용) ──
  python3 run_all.py --role all

  # ── 사전 점검 (본 실행 전 카메라/마이크 연결 확인) ──
  python3 run_all.py --role check              # 카메라 + 마이크
  python3 run_all.py --role check 1            # 카메라 선호 인덱스 1
  python3 run_all.py --role check --device 1   # 마이크 device 1

옵션:
  --role {cam|mic|all|check}   실행 묶음 (기본 all)
  --radar-host HOST         cam 기기 주소 (mic 기기에서 지정, 기본 localhost)
  --device N                mic 입력 장치 인덱스
  --no-calib                발화 캘리브레이션 생략
  --no-sigh / --no-speech   mic 파이프라인 일부 비활성

출력 prefix: [cam] [mic] [run]
────────────────────────────────────────────────────────────────
"""

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable
RADAR_PORT = 5000

_procs = []
_lock  = threading.Lock()


def _log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def _forward(proc, tag):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(line if line.startswith(f"[{tag}]") else f"[{tag}] {line}", flush=True)


def _start(cmd, tag, env=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=BASE_DIR, bufsize=1, env=env)
    with _lock:
        _procs.append(proc)
    threading.Thread(target=_forward, args=(proc, tag), daemon=True).start()
    return proc


def _wait_for_server(url, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _shutdown(sig=None, frame=None):
    _log("run", "종료 신호 — 모든 프로세스 정지...")
    with _lock:
        for p in _procs:
            try:
                p.terminate()
            except Exception:
                pass
    time.sleep(1.5)
    with _lock:
        for p in _procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
    _log("run", "완료")
    sys.exit(0)


def _parse_args(argv):
    a = {"cam_idx": "0", "role": "all", "radar_host": "localhost",
         "device": None, "no_calib": False, "no_sigh": False, "no_speech": False}
    it = iter(argv)
    for arg in it:
        if arg == "--role":
            a["role"] = next(it, "all")
        elif arg.startswith("--role="):
            a["role"] = arg.split("=", 1)[1]
        elif arg == "--radar-host":
            a["radar_host"] = next(it, "localhost")
        elif arg.startswith("--radar-host="):
            a["radar_host"] = arg.split("=", 1)[1]
        elif arg == "--device":
            a["device"] = next(it, None)
        elif arg == "--no-calib":
            a["no_calib"] = True
        elif arg == "--no-sigh":
            a["no_sigh"] = True
        elif arg == "--no-speech":
            a["no_speech"] = True
        elif arg.isdigit():
            a["cam_idx"] = arg
    if a["role"] not in ("cam", "mic", "all", "check"):
        _log("run", f"⚠ 알 수 없는 role '{a['role']}' → 'all'")
        a["role"] = "all"
    return a


def main():
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    a = _parse_args(sys.argv[1:])
    role = a["role"]
    radar_url = f"http://{a['radar_host']}:{RADAR_PORT}"
    mic_env = {**os.environ, "RADAR_URL": radar_url}

    # ── 사전 점검 모드: check_devices.py 로 위임하고 종료 ──
    if role == "check":
        cmd = [PYTHON, "check_devices.py", a["cam_idx"]]
        if a["device"] is not None:
            cmd += ["--device", str(a["device"])]
        _log("run", "장치 사전 점검 시작 (check_devices.py)...")
        _log("run", "(카메라만/마이크만 점검하려면 check_devices.py --cam-only / --mic-only 직접 실행)")
        try:
            rc = subprocess.call(cmd, cwd=BASE_DIR)
        except FileNotFoundError:
            _log("run", "❌ check_devices.py 를 찾을 수 없습니다 (같은 폴더에 두세요)")
            rc = 1
        sys.exit(rc)

    _log("run", "=" * 50)
    _log("run", "결재 레이더 통합 런처")
    _log("run", f"  role       : {role}")
    _log("run", f"  radar URL  : {radar_url}")
    if role in ("cam", "all"):
        _log("run", f"  카메라     : index {a['cam_idx']}")
    if role in ("mic", "all"):
        _log("run", f"  마이크     : device {a['device'] or 'default'} · "
                    f"calib {'off' if a['no_calib'] else 'on'} · "
                    f"sigh {'off' if a['no_sigh'] else 'on'} · "
                    f"speech {'off' if a['no_speech'] else 'on'}")
    _log("run", "=" * 50)

    # ── cam ──
    if role in ("cam", "all"):
        _log("run", "go_now_v2.py 시작...")
        _start([PYTHON, "go_now_v2.py", a["cam_idx"]], tag="cam")

    # ── 서버 준비 확인 (mic/all 은 에이전트 전에) ──
    if role == "cam":
        _log("run", "(cam 단독 — mic_agent 는 다른 기기에서 실행하세요)")
    else:
        target = radar_url if role == "mic" else f"http://localhost:{RADAR_PORT}"
        _log("run", f"Flask 응답 대기: {target} (최대 60초)...")
        if _wait_for_server(target, 60):
            _log("run", f"✅ Flask 준비 완료 ({target})")
        else:
            _log("run", "⚠ Flask 응답 없음 — mic_agent 는 전송 재시도합니다")

    # ── mic ──
    if role in ("mic", "all"):
        cmd = [PYTHON, "mic_agent.py"]
        if a["device"] is not None:
            cmd += ["--device", str(a["device"])]
        if a["no_calib"]:
            cmd.append("--no-calibrate")
        if a["no_sigh"]:
            cmd.append("--no-sigh")
        if a["no_speech"]:
            cmd.append("--no-speech")
        _log("run", "mic_agent.py 시작...")
        _start(cmd, tag="mic", env=mic_env)

    if not _procs:
        _log("run", "실행할 프로세스가 없습니다. --role 확인.")
        return

    _log("run", "")
    _log("run", "전체 실행 중 (Ctrl+C 종료)")
    if role in ("cam", "all"):
        _log("run", f"대시보드: {radar_url}/")
    _log("run", "")

    try:
        while True:
            time.sleep(5.0)
            with _lock:
                for p in list(_procs):
                    rc = p.poll()
                    if rc is not None:
                        _log("run", f"⚠ PID {p.pid} 종료됨 (코드 {rc})")
                        _procs.remove(p)
            with _lock:
                if not _procs:
                    _log("run", "모든 프로세스 종료 — 런처 종료")
                    return
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
