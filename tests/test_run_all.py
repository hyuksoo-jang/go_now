#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py 단위 테스트
────────────────────────────────────────────────────────────────
통합 런처. 실제 subprocess 기동/시그널/감시 루프는 mock 하고,
  · 인자 파싱 (_parse_args)
  · 서버 대기 (_wait_for_server)
  · 프로세스 시작/종료 (_start / _shutdown / _forward)
  · main() 부트스트랩 (check / mic 경로)
의 순수 로직만 검증한다.
"""
import sys, os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import run_all as ra


@pytest.fixture(autouse=True)
def reset_procs():
    """전역 _procs 리스트 초기화."""
    with ra._lock:
        ra._procs.clear()
    yield
    with ra._lock:
        ra._procs.clear()


# ═════════════════════════════════════════════════════════════
# A01: _parse_args
# ═════════════════════════════════════════════════════════════
class TestParseArgs:
    """A01: CLI 인자 파싱"""

    def test_A01_defaults(self):
        a = ra._parse_args([])
        assert a['role'] == 'all'
        assert a['cam_idx'] == '0'
        assert a['radar_host'] == 'localhost'
        assert a['device'] is None

    def test_A01_role_space_form(self):
        assert ra._parse_args(['--role', 'cam'])['role'] == 'cam'

    def test_A01_role_equals_form(self):
        assert ra._parse_args(['--role=mic'])['role'] == 'mic'

    def test_A01_unknown_role_falls_back_to_all(self):
        assert ra._parse_args(['--role', 'bogus'])['role'] == 'all'

    def test_A01_cam_idx_from_digit(self):
        assert ra._parse_args(['1'])['cam_idx'] == '1'

    def test_A01_radar_host_space_and_equals(self):
        assert ra._parse_args(['--radar-host', '10.0.0.5'])['radar_host'] == '10.0.0.5'
        assert ra._parse_args(['--radar-host=10.0.0.9'])['radar_host'] == '10.0.0.9'

    def test_A01_device(self):
        assert ra._parse_args(['--device', '2'])['device'] == '2'

    def test_A01_boolean_flags(self):
        a = ra._parse_args(['--no-calib', '--no-sigh', '--no-speech', '--vad-filter'])
        assert a['no_calib'] is True
        assert a['no_sigh'] is True
        assert a['no_speech'] is True
        assert a['vad_filter'] is True

    def test_A01_whisper_beam_compute_stt(self):
        a = ra._parse_args(['--whisper-size', 'small', '--beam-size=3',
                            '--compute-type', 'int8', '--stt-config=cfg.json'])
        assert a['whisper_size'] == 'small'
        assert a['beam_size'] == '3'
        assert a['compute_type'] == 'int8'
        assert a['stt_config'] == 'cfg.json'

    def test_A01_combined(self):
        a = ra._parse_args(['--role', 'mic', '--radar-host', 'h', '2'])
        assert a['role'] == 'mic'
        assert a['radar_host'] == 'h'
        assert a['cam_idx'] == '2'

    def test_A01_unknown_arg_ignored(self):
        """플래그/숫자가 아닌 인자는 무시 (148->113 분기)"""
        a = ra._parse_args(['--unknown-flag', 'garbage'])
        assert a['role'] == 'all'  # 기본값 유지

    def test_A01_remaining_equals_and_space_arms(self):
        """whisper= / beam(space) / compute= / stt(space) 분기 보강"""
        a = ra._parse_args(['--whisper-size=tiny', '--beam-size', '5',
                            '--compute-type=float16', '--stt-config', 'conf2.json'])
        assert a['whisper_size'] == 'tiny'
        assert a['beam_size'] == '5'
        assert a['compute_type'] == 'float16'
        assert a['stt_config'] == 'conf2.json'


# ═════════════════════════════════════════════════════════════
# A02: _wait_for_server
# ═════════════════════════════════════════════════════════════
class TestWaitForServer:
    """A02: Flask 응답 대기"""

    def test_A02_returns_true_on_success(self):
        with patch.object(ra.urllib.request, 'urlopen', return_value=MagicMock()):
            assert ra._wait_for_server('http://x', timeout=5) is True

    def test_A02_returns_false_on_timeout(self):
        # timeout=0 → deadline 즉시 경과 → urlopen 호출 없이 False
        assert ra._wait_for_server('http://x', timeout=0) is False

    def test_A02_retries_then_succeeds(self):
        calls = {'n': 0}
        def _flaky(url, timeout):
            calls['n'] += 1
            if calls['n'] < 2:
                raise OSError("not ready")
            return MagicMock()
        with patch.object(ra.urllib.request, 'urlopen', side_effect=_flaky), \
             patch.object(ra.time, 'sleep', lambda s: None):
            assert ra._wait_for_server('http://x', timeout=5) is True
        assert calls['n'] >= 2


# ═════════════════════════════════════════════════════════════
# A03: _start / _forward
# ═════════════════════════════════════════════════════════════
class TestStartAndForward:
    """A03: 프로세스 시작 및 출력 포워딩"""

    def test_A03_start_registers_proc(self):
        fake_proc = MagicMock()
        with patch.object(ra.subprocess, 'Popen', return_value=fake_proc) as popen, \
             patch.object(ra.threading, 'Thread') as thread:
            proc = ra._start(['python', 'x.py'], tag='cam')
        assert proc is fake_proc
        assert fake_proc in ra._procs
        popen.assert_called_once()
        thread.assert_called_once()  # _forward 스레드 시작

    def test_A03_forward_prints_lines(self, capsys):
        proc = MagicMock()
        proc.stdout = iter(['[cam] hello\n', 'world\n', '\n'])
        ra._forward(proc, 'cam')
        out = capsys.readouterr().out
        assert '[cam] hello' in out
        assert '[cam] world' in out  # tag 없는 줄엔 prefix 부착


# ═════════════════════════════════════════════════════════════
# A04: _shutdown
# ═════════════════════════════════════════════════════════════
class TestShutdown:
    """A04: 종료 처리"""

    def test_A04_terminates_and_kills(self):
        running = MagicMock()
        running.poll.return_value = None      # 아직 살아있음 → kill 대상
        stopped = MagicMock()
        stopped.poll.return_value = 0          # 이미 종료
        with ra._lock:
            ra._procs.extend([running, stopped])
        with patch.object(ra.time, 'sleep', lambda s: None), \
             pytest.raises(SystemExit):
            ra._shutdown()
        running.terminate.assert_called_once()
        running.kill.assert_called_once()
        stopped.kill.assert_not_called()

    def test_A04_swallows_terminate_kill_errors(self):
        """terminate/kill 이 예외를 던져도 종료 흐름 유지"""
        bad = MagicMock()
        bad.terminate.side_effect = OSError("x")
        bad.poll.return_value = None
        bad.kill.side_effect = OSError("y")
        with ra._lock:
            ra._procs.append(bad)
        with patch.object(ra.time, 'sleep', lambda s: None), \
             pytest.raises(SystemExit):
            ra._shutdown()
        bad.terminate.assert_called_once()
        bad.kill.assert_called_once()


# ═════════════════════════════════════════════════════════════
# A05: main
# ═════════════════════════════════════════════════════════════
class TestMain:
    """A05: main 부트스트랩"""

    def test_A05_check_role_delegates_and_exits(self, monkeypatch):
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        monkeypatch.setattr(ra.subprocess, 'call', lambda cmd, cwd: 0)
        monkeypatch.setattr(sys, 'argv', ['run_all.py', '--role', 'check'])
        with pytest.raises(SystemExit) as exc:
            ra.main()
        assert exc.value.code == 0

    def test_A05_check_role_missing_script(self, monkeypatch):
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        def _raise(cmd, cwd):
            raise FileNotFoundError()
        monkeypatch.setattr(ra.subprocess, 'call', _raise)
        monkeypatch.setattr(sys, 'argv', ['run_all.py', '--role', 'check'])
        with pytest.raises(SystemExit) as exc:
            ra.main()
        assert exc.value.code == 1

    def test_A05_mic_role_starts_then_monitors(self, monkeypatch):
        """mic 역할: 서버 대기 통과 후 mic_agent 시작, 감시 루프에서 종료 감지 후 반환"""
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        monkeypatch.setattr(ra, '_wait_for_server', lambda url, t: True)
        monkeypatch.setattr(ra.time, 'sleep', lambda s: None)

        proc = MagicMock()
        proc.poll.return_value = 0  # 즉시 종료된 것으로 → 감시 루프가 1회 만에 반환
        def _fake_start(cmd, tag, env=None):
            with ra._lock:
                ra._procs.append(proc)
            return proc
        monkeypatch.setattr(ra, '_start', _fake_start)
        monkeypatch.setattr(sys, 'argv', ['run_all.py', '--role', 'mic', '--device', '1'])

        ra.main()  # 블로킹 없이 반환되어야 함
        # 감시 루프가 종료된 proc 을 제거하고 빈 목록이 되면 반환
        with ra._lock:
            assert ra._procs == []

    def test_A05_check_role_with_device(self, monkeypatch):
        """check 역할 + --device → check_devices.py 에 device 전달"""
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        captured = {}
        monkeypatch.setattr(ra.subprocess, 'call',
                            lambda cmd, cwd: captured.update(cmd=cmd) or 0)
        monkeypatch.setattr(sys, 'argv',
                            ['run_all.py', '--role', 'check', '--device', '3'])
        with pytest.raises(SystemExit):
            ra.main()
        assert '--device' in captured['cmd']
        assert '3' in captured['cmd']

    def test_A05_all_role_builds_mic_cmd_with_flags(self, monkeypatch):
        """all 역할: cam+mic 시작, mic 커맨드에 모든 옵션 플래그 포함"""
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        monkeypatch.setattr(ra, '_wait_for_server', lambda url, t: True)
        monkeypatch.setattr(ra.time, 'sleep', lambda s: None)

        starts = []
        def _fake_start(cmd, tag, env=None):
            proc = MagicMock()
            proc.poll.return_value = 0  # 즉시 종료 → 감시 루프 1회 후 반환
            starts.append((tag, cmd))
            with ra._lock:
                ra._procs.append(proc)
            return proc
        monkeypatch.setattr(ra, '_start', _fake_start)
        monkeypatch.setattr(sys, 'argv', [
            'run_all.py', '--role', 'all', '1', '--device', '2',
            '--no-calib', '--no-sigh', '--no-speech',
            '--whisper-size', 'small', '--beam-size', '3',
            '--vad-filter', '--compute-type', 'int8', '--stt-config', 'c.json',
        ])
        ra.main()

        tags = [t for t, _ in starts]
        assert 'cam' in tags and 'mic' in tags
        mic_cmd = next(cmd for t, cmd in starts if t == 'mic')
        for flag in ('--no-calibrate', '--no-sigh', '--no-speech',
                     '--whisper-size', '--beam-size', '--vad-filter',
                     '--compute-type', '--stt-config'):
            assert flag in mic_cmd

    def test_A05_no_procs_returns_early(self, monkeypatch):
        """_start 가 프로세스를 등록하지 못하면 '실행할 프로세스 없음' 으로 반환"""
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        monkeypatch.setattr(ra, '_start', lambda cmd, tag, env=None: MagicMock())  # append 안 함
        monkeypatch.setattr(sys, 'argv', ['run_all.py', '--role', 'cam'])
        ra.main()  # 예외 없이 반환
        with ra._lock:
            assert ra._procs == []

    def test_A05_server_not_ready_logs_warning(self, monkeypatch):
        """_wait_for_server False → 경고 로그 후 mic 계속 진행 (206)"""
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        monkeypatch.setattr(ra, '_wait_for_server', lambda url, t: False)
        monkeypatch.setattr(ra.time, 'sleep', lambda s: None)
        proc = MagicMock()
        proc.poll.return_value = 0
        def _fake_start(cmd, tag, env=None):
            with ra._lock:
                ra._procs.append(proc)
            return proc
        monkeypatch.setattr(ra, '_start', _fake_start)
        monkeypatch.setattr(sys, 'argv', ['run_all.py', '--role', 'mic'])
        ra.main()
        with ra._lock:
            assert ra._procs == []

    def test_A05_monitor_keeps_alive_proc(self, monkeypatch):
        """감시 루프: 살아있는 proc 은 유지, 죽은 proc 은 제거 (248->246, 252->243)"""
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        monkeypatch.setattr(ra, '_wait_for_server', lambda url, t: True)

        alive = MagicMock(); alive.poll.return_value = None   # 계속 살아있음
        dead = MagicMock(); dead.poll.return_value = 0        # 종료됨

        def _fake_start(cmd, tag, env=None):
            with ra._lock:
                ra._procs.extend([alive, dead])
            return alive
        monkeypatch.setattr(ra, '_start', _fake_start)

        # 첫 sleep 통과, 두 번째 sleep 에서 KeyboardInterrupt 로 탈출
        state = {'n': 0}
        def _sleep(s):
            state['n'] += 1
            if state['n'] >= 2:
                raise KeyboardInterrupt()
        monkeypatch.setattr(ra.time, 'sleep', _sleep)
        monkeypatch.setattr(ra, '_shutdown', lambda *a: None)
        monkeypatch.setattr(sys, 'argv', ['run_all.py', '--role', 'mic'])

        ra.main()
        # 죽은 proc 은 제거되고 살아있는 proc 만 남음
        with ra._lock:
            assert dead not in ra._procs
            assert alive in ra._procs

    def test_A05_keyboard_interrupt_triggers_shutdown(self, monkeypatch):
        """감시 루프 중 KeyboardInterrupt → _shutdown 호출"""
        monkeypatch.setattr(ra.signal, 'signal', lambda *a: None)
        monkeypatch.setattr(ra, '_wait_for_server', lambda url, t: True)

        proc = MagicMock()
        proc.poll.return_value = None  # 계속 살아있음 → sleep 까지 도달
        def _fake_start(cmd, tag, env=None):
            with ra._lock:
                ra._procs.append(proc)
            return proc
        monkeypatch.setattr(ra, '_start', _fake_start)

        def _boom(s):
            raise KeyboardInterrupt()
        monkeypatch.setattr(ra.time, 'sleep', _boom)

        shutdown_called = {'n': 0}
        monkeypatch.setattr(ra, '_shutdown',
                            lambda *a: shutdown_called.__setitem__('n', 1))
        monkeypatch.setattr(sys, 'argv', ['run_all.py', '--role', 'mic'])
        ra.main()
        assert shutdown_called['n'] == 1
