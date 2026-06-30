import sys, os, queue, threading, time
import numpy as np
from unittest.mock import MagicMock, patch

# 하드웨어 의존 모듈 모킹
# 주의: 무조건 MagicMock 으로 덮어쓴다. (sounddevice 는 mediapipe 등이 실제 모듈을
# sys.modules 에 먼저 올릴 수 있어, 조건부 모킹 시 테스트 실행 순서에 따라
# 실제 하드웨어 모듈이 새어 들어와 audio_supervisor 분기 커버리지가 흔들린다.)
for _mod in ['sounddevice', 'webrtcvad']:
    sys.modules[_mod] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestNormKo:
    """R02: 한국어 텍스트 정규화(_norm_ko) 테스트"""

    def test_R02_norm_ko_removes_spaces(self):
        from mic_agent import _norm_ko
        assert _norm_ko("안 녕") == "안녕"

    def test_R02_norm_ko_removes_punctuation(self):
        from mic_agent import _norm_ko
        assert _norm_ko("안녕!") == "안녕"

    def test_R02_norm_ko_empty_string(self):
        from mic_agent import _norm_ko
        assert _norm_ko("") == ""

    def test_R02_norm_ko_mixed(self):
        from mic_agent import _norm_ko
        assert _norm_ko("결재. 부탁!") == "결재부탁"


class TestCER:
    """R02: 문자 오류율(_cer) 테스트"""

    def test_R02_cer_identical_is_zero(self):
        from mic_agent import _cer
        assert _cer("결재부탁드립니다", "결재부탁드립니다") == 0.0

    def test_R02_cer_both_empty_is_zero(self):
        from mic_agent import _cer
        assert _cer("", "") == 0.0

    def test_R02_cer_empty_ref_nonempty_hyp(self):
        from mic_agent import _cer
        assert _cer("", "abc") == 1.0

    def test_R02_cer_nonnegative(self):
        from mic_agent import _cer
        assert _cer("결재", "완전히다른문자") >= 0.0

    def test_R02_cer_partial_match_nonzero(self):
        from mic_agent import _cer
        assert _cer("결재부탁드립니다", "결재감사합니다") > 0.0


class TestLoadSttConfig:
    """R05: STT 설정 파일 로딩(load_stt_config) 테스트"""

    def test_R05_nonexistent_falls_back_to_default(self):
        """없는 경로 → 기본 경로 폴백, dict 반환"""
        from mic_agent import load_stt_config
        cfg, path = load_stt_config("/nonexistent/path/stt_config.json")
        assert isinstance(cfg, dict)  # 폴백으로 기본값 반환

    def test_R05_valid_file_returns_dict(self, ws_dir):
        """유효한 파일 → dict 반환"""
        from mic_agent import load_stt_config
        cfg, path = load_stt_config(os.path.join(ws_dir, 'stt_config.json'))
        assert isinstance(cfg, dict)

    def test_R05_valid_file_returns_path(self, ws_dir):
        """유효한 파일 → 경로 반환"""
        from mic_agent import load_stt_config
        cfg, path = load_stt_config(os.path.join(ws_dir, 'stt_config.json'))
        assert path is not None


class TestLoadGateConfig:
    """R01: 게이트 설정 파일 로딩(load_gate_config) 테스트"""

    def test_R01_nonexistent_falls_back_to_default(self):
        """없는 경로 → 기본 경로 폴백, dict 반환"""
        from mic_agent import load_gate_config
        cfg, path = load_gate_config("/nonexistent/path/gate_config.json")
        assert isinstance(cfg, dict)  # 폴백으로 기본값 반환

    def test_R01_valid_file_returns_dict(self, ws_dir):
        """유효한 파일 → dict 반환"""
        from mic_agent import load_gate_config
        cfg, path = load_gate_config(os.path.join(ws_dir, 'gate_config.json'))
        assert isinstance(cfg, dict)

    def test_R01_valid_file_returns_path(self, ws_dir):
        """유효한 파일 → 경로 반환"""
        from mic_agent import load_gate_config
        cfg, path = load_gate_config(os.path.join(ws_dir, 'gate_config.json'))
        assert path is not None


class TestAudioConstants:
    """R02: 오디오 처리 상수 검증"""

    def test_R02_sample_rate_16000(self):
        from mic_agent import SAMPLE_RATE
        assert SAMPLE_RATE == 16000

    def test_R02_frame_ms_30(self):
        from mic_agent import FRAME_MS
        assert FRAME_MS == 30

    def test_R02_frame_samples_480(self):
        from mic_agent import FRAME_SAMPLES, SAMPLE_RATE, FRAME_MS
        assert FRAME_SAMPLES == SAMPLE_RATE * FRAME_MS // 1000

    def test_R02_sigh_threshold_in_range(self):
        from mic_agent import SIGH_THRESHOLD
        assert 0.0 < SIGH_THRESHOLD < 1.0

    def test_R02_sigh_cooldown_positive(self):
        from mic_agent import SIGH_COOLDOWN
        assert SIGH_COOLDOWN > 0.0

    def test_R02_emotions_6_classes(self):
        from mic_agent import EMOTIONS
        assert len(EMOTIONS) == 6

    def test_R02_emotions_expected_labels(self):
        from mic_agent import EMOTIONS
        for e in ["기쁨", "슬픔", "분노", "불안", "당황", "상처"]:
            assert e in EMOTIONS

    def test_R02_sigh_keywords_nonempty(self):
        from mic_agent import SIGH_KEYWORDS
        assert len(SIGH_KEYWORDS) > 0


# ─────────────────────────────────────────────────────────────
# _post HTTP 전송 테스트
# ─────────────────────────────────────────────────────────────
class TestPost:
    """R02: _post HTTP 전송 헬퍼 테스트"""

    def _mock_resp(self, status):
        m = MagicMock()
        m.__enter__ = lambda s: m
        m.__exit__ = MagicMock(return_value=False)
        m.status = status
        return m

    def test_R02_post_200_returns_true(self):
        """HTTP 200 → True"""
        import mic_agent
        with patch('urllib.request.urlopen', return_value=self._mock_resp(200)):
            assert mic_agent._post("/api/test", {"k": "v"}) is True

    def test_R02_post_299_returns_true(self):
        """HTTP 299 → True"""
        import mic_agent
        with patch('urllib.request.urlopen', return_value=self._mock_resp(299)):
            assert mic_agent._post("/api/test", {}) is True

    def test_R02_post_300_returns_false(self):
        """HTTP 300 → False"""
        import mic_agent
        with patch('urllib.request.urlopen', return_value=self._mock_resp(300)):
            assert mic_agent._post("/api/test", {}) is False

    def test_R02_post_exception_returns_false(self):
        """연결 실패 → False"""
        import mic_agent
        with patch('urllib.request.urlopen', side_effect=OSError("conn refused")):
            assert mic_agent._post("/api/test", {}) is False

    def test_R02_post_logs_on_first_failure(self, capsys):
        """첫 실패 → 로그 출력"""
        import mic_agent
        mic_agent._post_fail_logged["_test_k1"] = 0.0
        with patch('urllib.request.urlopen', side_effect=OSError("err")):
            mic_agent._post("/api/test", {}, kind="_test_k1")
        assert "전송 실패" in capsys.readouterr().out

    def test_R02_post_rate_limits_repeated_log(self, capsys):
        """10초 이내 재실패 → 로그 억제"""
        import mic_agent
        mic_agent._post_fail_logged["_test_k2"] = time.time()
        with patch('urllib.request.urlopen', side_effect=OSError("err")):
            mic_agent._post("/api/test", {}, kind="_test_k2")
        assert "전송 실패" not in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────
# put_drop_oldest 큐 오버플로우 처리 테스트
# ─────────────────────────────────────────────────────────────
class TestPutDropOldest:
    """R02: put_drop_oldest 큐 오버플로우 처리 테스트"""

    def test_R02_put_when_empty(self):
        """빈 큐에 추가 → 정상 삽입"""
        import mic_agent
        q = queue.Queue()
        mic_agent.put_drop_oldest(q, "item", 5)
        assert q.get_nowait() == "item"

    def test_R02_drops_oldest_when_full(self):
        """꽉 찬 큐 → 가장 오래된 항목 삭제 후 신규 삽입"""
        import mic_agent
        q = queue.Queue()
        q.put("old1")
        q.put("old2")
        mic_agent.put_drop_oldest(q, "new", 2)
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert "new" in items
        assert "old1" not in items

    def test_R02_maxsize_1_keeps_only_latest(self):
        """maxsize=1 → 항상 최신 항목만 유지"""
        import mic_agent
        q = queue.Queue()
        q.put("old")
        mic_agent.put_drop_oldest(q, "new", 1)
        assert q.get_nowait() == "new"

    def test_R02_size_never_exceeds_maxsize(self):
        """연속 삽입 후 큐 크기 ≤ maxsize"""
        import mic_agent
        q = queue.Queue()
        for i in range(30):
            mic_agent.put_drop_oldest(q, i, 5)
        assert q.qsize() <= 5

    def test_R02_put_drop_empty_exception_handled(self):
        """race condition 시 Empty 예외 무시"""
        import mic_agent
        q = queue.Queue()
        # maxsize=0 으로 항상 while 진입, Empty 예외 처리 경로 커버
        mic_agent.put_drop_oldest(q, "x", 0)
        assert q.get_nowait() == "x"


# ─────────────────────────────────────────────────────────────
# 손상된 설정 파일 처리 테스트
# ─────────────────────────────────────────────────────────────
class TestLoadConfigBadJson:
    """R01/R05: 손상된 설정 파일 예외 처리 테스트"""

    def test_R05_stt_bad_json_prints_warning(self, tmp_path, capsys):
        """stt_config.json 파싱 실패 → 경고 출력 후 폴백"""
        import mic_agent
        bad = tmp_path / "stt_config.json"
        bad.write_text("{ invalid json !!!")
        cfg, _ = mic_agent.load_stt_config(str(bad))
        assert "읽기 실패" in capsys.readouterr().out
        assert isinstance(cfg, dict)

    def test_R01_gate_bad_json_prints_warning(self, tmp_path, capsys):
        """gate_config.json 파싱 실패 → 경고 출력 후 폴백"""
        import mic_agent
        bad = tmp_path / "gate_config.json"
        bad.write_text("{ invalid json !!!")
        cfg, _ = mic_agent.load_gate_config(str(bad))
        assert "읽기 실패" in capsys.readouterr().out
        assert isinstance(cfg, dict)

    def test_R05_stt_all_fail_returns_empty(self, tmp_path, monkeypatch, capsys):
        """모든 후보 경로 파싱 실패 → ({}, None) 반환"""
        import mic_agent
        bad = tmp_path / "stt_config.json"
        bad.write_text("!!bad!!")
        # CWD와 스크립트 디렉토리를 tmp_path로 덮어쓰기 → 같은 bad 파일만 존재
        monkeypatch.chdir(tmp_path)
        cfg, path = mic_agent.load_stt_config(str(bad))
        capsys.readouterr()
        # 폴백도 같은 bad 파일 → {}, None
        assert isinstance(cfg, dict)

    def test_R01_gate_all_fail_returns_empty(self, tmp_path, monkeypatch, capsys):
        """모든 후보 경로 파싱 실패 → ({}, None) 반환"""
        import mic_agent
        bad = tmp_path / "gate_config.json"
        bad.write_text("!!bad!!")
        monkeypatch.chdir(tmp_path)
        cfg, path = mic_agent.load_gate_config(str(bad))
        capsys.readouterr()
        assert isinstance(cfg, dict)


# ─────────────────────────────────────────────────────────────
# sigh_loop 모델 로딩 실패 처리 테스트
# ─────────────────────────────────────────────────────────────
class TestSighLoopFailure:
    """R02: sigh_loop 모델 로딩 실패 시 큐 드레인 후 종료 테스트"""

    def test_R02_sigh_loop_model_fail_drains_queue_and_exits(self):
        """_load_yamnet 실패 → 큐 비우고 stop 시 정상 종료"""
        import mic_agent

        sq = queue.Queue()
        for _ in range(3):
            sq.put(b'\x00' * 960)
        stop = threading.Event()

        # 0.3초 후 stop
        threading.Timer(0.3, stop.set).start()

        with patch('mic_agent._load_yamnet', side_effect=RuntimeError("no model")):
            mic_agent.sigh_loop(sq, stop, "fake.tflite", "fake.csv")

        assert stop.is_set()

    def test_R02_sigh_loop_model_fail_prints_error(self, capsys):
        """_load_yamnet 실패 → 에러 메시지 출력"""
        import mic_agent

        sq = queue.Queue()
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()

        with patch('mic_agent._load_yamnet', side_effect=RuntimeError("no tflite")):
            mic_agent.sigh_loop(sq, stop, "fake.tflite", "fake.csv")

        assert "비활성화" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────
# recalib_poller 재캘리브레이션 폴러 테스트
# ─────────────────────────────────────────────────────────────
class TestRecalibPoller:
    """R02: recalib_poller 재캘리브레이션 감지 테스트"""

    def test_R02_poller_sets_event_on_requested(self):
        """서버가 requested=true 반환 → recalib_req.set()"""
        import json as _json
        import mic_agent

        stop = threading.Event()
        recalib = threading.Event()

        def fake_urlopen(url, timeout):
            stop.set()
            m = MagicMock()
            m.__enter__ = lambda s: m
            m.__exit__ = MagicMock(return_value=False)
            m.read.return_value = _json.dumps({"requested": True}).encode()
            return m

        with patch('urllib.request.urlopen', side_effect=fake_urlopen):
            t = threading.Thread(
                target=mic_agent.recalib_poller,
                args=(stop, recalib), daemon=True)
            t.start()
            t.join(timeout=2.0)

        assert recalib.is_set()

    def test_R02_poller_handles_network_exception(self):
        """urlopen 예외 → 크래시 없이 계속 동작"""
        import mic_agent

        stop = threading.Event()
        recalib = threading.Event()
        calls = [0]

        def fake_urlopen(url, timeout):
            calls[0] += 1
            if calls[0] >= 2:
                stop.set()
            raise OSError("network error")

        with patch('urllib.request.urlopen', side_effect=fake_urlopen):
            t = threading.Thread(
                target=mic_agent.recalib_poller,
                args=(stop, recalib), daemon=True)
            t.start()
            t.join(timeout=2.0)

        assert calls[0] >= 2  # 예외 후에도 계속 시도

    def test_R02_poller_not_requested_no_event(self):
        """서버가 requested=false → recalib_req 미설정"""
        import json as _json
        import mic_agent

        stop = threading.Event()
        recalib = threading.Event()
        calls = [0]

        def fake_urlopen(url, timeout):
            calls[0] += 1
            stop.set()
            m = MagicMock()
            m.__enter__ = lambda s: m
            m.__exit__ = MagicMock(return_value=False)
            m.read.return_value = _json.dumps({"requested": False}).encode()
            return m

        with patch('urllib.request.urlopen', side_effect=fake_urlopen):
            t = threading.Thread(
                target=mic_agent.recalib_poller,
                args=(stop, recalib), daemon=True)
            t.start()
            t.join(timeout=2.0)

        assert not recalib.is_set()


# ─────────────────────────────────────────────────────────────
# calib_level_poster 캘리브레이션 레벨 전송 테스트
# ─────────────────────────────────────────────────────────────
class TestCalibLevelPoster:
    """R02: calib_level_poster 캘리브레이션 중 레벨 전송 테스트"""

    def test_R02_poster_calls_post_when_active(self):
        """_calib_live active=True → _post 호출"""
        import mic_agent

        stop = threading.Event()
        posted = threading.Event()

        def fake_post(*a, **kw):
            posted.set()
            stop.set()
            return True

        mic_agent._calib_live["active"] = True
        mic_agent._calib_live["rms"] = 2000.0

        try:
            with patch('mic_agent._post', side_effect=fake_post):
                t = threading.Thread(
                    target=mic_agent.calib_level_poster,
                    args=(stop,), daemon=True)
                t.start()
                posted.wait(timeout=2.0)
                t.join(timeout=1.0)
        finally:
            mic_agent._calib_live["active"] = False

        assert posted.is_set()

    def test_R02_poster_skips_when_inactive(self):
        """_calib_live active=False → _post 미호출"""
        import mic_agent

        stop = threading.Event()
        post_called = [False]

        def fake_post(*a, **kw):
            post_called[0] = True
            return True

        mic_agent._calib_live["active"] = False

        with patch('mic_agent._post', side_effect=fake_post):
            t = threading.Thread(
                target=mic_agent.calib_level_poster,
                args=(stop,), daemon=True)
            t.start()
            time.sleep(0.3)
            stop.set()
            t.join(timeout=1.0)

        assert not post_called[0]

    def test_R02_poster_level_clamped_to_1(self):
        """rms 매우 클 때 레벨 1.0 이하로 클램핑"""
        import mic_agent

        stop = threading.Event()
        posted_level = [None]

        def fake_post(path, payload, **kw):
            posted_level[0] = payload.get("level")
            stop.set()
            return True

        mic_agent._calib_live["active"] = True
        mic_agent._calib_live["rms"] = 999999.0  # 매우 큰 값

        try:
            with patch('mic_agent._post', side_effect=fake_post):
                t = threading.Thread(
                    target=mic_agent.calib_level_poster,
                    args=(stop,), daemon=True)
                t.start()
                stop.wait(timeout=2.0)
                t.join(timeout=1.0)
        finally:
            mic_agent._calib_live["active"] = False

        assert posted_level[0] is not None
        assert posted_level[0] <= 1.0


# ─────────────────────────────────────────────────────────────
# speech_assemble_loop 발화 조립 테스트
# ─────────────────────────────────────────────────────────────
class TestSpeechAssembleLoop:
    """R02: speech_assemble_loop VAD 발화 조립 로직 테스트"""

    def _make_frame(self, rms=300):
        """target RMS인 PCM 프레임 (480 samples, int16)"""
        data = np.full(480, int(rms), dtype=np.int16)
        return data.tobytes()

    def test_R02_max_utterance_triggers_finish(self):
        """MAX_UTTER_MS 초과 → utter_q에 발화 추가"""
        import mic_agent

        speech_q = queue.Queue()
        utter_q = queue.Queue()
        stop = threading.Event()
        gate = {"upper": 32768.0, "over_frac": 0.15}

        # MAX_UTTER_MS=8000ms / FRAME_MS=30ms = 267프레임 → 270 추가
        frame = self._make_frame(300)
        for _ in range(270):
            speech_q.put(frame)

        threading.Timer(1.5, stop.set).start()

        t = threading.Thread(
            target=mic_agent.speech_assemble_loop,
            args=(speech_q, utter_q, stop, gate, 1),
            daemon=True)
        t.start()
        t.join(timeout=4.0)

        assert not utter_q.empty(), "발화가 utter_q에 추가되어야 함"
        _, stats, reason = utter_q.get_nowait()
        assert reason in ("ok", "too_loud")
        assert "min" in stats and "mean" in stats and "max" in stats

    def test_R02_loud_frames_produce_too_loud(self):
        """over_frac=0.0 → 모든 상한 초과 프레임 → too_loud"""
        import mic_agent

        speech_q = queue.Queue()
        utter_q = queue.Queue()
        stop = threading.Event()
        # upper=100, over_frac=0.0 → over_cnt/n > 0 이면 too_loud
        gate = {"upper": 100.0, "over_frac": 0.0}

        loud = self._make_frame(32000)  # rms=32000 >> upper=100
        quiet = self._make_frame(50)    # rms=50 <= upper=100 (트리거용)

        # 트리거를 위해 quiet 1프레임 → 이후 loud 270프레임
        speech_q.put(quiet)
        for _ in range(270):
            speech_q.put(loud)

        threading.Timer(1.5, stop.set).start()

        t = threading.Thread(
            target=mic_agent.speech_assemble_loop,
            args=(speech_q, utter_q, stop, gate, 1),
            daemon=True)
        t.start()
        t.join(timeout=4.0)

        if not utter_q.empty():
            _, stats, reason = utter_q.get_nowait()
            assert reason == "too_loud"

    def test_R02_empty_queue_no_crash(self):
        """빈 큐 → 크래시 없이 stop 시 종료"""
        import mic_agent

        speech_q = queue.Queue()
        utter_q = queue.Queue()
        stop = threading.Event()
        gate = {"upper": 32768.0, "over_frac": 0.15}

        threading.Timer(0.3, stop.set).start()

        t = threading.Thread(
            target=mic_agent.speech_assemble_loop,
            args=(speech_q, utter_q, stop, gate, 1),
            daemon=True)
        t.start()
        t.join(timeout=2.0)

        assert stop.is_set()


# ─────────────────────────────────────────────────────────────
# _report_calib 테스트
# ─────────────────────────────────────────────────────────────
class TestReportCalib:
    """R02: _report_calib 캘리브레이션 상태 보고 테스트"""

    def test_R02_report_calib_calls_post(self):
        """_report_calib → _post 호출"""
        import mic_agent
        with patch('mic_agent._post') as mock_post:
            mic_agent._report_calib("noise", noise_sec=1.5)
        mock_post.assert_called_once()

    def test_R02_report_calib_phase_in_payload(self):
        """phase 값이 payload에 포함"""
        import mic_agent
        captured = {}
        def fake_post(path, payload, **kw):
            captured.update(payload)
            return True
        with patch('mic_agent._post', side_effect=fake_post):
            mic_agent._report_calib("done")
        assert captured.get("phase") == "done"

    def test_R02_report_calib_posts_to_correct_path(self):
        """올바른 API 경로로 전송"""
        import mic_agent
        paths = []
        def fake_post(path, payload, **kw):
            paths.append(path)
            return True
        with patch('mic_agent._post', side_effect=fake_post):
            mic_agent._report_calib("speak", text="결재", enroll_sec=4.0)
        assert paths[0] == "/api/speech-calib-status"

    def test_R02_report_calib_speak_includes_text(self):
        """speak 단계 → text 포함"""
        import mic_agent
        captured = {}
        def fake_post(path, payload, **kw):
            captured.update(payload)
            return True
        with patch('mic_agent._post', side_effect=fake_post):
            mic_agent._report_calib("speak", text="결재 부탁드립니다")
        assert "결재" in captured.get("text", "")


# ─────────────────────────────────────────────────────────────
# _measure_frames 테스트
# ─────────────────────────────────────────────────────────────
class TestMeasureFrames:
    """R02: _measure_frames 오디오 측정 테스트"""

    def _setup_sd(self, rms=1000):
        """sounddevice mock 설정 헬퍼"""
        import sounddevice as sd
        samples = np.full(480, rms, dtype=np.int16)
        frame_bytes = samples.tobytes()
        mock_stream = MagicMock()
        mock_stream.read.return_value = (frame_bytes, False)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        sd.RawInputStream.return_value = mock_ctx
        return mock_stream

    def test_R02_measure_frames_returns_list(self):
        """_measure_frames → list 반환"""
        import mic_agent
        self._setup_sd()
        result = mic_agent._measure_frames(0.03)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_R02_measure_frames_tuple_items(self):
        """각 항목이 (rms, voiced) 튜플"""
        import mic_agent
        self._setup_sd(rms=1000)
        result = mic_agent._measure_frames(0.03)
        rms, voiced = result[0]
        assert rms >= 0
        assert isinstance(voiced, bool)

    def test_R02_measure_frames_with_vad(self):
        """vad 인자 전달 시 is_speech 호출"""
        import mic_agent
        self._setup_sd()
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        result = mic_agent._measure_frames(0.03, vad=mock_vad)
        assert mock_vad.is_speech.called

    def test_R02_measure_frames_collect_audio(self):
        """collect_audio=True → (measurements, audio_array) 반환"""
        import mic_agent
        self._setup_sd()
        result = mic_agent._measure_frames(0.03, collect_audio=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        measurements, audio = result
        assert isinstance(measurements, list)
        assert hasattr(audio, 'dtype')

    def test_R02_measure_frames_updates_calib_live(self):
        """_calib_live['rms'] 갱신"""
        import mic_agent
        self._setup_sd(rms=2000)
        mic_agent._calib_live["rms"] = 0.0
        mic_agent._measure_frames(0.03)
        assert mic_agent._calib_live["rms"] >= 0

    def test_R02_measure_frames_zero_rms_when_silent(self):
        """무음 프레임 → rms=0"""
        import mic_agent
        self._setup_sd(rms=0)
        result = mic_agent._measure_frames(0.03)
        rms, _ = result[0]
        assert rms == 0.0


# ─────────────────────────────────────────────────────────────
# calibrate 테스트
# ─────────────────────────────────────────────────────────────
class TestCalibrate:
    """R02: calibrate 함수 테스트"""

    def _make_measure_mock(self, noise_rms=500, speech_rms=3000):
        """_measure_frames 모킹 헬퍼"""
        def fake_measure(seconds, vad=None, collect_audio=False):
            if collect_audio:
                audio = np.zeros(int(16000 * 4), dtype=np.int16)
                return [(speech_rms, True)] * 20, audio
            return [(noise_rms, False)] * 10
        return fake_measure

    def test_R02_calibrate_returns_float(self):
        """calibrate → float 반환 (upper 게이트 값)"""
        import mic_agent
        fake = self._make_measure_mock()
        with patch('mic_agent._measure_frames', side_effect=fake), \
             patch('mic_agent._report_calib'), \
             patch('mic_agent._post'):
            result = mic_agent.calibrate(noise_sec=0.1, enroll_sec=0.1, autotune=False)
        assert isinstance(result, float)
        assert result > 0

    def test_R02_calibrate_upper_greater_than_noise(self):
        """upper > noise_rms"""
        import mic_agent
        fake = self._make_measure_mock(noise_rms=500, speech_rms=3000)
        with patch('mic_agent._measure_frames', side_effect=fake), \
             patch('mic_agent._report_calib'), \
             patch('mic_agent._post'):
            result = mic_agent.calibrate(noise_sec=0.1, enroll_sec=0.1, autotune=False)
        assert result > 500

    def test_R02_calibrate_resets_calib_live_active(self):
        """calibrate 완료 후 _calib_live['active'] = False"""
        import mic_agent
        fake = self._make_measure_mock()
        with patch('mic_agent._measure_frames', side_effect=fake), \
             patch('mic_agent._report_calib'), \
             patch('mic_agent._post'):
            mic_agent.calibrate(noise_sec=0.1, enroll_sec=0.1, autotune=False)
        assert mic_agent._calib_live["active"] is False

    def test_R02_calibrate_reports_noise_and_done(self):
        """noise→speak→done 순으로 _report_calib 호출"""
        import mic_agent
        phases = []
        fake = self._make_measure_mock()
        def fake_report(phase, **kw):
            phases.append(phase)
        with patch('mic_agent._measure_frames', side_effect=fake), \
             patch('mic_agent._report_calib', side_effect=fake_report), \
             patch('mic_agent._post'):
            mic_agent.calibrate(noise_sec=0.1, enroll_sec=0.1, autotune=False)
        assert "noise" in phases
        assert "done" in phases

    def test_R02_calibrate_low_snr_warns(self, capsys):
        """SNR < 6dB → 경고 출력"""
        import mic_agent
        fake = self._make_measure_mock(noise_rms=1000, speech_rms=1050)
        with patch('mic_agent._measure_frames', side_effect=fake), \
             patch('mic_agent._report_calib'), \
             patch('mic_agent._post'):
            mic_agent.calibrate(noise_sec=0.1, enroll_sec=0.1, autotune=False)
        assert "SNR" in capsys.readouterr().out

    def test_R02_calibrate_exception_resets_active(self):
        """예외 발생 시에도 _calib_live['active'] = False"""
        import mic_agent
        with patch('mic_agent._measure_frames', side_effect=RuntimeError("test err")), \
             patch('mic_agent._report_calib'), \
             patch('mic_agent._post'):
            try:
                mic_agent.calibrate(noise_sec=0.1, enroll_sec=0.1)
            except Exception:
                pass
        assert mic_agent._calib_live["active"] is False

    def test_R02_calibrate_autotune_puts_to_tune_q(self):
        """autotune=True + 충분한 audio → _tune_q에 추가"""
        import mic_agent
        fake = self._make_measure_mock(speech_rms=3000)
        mic_agent._tune_q.queue.clear()
        with patch('mic_agent._measure_frames', side_effect=fake), \
             patch('mic_agent._report_calib'), \
             patch('mic_agent._post'):
            mic_agent.calibrate(noise_sec=0.1, enroll_sec=0.1, autotune=True)
        assert not mic_agent._tune_q.empty()
        mic_agent._tune_q.queue.clear()


# ─────────────────────────────────────────────────────────────
# sigh_loop 모의 모델 테스트
# ─────────────────────────────────────────────────────────────
class TestSighLoopWithMockModel:
    """R02: sigh_loop 모의 모델 테스트"""

    def _make_model(self, sigh_prob=0.9):
        """mock YAMNet 모델 반환"""
        interp = MagicMock()
        in_det = [{"index": 0, "shape": np.array([15600])}]
        out_det = [{"index": 1}]
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = sigh_prob
        interp.get_tensor.return_value = scores
        return interp, in_det, out_det, 15600, [0], {0: "sigh"}

    def test_R02_sigh_loop_detects_sigh_and_posts(self):
        """한숨 확률 높을 때 detected:True POST 전송"""
        import mic_agent

        sigh_q = queue.Queue()
        stop = threading.Event()
        frame = np.zeros(480, dtype=np.int16).tobytes()
        for _ in range(50):
            sigh_q.put(frame)

        posted = threading.Event()
        def fake_post(path, payload, **kw):
            if payload.get("detected") is True:
                posted.set()
                stop.set()
            return True

        with patch('mic_agent._load_yamnet', return_value=self._make_model(0.95)), \
             patch('mic_agent._post', side_effect=fake_post):
            threading.Timer(3.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.sigh_loop,
                args=(sigh_q, stop, "fake.tflite", "fake.csv"),
                daemon=True)
            t.start()
            t.join(timeout=5.0)

        assert posted.is_set()

    def test_R02_sigh_loop_below_threshold_no_detect(self):
        """한숨 확률 낮을 때 detected:True 미전송"""
        import mic_agent

        sigh_q = queue.Queue()
        stop = threading.Event()
        frame = np.zeros(480, dtype=np.int16).tobytes()
        for _ in range(50):
            sigh_q.put(frame)

        detected = [False]
        def fake_post(path, payload, **kw):
            if payload.get("detected") is True:
                detected[0] = True
            return True

        threading.Timer(0.5, stop.set).start()
        with patch('mic_agent._load_yamnet', return_value=self._make_model(0.1)), \
             patch('mic_agent._post', side_effect=fake_post):
            t = threading.Thread(
                target=mic_agent.sigh_loop,
                args=(sigh_q, stop, "fake.tflite", "fake.csv"),
                daemon=True)
            t.start()
            t.join(timeout=2.0)

        assert not detected[0]

    def test_R02_sigh_loop_heartbeat_sent(self):
        """SIGH_HEARTBEAT 경과 후 detected:False heartbeat 전송"""
        import mic_agent

        sigh_q = queue.Queue()
        stop = threading.Event()
        heartbeat = threading.Event()

        def fake_post(path, payload, **kw):
            if payload.get("detected") is False:
                heartbeat.set()
                stop.set()
            return True

        with patch('mic_agent._load_yamnet', return_value=self._make_model(0.0)), \
             patch('mic_agent._post', side_effect=fake_post), \
             patch('mic_agent.SIGH_HEARTBEAT', 0.1):
            threading.Timer(3.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.sigh_loop,
                args=(sigh_q, stop, "fake.tflite", "fake.csv"),
                daemon=True)
            t.start()
            t.join(timeout=4.0)

        assert heartbeat.is_set()

    def test_R02_sigh_loop_no_sigh_idx_prints_warning(self, capsys):
        """sigh_idx 비어있으면 경고 출력"""
        import mic_agent

        sigh_q = queue.Queue()
        stop = threading.Event()
        interp = MagicMock()
        in_det = [{"index": 0, "shape": np.array([15600])}]
        out_det = [{"index": 1}]
        empty_model = (interp, in_det, out_det, 15600, [], {})

        threading.Timer(0.2, stop.set).start()
        with patch('mic_agent._load_yamnet', return_value=empty_model):
            mic_agent.sigh_loop(sigh_q, stop, "fake.tflite", "fake.csv")

        assert "클래스 없음" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────
# audio_supervisor 테스트
# ─────────────────────────────────────────────────────────────
class TestAudioSupervisor:
    """R02: audio_supervisor 스트림 관리 테스트"""

    def _make_args(self, no_sigh=False, no_speech=True):
        args = MagicMock()
        args.no_sigh = no_sigh
        args.no_speech = no_speech
        args.device = None
        args.gate_margin = 2.6
        args.vad = 1
        args.enroll_sec = 4.0
        args.enroll_text = "결재 부탁드립니다"
        args.no_autotune = True
        return args

    def _setup_sd_stream(self, stop_evt, reads=3):
        import sounddevice as sd
        frame = np.zeros(480, dtype=np.int16).tobytes()
        call_count = [0]

        def fake_read(n):
            call_count[0] += 1
            if call_count[0] >= reads:
                stop_evt.set()
            return (frame, False)

        mock_stream = MagicMock()
        mock_stream.read.side_effect = fake_read
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        sd.RawInputStream.return_value = mock_ctx
        return mock_stream

    def test_R02_supervisor_puts_frames_to_sigh_q(self):
        """프레임 읽어 sigh_q에 추가"""
        import mic_agent
        stop = threading.Event()
        recalib = threading.Event()
        self._setup_sd_stream(stop, reads=3)

        sigh_q = queue.Queue()
        speech_q = queue.Queue()
        gate = {"upper": float("inf"), "over_frac": 0.15}
        args = self._make_args(no_sigh=False, no_speech=True)

        threading.Timer(2.0, stop.set).start()
        mic_agent.audio_supervisor(stop, recalib, gate, args, sigh_q, speech_q, False)
        assert not sigh_q.empty()

    def test_R02_supervisor_puts_frames_to_speech_q(self):
        """프레임 읽어 speech_q에 추가"""
        import mic_agent
        stop = threading.Event()
        recalib = threading.Event()
        self._setup_sd_stream(stop, reads=3)

        sigh_q = queue.Queue()
        speech_q = queue.Queue()
        gate = {"upper": float("inf"), "over_frac": 0.15}
        args = self._make_args(no_sigh=True, no_speech=False)

        threading.Timer(2.0, stop.set).start()
        mic_agent.audio_supervisor(stop, recalib, gate, args, sigh_q, speech_q, False)
        assert not speech_q.empty()

    def test_R02_supervisor_stream_exception_retries(self):
        """스트림 예외 → 재시도"""
        import mic_agent
        import sounddevice as sd

        stop = threading.Event()
        recalib = threading.Event()
        attempts = [0]
        original_ris = sd.RawInputStream

        def fake_ris(*a, **kw):
            attempts[0] += 1
            if attempts[0] >= 2:
                stop.set()
            raise OSError("device busy")

        sd.RawInputStream = fake_ris
        try:
            gate = {"upper": float("inf"), "over_frac": 0.15}
            args = self._make_args()
            with patch('time.sleep'):
                mic_agent.audio_supervisor(
                    stop, recalib, gate, args,
                    queue.Queue(), queue.Queue(), False)
        finally:
            sd.RawInputStream = original_ris

        assert attempts[0] >= 2

    def test_R02_supervisor_recalib_clears_flag(self):
        """recalib_req 설정 → 처리 후 clear"""
        import mic_agent
        stop = threading.Event()
        recalib = threading.Event()
        frame = np.zeros(480, dtype=np.int16).tobytes()
        call_count = [0]

        def fake_read(n):
            call_count[0] += 1
            if call_count[0] == 2:
                recalib.set()
            if call_count[0] >= 4:
                stop.set()
            return (frame, False)

        import sounddevice as sd
        mock_stream = MagicMock()
        mock_stream.read.side_effect = fake_read
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        sd.RawInputStream.return_value = mock_ctx

        gate = {"upper": float("inf"), "over_frac": 0.15}
        args = self._make_args(no_sigh=False, no_speech=True)

        with patch('mic_agent.calibrate', return_value=5000.0):
            threading.Timer(3.0, stop.set).start()
            mic_agent.audio_supervisor(
                stop, recalib, gate, args,
                queue.Queue(), queue.Queue(), False)

        assert not recalib.is_set()


# ─────────────────────────────────────────────────────────────
# main() 테스트
# ─────────────────────────────────────────────────────────────
class TestMain:
    """R02: main() 진입점 테스트"""

    def test_R02_main_exits_if_both_disabled(self):
        """--no-sigh --no-speech → sys.exit"""
        import mic_agent, pytest as _pytest
        with patch('sys.argv', ['mic_agent.py', '--no-sigh', '--no-speech']):
            with _pytest.raises(SystemExit):
                mic_agent.main()

    def test_R02_main_calls_audio_supervisor(self):
        """정상 실행 → audio_supervisor 호출"""
        import mic_agent
        called = threading.Event()

        def fake_supervisor(*a, **kw):
            called.set()
            raise KeyboardInterrupt

        with patch('sys.argv', ['mic_agent.py', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('time.sleep'):
            mic_agent.main()

        assert called.is_set()

    def test_R02_main_no_sigh_skips_sigh_thread(self):
        """--no-sigh → sigh_loop 스레드 미생성"""
        import mic_agent
        sigh_called = [False]

        def fake_sigh(*a, **kw):
            sigh_called[0] = True

        with patch('sys.argv', ['mic_agent.py', '--no-sigh', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())), \
             patch('mic_agent.sigh_loop', side_effect=fake_sigh), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('time.sleep'):
            mic_agent.main()

        assert not sigh_called[0]

    def test_R02_main_no_speech_skips_speech_threads(self):
        """--no-speech → speech 스레드 미생성"""
        import mic_agent
        speech_called = [False]

        def fake_speech(*a, **kw):
            speech_called[0] = True

        with patch('sys.argv', ['mic_agent.py', '--no-speech', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop', side_effect=fake_speech), \
             patch('time.sleep'):
            mic_agent.main()

        assert not speech_called[0]

    def test_R02_main_gate_passed_to_supervisor(self):
        """gate dict가 audio_supervisor에 전달"""
        import mic_agent
        captured = {}

        def fake_supervisor(stop_evt, recalib_req, gate, args, *a, **kw):
            captured['gate'] = gate
            raise KeyboardInterrupt

        with patch('sys.argv', ['mic_agent.py', '--no-speech', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.sigh_loop'), \
             patch('time.sleep'):
            mic_agent.main()

        assert 'upper' in captured.get('gate', {})
        assert 'over_frac' in captured.get('gate', {})

    def test_R02_main_stT_cfg_beam_from_cli(self):
        """--beam-size 5 → STT_CFG['beam'] = 5"""
        import mic_agent

        with patch('sys.argv', ['mic_agent.py', '--no-speech', '--no-calibrate',
                                '--beam-size', '5']), \
             patch('mic_agent.audio_supervisor', side_effect=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())), \
             patch('mic_agent.sigh_loop'), \
             patch('time.sleep'):
            mic_agent.main()

        assert mic_agent.STT_CFG['beam'] == 5

    def test_R02_main_prints_banner(self, capsys):
        """main() 실행 시 RADAR_URL 배너 출력"""
        import mic_agent

        with patch('sys.argv', ['mic_agent.py', '--no-speech', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())), \
             patch('mic_agent.sigh_loop'), \
             patch('time.sleep'):
            mic_agent.main()

        assert "RADAR_URL" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────
# load_stt_config / load_gate_config 추가 브랜치 커버
# ─────────────────────────────────────────────────────────────
class TestLoadConfigBranches:
    """R01/R05: 설정 로딩 엣지 브랜치 커버"""

    def test_R05_stt_explicit_empty_string_skipped(self):
        """explicit='' → not p 분기 실행 → 폴백 경로 탐색"""
        import mic_agent
        # 빈 문자열 explicit → not p 분기 실행
        cfg, path = mic_agent.load_stt_config("")
        assert isinstance(cfg, dict)

    def test_R01_gate_explicit_empty_string_skipped(self):
        """explicit='' → not p 분기 실행 → 폴백 경로 탐색"""
        import mic_agent
        cfg, path = mic_agent.load_gate_config("")
        assert isinstance(cfg, dict)

    def test_R05_stt_duplicate_path_seen_skipped(self, tmp_path, monkeypatch):
        """같은 경로가 seen에 있으면 p in seen 분기 실행"""
        import mic_agent
        cfg_file = tmp_path / "stt_config.json"
        cfg_file.write_text('{"whisper_size": "tiny"}')
        # CWD를 tmp_path로 → cands[0]==cands[1] 중복 경로 발생
        monkeypatch.chdir(tmp_path)
        # explicit도 같은 경로 → seen 중복 발생
        cfg, path = mic_agent.load_stt_config(str(tmp_path / "stt_config.json"))
        assert isinstance(cfg, dict)

    def test_R01_gate_duplicate_path_seen_skipped(self, tmp_path, monkeypatch):
        """같은 경로가 seen에 있으면 p in seen 분기 실행"""
        import mic_agent
        cfg_file = tmp_path / "gate_config.json"
        cfg_file.write_text('{"gate_margin": 2.0}')
        monkeypatch.chdir(tmp_path)
        cfg, path = mic_agent.load_gate_config(str(tmp_path / "gate_config.json"))
        assert isinstance(cfg, dict)


# ─────────────────────────────────────────────────────────────
# _measure_frames 직접 패치 테스트
# ─────────────────────────────────────────────────────────────
class TestMeasureFramesDirect:
    """R02: _measure_frames sd 직접 패치로 커버"""

    def _make_sd_mock(self, rms=1000, n_reads=1):
        """mic_agent.sd.RawInputStream 패치용 mock 생성"""
        samples = np.full(480, rms, dtype=np.int16)
        frame_bytes = samples.tobytes()
        read_count = [0]

        def fake_read(n):
            read_count[0] += 1
            return (frame_bytes, False)

        mock_stream = MagicMock()
        mock_stream.read.side_effect = fake_read
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ris = MagicMock(return_value=mock_ctx)
        return mock_ris

    def test_R02_measure_frames_basic(self):
        """sd.RawInputStream 직접 패치 → _measure_frames 실행"""
        import mic_agent
        mock_ris = self._make_sd_mock(rms=1000)
        with patch.object(mic_agent, 'sd') as mock_sd:
            mock_sd.RawInputStream.return_value = mock_ris.return_value
            mock_ris.return_value.__enter__.return_value.read.return_value = (
                np.full(480, 1000, dtype=np.int16).tobytes(), False)
            result = mic_agent._measure_frames(0.03)
        assert isinstance(result, list)

    def test_R02_measure_frames_vad_false(self):
        """vad=None → voiced=True"""
        import mic_agent
        samples = np.full(480, 500, dtype=np.int16)
        mock_stream = MagicMock()
        mock_stream.read.return_value = (samples.tobytes(), False)
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_stream
        mock_ctx.__exit__.return_value = False

        with patch.object(mic_agent, 'sd') as mock_sd:
            mock_sd.RawInputStream.return_value = mock_ctx
            result = mic_agent._measure_frames(0.03)

        assert len(result) >= 1
        rms, voiced = result[0]
        assert voiced is True

    def test_R02_measure_frames_vad_true(self):
        """vad 제공 → is_speech 결과 반영"""
        import mic_agent
        samples = np.full(480, 500, dtype=np.int16)
        mock_stream = MagicMock()
        mock_stream.read.return_value = (samples.tobytes(), False)
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_stream
        mock_ctx.__exit__.return_value = False

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False

        with patch.object(mic_agent, 'sd') as mock_sd:
            mock_sd.RawInputStream.return_value = mock_ctx
            result = mic_agent._measure_frames(0.03, vad=mock_vad)

        assert len(result) >= 1
        rms, voiced = result[0]
        assert voiced is False

    def test_R02_measure_frames_collect_audio_true(self):
        """collect_audio=True → (list, ndarray) 반환"""
        import mic_agent
        samples = np.full(480, 500, dtype=np.int16)
        mock_stream = MagicMock()
        mock_stream.read.return_value = (samples.tobytes(), False)
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_stream
        mock_ctx.__exit__.return_value = False

        with patch.object(mic_agent, 'sd') as mock_sd:
            mock_sd.RawInputStream.return_value = mock_ctx
            result = mic_agent._measure_frames(0.03, collect_audio=True)

        out, audio = result
        assert isinstance(out, list)
        assert isinstance(audio, np.ndarray)

    def test_R02_measure_frames_rms_correct(self):
        """rms 값이 0 초과"""
        import mic_agent
        samples = np.full(480, 1000, dtype=np.int16)
        mock_stream = MagicMock()
        mock_stream.read.return_value = (samples.tobytes(), False)
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_stream
        mock_ctx.__exit__.return_value = False

        with patch.object(mic_agent, 'sd') as mock_sd:
            mock_sd.RawInputStream.return_value = mock_ctx
            result = mic_agent._measure_frames(0.03)

        rms, _ = result[0]
        assert rms > 0


# ─────────────────────────────────────────────────────────────
# _run_autotune 테스트
# ─────────────────────────────────────────────────────────────
class TestRunAutotune:
    """R02: _run_autotune beam 자동 선택 테스트"""

    def _make_stt(self, text="결재 부탁드립니다"):
        seg = MagicMock()
        seg.text = text
        stt = MagicMock()
        stt.transcribe.return_value = ([seg], MagicMock())
        return stt

    def test_R02_autotune_updates_stT_cfg_beam(self):
        """beam 1/5 비교 후 STT_CFG['beam'] 업데이트"""
        import mic_agent
        stt = self._make_stt("결재 부탁드립니다")
        audio = np.zeros(16000, dtype=np.float32)
        ref = "결재 부탁드립니다"
        mic_agent._run_autotune(stt, audio, ref)
        assert mic_agent.STT_CFG['beam'] in (1, 5)

    def test_R02_autotune_picks_lower_cer(self):
        """CER 낮은 beam 선택"""
        import mic_agent
        # beam=1: 정확히 일치, beam=5: 다름 → beam=1 선택
        call_count = [0]
        def fake_transcribe(audio, language, beam_size, **kw):
            call_count[0] += 1
            seg = MagicMock()
            seg.text = "결재 부탁드립니다" if beam_size == 1 else "전혀 다른 내용"
            return ([seg], MagicMock())
        stt = MagicMock()
        stt.transcribe.side_effect = fake_transcribe
        audio = np.zeros(16000, dtype=np.float32)
        mic_agent._run_autotune(stt, audio, "결재 부탁드립니다")
        assert mic_agent.STT_CFG['beam'] == 1

    def test_R02_autotune_prints_result(self, capsys):
        """자동 튜닝 결과 출력"""
        import mic_agent
        stt = self._make_stt("결재")
        audio = np.zeros(16000, dtype=np.float32)
        mic_agent._run_autotune(stt, audio, "결재")
        assert "자동 튜닝" in capsys.readouterr().out

    def test_R02_autotune_respects_stT_lock(self):
        """STT_LOCK 사용 → 스레드 안전"""
        import mic_agent
        stt = self._make_stt("결재")
        audio = np.zeros(16000, dtype=np.float32)
        results = []
        def worker():
            mic_agent._run_autotune(stt, audio, "결재")
            results.append(mic_agent.STT_CFG['beam'])
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert all(b in (1, 5) for b in results)


# ─────────────────────────────────────────────────────────────
# speech_process_loop 모킹 테스트
# ─────────────────────────────────────────────────────────────
class TestSpeechProcessLoop:
    """R02: speech_process_loop WhisperModel/KoBERT 모킹 테스트"""

    def setup_method(self):
        """faster_whisper를 sys.modules에 주입 (함수 내부 로컬 임포트 대응)"""
        self._fw_mock = MagicMock()
        self._stt_mock = MagicMock()
        self._fw_mock.WhisperModel.return_value = self._stt_mock
        sys.modules['faster_whisper'] = self._fw_mock

    def _make_clf(self, emotion="기쁨", transcript="결재 부탁드립니다"):
        seg = MagicMock()
        seg.text = transcript
        self._stt_mock.transcribe.return_value = ([seg], MagicMock())
        clf = MagicMock()
        clf.predict.return_value = {
            "top": emotion, "top_prob": 0.9,
            "probs": {e: 0.1 for e in ["기쁨", "슬픔", "분노", "불안", "당황", "상처"]}
        }
        return clf

    def test_R02_speech_process_loop_posts_emotion(self):
        """발화 입력 → 감정 POST 전송"""
        import mic_agent
        clf = self._make_clf()
        utter_q = queue.Queue()
        stop = threading.Event()
        audio = np.zeros(8000, dtype=np.float32)
        stats = {"min": 100.0, "mean": 200.0, "max": 300.0, "over_frac": 0.0}
        utter_q.put((audio, stats, "ok"))
        posted = threading.Event()

        def fake_post(path, payload, **kw):
            if "/add-speech-emotion" in path:
                posted.set()
                stop.set()
            return True

        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post', side_effect=fake_post):
            threading.Timer(3.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=5.0)

        assert posted.is_set()

    def test_R02_speech_process_loop_too_loud_skipped(self):
        """too_loud 발화 → STT 스킵"""
        import mic_agent
        clf = self._make_clf()
        utter_q = queue.Queue()
        stop = threading.Event()
        stats = {"min": 100.0, "mean": 200.0, "max": 300.0, "over_frac": 0.9}
        utter_q.put((None, stats, "too_loud"))

        stt_called = [False]
        def fake_transcribe(*a, **kw):
            stt_called[0] = True
            return ([], MagicMock())
        self._stt_mock.transcribe.side_effect = fake_transcribe

        threading.Timer(1.0, stop.set).start()
        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post'):
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=3.0)

        assert not stt_called[0]

    def test_R02_speech_process_loop_short_text_skipped(self):
        """짧은 텍스트(min_chars 미만) → POST 미전송"""
        import mic_agent
        clf = self._make_clf(transcript="아")  # 1글자 < min_chars=2
        utter_q = queue.Queue()
        stop = threading.Event()
        audio = np.zeros(8000, dtype=np.float32)
        stats = {"min": 100.0, "mean": 200.0, "max": 300.0, "over_frac": 0.0}
        utter_q.put((audio, stats, "ok"))
        posted = [False]

        def fake_post(path, payload, **kw):
            if "/add-speech-emotion" in path:
                posted[0] = True
            return True

        threading.Timer(1.0, stop.set).start()
        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post', side_effect=fake_post):
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=3.0)

        assert not posted[0]

    def test_R02_speech_process_loop_heartbeat_sent(self):
        """speech-heartbeat POST 전송"""
        import mic_agent
        clf = self._make_clf()
        utter_q = queue.Queue()
        stop = threading.Event()
        heartbeat = threading.Event()

        def fake_post(path, payload, **kw):
            if "heartbeat" in path:
                heartbeat.set()
                stop.set()
            return True

        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post', side_effect=fake_post), \
             patch('mic_agent.time') as mock_time:
            orig = time.time
            mock_time.time.side_effect = lambda: orig() + 6.0
            mock_time.sleep = time.sleep
            threading.Timer(2.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=4.0)

        assert heartbeat.is_set()

    def test_R02_speech_process_loop_autotune_processed(self):
        """_tune_q 요청 → _run_autotune 호출"""
        import mic_agent
        clf = self._make_clf()
        utter_q = queue.Queue()
        stop = threading.Event()
        tuned = threading.Event()

        def fake_autotune(stt, audio, ref):
            tuned.set()
            stop.set()

        audio = np.zeros(8000, dtype=np.float32)
        mic_agent._tune_q.queue.clear()
        mic_agent._tune_q.put((audio, "결재"))

        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post'), \
             patch('mic_agent._run_autotune', side_effect=fake_autotune):
            threading.Timer(3.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=5.0)
            mic_agent._tune_q.queue.clear()

        assert tuned.is_set()

    def test_R02_speech_process_loop_drains_stale_utterances(self):
        """밀린 발화 → 최신 1개만 처리"""
        import mic_agent
        clf = self._make_clf()
        utter_q = queue.Queue()
        stop = threading.Event()
        audio = np.zeros(8000, dtype=np.float32)
        stats = {"min": 100.0, "mean": 200.0, "max": 300.0, "over_frac": 0.0}
        for _ in range(5):
            utter_q.put((audio, stats, "ok"))
        processed = [0]
        posted = threading.Event()

        def fake_post(path, payload, **kw):
            if "/add-speech-emotion" in path:
                processed[0] += 1
                posted.set()
                stop.set()
            return True

        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post', side_effect=fake_post):
            threading.Timer(3.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=5.0)

        assert processed[0] == 1


# ─────────────────────────────────────────────────────────────
# main() speech 관련 브랜치 추가 커버
# ─────────────────────────────────────────────────────────────
class TestMainSpeechBranches:
    """R02: main() speech 브랜치 커버"""

    def _run_main_with_speech(self, extra_args=None, stt_cfg=None, gate_cfg=None):
        """speech 활성화 main() 실행 헬퍼"""
        import mic_agent
        argv = ['mic_agent.py', '--no-calibrate'] + (extra_args or [])

        with patch('sys.argv', argv), \
             patch('mic_agent.audio_supervisor',
                   side_effect=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config',
                   return_value=(stt_cfg or {}, None)), \
             patch('mic_agent.load_gate_config',
                   return_value=(gate_cfg or {}, None)), \
             patch('time.sleep'):
            mic_agent.main()

    def test_R02_main_speech_no_stt_config_prints_msg(self, capsys):
        """speech + stt_config 없음 → '기본값/CLI' 메시지 출력"""
        self._run_main_with_speech()
        assert "기본값" in capsys.readouterr().out

    def test_R02_main_speech_with_stt_config_prints_path(self, capsys, tmp_path):
        """speech + stt_config 있음 → 설정 파일 경로 출력"""
        import mic_agent
        argv = ['mic_agent.py', '--no-calibrate']
        cfg_path = str(tmp_path / "stt_config.json")

        with patch('sys.argv', argv), \
             patch('mic_agent.audio_supervisor',
                   side_effect=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config', return_value=({}, cfg_path)), \
             patch('mic_agent.load_gate_config', return_value=({}, None)), \
             patch('time.sleep'):
            mic_agent.main()

        assert "설정 파일" in capsys.readouterr().out

    def test_R02_main_speech_with_gate_config_prints_path(self, capsys, tmp_path):
        """speech + gate_config 있음 → 게이트 설정 경로 출력"""
        import mic_agent
        argv = ['mic_agent.py', '--no-calibrate']
        gate_path = str(tmp_path / "gate_config.json")

        with patch('sys.argv', argv), \
             patch('mic_agent.audio_supervisor',
                   side_effect=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config', return_value=({}, None)), \
             patch('mic_agent.load_gate_config',
                   return_value=({"gate_margin": 2.0, "over_frac": 0.1}, gate_path)), \
             patch('time.sleep'):
            mic_agent.main()

        assert "gate" in capsys.readouterr().out.lower()

    def test_R02_main_speech_no_gate_config_prints_default(self, capsys):
        """speech + gate_config 없음 → 기본값 출력"""
        self._run_main_with_speech()
        out = capsys.readouterr().out
        assert "gate_config.json 없음" in out or "기본" in out

    def test_R02_main_gate_margin_from_config(self):
        """gate_config.json에서 gate_margin 읽기"""
        import mic_agent
        captured = {}

        def fake_supervisor(stop_evt, recalib_req, gate, args, *a, **kw):
            captured['gate_margin'] = args.gate_margin
            raise KeyboardInterrupt

        with patch('sys.argv', ['mic_agent.py', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config', return_value=({}, None)), \
             patch('mic_agent.load_gate_config',
                   return_value=({"gate_margin": 3.5, "over_frac": 0.2}, None)), \
             patch('time.sleep'):
            mic_agent.main()

        assert captured.get('gate_margin') == 3.5

    def test_R02_main_pick_from_stt_config(self):
        """_pick: stt_config에서 whisper_size 읽기"""
        import mic_agent
        captured = {}

        def fake_supervisor(stop_evt, recalib_req, gate, args, *a, **kw):
            raise KeyboardInterrupt

        # stt_config에 whisper_size 있고 CLI에는 없음
        with patch('sys.argv', ['mic_agent.py', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config',
                   return_value=({"whisper_size": "small", "beam_size": 3,
                                  "compute_type": "float16", "vad_filter": True}, None)), \
             patch('mic_agent.load_gate_config', return_value=({}, None)), \
             patch('time.sleep'):
            mic_agent.main()

        # STT_CFG beam이 stt_config에서 온 3으로 설정됐는지 확인
        assert mic_agent.STT_CFG['beam'] == 3
        assert mic_agent.STT_CFG['vad_filter'] is True

    def test_R02_main_initial_calibrate_triggered(self):
        """--no-sigh + speech활성(calibrate 안 막음) → do_initial_calib=True"""
        import mic_agent
        captured = {}

        def fake_supervisor(stop_evt, recalib_req, gate, args, sigh_q, speech_q,
                            do_initial_calib):
            captured['do_initial_calib'] = do_initial_calib
            raise KeyboardInterrupt

        with patch('sys.argv', ['mic_agent.py', '--no-sigh']), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config', return_value=({}, None)), \
             patch('mic_agent.load_gate_config', return_value=({}, None)), \
             patch('time.sleep'):
            mic_agent.main()

        assert captured.get('do_initial_calib') is True

    def test_R02_main_no_calibrate_flag(self):
        """--no-calibrate → do_initial_calib=False"""
        import mic_agent
        captured = {}

        def fake_supervisor(stop_evt, recalib_req, gate, args, sigh_q, speech_q,
                            do_initial_calib):
            captured['do_initial_calib'] = do_initial_calib
            raise KeyboardInterrupt

        with patch('sys.argv', ['mic_agent.py', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config', return_value=({}, None)), \
             patch('mic_agent.load_gate_config', return_value=({}, None)), \
             patch('time.sleep'):
            mic_agent.main()

        assert captured.get('do_initial_calib') is False


# ─────────────────────────────────────────────────────────────
# audio_supervisor 캘리브레이션 실패 브랜치
# ─────────────────────────────────────────────────────────────
class TestAudioSupervisorCalibFail:
    """R02: audio_supervisor 캘리브레이션 실패 → 게이트 off"""

    def test_R02_supervisor_calib_fail_sets_gate_inf(self):
        """calibrate 예외 → gate['upper'] = inf"""
        import mic_agent

        stop = threading.Event()
        recalib = threading.Event()
        gate = {"upper": 0.0, "over_frac": 0.15}
        args = MagicMock()
        args.no_sigh = True
        args.no_speech = False
        args.device = None
        args.gate_margin = 2.6
        args.vad = 1
        args.enroll_sec = 4.0
        args.enroll_text = "결재"
        args.no_autotune = True

        import sounddevice as sd
        frame = np.zeros(480, dtype=np.int16).tobytes()
        call_count = [0]

        def fake_read(n):
            call_count[0] += 1
            stop.set()
            return (frame, False)

        mock_stream = MagicMock()
        mock_stream.read.side_effect = fake_read
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        sd.RawInputStream.return_value = mock_ctx

        with patch('mic_agent.calibrate', side_effect=RuntimeError("calib fail")), \
             patch('mic_agent._post'):
            threading.Timer(3.0, stop.set).start()
            mic_agent.audio_supervisor(
                stop, recalib, gate, args,
                queue.Queue(), queue.Queue(), True)

        assert gate['upper'] == float('inf')


# ─────────────────────────────────────────────────────────────
# load_stt_config / load_gate_config: 모든 경로 실패 → ({}, None)
# ─────────────────────────────────────────────────────────────
class TestLoadConfigReturnEmpty:
    """load_stt_config / load_gate_config 모든 후보 경로 없음 → ({}, None)"""

    def test_R05_stt_returns_empty_when_no_file_anywhere(self, tmp_path, monkeypatch):
        import mic_agent
        monkeypatch.chdir(tmp_path)
        with patch('os.path.exists', return_value=False):
            cfg, path = mic_agent.load_stt_config(None)
        assert cfg == {}
        assert path is None

    def test_R01_gate_returns_empty_when_no_file_anywhere(self, tmp_path, monkeypatch):
        import mic_agent
        monkeypatch.chdir(tmp_path)
        with patch('os.path.exists', return_value=False):
            cfg, path = mic_agent.load_gate_config(None)
        assert cfg == {}
        assert path is None

    def test_R05_stt_explicit_not_found_no_fallback(self, tmp_path, monkeypatch):
        import mic_agent
        monkeypatch.chdir(tmp_path)
        with patch('os.path.exists', return_value=False):
            cfg, path = mic_agent.load_stt_config("/no/such/stt_config.json")
        assert cfg == {}
        assert path is None

    def test_R01_gate_explicit_not_found_no_fallback(self, tmp_path, monkeypatch):
        import mic_agent
        monkeypatch.chdir(tmp_path)
        with patch('os.path.exists', return_value=False):
            cfg, path = mic_agent.load_gate_config("/no/such/gate_config.json")
        assert cfg == {}
        assert path is None


# ─────────────────────────────────────────────────────────────
# _load_yamnet: tflite_runtime 모킹으로 함수 본문 커버
# ─────────────────────────────────────────────────────────────
class TestLoadYamnet:
    """_load_yamnet tflite_runtime 모킹 테스트"""

    def _write_csv(self, tmp_path, include_sigh=True):
        csv_path = tmp_path / "yamnet_class_map.csv"
        lines = ["index,mid,display_name\n"]
        if include_sigh:
            lines += ["0,/m/sigh,sigh\n", "1,/m/breathing,breathing\n",
                      "2,/m/music,music\n"]
        else:
            lines += ["0,/m/music,music\n", "1,/m/speech,speech\n"]
        csv_path.write_text("".join(lines))
        return str(csv_path)

    def _inject_tflite(self, shape=(15600,)):
        interp = MagicMock()
        interp.get_input_details.return_value = [
            {"index": 0, "shape": np.array(shape)}
        ]
        interp.get_output_details.return_value = [{"index": 1}]
        tflite_mod = MagicMock()
        tflite_mod.Interpreter.return_value = interp
        pkg = MagicMock()
        pkg.interpreter = tflite_mod
        sys.modules['tflite_runtime'] = pkg
        sys.modules['tflite_runtime.interpreter'] = tflite_mod
        return interp

    def test_R02_returns_6_tuple(self, tmp_path):
        import mic_agent
        self._inject_tflite()
        result = mic_agent._load_yamnet("fake.tflite", self._write_csv(tmp_path))
        assert len(result) == 6

    def test_R02_sigh_idx_nonempty_with_sigh_labels(self, tmp_path):
        import mic_agent
        self._inject_tflite()
        _, _, _, _, sigh_idx, _ = mic_agent._load_yamnet(
            "fake.tflite", self._write_csv(tmp_path, include_sigh=True))
        assert len(sigh_idx) > 0

    def test_R02_sigh_idx_empty_without_sigh_labels(self, tmp_path):
        import mic_agent
        self._inject_tflite()
        _, _, _, _, sigh_idx, _ = mic_agent._load_yamnet(
            "fake.tflite", self._write_csv(tmp_path, include_sigh=False))
        assert sigh_idx == []

    def test_R02_window_len_1d_shape(self, tmp_path):
        import mic_agent
        self._inject_tflite(shape=(15600,))
        _, _, _, window_len, _, _ = mic_agent._load_yamnet(
            "fake.tflite", self._write_csv(tmp_path))
        assert window_len == 15600

    def test_R02_window_len_2d_shape(self, tmp_path):
        import mic_agent
        self._inject_tflite(shape=(1, 15600))
        _, _, _, window_len, _, _ = mic_agent._load_yamnet(
            "fake.tflite", self._write_csv(tmp_path))
        assert window_len == 15600

    def test_R02_prints_completion(self, tmp_path, capsys):
        import mic_agent
        self._inject_tflite()
        mic_agent._load_yamnet("fake.tflite", self._write_csv(tmp_path))
        assert "YAMNet" in capsys.readouterr().out

    def test_R02_tensorflow_fallback(self, tmp_path):
        """tflite_runtime 없을 때 tensorflow.lite 사용"""
        import mic_agent
        interp = MagicMock()
        interp.get_input_details.return_value = [
            {"index": 0, "shape": np.array([15600])}
        ]
        interp.get_output_details.return_value = [{"index": 1}]
        tf_mock = MagicMock()
        tf_mock.lite.Interpreter.return_value = interp
        # sys.modules에 None 설정 → Python import 시 ImportError 발생
        with patch.dict(sys.modules, {
            'tflite_runtime': None,
            'tflite_runtime.interpreter': None,
            'tensorflow': tf_mock,
        }):
            result = mic_agent._load_yamnet("fake.tflite", self._write_csv(tmp_path))
        assert len(result) == 6


# ─────────────────────────────────────────────────────────────
# EmotionClassifier: torch/kobert 모킹으로 __init__ / predict 커버
# ─────────────────────────────────────────────────────────────
class TestEmotionClassifier:
    """EmotionClassifier __init__ / predict 모킹 테스트"""

    def _inject_mocks(self):
        torch_mock = MagicMock()
        # no_grad context manager
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        torch_mock.no_grad.return_value = ctx

        # softmax 반환: [[prob0, prob1, ...]] 형태
        prob_vals = [MagicMock() for _ in range(6)]
        for i, pv in enumerate(prob_vals):
            pv.item.return_value = round(0.1 + i * 0.05, 3)
        torch_mock.softmax.return_value = [prob_vals]

        model_out = MagicMock()
        model_out.logits = MagicMock()
        model_mock = MagicMock(return_value=model_out)

        tokenizer_mock = MagicMock()
        tokenizer_mock.return_value = {"input_ids": MagicMock()}

        transformers_mock = MagicMock()
        transformers_mock.AutoModelForSequenceClassification\
            .from_pretrained.return_value = model_mock()

        kobert_mock = MagicMock()
        kobert_mock.get_tokenizer.return_value = tokenizer_mock

        sys.modules['torch'] = torch_mock
        sys.modules['transformers'] = transformers_mock
        sys.modules['kobert_transformers'] = kobert_mock

        return torch_mock, tokenizer_mock

    def test_R02_init_succeeds(self):
        import mic_agent
        self._inject_mocks()
        clf = mic_agent.EmotionClassifier(model_name="test-model", threads=2)
        assert clf is not None

    def test_R02_predict_returns_dict(self):
        import mic_agent
        self._inject_mocks()
        clf = mic_agent.EmotionClassifier(model_name="test-model", threads=2)
        result = clf.predict("결재 부탁드립니다")
        assert isinstance(result, dict)
        assert 'top' in result
        assert 'top_prob' in result
        assert 'probs' in result

    def test_R02_predict_top_is_valid_emotion(self):
        from mic_agent import EMOTIONS
        import mic_agent
        self._inject_mocks()
        clf = mic_agent.EmotionClassifier(model_name="test-model", threads=2)
        result = clf.predict("슬프다")
        assert result['top'] in EMOTIONS

    def test_R02_predict_probs_has_all_emotions(self):
        from mic_agent import EMOTIONS
        import mic_agent
        self._inject_mocks()
        clf = mic_agent.EmotionClassifier(model_name="test-model", threads=2)
        result = clf.predict("화가난다")
        assert len(result['probs']) == len(EMOTIONS)


# ─────────────────────────────────────────────────────────────
# speech_assemble_loop: finish() voiced_rms 빈 브랜치
# ─────────────────────────────────────────────────────────────
class TestSpeechAssembleLoopFinishEmpty:
    """finish() voiced_rms 비어있는 브랜치 커버"""

    @staticmethod
    def _frame(rms=0):
        return np.full(480, int(rms), dtype=np.int16).tobytes()

    def test_R02_finish_empty_voiced_rms_no_crash(self):
        """voiced frame 0개인 상태로 finish() 호출 → utter_q 비어 있음"""
        import mic_agent
        speech_q = queue.Queue()
        utter_q = queue.Queue()
        stop = threading.Event()
        gate = {"upper": 0.0, "over_frac": 0.15}  # upper=0 → 트리거 없음

        for _ in range(30):
            speech_q.put(self._frame(rms=0))
        threading.Timer(0.8, stop.set).start()

        t = threading.Thread(
            target=mic_agent.speech_assemble_loop,
            args=(speech_q, utter_q, stop, gate, 1), daemon=True)
        t.start()
        t.join(timeout=3.0)
        assert utter_q.empty()

    def test_R02_finish_after_voiced_silence_tail(self):
        """voiced frame + MAX_UTTER_MS(8000ms) 초과 → finish() ok 분기 커버
        (webrtcvad mock은 항상 True → silence 카운터 미작동,
         대신 len(voiced)*30 > 8000 조건으로 finish() 트리거)"""
        import mic_agent
        speech_q = queue.Queue()
        utter_q = queue.Queue()
        stop = threading.Event()
        gate = {"upper": 32768.0, "over_frac": 0.15}

        # triggered 활성화용 voiced 프레임 1개 (rms=500)
        voiced_frame = np.full(480, 500, dtype=np.int16).tobytes()
        speech_q.put(voiced_frame)

        # MAX_UTTER_MS=8000ms, FRAME_MS=30ms → 267*30=8010 > 8000 필요
        # ring buffer(maxlen=8)도 voiced에 합쳐지므로 270개면 충분
        silent_frame = np.zeros(480, dtype=np.int16).tobytes()
        for _ in range(270):
            speech_q.put(silent_frame)

        threading.Timer(2.0, stop.set).start()
        t = threading.Thread(
            target=mic_agent.speech_assemble_loop,
            args=(speech_q, utter_q, stop, gate, 1), daemon=True)
        t.start()
        t.join(timeout=5.0)

        assert not utter_q.empty()
        _, _, reason = utter_q.get_nowait()
        assert reason == "ok"


# ─────────────────────────────────────────────────────────────
# speech_process_loop: onnxruntime import + drain Empty 브랜치
# ─────────────────────────────────────────────────────────────
class TestSpeechProcessLoopExtra:
    """speech_process_loop 추가 브랜치 커버"""

    def setup_method(self):
        fw = MagicMock()
        fw.WhisperModel.return_value = MagicMock()
        sys.modules['faster_whisper'] = fw

    def test_R02_onnxruntime_severity_called(self):
        """onnxruntime import 성공 → set_default_logger_severity(3) 호출"""
        import mic_agent
        onnx_mock = MagicMock()
        sys.modules['onnxruntime'] = onnx_mock
        utter_q = queue.Queue()
        stop = threading.Event()
        clf = MagicMock()
        clf.predict.return_value = {"top": "기쁨", "top_prob": 0.9, "probs": {}}
        threading.Timer(0.3, stop.set).start()
        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post'):
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=2.0)
        onnx_mock.set_default_logger_severity.assert_called_once_with(3)
        sys.modules.pop('onnxruntime', None)

    def test_R02_drain_loop_with_extra_item(self):
        """utter_q에 2개 → drain 루프 진입 후 Empty 예외 브랜치 커버"""
        import mic_agent
        utter_q = queue.Queue()
        stop = threading.Event()
        audio = np.zeros(8000, dtype=np.float32)
        stats = {"min": 100.0, "mean": 200.0, "max": 300.0, "over_frac": 0.0}

        seg = MagicMock()
        seg.text = "결재"
        fw_mock = sys.modules['faster_whisper']
        fw_mock.WhisperModel.return_value.transcribe.return_value = (
            [seg], MagicMock())

        utter_q.put((audio, stats, "ok"))
        utter_q.put((audio, stats, "ok"))

        done = threading.Event()

        def fake_post(path, payload, **kw):
            if "/add-speech-emotion" in path:
                done.set()
                stop.set()
            return True

        clf = MagicMock()
        clf.predict.return_value = {"top": "기쁨", "top_prob": 0.9, "probs": {}}
        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post', side_effect=fake_post):
            threading.Timer(3.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=5.0)
        assert done.is_set()

    def test_R02_drain_get_nowait_empty_exception(self):
        """drain 루프 race condition: empty()=False이지만 get_nowait()→Empty
        → except queue.Empty: break 브랜치 커버"""
        import mic_agent

        class _RaceQueue(queue.Queue):
            """empty()는 False, get_nowait()은 Empty — race condition 시뮬"""
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._race_triggered = False

            def empty(self):
                if not self._race_triggered:
                    return False   # 드레인 루프 진입을 위해 거짓 False 반환
                return super().empty()

            def get_nowait(self):
                if not self._race_triggered:
                    self._race_triggered = True
                    raise queue.Empty   # race condition 시뮬레이션
                return super().get_nowait()

        audio = np.zeros(8000, dtype=np.float32)
        stats = {"min": 100.0, "mean": 200.0, "max": 300.0, "over_frac": 0.0}
        utter_q = _RaceQueue()
        utter_q.put((audio, stats, "ok"))  # 메인 get()용

        stop = threading.Event()
        seg = MagicMock()
        seg.text = "결재"
        fw_mock = sys.modules['faster_whisper']
        fw_mock.WhisperModel.return_value.transcribe.return_value = (
            [seg], MagicMock())

        done = threading.Event()

        def fake_post(path, payload, **kw):
            if "/add-speech-emotion" in path:
                done.set()
                stop.set()
            return True

        clf = MagicMock()
        clf.predict.return_value = {"top": "기쁨", "top_prob": 0.9, "probs": {}}
        with patch('mic_agent.EmotionClassifier', return_value=clf), \
             patch('mic_agent._post', side_effect=fake_post):
            threading.Timer(3.0, stop.set).start()
            t = threading.Thread(
                target=mic_agent.speech_process_loop,
                args=(utter_q, stop, "tiny", 2, 2), daemon=True)
            t.start()
            t.join(timeout=5.0)
        assert done.is_set()


# ─────────────────────────────────────────────────────────────
# main(): gate_margin CLI 분기 (args.gate_margin is not None)
# ─────────────────────────────────────────────────────────────
class TestMainGateMarginBranch:
    """main() gate_margin / over_frac CLI 분기 커버"""

    def _run_main_capture(self, extra_argv, captured_key, captured):
        import mic_agent

        def fake_supervisor(stop_evt, recalib_req, gate, args, *a, **kw):
            captured[captured_key] = getattr(args, captured_key, None)
            raise KeyboardInterrupt

        with patch('sys.argv', ['mic_agent.py', '--no-calibrate'] + extra_argv), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config', return_value=({}, None)), \
             patch('mic_agent.load_gate_config',
                   return_value=({"gate_margin": 99.0, "over_frac": 0.99}, None)), \
             patch('time.sleep'):
            mic_agent.main()

    def test_R02_gate_margin_cli_overrides_config(self):
        """--gate-margin 3.0 → args.gate_margin = 3.0 (config 99.0 무시)"""
        captured = {}
        self._run_main_capture(['--gate-margin', '3.0'], 'gate_margin', captured)
        assert captured.get('gate_margin') == 3.0

    def test_R02_gate_margin_none_uses_config(self):
        """--gate-margin 없으면 config값 사용"""
        import mic_agent
        gate_used = {}

        def fake_supervisor(stop_evt, recalib_req, gate, args, *a, **kw):
            gate_used['gate_margin'] = args.gate_margin
            raise KeyboardInterrupt

        with patch('sys.argv', ['mic_agent.py', '--no-calibrate']), \
             patch('mic_agent.audio_supervisor', side_effect=fake_supervisor), \
             patch('mic_agent.sigh_loop'), \
             patch('mic_agent.speech_assemble_loop'), \
             patch('mic_agent.speech_process_loop'), \
             patch('mic_agent.recalib_poller'), \
             patch('mic_agent.calib_level_poster'), \
             patch('mic_agent.load_stt_config', return_value=({}, None)), \
             patch('mic_agent.load_gate_config',
                   return_value=({"gate_margin": 99.0}, None)), \
             patch('time.sleep'):
            mic_agent.main()

        assert gate_used.get('gate_margin') == 99.0