import sys, os, time, threading
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ─────────────────────────────────────────────────────────────
# TimedBuffer
# ─────────────────────────────────────────────────────────────
class TestTimedBuffer:
    """R01: TimedBuffer — 타임스탬프 버퍼 테스트"""

    def _buf(self):
        from radar_signal_processor import TimedBuffer
        return TimedBuffer()

    def test_R01_initial_length_zero(self):
        """초기 버퍼 길이 0"""
        assert len(self._buf()) == 0

    def test_R01_add_increases_length(self):
        """add 후 길이 1 증가"""
        b = self._buf()
        b.add("분노")
        assert len(b) == 1

    def test_R01_add_multiple(self):
        """여러 항목 추가"""
        b = self._buf()
        for i in range(5):
            b.add(i)
        assert len(b) == 5

    def test_R01_get_window_returns_recent(self):
        """최근 N초 이내 항목만 반환"""
        b = self._buf()
        now = time.time()
        b.add("오래된", now - 100)
        b.add("최근", now - 5)
        result = b.get_window(10)
        values = [v for _, v in result]
        assert "최근" in values
        assert "오래된" not in values

    def test_R01_get_window_empty_if_all_old(self):
        """모두 오래된 항목 → 빈 리스트"""
        b = self._buf()
        b.add("오래된", time.time() - 200)
        assert b.get_window(10) == []

    def test_R01_tail_returns_last_n(self):
        """tail(n)이 마지막 n개 반환"""
        b = self._buf()
        now = time.time()
        for i in range(10):
            b.add(i, now + i)
        tail = b.tail(3)
        assert [v for _, v in tail] == [7, 8, 9]

    def test_R01_first_ts_returns_oldest(self):
        """first_ts()가 가장 오래된 타임스탬프 반환"""
        b = self._buf()
        now = time.time()
        b.add("first", now - 100)
        b.add("second", now - 50)
        assert abs(b.first_ts() - (now - 100)) < 0.01

    def test_R01_first_ts_empty_returns_none(self):
        """빈 버퍼 first_ts() → None"""
        assert self._buf().first_ts() is None

    def test_R01_thread_safe_concurrent_add(self):
        """멀티스레드 동시 add 안전성"""
        from radar_signal_processor import TimedBuffer
        b = TimedBuffer()
        threads = [threading.Thread(target=lambda: b.add(1)) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(b) == 100


# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
class TestConstants:
    """R01: 모듈 상수 검증"""

    def test_R01_windows_keys(self):
        """WINDOWS 키: 1m, 5m, 10m"""
        from radar_signal_processor import WINDOWS
        assert set(WINDOWS.keys()) == {"1m", "5m", "10m"}

    def test_R01_windows_values(self):
        """WINDOWS 값(초): 60, 300, 600"""
        from radar_signal_processor import WINDOWS
        assert WINDOWS == {"1m": 60, "5m": 300, "10m": 600}

    def test_R01_default_weights_config_keys(self):
        """DEFAULT_WEIGHTS_CONFIG 필수 키 존재 및 기본값 확인"""
        from radar_signal_processor import DEFAULT_WEIGHTS_CONFIG
        for key in ("emotion_weights", "min_conf", "happy_gate",
                    "w_face", "w_speech", "anger_hard_th", "red_th", "green_th"):
            assert key in DEFAULT_WEIGHTS_CONFIG
        assert "기쁨" in DEFAULT_WEIGHTS_CONFIG["emotion_weights"]
        assert DEFAULT_WEIGHTS_CONFIG["red_th"] > DEFAULT_WEIGHTS_CONFIG["green_th"]

    def test_R01_positive_emotions_only_joy(self):
        """긍정 감정은 기쁨 1종"""
        from radar_signal_processor import POSITIVE_EMOTIONS
        assert POSITIVE_EMOTIONS == {"기쁨"}

    def test_R01_negative_emotions_5_classes(self):
        """부정 감정 5종 확인"""
        from radar_signal_processor import NEGATIVE_EMOTIONS
        for e in ["슬픔", "분노", "불안", "당황", "상처"]:
            assert e in NEGATIVE_EMOTIONS

    def test_R01_valid_speech_emotions_6_classes(self):
        """발화 감정 총 6종"""
        from radar_signal_processor import VALID_SPEECH_EMOTIONS
        assert len(VALID_SPEECH_EMOTIONS) == 6

    def test_R01_anger_ratio_threshold(self):
        """분노 임계값 0.30"""
        from radar_signal_processor import ANGER_RATIO_TH
        assert ANGER_RATIO_TH == 0.30

    def test_R01_happy_ratio_threshold(self):
        """행복 임계값 0.50"""
        from radar_signal_processor import HAPPY_RATIO_TH
        assert HAPPY_RATIO_TH == 0.50


# ─────────────────────────────────────────────────────────────
# 입력 메서드
# ─────────────────────────────────────────────────────────────
class TestInputMethods:
    """R01: add_expression / add_sigh / add_speech_emotion 테스트"""

    def _proc(self):
        from radar_signal_processor import RadarSignalProcessor
        return RadarSignalProcessor()

    def test_R01_add_expression_normal(self):
        """'일반' 표정 추가"""
        p = self._proc()
        p.add_expression("일반")
        assert len(p.expression_buffer) == 1

    def test_R01_add_expression_normalizes_보통(self):
        """'보통' → '일반'으로 정규화"""
        p = self._proc()
        p.add_expression("보통")
        items = p.expression_buffer.get_window(10)
        assert items[0][1] == "일반"

    def test_R01_add_expression_normalizes_중립(self):
        """'중립' → '일반'으로 정규화"""
        p = self._proc()
        p.add_expression("중립")
        items = p.expression_buffer.get_window(10)
        assert items[0][1] == "일반"

    def test_R01_add_expression_normalizes_neutral(self):
        """'neutral' → '일반'으로 정규화"""
        p = self._proc()
        p.add_expression("neutral")
        items = p.expression_buffer.get_window(10)
        assert items[0][1] == "일반"

    def test_R01_add_sigh_true(self):
        """한숨 감지(True) 추가"""
        p = self._proc()
        p.add_sigh(True)
        assert len(p.sigh_buffer) == 1

    def test_R01_add_sigh_false_heartbeat(self):
        """한숨 미감지(False) heartbeat도 버퍼에 추가됨"""
        p = self._proc()
        p.add_sigh(False)
        assert len(p.sigh_buffer) == 1

    def test_R01_add_speech_emotion_valid(self):
        """유효한 발화 감정 추가"""
        p = self._proc()
        for e in ["기쁨", "슬픔", "분노", "불안", "당황", "상처"]:
            p.add_speech_emotion(e)
        assert len(p.speech_buffer) == 6

    def test_R01_add_speech_emotion_invalid_raises(self):
        """유효하지 않은 발화 감정 → ValueError"""
        p = self._proc()
        with pytest.raises(ValueError):
            p.add_speech_emotion("무효감정")

    def test_R01_mark_sigh_alive_updates_ts(self):
        """mark_sigh_alive 호출 후 alive 상태 갱신"""
        p = self._proc()
        p.mark_sigh_alive()
        alive = p.device_alive(timeout=5.0)
        assert alive["sigh"] is True

    def test_R01_mark_speech_alive_updates_ts(self):
        """mark_speech_alive 호출 후 alive 상태 갱신"""
        p = self._proc()
        p.mark_speech_alive()
        alive = p.device_alive(timeout=5.0)
        assert alive["speech"] is True

    def test_R01_device_alive_false_without_input(self):
        """입력 없으면 alive=False"""
        p = self._proc()
        alive = p.device_alive(timeout=0.001)
        assert alive["sigh"] is False
        assert alive["speech"] is False


# ─────────────────────────────────────────────────────────────
# 표정 대표값
# ─────────────────────────────────────────────────────────────
class TestRepresentativeExpr:
    """R01: _representative_expr 표정 대표값 계산"""

    def _proc(self):
        from radar_signal_processor import RadarSignalProcessor
        return RadarSignalProcessor()

    def test_R01_empty_returns_불명(self):
        """샘플 없으면 '불명'"""
        assert self._proc()._representative_expr([]) == "불명"

    def test_R01_anger_ratio_30pct_returns_분노(self):
        """분노 비율 30% → '분노'"""
        samples = ["분노"] * 3 + ["일반"] * 7
        assert self._proc()._representative_expr(samples) == "분노"

    def test_R01_anger_below_30pct_not_분노(self):
        """분노 비율 29% → '분노' 아님"""
        samples = ["분노"] * 2 + ["일반"] * 5  # 2/7 ≈ 28.6%
        result = self._proc()._representative_expr(samples)
        assert result != "분노"

    def test_R01_happy_ratio_50pct_returns_행복(self):
        """행복 비율 50% → '행복'"""
        samples = ["행복"] * 5 + ["일반"] * 5
        assert self._proc()._representative_expr(samples) == "행복"

    def test_R01_happy_below_50pct_not_행복(self):
        """행복 비율 49% → '행복' 아님"""
        samples = ["행복"] * 4 + ["일반"] * 5  # 4/9 ≈ 44.4%
        result = self._proc()._representative_expr(samples)
        assert result != "행복"

    def test_R01_all_normal_returns_일반(self):
        """전부 일반 → '일반'"""
        samples = ["일반"] * 10
        assert self._proc()._representative_expr(samples) == "일반"

    def test_R01_anger_priority_over_happy(self):
        """분노 비율 >= 0.30이면 행복보다 우선"""
        # 분노 30%, 행복 50% 이상 동시 → 분노 우선
        samples = ["분노"] * 3 + ["행복"] * 6 + ["일반"] * 1
        assert self._proc()._representative_expr(samples) == "분노"


# ─────────────────────────────────────────────────────────────
# 신호 판정 우선순위
# ─────────────────────────────────────────────────────────────
class TestSignalPriority:
    """R01: _decide_window 신호 판정 우선순위 테스트"""

    def _proc_with_data(self, expressions=None, sigh_detected=False, neg_emotions=0):
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        now = time.time()

        exprs = expressions or ["일반"] * 10
        # ✅ now-50 ~ now-41: 1분 윈도우(60초) 이내에 전부 포함
        for i, e in enumerate(exprs):
            p.add_expression(e, now - 50 + i)

        if sigh_detected:
            p.add_sigh(True, now - 5)

        for _ in range(neg_emotions):
            p.add_speech_emotion("슬픔", now - 5)

        return p

    def test_R01_rule1_sigh_always_red(self):
        """규칙 1: 한숨 발생 → 빨강 (최우선)"""
        p = self._proc_with_data(
            expressions=["행복"] * 10,  # 행복이어도
            sigh_detected=True
        )
        result = p._decide_window("1m")
        assert result["signal"] == "red"

    def test_R01_rule2_anger_red(self):
        """규칙 2: 표정 분노 → 빨강"""
        p = self._proc_with_data(expressions=["분노"] * 10)
        result = p._decide_window("1m")
        assert result["signal"] == "red"

    def test_R01_rule2_happy_green(self):
        """규칙 2: 표정 행복 → 초록"""
        p = self._proc_with_data(expressions=["행복"] * 10)
        result = p._decide_window("1m")
        assert result["signal"] == "green"

    def test_R01_rule3_score_based_red(self):
        """규칙 3: 위험점수 R >= red_th → 빨강 (score reason)
        분노 50%(anger_hard_th=60% 미만) + 발화 없음 → R=face_risk=0.5=red_th → red"""
        p = self._proc_with_data(
            expressions=["분노"] * 5 + ["일반"] * 5,
        )
        result = p._decide_window("1m")
        assert result["signal"] == "red"
        assert result["reason"] == "score"

    def test_R01_rule4_default_yellow(self):
        """규칙 4: 기본값 → 노랑"""
        p = self._proc_with_data(expressions=["일반"] * 10)
        result = p._decide_window("1m")
        assert result["signal"] == "yellow"

    def test_R01_sigh_overrides_happy(self):
        """한숨은 행복 표정보다 우선 → 빨강"""
        p = self._proc_with_data(
            expressions=["행복"] * 10,
            sigh_detected=True
        )
        assert p._decide_window("1m")["signal"] == "red"

    def test_R01_decide_window_returns_required_keys(self):
        """_decide_window 반환값에 필수 키 존재"""
        p = self._proc_with_data()
        result = p._decide_window("1m")
        for key in ["signal", "expr", "sigh", "neg_count", "ready", "data_points"]:
            assert key in result, f"'{key}' 키 없음"

    def test_R01_decide_window_data_points_keys(self):
        """data_points에 expression/sigh/speech 키 존재"""
        p = self._proc_with_data()
        dp = p._decide_window("1m")["data_points"]
        for key in ["expression", "sigh", "speech"]:
            assert key in dp


# ─────────────────────────────────────────────────────────────
# get_all_signals / get_status / get_recent_data
# ─────────────────────────────────────────────────────────────
class TestPublicAPIs:
    """R01: 공개 API 테스트"""

    def _proc(self):
        from radar_signal_processor import RadarSignalProcessor
        return RadarSignalProcessor()

    def test_R01_get_all_signals_has_3_windows(self):
        """get_all_signals → 1m/5m/10m 키 포함"""
        p = self._proc()
        result = p.get_all_signals()
        for key in ["1m", "5m", "10m"]:
            assert key in result

    def test_R01_get_all_signals_has_updated_at(self):
        """get_all_signals → updated_at 포함"""
        p = self._proc()
        assert "updated_at" in p.get_all_signals()

    def test_R01_get_status_has_required_keys(self):
        """get_status → buffer_sizes/is_ready/uptime_sec 포함"""
        p = self._proc()
        status = p.get_status()
        for key in ["buffer_sizes", "is_ready", "uptime_sec"]:
            assert key in status

    def test_R01_get_status_is_ready_false_initially(self):
        """초기 is_ready=False"""
        assert self._proc().get_status()["is_ready"] is False

    def test_R01_get_status_is_ready_true_after_add(self):
        """표정 추가 후 is_ready=True"""
        p = self._proc()
        p.add_expression("일반")
        assert p.get_status()["is_ready"] is True

    def test_R01_get_status_uptime_positive(self):
        """uptime_sec >= 0"""
        assert self._proc().get_status()["uptime_sec"] >= 0

    def test_R01_get_recent_data_structure(self):
        """get_recent_data → expression/sigh/speech 키"""
        p = self._proc()
        data = p.get_recent_data()
        for key in ["expression", "sigh", "speech"]:
            assert key in data

    def test_R01_get_recent_data_returns_list(self):
        """get_recent_data 각 값은 리스트"""
        p = self._proc()
        data = p.get_recent_data()
        assert isinstance(data["expression"], list)

    def test_R01_get_recent_data_item_has_t_v(self):
        """get_recent_data 항목에 t, v 키 존재"""
        p = self._proc()
        p.add_expression("분노")
        items = p.get_recent_data(n=1)["expression"]
        assert len(items) == 1
        assert "t" in items[0] and "v" in items[0]


# ─────────────────────────────────────────────────────────────
# 싱글턴
# ─────────────────────────────────────────────────────────────
class TestSingleton:
    """R01: get_processor() 싱글턴 테스트"""

    def test_R01_get_processor_returns_instance(self):
        from radar_signal_processor import get_processor, RadarSignalProcessor
        assert isinstance(get_processor(), RadarSignalProcessor)

    def test_R01_get_processor_same_instance(self):
        from radar_signal_processor import get_processor
        assert get_processor() is get_processor()

    def test_R01_inner_lock_check_already_set(self, monkeypatch):
        """더블체크 락: 락 진입 시 이미 생성된 경우 → 기존 인스턴스 반환"""
        import radar_signal_processor

        existing = radar_signal_processor.RadarSignalProcessor()

        # 싱글턴 초기화
        monkeypatch.setattr(radar_signal_processor, '_processor', None)

        # 락 진입 순간 다른 스레드가 먼저 생성한 상황 시뮬레이션
        class MockLock:
            def __enter__(self):
                radar_signal_processor._processor = existing
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(radar_signal_processor, '_proc_lock', MockLock())
        result = radar_signal_processor.get_processor()
        assert result is existing
# ─────────────────────────────────────────────────────────────
class TestLoadWeightsConfig:
    """R01: load_weights_config 다양한 경로 브랜치 커버"""

    def test_R01_explicit_path_used(self, tmp_path):
        """explicit 경로로 파일 찾기 → cfg 반환 + 경로 확인"""
        import json as _json
        from radar_signal_processor import load_weights_config, DEFAULT_WEIGHTS_CONFIG
        cfg_file = tmp_path / "weights_config.json"
        cfg_file.write_text(_json.dumps({
            "emotion_weights": {"분노": 0.9, "기쁨": -0.5},
            "red_th": 0.4
        }), encoding="utf-8")
        cfg, path = load_weights_config(explicit=str(cfg_file))
        assert path == str(cfg_file)
        assert cfg["red_th"] == 0.4
        assert cfg["emotion_weights"]["분노"] == 0.9

    def test_R01_seen_dedup_continue(self, tmp_path, monkeypatch):
        """explicit == CWD 경로 → seen 중복 → continue 브랜치"""
        import json as _json
        from radar_signal_processor import load_weights_config
        # CWD를 tmp_path로 변경 (weights_config.json 없음)
        monkeypatch.chdir(tmp_path)
        # explicit = CWD path → 두 번째 같은 경로는 seen에서 continue
        cwd_path = str(tmp_path / "weights_config.json")
        # 파일 없어도 목적은 seen dedup 커버
        cfg, path = load_weights_config(explicit=cwd_path)
        # 경로 없으므로 None이거나 다른 경로
        assert cfg is not None

    def test_R01_no_emotion_weights_key_in_json(self, tmp_path, monkeypatch):
        """JSON에 emotion_weights 없음 → scalar 키만 반영"""
        import json as _json
        from radar_signal_processor import load_weights_config, DEFAULT_WEIGHTS_CONFIG
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "weights_config.json"
        cfg_file.write_text(_json.dumps({"red_th": 0.6}), encoding="utf-8")
        cfg, path = load_weights_config()
        # emotion_weights는 DEFAULT 유지, scalar만 덮어쓰기
        assert cfg["red_th"] == 0.6
        assert cfg["emotion_weights"] == DEFAULT_WEIGHTS_CONFIG["emotion_weights"]

    def test_R01_null_scalar_skipped(self, tmp_path, monkeypatch):
        """null scalar 키 → 덮어쓰지 않음 (default 유지)"""
        import json as _json
        from radar_signal_processor import load_weights_config, DEFAULT_WEIGHTS_CONFIG
        monkeypatch.chdir(tmp_path)
        cfg_file = tmp_path / "weights_config.json"
        cfg_file.write_text(_json.dumps({"red_th": None, "green_th": -0.2}),
                            encoding="utf-8")
        cfg, path = load_weights_config()
        assert cfg["red_th"] == DEFAULT_WEIGHTS_CONFIG["red_th"]  # null → 유지
        assert cfg["green_th"] == -0.2  # non-null → 덮어쓰기

    def test_R01_bad_json_exception_fallback(self, tmp_path, monkeypatch):
        """bad JSON → exception catch → 다음 후보 or return cfg, None"""
        from radar_signal_processor import load_weights_config, DEFAULT_WEIGHTS_CONFIG
        monkeypatch.chdir(tmp_path)
        bad_file = tmp_path / "weights_config.json"
        bad_file.write_text("{ this is not json }", encoding="utf-8")
        # 예외 후 다른 후보도 없으면 default 반환
        cfg, path = load_weights_config()
        # path는 None (모든 후보 실패)이거나 src/ 것일 수 있음
        assert cfg is not None
        assert "emotion_weights" in cfg

    def test_R01_no_config_returns_default(self, tmp_path, monkeypatch):
        """어디서도 weights_config.json 없음 → return cfg, None"""
        import os
        from radar_signal_processor import load_weights_config, DEFAULT_WEIGHTS_CONFIG
        monkeypatch.chdir(tmp_path)
        with pytest.MonkeyPatch().context() as m:
            m.setattr("os.path.exists", lambda p: False)
            cfg, path = load_weights_config()
        assert path is None
        assert cfg["red_th"] == DEFAULT_WEIGHTS_CONFIG["red_th"]


# ─────────────────────────────────────────────────────────────
# speech_utterance_risk 세부 브랜치 커버
# ─────────────────────────────────────────────────────────────
class TestSpeechUtteranceRisk:
    """R01: speech_utterance_risk 함수 브랜치 커버"""

    def test_R01_default_weights_when_none(self):
        """weights=None → DEFAULT_WEIGHTS_CONFIG 사용"""
        from radar_signal_processor import speech_utterance_risk, DEFAULT_WEIGHTS_CONFIG
        r = speech_utterance_risk({"분노": 1.0}, top_prob=1.0, weights=None)
        expected = DEFAULT_WEIGHTS_CONFIG["emotion_weights"]["분노"]
        assert abs(r - expected) < 0.01

    def test_R01_top_prob_computed_from_probs(self):
        """top_prob=None → max(probs.values()) 자동 계산"""
        from radar_signal_processor import speech_utterance_risk
        # top_prob=None + probs non-empty → 내부에서 max 계산
        r = speech_utterance_risk({"분노": 0.9, "기쁨": 0.1}, top_prob=None)
        assert r > 0

    def test_R01_empty_probs_top_prob_zero(self):
        """top_prob=None + probs 빈 dict → top_prob=0.0"""
        from radar_signal_processor import speech_utterance_risk
        r = speech_utterance_risk({}, top_prob=None)
        assert r == 0.0

    def test_R01_low_conf_attenuation(self):
        """top_prob < min_conf → r 감쇠 (r *= top_prob/min_conf)"""
        from radar_signal_processor import speech_utterance_risk
        # top_prob=0.3 < min_conf=0.45 → 감쇠 적용
        r_attenuated = speech_utterance_risk({"분노": 1.0}, top_prob=0.3, min_conf=0.45)
        r_full = speech_utterance_risk({"분노": 1.0}, top_prob=1.0, min_conf=0.45)
        assert r_attenuated < r_full

    def test_R01_happy_gate_skips_joy(self):
        """기쁨 확신도 < happy_gate → 기쁨 기여 0"""
        from radar_signal_processor import speech_utterance_risk
        # 기쁨 0.5 < happy_gate=0.70 → 기쁨 가중치 미적용
        r_no_gate = speech_utterance_risk({"기쁨": 0.5}, top_prob=0.5, happy_gate=0.3)
        r_with_gate = speech_utterance_risk({"기쁨": 0.5}, top_prob=0.5, happy_gate=0.70)
        # gate 적용 시 기쁨 기여 없음 → r=0
        assert r_with_gate == 0.0

    def test_R01_happy_above_gate_contributes(self):
        """기쁨 확신도 >= happy_gate → 음수 기여"""
        from radar_signal_processor import speech_utterance_risk
        r = speech_utterance_risk({"기쁨": 0.9}, top_prob=0.9, happy_gate=0.70)
        assert r < 0  # 기쁨 weight = -0.6


# ─────────────────────────────────────────────────────────────
# RadarSignalProcessor 신규 기능 커버
# ─────────────────────────────────────────────────────────────
class TestRadarSignalProcessorNewFeatures:
    """R01: 신규 기능 (no-config print, reload, probs speech) 커버"""

    def test_R01_no_config_prints_default_message(self, capsys):
        """weights_config.json 없을 때 '기본 가중치' 메시지 출력"""
        from radar_signal_processor import RadarSignalProcessor
        from unittest.mock import patch
        with patch("os.path.exists", return_value=False):
            p = RadarSignalProcessor()
        out = capsys.readouterr().out
        assert "기본 가중치" in out

    def test_R01_reload_weights_returns_path(self, tmp_path):
        """reload_weights() → 경로 반환"""
        import json as _json
        from radar_signal_processor import RadarSignalProcessor
        cfg_file = tmp_path / "weights_config.json"
        cfg_file.write_text(_json.dumps({"red_th": 0.4}), encoding="utf-8")
        p = RadarSignalProcessor()
        path = p.reload_weights(weights_config=str(cfg_file))
        assert path == str(cfg_file)
        assert p.cfg["red_th"] == 0.4

    def test_R01_reload_weights_updates_cfg(self, tmp_path):
        """reload_weights() 후 cfg와 weights 갱신"""
        import json as _json
        from radar_signal_processor import RadarSignalProcessor
        cfg_file = tmp_path / "w.json"
        cfg_file.write_text(_json.dumps({
            "emotion_weights": {"분노": 2.0, "기쁨": -1.0,
                                "슬픔": 0.2, "불안": 0.7, "당황": 0.5, "상처": 0.3}
        }), encoding="utf-8")
        p = RadarSignalProcessor()
        p.reload_weights(weights_config=str(cfg_file))
        assert p.weights["분노"] == 2.0

    def test_R01_add_speech_with_probs_stores_risk(self):
        """add_speech_emotion(probs=...) → risk 계산 후 dict 저장"""
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        p.add_speech_emotion("분노",
                             probs={"분노": 0.8, "슬픔": 0.1, "기쁨": 0.05,
                                    "불안": 0.02, "당황": 0.02, "상처": 0.01},
                             top_prob=0.8)
        items = p.speech_buffer.get_window(10)
        assert len(items) == 1
        stored = items[0][1]
        assert isinstance(stored, dict)
        assert stored["emotion"] == "분노"
        assert "risk" in stored
        assert stored["risk"] > 0

    def test_R01_add_speech_explicit_risk_skips_calc(self):
        """add_speech_emotion(risk=0.5) → 계산 없이 저장"""
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        p.add_speech_emotion("슬픔", risk=0.5)
        items = p.speech_buffer.get_window(10)
        assert items[0][1]["risk"] == 0.5

    def test_R01_add_speech_top_prob_stored(self):
        """add_speech_emotion(top_prob=0.8) → top_prob dict에 저장"""
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        p.add_speech_emotion("분노",
                             probs={"분노": 0.8}, top_prob=0.8)
        stored = p.speech_buffer.get_window(10)[0][1]
        assert stored["top_prob"] == 0.8


# ─────────────────────────────────────────────────────────────
# _decide_window 추가 브랜치 커버
# ─────────────────────────────────────────────────────────────
class TestDecideWindowExtra:
    """R01: _decide_window 미커버 브랜치 (legacy string, R-score red) 커버"""

    def _proc_base(self, n_expr=10, emotion="일반"):
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        now = time.time()
        for i in range(n_expr):
            p.add_expression(emotion, now - 50 + i)
        return p, now

    def test_R01_positive_speech_no_neg_count(self):
        """기쁨 발화(dict) → neg_count 증가 없음 (291→288 브랜치)"""
        from radar_signal_processor import RadarSignalProcessor
        p, now = self._proc_base()
        p.add_speech_emotion("기쁨",
                             probs={"기쁨": 0.9}, top_prob=0.9,
                             timestamp=now - 5)
        result = p._decide_window("1m")
        assert result["neg_count"] == 0

    def test_R01_negative_speech_dict_increments_neg_count(self):
        """부정 발화(dict) → neg_count 증가 (line 292 브랜치)"""
        from radar_signal_processor import RadarSignalProcessor
        p, now = self._proc_base()
        p.add_speech_emotion("분노",
                             probs={"분노": 0.8}, top_prob=0.8,
                             timestamp=now - 5)
        result = p._decide_window("1m")
        assert result["neg_count"] == 1

    def test_R01_legacy_string_speech_in_buffer(self):
        """과도기: speech_buffer에 문자열 직접 저장 → 294-298 브랜치"""
        from radar_signal_processor import RadarSignalProcessor
        p, now = self._proc_base()
        # 구 버전 호환: 문자열 직접 삽입 (일반 API 우회)
        p.speech_buffer.add("슬픔", now - 5)
        result = p._decide_window("1m")
        assert result["neg_count"] >= 1

    def test_R01_legacy_positive_string_no_neg_count(self):
        """과도기: 기쁨 문자열 → neg_count 증가 없음"""
        from radar_signal_processor import RadarSignalProcessor
        p, now = self._proc_base()
        p.speech_buffer.add("기쁨", now - 5)
        result = p._decide_window("1m")
        assert result["neg_count"] == 0

    def test_R01_red_via_score_not_hard_rule(self):
        """R >= red_th 이면서 anger_hard_th 미만 → red reason=score (line 320)"""
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        now = time.time()
        # 분노 50% (< anger_hard_th=0.6) + 발화 없음 → face_risk=0.5, R=0.5=red_th
        for i in range(5):
            p.add_expression("분노", now - 50 + i)
        for i in range(5):
            p.add_expression("일반", now - 45 + i)
        result = p._decide_window("1m")
        assert result["signal"] == "red"
        assert result["reason"] == "score"

    def test_R01_green_via_score(self):
        """R <= green_th → green reason=score"""
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        now = time.time()
        # 행복 100% → face_risk = _clamp(0 - 1.0, -1, 1) = -1.0, R=-1.0 <= -0.3 → green
        for i in range(10):
            p.add_expression("행복", now - 50 + i)
        result = p._decide_window("1m")
        assert result["signal"] == "green"
        assert result["reason"] == "score"

    def test_R01_get_all_signals_has_thresholds(self):
        """get_all_signals → thresholds 키 포함"""
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        out = p.get_all_signals()
        assert "thresholds" in out
        for k in ("red_th", "green_th", "anger_hard_th", "w_face", "w_speech"):
            assert k in out["thresholds"]

    def test_R01_decide_window_new_keys(self):
        """_decide_window 반환값에 신규 키 포함"""
        from radar_signal_processor import RadarSignalProcessor
        p = RadarSignalProcessor()
        now = time.time()
        for i in range(10):
            p.add_expression("일반", now - 50 + i)
        result = p._decide_window("1m")
        for key in ("risk", "face_risk", "speech_risk", "reason", "ready"):
            assert key in result, f"'{key}' 키 없음"