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

    def test_R01_neg_threshold_values(self):
        """NEG_THRESHOLD: 1m→1, 5m→2, 10m→3"""
        from radar_signal_processor import NEG_THRESHOLD
        assert NEG_THRESHOLD == {"1m": 1, "5m": 2, "10m": 3}

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

    def test_R01_rule3_neg_speech_1m_red(self):
        """규칙 3: 1분 구간 부정 발화 1회 이상 → 빨강"""
        p = self._proc_with_data(
            expressions=["일반"] * 10,
            neg_emotions=1
        )
        result = p._decide_window("1m")
        assert result["signal"] == "red"

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