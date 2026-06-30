import sys, os, json, tempfile
from datetime import datetime, timezone, timedelta, date
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestToNaiveKst:
    """R04: _to_naive_kst — datetime/date → KST naive 변환 테스트"""

    def test_R04_utc_datetime_converted_to_kst(self):
        """UTC datetime → KST +9시간으로 변환"""
        from outlook_sync import _to_naive_kst
        utc_dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _to_naive_kst(utc_dt)
        assert result == datetime(2025, 1, 1, 9, 0, 0)

    def test_R04_naive_datetime_unchanged(self):
        """timezone 없는 datetime → 그대로 반환"""
        from outlook_sync import _to_naive_kst
        naive_dt = datetime(2025, 6, 15, 10, 30, 0)
        result = _to_naive_kst(naive_dt)
        assert result == naive_dt

    def test_R04_date_converts_to_midnight(self):
        """date → 자정(00:00:00) datetime으로 변환"""
        from outlook_sync import _to_naive_kst
        d = date(2025, 6, 15)
        result = _to_naive_kst(d)
        assert result == datetime(2025, 6, 15, 0, 0, 0)

    def test_R04_result_has_no_timezone(self):
        """반환값에 timezone 정보 없음"""
        from outlook_sync import _to_naive_kst
        utc_dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _to_naive_kst(utc_dt)
        assert result.tzinfo is None

    def test_R04_kst_plus9_offset(self):
        """KST +09:00 datetime → 변환 후 동일 시각"""
        from outlook_sync import _to_naive_kst
        kst = timezone(timedelta(hours=9))
        kst_dt = datetime(2025, 6, 15, 10, 0, 0, tzinfo=kst)
        result = _to_naive_kst(kst_dt)
        assert result == datetime(2025, 6, 15, 10, 0, 0)


class TestParseIcsDatetime:
    """R04: _parse_ics_datetime — ICS 날짜 파싱 테스트"""

    def test_R04_date_only_8chars(self):
        """YYYYMMDD → 날짜 datetime"""
        from outlook_sync import _parse_ics_datetime
        result = _parse_ics_datetime("20250615")
        assert result == datetime(2025, 6, 15)

    def test_R04_utc_z_suffix(self):
        """YYYYMMDDTHHMMSSz → KST +9시간"""
        from outlook_sync import _parse_ics_datetime
        result = _parse_ics_datetime("20250615T000000Z")
        assert result == datetime(2025, 6, 15, 9, 0, 0)

    def test_R04_local_datetime(self):
        """YYYYMMDDTHHMMss (timezone 없음) → 그대로"""
        from outlook_sync import _parse_ics_datetime
        result = _parse_ics_datetime("20250615T143000")
        assert result == datetime(2025, 6, 15, 14, 30, 0)

    def test_R04_invalid_returns_none(self):
        """잘못된 형식 → None 반환"""
        from outlook_sync import _parse_ics_datetime
        result = _parse_ics_datetime("invalid")
        assert result is None


class TestParseIcsManual:
    """R04: _parse_ics_manual — ICS 직접 파싱 테스트"""

    def _sample_ics(self):
        return (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\n"
            "SUMMARY:팀 회의\r\n"
            "DTSTART:20250615T100000\r\n"
            "DESCRIPTION:주간 스프린트 회의\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

    def test_R04_parses_one_event(self):
        """이벤트 1개 파싱"""
        from outlook_sync import _parse_ics_manual
        events = _parse_ics_manual(self._sample_ics())
        assert len(events) == 1

    def test_R04_parses_summary(self):
        """SUMMARY 정상 파싱"""
        from outlook_sync import _parse_ics_manual
        events = _parse_ics_manual(self._sample_ics())
        assert events[0]['summary'] == '팀 회의'

    def test_R04_parses_datetime(self):
        """DTSTART 날짜 파싱"""
        from outlook_sync import _parse_ics_manual
        events = _parse_ics_manual(self._sample_ics())
        assert events[0]['dt'] == datetime(2025, 6, 15, 10, 0, 0)

    def test_R04_parses_description(self):
        """DESCRIPTION 정상 파싱"""
        from outlook_sync import _parse_ics_manual
        events = _parse_ics_manual(self._sample_ics())
        assert '스프린트' in events[0].get('description', '')

    def test_R04_empty_ics_returns_empty_list(self):
        """빈 ICS → 빈 리스트"""
        from outlook_sync import _parse_ics_manual
        events = _parse_ics_manual("BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n")
        assert events == []

    def test_R04_event_without_dtstart_skipped(self):
        """DTSTART 없는 이벤트 → 무시"""
        from outlook_sync import _parse_ics_manual
        ics = (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\n"
            "SUMMARY:날짜없는 이벤트\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        events = _parse_ics_manual(ics)
        assert len(events) == 0

    def test_R04_multiple_events(self):
        """여러 이벤트 파싱"""
        from outlook_sync import _parse_ics_manual
        ics = (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\nSUMMARY:이벤트1\r\nDTSTART:20250615T100000\r\nEND:VEVENT\r\n"
            "BEGIN:VEVENT\r\nSUMMARY:이벤트2\r\nDTSTART:20250616T140000\r\nEND:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        events = _parse_ics_manual(ics)
        assert len(events) == 2


class TestReadWriteCalendar:
    """R04: _read_calendar / _write_calendar 테스트"""

    def test_R04_write_and_read_roundtrip(self, tmp_path, monkeypatch):
        """쓰기 → 읽기 라운드트립"""
        import outlook_sync
        test_file = str(tmp_path / 'calendar_data.json')
        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', test_file)

        data = [{'id': '1', 'date': '2025-06-15', 'title': '테스트'}]
        outlook_sync._write_calendar(data)
        result = outlook_sync._read_calendar()
        assert result == data

    def test_R04_read_nonexistent_returns_empty_list(self, tmp_path, monkeypatch):
        """파일 없으면 빈 리스트 반환"""
        import outlook_sync
        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', str(tmp_path / 'nofile.json'))
        result = outlook_sync._read_calendar()
        assert result == []


class TestGetSyncStatus:
    """R04: get_sync_status 테스트"""

    def test_R04_returns_dict(self):
        """dict 타입 반환 확인"""
        from outlook_sync import get_sync_status
        result = get_sync_status()
        assert isinstance(result, dict)

    def test_R04_has_required_keys(self):
        """ok, message, ts, count 키 존재"""
        from outlook_sync import get_sync_status
        result = get_sync_status()
        for key in ['ok', 'message', 'ts', 'count']:
            assert key in result, f"'{key}' 키 없음"


class TestSyncNowNoUrl:
    """R04: sync_now — ics_url 미설정 시 동작"""

    def test_R04_sync_now_no_url_returns_false(self, tmp_path, monkeypatch):
        """ics_url 없으면 (False, 메시지) 반환"""
        import outlook_sync
        empty_cfg = {"ics_url": "", "sync_days_ahead": 30, "sync_days_past": 0}
        monkeypatch.setattr(outlook_sync, '_load_config', lambda: empty_cfg)

        ok, msg = outlook_sync.sync_now()
        assert ok is False
        assert 'ics_url' in msg


class TestLoadConfig:
    """R04: _load_config 테스트"""

    def test_R04_load_config_returns_dict(self):
        """_load_config → dict 반환"""
        from outlook_sync import _load_config
        cfg = _load_config()
        assert isinstance(cfg, dict)

    def test_R04_load_config_has_ics_url_key(self):
        """_load_config → ics_url 키 존재"""
        from outlook_sync import _load_config
        cfg = _load_config()
        assert 'ics_url' in cfg

    def test_R04_load_config_has_sync_days_ahead(self):
        """_load_config → sync_days_ahead 키 존재"""
        from outlook_sync import _load_config
        cfg = _load_config()
        assert 'sync_days_ahead' in cfg

    def test_R04_load_config_has_sync_days_past(self):
        """_load_config → sync_days_past 키 존재"""
        from outlook_sync import _load_config
        cfg = _load_config()
        assert 'sync_days_past' in cfg

    def test_R04_sync_days_ahead_positive(self):
        """sync_days_ahead >= 0"""
        from outlook_sync import _load_config
        assert _load_config()['sync_days_ahead'] >= 0

    def test_R04_custom_config_file(self, tmp_path, monkeypatch):
        """커스텀 config 파일 로딩"""
        import json, outlook_sync
        cfg_file = tmp_path / 'outlook_config.json'
        cfg_file.write_text(json.dumps({
            "ics_url": "https://example.com/cal.ics",
            "sync_days_ahead": 14,
            "sync_days_past": 7
        }))
        monkeypatch.setattr(outlook_sync, '_CONFIG_FILE', str(cfg_file))
        cfg = outlook_sync._load_config()
        assert cfg['ics_url'] == "https://example.com/cal.ics"
        assert cfg['sync_days_ahead'] == 14


class TestSyncNowWithMock:
    """R04: sync_now — HTTP 모킹 테스트"""

    def _make_ics(self):
        return (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\n"
            "SUMMARY:결재 회의\r\n"
            "DTSTART:20250615T100000\r\n"
            "DTEND:20250615T110000\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        ).encode('utf-8')

    def test_R04_sync_now_success_returns_true(self, tmp_path, monkeypatch):
        """유효한 ICS → (True, 메시지) 반환"""
        import outlook_sync
        from unittest.mock import patch, MagicMock

        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', str(tmp_path / 'cal.json'))
        mock_cfg = {"ics_url": "https://example.com/cal.ics", "sync_days_ahead": 30, "sync_days_past": 0}
        monkeypatch.setattr(outlook_sync, '_load_config', lambda: mock_cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = self._make_ics()

        with patch('requests.get', return_value=mock_resp):
            ok, msg = outlook_sync.sync_now()
        assert ok is True

    def test_R04_sync_now_updates_calendar_file(self, tmp_path, monkeypatch):
        """sync_now 성공 → calendar_data.json 파일 생성"""
        import outlook_sync, os
        from unittest.mock import patch, MagicMock

        cal_file = str(tmp_path / 'cal.json')
        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', cal_file)
        mock_cfg = {"ics_url": "https://example.com/cal.ics", "sync_days_ahead": 30, "sync_days_past": 0}
        monkeypatch.setattr(outlook_sync, '_load_config', lambda: mock_cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = self._make_ics()

        with patch('requests.get', return_value=mock_resp):
            outlook_sync.sync_now()
        assert os.path.exists(cal_file)

    def test_R04_sync_now_http_error_returns_false(self, tmp_path, monkeypatch):
        """HTTP 오류 → (False, 메시지) 반환"""
        import outlook_sync
        from unittest.mock import patch

        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', str(tmp_path / 'cal.json'))
        mock_cfg = {"ics_url": "https://example.com/cal.ics", "sync_days_ahead": 30, "sync_days_past": 0}
        monkeypatch.setattr(outlook_sync, '_load_config', lambda: mock_cfg)

        with patch('requests.get', side_effect=Exception("connection error")):
            ok, msg = outlook_sync.sync_now()
        assert ok is False

    def test_R04_sync_status_updated_after_sync(self, tmp_path, monkeypatch):
        """sync_now 후 get_sync_status 갱신 확인"""
        import outlook_sync
        from unittest.mock import patch, MagicMock

        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', str(tmp_path / 'cal.json'))
        mock_cfg = {"ics_url": "https://example.com/cal.ics", "sync_days_ahead": 30, "sync_days_past": 0}
        monkeypatch.setattr(outlook_sync, '_load_config', lambda: mock_cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = self._make_ics()

        with patch('requests.get', return_value=mock_resp):
            outlook_sync.sync_now()

        status = outlook_sync.get_sync_status()
        assert status['ts'] > 0


# ─────────────────────────────────────────────────────────────
# 추가: 커버리지 보강 (config 생성 / lib 파서 / 필터 / 예외 / auto-sync)
# ─────────────────────────────────────────────────────────────
class TestLoadConfigCreate:
    """R04: 설정 파일이 없으면 기본값으로 생성"""

    def test_R04_creates_default_config_when_missing(self, tmp_path, monkeypatch):
        import outlook_sync, os
        cfg_file = str(tmp_path / 'outlook_config.json')
        monkeypatch.setattr(outlook_sync, '_CONFIG_FILE', cfg_file)
        cfg = outlook_sync._load_config()
        assert os.path.exists(cfg_file)
        assert cfg['ics_url'] == ''
        assert 'sync_days_ahead' in cfg


class TestToNaiveKstNone:
    """R04: _to_naive_kst — datetime/date 가 아니면 None"""

    def test_R04_non_date_returns_none(self):
        from outlook_sync import _to_naive_kst
        assert _to_naive_kst("2025-01-01") is None
        assert _to_naive_kst(None) is None


class TestParseIcsWithLib:
    """R04: icalendar 라이브러리 파서 경로"""

    def test_R04_lib_parses_event(self):
        from outlook_sync import _parse_ics_with_lib
        raw = (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\nSUMMARY:회의\r\nDTSTART:20250615T100000\r\nEND:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        ).encode('utf-8')
        events = _parse_ics_with_lib(raw)
        assert len(events) == 1
        assert events[0]['summary'] == '회의'

    def test_R04_lib_skips_event_without_dtstart(self):
        """DTSTART 없는 VEVENT 는 건너뜀"""
        from outlook_sync import _parse_ics_with_lib
        raw = (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\nSUMMARY:DTSTART없음\r\nEND:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        ).encode('utf-8')
        assert _parse_ics_with_lib(raw) == []

    def test_R04_lib_skips_when_dt_unparseable(self, monkeypatch):
        """DTSTART 는 있으나 _to_naive_kst 가 None → 건너뜀 (83)"""
        import outlook_sync
        monkeypatch.setattr(outlook_sync, '_to_naive_kst', lambda v: None)
        raw = (
            "BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:x\r\n"
            "DTSTART:20250615T100000\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n"
        ).encode('utf-8')
        assert outlook_sync._parse_ics_with_lib(raw) == []

    def test_R04_dispatch_uses_lib_when_available(self):
        """_parse_ics 디스패처 (icalendar 설치 시 lib 경로)"""
        from outlook_sync import _parse_ics
        raw = (
            "BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:x\r\n"
            "DTSTART:20250615T100000\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n"
        ).encode('utf-8')
        events = _parse_ics(raw)
        assert len(events) == 1

    def test_R04_dispatch_falls_back_to_manual_without_icalendar(self, monkeypatch):
        """icalendar ImportError → 직접 파서 폴백 (141-142)"""
        import outlook_sync, sys
        monkeypatch.setitem(sys.modules, 'icalendar', None)  # import 시 ImportError
        raw = (
            "BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:수동\r\n"
            "DTSTART:20250615T100000\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n"
        ).encode('utf-8')
        events = outlook_sync._parse_ics(raw)
        assert len(events) == 1
        assert events[0]['summary'] == '수동'


class TestParseIcsManualEdge:
    """R04: 직접 파서 엣지"""

    def test_R04_malformed_line_skipped(self):
        from outlook_sync import _parse_ics_manual
        raw = (
            "BEGIN:VEVENT\n"
            "이건콜론없는잘못된줄\n"          # 정규식 불일치 → 건너뜀
            "UID:abc-123\n"                  # SUMMARY/DESC/DTSTART 아님 → 무시 (132->114)
            "SUMMARY:정상\n"
            "DTSTART:20250615T100000\n"
            "END:VEVENT\n"
        )
        events = _parse_ics_manual(raw)
        assert len(events) == 1
        assert events[0]['summary'] == '정상'


class TestReadCalendarCorrupt:
    """R04: _read_calendar — 손상 파일은 빈 리스트"""

    def test_R04_corrupt_file_returns_empty(self, tmp_path, monkeypatch):
        import outlook_sync
        p = tmp_path / 'cal.json'
        p.write_text('{깨진 json', encoding='utf-8')
        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', str(p))
        assert outlook_sync._read_calendar() == []


class TestSyncNowFilteringAndErrors:
    """R04: sync_now 날짜 필터 / 예외 분기"""

    def _cfg(self):
        return {"ics_url": "https://x/cal.ics", "sync_days_ahead": 30, "sync_days_past": 0}

    def test_R04_filters_none_and_out_of_range(self, tmp_path, monkeypatch):
        import outlook_sync
        from unittest.mock import patch, MagicMock
        from datetime import datetime, timedelta
        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', str(tmp_path / 'c.json'))
        monkeypatch.setattr(outlook_sync, '_load_config', self._cfg)
        far = datetime.now() + timedelta(days=999)
        monkeypatch.setattr(outlook_sync, '_parse_ics',
                            lambda raw: [{'dt': None, 'summary': 'a', 'description': ''},
                                         {'dt': far, 'summary': 'b', 'description': ''}])
        resp = MagicMock(); resp.content = b'x'
        with patch('requests.get', return_value=resp):
            ok, msg = outlook_sync.sync_now()
        assert ok is True
        assert '0개' in msg  # 둘 다 필터링됨

    def test_R04_request_exception_returns_false(self, tmp_path, monkeypatch):
        import outlook_sync, requests
        from unittest.mock import patch
        monkeypatch.setattr(outlook_sync, '_CALENDAR_FILE', str(tmp_path / 'c.json'))
        monkeypatch.setattr(outlook_sync, '_load_config', self._cfg)
        with patch('requests.get',
                   side_effect=requests.exceptions.ConnectionError("net down")):
            ok, msg = outlook_sync.sync_now()
        assert ok is False
        assert '네트워크 오류' in msg


class TestStartAutoSync:
    """R04: start_auto_sync — 백그라운드 스레드 시작 + 즉시 1회 동기화"""

    def test_R04_starts_thread_and_first_sync(self, monkeypatch):
        import outlook_sync
        from unittest.mock import MagicMock

        synced = {'n': 0}
        # _loop 내부 sync_now 가 무한루프 돌지 않도록, Thread 를 즉시 target 실행 대신
        # 호출만 기록하는 mock 으로 대체
        started = {'n': 0}
        class _FakeThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target
            def start(self):
                started['n'] += 1
        monkeypatch.setattr(outlook_sync.threading, 'Thread', _FakeThread)
        outlook_sync.start_auto_sync(interval_minutes=5)
        assert started['n'] == 1

    def test_R04_loop_body_runs_initial_and_periodic_sync(self, monkeypatch):
        """_loop 내부: 즉시 1회 + 주기 동기화 (248-251) — sleep 으로 루프 탈출"""
        import outlook_sync

        captured = {}
        class _FakeThread:
            def __init__(self, target=None, name=None, daemon=None):
                captured['target'] = target
            def start(self):
                pass
        monkeypatch.setattr(outlook_sync.threading, 'Thread', _FakeThread)

        calls = {'sync': 0}
        monkeypatch.setattr(outlook_sync, 'sync_now',
                            lambda: calls.__setitem__('sync', calls['sync'] + 1))

        class _Break(Exception):
            pass
        sleeps = {'n': 0}
        def _sleep(sec):
            sleeps['n'] += 1
            if sleeps['n'] >= 2:
                raise _Break()  # 두 번째 sleep 에서 탈출 → 주기 sync_now(251) 1회 실행됨
        monkeypatch.setattr(outlook_sync.time, 'sleep', _sleep)

        outlook_sync.start_auto_sync(interval_minutes=1)
        with __import__('pytest').raises(_Break):
            captured['target']()   # _loop 직접 실행
        # 즉시 1회(248) + 주기 1회(251) = 최소 2회
        assert calls['sync'] >= 2


