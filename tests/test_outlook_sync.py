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
    

