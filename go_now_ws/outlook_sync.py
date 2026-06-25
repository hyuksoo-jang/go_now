#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outlook ICS → calendar_data.json 동기화 모듈

설정:
  outlook_config.json 의 ics_url 에 Outlook ICS 구독 URL 입력
  Outlook.com → 설정 → 캘린더 → 공유 캘린더 → ICS 링크 복사

의존: requests, (선택) icalendar
  pip install requests icalendar
"""

import os
import re
import json
import uuid
import time
import threading
import requests
from datetime import datetime, timezone, timedelta, date

_DIR           = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE   = os.path.join(_DIR, 'outlook_config.json')
_CALENDAR_FILE = os.path.join(_DIR, 'calendar_data.json')

_DEFAULT_CONFIG = {
    "ics_url":        "",   # Outlook ICS 구독 URL (필수)
    "sync_days_ahead": 30,  # 오늘부터 몇 일 후까지 저장할지
    "sync_days_past":  0,   # 오늘로부터 며칠 과거 이벤트까지 저장할지
}

_sync_lock   = threading.Lock()
_sync_status = {"ok": None, "message": "아직 동기화하지 않았습니다.", "ts": 0, "count": 0}


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

def _load_config():
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        for k, v in _DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    with open(_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(_DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    print('[Outlook] outlook_config.json 생성됨. ics_url을 입력하세요.')
    return dict(_DEFAULT_CONFIG)


# ─────────────────────────────────────────────────────────────
# ICS 파싱 (icalendar 라이브러리 우선, 없으면 직접 파싱)
# ─────────────────────────────────────────────────────────────

def _to_naive_kst(dt_val):
    """datetime 또는 date → KST naive datetime 으로 통일."""
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is not None:
            kst = timezone(timedelta(hours=9))
            dt_val = dt_val.astimezone(kst).replace(tzinfo=None)
        return dt_val
    if isinstance(dt_val, date):
        return datetime(dt_val.year, dt_val.month, dt_val.day, 0, 0, 0)
    return None


def _parse_ics_with_lib(raw_bytes):
    """icalendar 라이브러리로 파싱."""
    from icalendar import Calendar
    cal = Calendar.from_ical(raw_bytes)
    events = []
    for comp in cal.walk('VEVENT'):
        summary = str(comp.get('SUMMARY', '')).strip() or '(제목 없음)'
        desc    = str(comp.get('DESCRIPTION', '')).strip()[:120]
        dtstart = comp.get('DTSTART')
        if dtstart is None:
            continue
        dt = _to_naive_kst(dtstart.dt)
        if dt is None:
            continue
        events.append({'summary': summary, 'description': desc, 'dt': dt})
    return events


def _unescape_ics(val):
    return (val.replace('\\n', '\n')
               .replace('\\,', ',')
               .replace('\\;', ';')
               .replace('\\\\', '\\'))


def _parse_ics_datetime(val):
    """ICS 날짜/시간 문자열 → KST naive datetime."""
    val = val.strip()
    try:
        if len(val) == 8:
            return datetime.strptime(val, '%Y%m%d')
        elif val.endswith('Z'):
            dt = datetime.strptime(val, '%Y%m%dT%H%M%SZ')
            return dt + timedelta(hours=9)
        else:
            return datetime.strptime(val[:15], '%Y%m%dT%H%M%S')
    except Exception:
        return None


def _parse_ics_manual(raw_text):
    """icalendar 라이브러리 없을 때 직접 파싱 (RFC 5545 기본)."""
    raw_text = re.sub(r'\r?\n[ \t]', '', raw_text)  # line folding 해제
    events, in_event, cur = [], False, {}
    for line in raw_text.splitlines():
        line = line.strip()
        if line == 'BEGIN:VEVENT':
            in_event, cur = True, {}
        elif line == 'END:VEVENT':
            in_event = False
            if cur.get('dt'):
                events.append(cur)
            cur = {}
        elif in_event:
            m = re.match(r'^([A-Z\-]+)(?:;[^:]*)?:(.*)$', line)
            if not m:
                continue
            prop, val = m.group(1), _unescape_ics(m.group(2))
            if prop == 'SUMMARY':
                cur['summary'] = val.strip() or '(제목 없음)'
            elif prop == 'DESCRIPTION':
                cur['description'] = val.strip()[:120]
            elif prop.startswith('DTSTART'):
                cur['dt'] = _parse_ics_datetime(val)
    return events


def _parse_ics(raw_bytes):
    try:
        import icalendar  # noqa: F401
        return _parse_ics_with_lib(raw_bytes)
    except ImportError:
        return _parse_ics_manual(raw_bytes.decode('utf-8', errors='replace'))


# ─────────────────────────────────────────────────────────────
# 캘린더 파일 I/O
# ─────────────────────────────────────────────────────────────

def _read_calendar():
    if os.path.exists(_CALENDAR_FILE):
        try:
            with open(_CALENDAR_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _write_calendar(data):
    with open(_CALENDAR_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────

def sync_now():
    """
    Outlook ICS → calendar_data.json 동기화.
    반환: (성공여부: bool, 메시지: str)
    """
    global _sync_status

    with _sync_lock:
        try:
            cfg = _load_config()
            ics_url = cfg.get('ics_url', '').strip()
            if not ics_url:
                msg = 'outlook_config.json에 ics_url을 설정하세요.'
                _sync_status = {'ok': False, 'message': msg, 'ts': time.time(), 'count': 0}
                return False, msg

            # ICS 다운로드
            resp = requests.get(ics_url, timeout=20, headers={'User-Agent': 'go_now/1.0'})
            resp.raise_for_status()
            raw = resp.content

            # 파싱
            parsed = _parse_ics(raw)

            # 날짜 범위 필터
            today      = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            days_past  = int(cfg.get('sync_days_past', 0))
            days_ahead = int(cfg.get('sync_days_ahead', 30))
            cutoff_lo  = today - timedelta(days=days_past)
            cutoff_hi  = today + timedelta(days=days_ahead)

            new_items = []
            for ev in parsed:
                dt = ev.get('dt')
                if dt is None:
                    continue
                if not (cutoff_lo <= dt <= cutoff_hi):
                    continue
                new_items.append({
                    'id':       str(uuid.uuid4()),
                    'date':     dt.strftime('%Y-%m-%d'),
                    'time':     dt.strftime('%H:%M'),
                    'title':    ev.get('summary', '(제목 없음)'),
                    'note':     ev.get('description', ''),
                    '_outlook': True,
                })
            new_items.sort(key=lambda e: (e['date'], e['time']))

            # 수동 추가 항목 보존, Outlook 항목 교체
            existing = _read_calendar()
            manual   = [e for e in existing if not e.get('_outlook')]
            merged   = sorted(manual + new_items, key=lambda e: (e.get('date', ''), e.get('time', '')))
            _write_calendar(merged)

            msg = f'동기화 완료 ({len(new_items)}개 이벤트)'
            _sync_status = {'ok': True, 'message': msg, 'ts': time.time(), 'count': len(new_items)}
            print(f'[Outlook] {msg}')
            return True, msg

        except requests.exceptions.RequestException as exc:
            msg = f'네트워크 오류: {exc}'
            _sync_status = {'ok': False, 'message': msg, 'ts': time.time(), 'count': 0}
            print(f'[Outlook] {msg}')
            return False, msg
        except Exception as exc:
            msg = f'동기화 실패: {exc}'
            _sync_status = {'ok': False, 'message': msg, 'ts': time.time(), 'count': 0}
            print(f'[Outlook] {msg}')
            return False, msg


def get_sync_status():
    """최근 동기화 상태 반환."""
    return dict(_sync_status)


def start_auto_sync(interval_minutes=10):
    """백그라운드 자동 동기화 스레드 시작 (프로세스 시작 즉시 첫 동기화 수행)."""
    def _loop():
        sync_now()   # 시작 즉시 첫 동기화
        while True:
            time.sleep(interval_minutes * 60)
            sync_now()

    t = threading.Thread(target=_loop, name='outlook-auto-sync', daemon=True)
    t.start()
    print(f"[Outlook] 자동 동기화 시작 ({interval_minutes}분 간격, 즉시 첫 실행)")
