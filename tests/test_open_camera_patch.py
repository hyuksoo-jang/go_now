#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
open_camera_patch.py 단위 테스트
────────────────────────────────────────────────────────────────
go_now_v2.open_camera() 교체용 패치 스니펫. 실제 cv2.VideoCapture
(하드웨어)는 mock 으로 대체하여 분기만 검증한다.
  · 첫 인덱스/백엔드에서 즉시 성공
  · 워밍업 재시도
  · 모든 인덱스 실패 → GStreamer 폴백
  · 전부 실패 → RuntimeError
"""
import sys, os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import open_camera_patch as ocp


def _make_cap(opened=True, frames=None):
    """frames: read() 가 순차 반환할 (ret, frame) 리스트. 없으면 항상 실패."""
    cap = MagicMock()
    cap.isOpened.return_value = opened
    if frames is None:
        cap.read.return_value = (False, None)
    else:
        cap.read.side_effect = frames
    return cap


@pytest.fixture(autouse=True)
def no_sleep():
    """time.sleep 무력화 (워밍업 재시도 대기 제거)."""
    with patch.object(ocp.time, 'sleep', lambda s: None):
        yield


class TestOpenCamera:
    """C01: open_camera 분기"""

    def test_C01_first_index_success(self):
        good = _make_cap(opened=True, frames=[(True, MagicMock())])
        with patch.object(ocp.cv2, 'VideoCapture', return_value=good):
            cap = ocp.open_camera(0)
        assert cap is good

    def test_C01_skips_unopened_then_succeeds(self):
        bad = _make_cap(opened=False)
        good = _make_cap(opened=True, frames=[(True, MagicMock())])
        seq = [bad, good]
        with patch.object(ocp.cv2, 'VideoCapture', side_effect=lambda *a, **k: seq.pop(0)):
            cap = ocp.open_camera(0)
        assert cap is good
        bad.release.assert_called()

    def test_C01_warmup_retry_then_frame(self):
        """첫 read 는 빈 프레임, 이후 정상 → 워밍업 재시도로 성공"""
        cap = _make_cap(opened=True,
                        frames=[(False, None), (False, None), (True, MagicMock())])
        with patch.object(ocp.cv2, 'VideoCapture', return_value=cap):
            result = ocp.open_camera(0)
        assert result is cap
        assert cap.read.call_count >= 3

    def test_C01_gstreamer_fallback(self):
        """모든 일반 인덱스 실패 후 GStreamer 파이프라인으로 성공"""
        dead = _make_cap(opened=False)
        gst = _make_cap(opened=True, frames=[(True, MagicMock())])

        def _factory(arg, backend=None):
            # GStreamer 파이프라인 문자열이면 gst cap 반환
            if isinstance(arg, str):
                return gst
            return dead

        with patch.object(ocp.cv2, 'VideoCapture', side_effect=_factory):
            cap = ocp.open_camera(0)
        assert cap is gst

    def test_C01_all_fail_raises(self):
        dead = _make_cap(opened=False)
        with patch.object(ocp.cv2, 'VideoCapture', return_value=dead):
            with pytest.raises(RuntimeError):
                ocp.open_camera(0)

    def test_C01_warmup_exhausted_then_gstreamer_fails(self):
        """열리지만 프레임이 안 들어옴 → 워밍업 소진 후 release(38),
        GStreamer 폴백도 실패 → release(51) 후 RuntimeError"""
        opened_no_frame = _make_cap(opened=True)  # read 항상 (False, None)
        with patch.object(ocp.cv2, 'VideoCapture', return_value=opened_no_frame):
            with pytest.raises(RuntimeError):
                ocp.open_camera(0)
        assert opened_no_frame.release.called
