#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
go_now_v2.py 의 open_camera() 를 이 함수로 통째 교체하세요.
(go_now_v2.py 에서 바꿀 곳은 여기 한 곳뿐입니다 — 나머지는 radar_signal_processor.py
 가 같은 폴더에 있으면 그대로 동작합니다.)

변경점:
  · 첫 read 실패로 카메라를 버리지 않고 워밍업 재시도(최대 10회)
  · V4L2 백엔드 명시 후 CAP_ANY 폴백
  · 인덱스 탐색 범위 0~7 로 확장
  · 마지막 폴백으로 libcamera GStreamer 파이프라인 시도 (CSI 모듈용)
"""

import time
import cv2


def open_camera(preferred_idx: int) -> cv2.VideoCapture:
    """카메라를 열고 실제 프레임이 들어오는지 확인 (워밍업/백엔드 강화)."""
    backends  = [cv2.CAP_V4L2, cv2.CAP_ANY]
    idx_order = [preferred_idx] + [i for i in range(8) if i != preferred_idx]

    for backend in backends:
        for idx in idx_order:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # 워밍업: 첫 프레임이 비어 나올 수 있으므로 최대 10회 재시도
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    bname = "V4L2" if backend == cv2.CAP_V4L2 else "ANY"
                    print(f"[INFO] 카메라 인덱스 {idx} 사용 (backend={bname})")
                    return cap
                time.sleep(0.1)
            cap.release()

    # 마지막 폴백: libcamera GStreamer (OpenCV 가 GStreamer 지원으로 빌드된 경우)
    gst = ("libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 "
           "! videoconvert ! appsink")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                print("[INFO] libcamera GStreamer 파이프라인 사용")
                return cap
            time.sleep(0.1)
        cap.release()

    raise RuntimeError("사용 가능한 카메라를 찾을 수 없습니다.")
