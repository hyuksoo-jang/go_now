import os
import numpy as np

class TestModelFiles:
    """R02: AI 모델 파일 테스트"""

    def test_R02_yamnet_model_exists(self, model_dir):
        """R02: yamnet.tflite 모델 파일 존재 확인"""
        assert os.path.exists(os.path.join(model_dir, 'yamnet.tflite'))

    def test_R02_classifier_exists(self, model_dir):
        """R02: classifier.npz 분류기 파일 존재 확인"""
        assert os.path.exists(os.path.join(model_dir, 'classifier.npz'))

    def test_R02_threshold_exists(self, model_dir):
        """R02: threshold.txt 임계값 파일 존재 확인"""
        assert os.path.exists(os.path.join(model_dir, 'threshold.txt'))

    def test_R02_class_map_exists(self, model_dir):
        """R02: yamnet_class_map.csv 존재 확인"""
        assert os.path.exists(os.path.join(model_dir, 'yamnet_class_map.csv'))

    def test_R02_threshold_valid_range(self, model_dir):
        """R02: 임계값이 0.0~1.0 범위인지 확인"""
        with open(os.path.join(model_dir, 'threshold.txt')) as f:
            threshold = float(f.read().strip())
        assert 0.0 <= threshold <= 1.0, f"임계값 {threshold}이 범위를 벗어남"

    def test_R02_classifier_loadable(self, model_dir):
        """R02: classifier.npz 로딩 성공"""
        data = np.load(os.path.join(model_dir, 'classifier.npz'), allow_pickle=True)
        assert len(data.files) > 0

    def test_R02_class_map_readable(self, model_dir):
        """R02: yamnet_class_map.csv 읽기 및 클래스 존재 확인"""
        import csv
        with open(os.path.join(model_dir, 'yamnet_class_map.csv'), newline='') as f:
            rows = list(csv.reader(f))
        assert len(rows) > 1  # 헤더 + 데이터 최소 2줄