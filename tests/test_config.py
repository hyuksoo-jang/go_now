import os, json

class TestConfigFiles:
    """R01/R04/R05: 설정 파일 유효성 테스트"""

    def test_R01_gate_config_exists(self, ws_dir):
        """R01: gate_config.json 존재 확인"""
        assert os.path.exists(os.path.join(ws_dir, 'gate_config.json'))

    def test_R01_gate_config_valid_json(self, gate_config):
        """R01: gate_config.json 유효한 JSON 형식"""
        assert isinstance(gate_config, dict)

    def test_R04_outlook_config_exists(self, ws_dir):
        """R04: outlook_config.json 존재 확인"""
        assert os.path.exists(os.path.join(ws_dir, 'outlook_config.json'))

    def test_R04_outlook_config_valid_json(self, outlook_config):
        """R04: outlook_config.json 유효한 JSON 형식"""
        assert isinstance(outlook_config, dict)

    def test_R04_calendar_data_exists(self, ws_dir):
        """R04: calendar_data.json 존재 확인"""
        assert os.path.exists(os.path.join(ws_dir, 'calendar_data.json'))

    def test_R04_calendar_data_valid_structure(self, calendar_data):
        """R04: calendar_data.json 유효한 데이터 구조"""
        assert isinstance(calendar_data, (list, dict))

    def test_R05_stt_config_exists(self, ws_dir):
        """R05: stt_config.json 존재 확인"""
        assert os.path.exists(os.path.join(ws_dir, 'stt_config.json'))

    def test_R05_stt_config_valid_json(self, stt_config):
        """R05: stt_config.json 유효한 JSON 형식"""
        assert isinstance(stt_config, dict)