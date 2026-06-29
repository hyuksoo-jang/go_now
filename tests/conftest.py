import sys, os, json, pytest

WS = os.path.join(os.path.dirname(__file__), '..', 'go_now_ws')
sys.path.insert(0, os.path.abspath(WS))

@pytest.fixture
def ws_dir():
    return os.path.abspath(WS)

@pytest.fixture
def model_dir():
    return os.path.join(os.path.abspath(WS), 'model')

@pytest.fixture
def gate_config():
    with open(os.path.join(WS, 'gate_config.json'), encoding='utf-8') as f:
        return json.load(f)

@pytest.fixture
def outlook_config():
    with open(os.path.join(WS, 'outlook_config.json'), encoding='utf-8') as f:
        return json.load(f)

@pytest.fixture
def stt_config():
    with open(os.path.join(WS, 'stt_config.json'), encoding='utf-8') as f:
        return json.load(f)

@pytest.fixture
def calendar_data():
    with open(os.path.join(WS, 'calendar_data.json'), encoding='utf-8') as f:
        return json.load(f)