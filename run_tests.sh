#!/bin/bash
cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/willtek/work/env/bin/python}

echo "===== pytest / pytest-cov 설치 확인 ====="
$PYTHON -m pip install pytest pytest-html pytest-cov -q

echo "===== 테스트 실행 ====="
$PYTHON -m pytest tests/ \
    -v \
    --junitxml=test-results/junit.xml \
    --html=test-results/report.html \
    --self-contained-html \
    --cov=src \
    --cov-config=.coveragerc \
    --cov-report=html:test-results/coverage_html \
    --cov-report=xml:test-results/coverage.xml \
    --cov-report=term-missing \
    2>&1 | tee test-results/test_log.txt

PYTEST_EXIT=${PIPESTATUS[0]}

echo ""
echo "════════════════════════════════════════"
echo "         Pass Rate / Coverage 요약      "
echo "════════════════════════════════════════"
python3 - << 'PYEOF'
import xml.etree.ElementTree as ET, os

# ── Pass Rate ──────────────────────────────
try:
    tree  = ET.parse('test-results/junit.xml')
    root  = tree.getroot()
    suite = root if root.tag == 'testsuite' else root.find('testsuite')
    total    = int(suite.get('tests',    0))
    failures = int(suite.get('failures', 0))
    errors   = int(suite.get('errors',   0))
    skipped  = int(suite.get('skipped',  0))
    passed   = total - failures - errors - skipped
    rate     = passed / total * 100 if total else 0
    bar_p    = int(rate / 5)
    bar      = '█' * bar_p + '░' * (20 - bar_p)
    print(f"  ✅ Pass Rate   : {passed}/{total} [{bar}] {rate:.1f}%")
    if failures: print(f"  ❌ Failures    : {failures}")
    if errors:   print(f"  💥 Errors      : {errors}")
    if skipped:  print(f"  ⏭  Skipped     : {skipped}")
except Exception as e:
    print(f"  Pass Rate 계산 실패: {e}")

print()

# ── Coverage ───────────────────────────────
TARGET = ['mic_agent', 'radar_signal_processor', 'outlook_sync']
try:
    tree  = ET.parse('test-results/coverage.xml')
    root  = tree.getroot()
    lr    = float(root.get('line-rate',   0)) * 100
    br    = float(root.get('branch-rate', 0)) * 100
    bar_l = int(lr / 5)
    bar_b = int(br / 5)
    print(f"  📊 Line   Coverage : [{'█'*bar_l}{'░'*(20-bar_l)}] {lr:.1f}%")
    print(f"  🌿 Branch Coverage : [{'█'*bar_b}{'░'*(20-bar_b)}] {br:.1f}%")
    print()
    print(f"  {'파일명':<35} {'Line':>8}  {'Branch':>8}")
    print(f"  {'─'*35} {'─'*8}  {'─'*8}")
    for cls in root.iter('class'):
        fname = cls.get('filename', '')
        if any(t in fname for t in TARGET):
            clr = float(cls.get('line-rate',   0)) * 100
            cbr = float(cls.get('branch-rate', 0)) * 100
            icon = '✅' if clr >= 80 else ('⚠️ ' if clr >= 50 else '❌')
            print(f"  {icon} {os.path.basename(fname):<33} {clr:>7.1f}%  {cbr:>7.1f}%")
except Exception as e:
    print(f"  Coverage 계산 실패: {e}")

PYEOF

echo ""
echo "════════════════════════════════════════"
echo "  HTML 리포트   : test-results/report.html"
echo "  커버리지 HTML  : test-results/coverage_html/index.html"
echo "  JUnit XML      : test-results/junit.xml"
echo "  Coverage XML   : test-results/coverage.xml"
echo "════════════════════════════════════════"

exit $PYTEST_EXIT