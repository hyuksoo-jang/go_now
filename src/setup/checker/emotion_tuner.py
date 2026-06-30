#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emotion_tuner.py
────────────────────────────────────────────────────────────────
발화 위험점수의 '임계(red_th/green_th)'를 라벨 데이터로 보정한다.
speech_tuner.py 와 같은 컨셉: 튜너 1회 실행 → weights_config.json 생성/갱신
→ radar_signal_processor.py 가 자동 로드.

입력: 문장 텍스트 + 3단계 라벨(안전/주의/위험).
  · 라벨은 '그 발화 자체의 언어적 위험도'를 매긴다(최종 신호등이 아니라).
    - 안전 → green, 주의 → yellow, 위험 → red
방법:
  · KoBERT로 각 문장의 6감정 확률을 구하고,
  · 현재 가중치(weights_config 또는 기본값)로 발화 위험 r_i 를 계산한 뒤,
  · (red_th, green_th) 격자를 훑어 라벨과 가장 잘 맞는 임계를 고른다.
  · 감정 '가중치' 자체는 도메인 사전값이라 자동탐색하지 않고,
    위험 vs 안전 표본의 감정 평균확률을 리포트로 보여줘 수동 조정을 돕는다.

실행:
  python3 emotion_tuner.py                       # 대화식 입력
  python3 emotion_tuner.py --samples samples.json
  python3 emotion_tuner.py --samples samples.json --out weights_config.json

samples.json 형식:
  [ {"text": "지금 바빠 죽겠는데", "label": "위험"},
    {"text": "네 확인했습니다",   "label": "주의"},
    {"text": "좋아요 수고했어요",  "label": "안전"} ]
────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys

# radar 는 표준 라이브러리만 쓰므로 가볍게 import (위험점수 함수/설정 재사용)
from work.main_project.go_now.go_now_ws.radar_signal_processor_org import (
    speech_utterance_risk, load_weights_config, DEFAULT_WEIGHTS_CONFIG,
)

LABEL_TO_SIGNAL = {"안전": "green", "주의": "yellow", "위험": "red"}
SIGNAL_ORDER = ["green", "yellow", "red"]
EMO_ORDER = ["분노", "불안", "당황", "상처", "슬픔", "기쁨"]


# ─────────────────────────────────────────────────────────────
# 임계 격자 탐색 (순수 함수 — 모델 없이 테스트 가능)
# ─────────────────────────────────────────────────────────────
def predict_signal(risk, red_th, green_th):
    if risk >= red_th:
        return "red"
    if risk <= green_th:
        return "green"
    return "yellow"


def grid_search_thresholds(samples, cfg, red_grid=None, green_grid=None):
    """samples: [(probs, top_prob, gold_signal), ...]
    cfg: weights config(가중치/게이트). 임계만 탐색해 라벨 일치율 최대화.
    반환: dict(best_red, best_green, accuracy, n, risks, confusion)."""
    weights = cfg["emotion_weights"]
    min_conf, happy_gate = cfg["min_conf"], cfg["happy_gate"]

    scored = []  # (risk, gold_signal)
    for probs, tp, gold in samples:
        r = speech_utterance_risk(probs, tp, weights, min_conf, happy_gate)
        scored.append((r, gold))

    if red_grid is None:
        red_grid = [round(0.20 + 0.05 * i, 2) for i in range(11)]    # 0.20..0.70
    if green_grid is None:
        green_grid = [round(-0.60 + 0.05 * i, 2) for i in range(11)]  # -0.60..-0.10

    n = len(scored)
    best = {"best_red": cfg["red_th"], "best_green": cfg["green_th"],
            "accuracy": -1.0}
    for red in red_grid:
        for green in green_grid:
            if green >= red:
                continue
            correct = sum(1 for r, gold in scored
                          if predict_signal(r, red, green) == gold)
            acc = correct / n if n else 0.0
            if acc > best["accuracy"]:
                best = {"best_red": red, "best_green": green, "accuracy": acc}

    # 혼동행렬 (gold x pred)
    conf = {g: {p: 0 for p in SIGNAL_ORDER} for g in SIGNAL_ORDER}
    for r, gold in scored:
        pred = predict_signal(r, best["best_red"], best["best_green"])
        conf[gold][pred] += 1

    best["n"] = n
    best["risks"] = scored
    best["confusion"] = conf
    return best


def emotion_prob_report(samples):
    """위험 vs 안전 표본에서 감정별 평균확률 — 가중치 수동조정 힌트."""
    buckets = {"위험": [], "주의": [], "안전": []}
    sig_to_label = {v: k for k, v in LABEL_TO_SIGNAL.items()}
    for probs, _tp, gold in samples:
        buckets[sig_to_label[gold]].append(probs)
    out = {}
    for emo in EMO_ORDER:
        row = {}
        for lab in ("위험", "주의", "안전"):
            ps = buckets[lab]
            row[lab] = round(sum(p.get(emo, 0.0) for p in ps) / len(ps), 3) if ps else None
        out[emo] = row
    return out


# ─────────────────────────────────────────────────────────────
# 모델 (지연 import — 텍스트→감정확률)
# ─────────────────────────────────────────────────────────────
def classify_texts(texts, threads=4):
    """KoBERT로 각 문장의 6감정 확률 계산 → [(probs, top_prob), ...].
    무거운 의존성은 여기서만 로드(mic_agent.EmotionClassifier 재사용)."""
    from mic_agent import EmotionClassifier
    clf = EmotionClassifier(threads=threads)
    out = []
    for t in texts:
        res = clf.predict(t)
        out.append((res["probs"], res["top_prob"]))
    return out


# ─────────────────────────────────────────────────────────────
# 입력 수집
# ─────────────────────────────────────────────────────────────
def load_samples_file(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for r in raw:
        text = (r.get("text") or "").strip()
        label = (r.get("label") or "").strip()
        if not text or label not in LABEL_TO_SIGNAL:
            print(f"[tuner] 건너뜀(형식오류): {r}", flush=True)
            continue
        items.append((text, label))
    return items


def collect_interactive():
    print("문장과 라벨을 입력하세요. 라벨: 1=안전 2=주의 3=위험 · 빈 줄이면 종료\n")
    keymap = {"1": "안전", "2": "주의", "3": "위험"}
    items = []
    while True:
        text = input(f"[{len(items)+1}] 문장> ").strip()
        if not text:
            break
        lab = input("    라벨(1/2/3)> ").strip()
        if lab not in keymap:
            print("    ⚠ 1/2/3 중 하나"); continue
        items.append((text, keymap[lab]))
    return items


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="발화 위험 임계 튜너 (텍스트+3단계 라벨)")
    ap.add_argument("--samples", help="라벨 샘플 JSON 경로(없으면 대화식 입력)")
    ap.add_argument("--out", default="../../weights_config.json", help="출력 설정 파일")
    ap.add_argument("--weights-config", default=None,
                    help="시작 가중치/게이트를 읽을 설정(기본: 자동 탐색)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    # 시작 설정(가중치/게이트/현재 임계)
    cfg, cfg_path = load_weights_config(args.weights_config)
    print(f"[tuner] 시작 설정: {cfg_path or '기본값'}")
    print(f"[tuner] 가중치 {cfg['emotion_weights']}")
    print(f"[tuner] 게이트 min_conf={cfg['min_conf']} happy_gate={cfg['happy_gate']}\n")

    # 샘플 수집
    if args.samples:
        labeled = load_samples_file(args.samples)
    else:
        labeled = collect_interactive()
    if len(labeled) < 3:
        sys.exit("[tuner] 표본이 너무 적습니다(>=3 권장). 종료.")
    print(f"\n[tuner] 표본 {len(labeled)}개 · KoBERT 추론 중...", flush=True)

    texts = [t for t, _ in labeled]
    probs_list = classify_texts(texts, threads=args.threads)
    samples = [(probs_list[i][0], probs_list[i][1], LABEL_TO_SIGNAL[lab])
               for i, (_t, lab) in enumerate(labeled)]

    # 임계 탐색
    best = grid_search_thresholds(samples, cfg)

    # 리포트
    print("\n── 샘플별 위험점수 ─────────────────────────────")
    print(f"{'gold':<5} {'risk':>7}  문장")
    sig_to_lab = {v: k for k, v in LABEL_TO_SIGNAL.items()}
    for (text, lab), (risk, gold) in zip(labeled, best["risks"]):
        print(f"{lab:<5} {risk:>7.3f}  {text[:30]}")

    print("\n── 추천 임계 ───────────────────────────────────")
    print(f"red_th={best['best_red']}  green_th={best['best_green']}  "
          f"라벨일치율={best['accuracy']*100:.0f}% (n={best['n']})")

    print("\n── 혼동행렬 (행=정답, 열=예측) ─────────────────")
    print(f"{'':<8}" + "".join(f"{p:>8}" for p in SIGNAL_ORDER))
    for g in SIGNAL_ORDER:
        print(f"{g:<8}" + "".join(f"{best['confusion'][g][p]:>8}" for p in SIGNAL_ORDER))

    print("\n── 감정별 평균확률 (가중치 수동조정 힌트) ───────")
    rep = emotion_prob_report(samples)
    print(f"{'감정':<6}{'위험':>8}{'주의':>8}{'안전':>8}")
    for emo in EMO_ORDER:
        row = rep[emo]
        cells = "".join(f"{(row[l] if row[l] is not None else '-'):>8}" for l in ("위험","주의","안전"))
        print(f"{emo:<6}{cells}")

    # 저장 (가중치/게이트는 유지, 임계만 갱신)
    out_cfg = {
        "emotion_weights": cfg["emotion_weights"],
        "min_conf": cfg["min_conf"],
        "happy_gate": cfg["happy_gate"],
        "w_face": cfg["w_face"],
        "w_speech": cfg["w_speech"],
        "anger_hard_th": cfg["anger_hard_th"],
        "red_th": best["best_red"],
        "green_th": best["best_green"],
        "_meta": {"tuned_by": "emotion_tuner.py", "n_samples": best["n"],
                  "label_accuracy": round(best["accuracy"], 3)},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)
    print(f"\n[tuner] 저장 완료 → {args.out}")
    print("[tuner] radar_signal_processor.py 가 다음 기동 시 자동 로드합니다.")
    print("[tuner] (감정 가중치를 바꾸려면 위 '감정별 평균확률'을 보고 JSON을 직접 수정 후 재실행)")


if __name__ == "__main__":
    main()
