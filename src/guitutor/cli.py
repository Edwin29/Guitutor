from __future__ import annotations
import argparse, logging
from pathlib import Path
import yaml

from .vision.calibration import Calibrator
from .vision.hand_tracker import HandTracker
from .domain.fretboard import FretboardMapper
from .eval.fingering import FingeringRules
from .eval.evaluator import FingeringEvaluator
from .vision.overlay import Overlay
from .vision.runner import VideoRunner


def _build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="guitutor",
        description="Guitar fingering vision tutor (MVP)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- calibrate: --video 또는 --source 중 하나를 필수로 받기 ---
    c = sub.add_parser("calibrate", help="프렛보드 4점 수동 캘리브레이션")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="비디오/이미지 경로 (예: data/samples/fretboard.mp4)")
    g.add_argument("--source", type=int, help="웹캠 인덱스 (예: 0)")
    c.add_argument("--save", required=True, help="캘리브 결과 저장 경로 (yaml)")

    # --- run: 실시간/파일 분석 ---
    r = sub.add_parser("run", help="실시간/파일 분석")
    r.add_argument("--source", required=True, help="0(웹캠) 또는 파일 경로")
    r.add_argument("--config", default="config/default.yaml", help="설정 yaml")
    r.add_argument("--rules",  default="config/fingerings.yaml", help="운지 룰 yaml")

    return p


def main() -> None:
    args = _build().parse_args()

    if args.cmd == "calibrate":
        # --source가 있으면 정수(웹캠), 없으면 --video 경로 사용
        input_src = args.source if getattr(args, "source", None) is not None else args.video
        Calibrator().interactive_calibrate(input_src, args.save)
        return

    if args.cmd == "run":
        cfg_path  = Path(args.config)
        rules_path = Path(args.rules)

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        rules = FingeringRules.from_yaml(str(rules_path))

        logging.basicConfig(level=getattr(logging, cfg["app"]["log_level"], logging.INFO))

        fb = FretboardMapper(
            strings=cfg["board"]["strings"],
            frets=cfg["board"]["frets"],
            calib_path=cfg["board"].get("calibration_path", "config/calib.yaml"),  # ✅
        )
        tracker = HandTracker(kind=cfg["hand"]["tracker"], min_conf=cfg["hand"]["min_conf"])
        evaluator = FingeringEvaluator(rules, hit_margin=cfg["board"]["hit_margin"])
        overlay = Overlay(draw=cfg["video"]["draw_overlay"])

        # int인지 파일 경로인지 판별
        src_arg = args.source
        source = int(src_arg) if str(src_arg).isdigit() else src_arg

        VideoRunner(
            tracker, fb, evaluator, overlay, 
            cfg=cfg,
            smoothing_window=cfg["video"]["smoothing_window"]
        ).run(source)
