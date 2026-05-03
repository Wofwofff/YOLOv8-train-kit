import argparse
import logging
import sys
from pathlib import Path

from yodet.config import Config, InferenceConfig
from yodet.predict import predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main(args) -> None:
    cfg = Config.from_yaml(args.config)

    inference_cfg: InferenceConfig = cfg.inference
    if args.conf is not None:
        inference_cfg.confidence = args.conf
    if args.slicing:
        inference_cfg.use_slicing = True

    if not args.weights.exists():
        log.critical(f"Weights file not found: {args.weights}")
        sys.exit(1)

    if not args.source.exists():
        log.critical(f"Source not found: {args.source}")
        sys.exit(1)

    predict(
        source=args.source,
        weights=args.weights,
        output_dir=args.output,
        cfg=inference_cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run tunnel-defect detection on images.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument(
        "--source", type=Path, required=True, help="Image or directory")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/predict"))
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override confidence threshold")
    parser.add_argument(
        "--slicing", action="store_true", help="Use SAHI sliced inference")
    args = parser.parse_args()

    main(args)
