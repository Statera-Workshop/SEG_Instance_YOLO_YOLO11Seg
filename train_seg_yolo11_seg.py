"""Train seg YOLO11-seg, then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.yolo11_seg.build import build_model

if __name__ == "__main__":
    main("train", task="seg", version="yolo11_seg", build_model=build_model)
