"""Train seg yolo11_seg on KITTI-MOTS from the MOT-stage v11 weights (vseg02), then evaluate.

    UNIFIED_SEG_DATASET=kittimots UNIFIED_IMAGE_H=384 python train_seg_yolo11_seg_km.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from common.seg.init_from_det import with_init
from detectors.yolo11_seg.build import build_model

if __name__ == "__main__":
    main("train", task="seg", version="yolo11_seg_km", build_model=with_init(build_model, "yolo11_seg", "v11"))
