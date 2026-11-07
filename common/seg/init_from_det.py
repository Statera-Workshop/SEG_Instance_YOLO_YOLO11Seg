"""
vseg02 (2026-10-31): start a KITTI-MOTS instance-segmentation model from the MOT-stage detector of the same box
architecture (study dev3_mot_det_wide, 1216 x 384, KITTI-MOTS train), so the box branch begins where the MOT
comparison ended and only the mask branch is new:

    mask_rcnn_km          <- image_faster_rcnn   (Mask R-CNN = Faster R-CNN + mask head)
    cascade_mask_rcnn_km  <- image_cascade_rcnn  (Cascade Mask R-CNN = Cascade R-CNN + mask head)
    yolo11_seg_km         <- image_v11           (YOLO11-seg = YOLO11 + Proto + coefficient heads)

The checkpoint is loaded with strict=False and then checked: no unexpected key, every missing key must start with
one of the model's `init_new_prefixes` (the mask branch), and at least 95 % of the model's other tensors must load.
Otherwise the run stops. Checkpoint: $UNIFIED_INIT_CKPT if set, else the newest
<PHYSICAL_AI_ROOT>/projects/dev3_mot_det_wide/runs/image_<det>/checkpoints/image_<det>_kittimots_e*.pth.
[Unified] the FL36 seg models start from scratch; these *_km variants are a protocol difference, recorded in the
run log ("Init from") and the checkpoint metadata.
"""

import os
from pathlib import Path

import torch

NEW_PREFIXES = {
    "mask_rcnn": ("mask_roi_pool.", "mask_head.", "mask_predictor."),
    "cascade_mask_rcnn": ("mask_roi_pool.", "mask_head.", "mask_predictor."),
    "yolo11_seg": ("proto.", "cv4."),
}


def detector_checkpoint(det):
    env = os.environ.get("UNIFIED_INIT_CKPT")
    if env:
        return Path(env).expanduser()
    from common import config as C
    d = C.PHYSICAL_AI_ROOT / "projects" / "dev3_mot_det_wide" / "runs" / f"image_{det}" / "checkpoints"
    found = sorted(d.glob(f"image_{det}_kittimots_e*.pth"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no KITTI-MOTS checkpoint for {det} in {d} (set UNIFIED_INIT_CKPT)")
    return found[-1]


def load_detector_init(model):
    det, prefixes = model.init_detector, tuple(model.init_new_prefixes)
    path = detector_checkpoint(det)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    res = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith(prefixes)]
    if res.unexpected_keys or bad_missing:
        raise RuntimeError(f"init from {path}: unexpected {res.unexpected_keys[:5]} ({len(res.unexpected_keys)}), "
                           f"missing outside the mask branch {bad_missing[:5]} ({len(bad_missing)})")
    own = [k for k in model.state_dict() if not k.startswith(prefixes)]
    loaded = [k for k in own if k in state]
    if len(loaded) < 0.95 * len(own):
        raise RuntimeError(f"init from {path}: only {len(loaded)} / {len(own)} non-mask tensors loaded")
    new = sum(1 for k in model.state_dict() if k.startswith(prefixes))
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    return (f"{path} (epoch {ck.get('epoch', meta.get('epoch', '?'))}; {len(loaded)} tensors loaded, "
            f"{new} new mask-branch tensors keep the common init)")


def with_init(build_model, kind, det):
    """Wrap a seg build_model so the runner loads `det`'s KITTI-MOTS weights after the common init (training)."""
    def build(task, num_classes):
        model = build_model(task, num_classes)
        model.init_detector, model.init_new_prefixes = det, NEW_PREFIXES[kind]
        return model
    return build
