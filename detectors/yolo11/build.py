"""
YOLO11 builder (Image task only).

Reference: YOLO11 has no paper; the original is the Ultralytics release (2024,
ultralytics/cfg/models/11/yolo11.yaml, AGPL-3.0). Blocks in modules.py are copied verbatim.

Delta vs YOLOv8 (detectors/yolov8):
    backbone / neck : C3k2 instead of C2f (C3k inner blocks at P4 / P5), C2PSA (position-sensitive
                      attention) after SPPF                                     modules.py
    head            : class branch DWConv 3x3 + Conv 1x1 (x2) instead of Conv 3x3 (x2); box branch as v8
    shared with v8  : 1x1 prediction convs + DFL decode (yolov8_pred), Task-Aligned Assigner and
                      BCE + CIoU + DFL loss (Ultralytics uses the same v8DetectionLoss for YOLO11)

Size: n (depth 0.50, width 0.25, max channels 1024) - the official yolo11n: 2,624,080 parameters
with 80 classes (incl. the frozen 16-weight DFL conv, kept here as a buffer).
"""

import math

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.yolo11.modules import C2PSA, C3k2, Concat, Conv, DWConv, SPPF
from detectors.yolov8.loss import Criterion
from detectors.yolov8.yolov8_config import yolov8_cfg
from detectors.yolov8.yolov8_pred import build_pred_layer

SCALES = {"n": (0.50, 0.25, 1024), "s": (0.50, 0.50, 1024), "m": (0.50, 1.00, 512),
          "l": (1.00, 1.00, 512), "x": (1.00, 1.50, 512)}
VARIANT = "n"
STRIDES = [8, 16, 32]
REG_MAX = 16


def make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


class YOLO11(nn.Module):
    def __init__(self, num_classes, variant=VARIANT):
        super().__init__()
        depth, width, max_ch = SCALES[variant]
        ch = lambda c: make_divisible(min(c, max_ch) * width, 8)            # noqa: E731
        n = lambda r: max(round(r * depth), 1) if r > 1 else r              # noqa: E731
        c3k_big = variant in {"m", "l", "x"}                               # parse_model: C3k2 c3k for M/L/X
        self.num_classes = num_classes
        # ---- backbone (yolo11.yaml, layers 0-10)
        self.b0 = Conv(3, ch(64), 3, 2)
        self.b1 = Conv(ch(64), ch(128), 3, 2)
        self.b2 = C3k2(ch(128), ch(256), n(2), c3k_big or False, 0.25)
        self.b3 = Conv(ch(256), ch(256), 3, 2)
        self.b4 = C3k2(ch(256), ch(512), n(2), c3k_big or False, 0.25)    # P3 out
        self.b5 = Conv(ch(512), ch(512), 3, 2)
        self.b6 = C3k2(ch(512), ch(512), n(2), True)                       # P4 out
        self.b7 = Conv(ch(512), ch(1024), 3, 2)
        self.b8 = C3k2(ch(1024), ch(1024), n(2), True)
        self.b9 = SPPF(ch(1024), ch(1024), 5)
        self.b10 = C2PSA(ch(1024), ch(1024), n(2))                         # P5 out
        # ---- neck (layers 11-22)
        self.up = nn.Upsample(None, 2, "nearest")
        self.cat = Concat(1)
        self.h13 = C3k2(ch(1024) + ch(512), ch(512), n(2), c3k_big or False)
        self.h16 = C3k2(ch(512) + ch(512), ch(256), n(2), c3k_big or False)       # P3/8
        self.h17 = Conv(ch(256), ch(256), 3, 2)
        self.h19 = C3k2(ch(256) + ch(512), ch(512), n(2), c3k_big or False)       # P4/16
        self.h20 = Conv(ch(512), ch(512), 3, 2)
        self.h22 = C3k2(ch(512) + ch(1024), ch(1024), n(2), True)                 # P5/32
        # ---- Detect head (cv2 / cv3 without their last 1x1 conv, which lives in the pred layer)
        chs = [ch(256), ch(512), ch(1024)]
        c2, c3 = max(16, chs[0] // 4, REG_MAX * 4), max(chs[0], min(num_classes, 100))
        self.reg_heads = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3)) for x in chs)
        self.cls_heads = nn.ModuleList(nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                                                     nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1))) for x in chs)
        self.pred_layers = build_pred_layer(cls_dim=c3, reg_dim=c2, strides=STRIDES, num_classes=num_classes,
                                            num_coords=4, num_levels=3, reg_max=REG_MAX)
        self.criterion = Criterion(yolov8_cfg["yolov8_n"], None, num_classes)   # same loss as Ultralytics v8 / 11
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def features(self, x):
        x = self.b2(self.b1(self.b0(x)))
        p3 = self.b4(self.b3(x))
        p4 = self.b6(self.b5(p3))
        p5 = self.b10(self.b9(self.b8(self.b7(p4))))
        h13 = self.h13(self.cat([self.up(p5), p4]))
        h16 = self.h16(self.cat([self.up(h13), p3]))
        h19 = self.h19(self.cat([self.h17(h16), h13]))
        h22 = self.h22(self.cat([self.h20(h19), p5]))
        return [h16, h19, h22]

    def forward(self, x, targets=None):
        feats = self.features(x)
        out = self.pred_layers([h(f) for h, f in zip(self.cls_heads, feats)],
                               [h(f) for h, f in zip(self.reg_heads, feats)])
        for k in ("anchors", "stride_tensor"):
            out[k] = [a.to(x.device) for a in out[k]]
        cls = torch.cat(out["pred_cls"], 1)
        raw = to_raw(torch.cat(out["pred_box"], 1), cls.sigmoid())
        if targets is None:
            return raw
        loss_dict, fg_mask, fg_labels = self.criterion(out, split_targets(targets, x.shape[0], x.shape[2]))
        flat = cls.reshape(-1, self.num_classes)
        self.stats.update(loss_dict, flat.sigmoid().max(-1).values, flat, fg_mask, fg_labels)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("YOLO11 has no Darknet weights; train from scratch.")
