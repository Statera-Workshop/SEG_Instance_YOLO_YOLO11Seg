"""
YOLO11-seg builder (seg task) - YOLO11 (detectors/yolo11, unchanged) + the Ultralytics Segment head.

Reference: Ultralytics release (2024, ultralytics/cfg/models/11/yolo11-seg.yaml: head `Segment, [nc, 32, 256]`;
ultralytics/nn/modules/head.py Segment, block.py Proto; utils/loss.py v8SegmentationLoss; utils/ops.py
process_mask / crop_mask), AGPL-3.0. Reimplemented here on top of the framework's YOLO11 (same blocks).

    Proto      on P3 (neck stride 8): Conv 3x3 (c_) -> ConvTranspose 2x2 s2 (c_) -> Conv 3x3 (c_) -> Conv 1x1 (32)
               -> 32 prototypes at stride 4 (152 x 152 for a 608 input); c_ = npr 256 x width (n: 64)
    cv4        per level: Conv 3x3 (c4) -> Conv 3x3 (c4) -> Conv2d 1x1 (32) mask coefficients, c4 = max(P3 ch / 4, 32)
    mask       sigmoid(coefficients . prototypes), cropped with the box
    loss       v8DetectionLoss (detectors/yolov8/loss.Criterion, unchanged) + v8SegmentationLoss mask term:
               per positive anchor, BCE(coef . proto, gt mask of the assigned object at proto resolution),
               averaged inside the assigned gt box (crop_mask), divided by the normalised box area, summed and
               divided by the number of positive anchors; gain = box gain 7.5 (Ultralytics loss[1] *= hyp.box)

Raw contract: YOLO11 rows [B, M, 5 + C] (one per anchor); aux = coefficients [B, M, 32] + prototypes.
decode_masks returns kind "full" at proto resolution; tasks/seg_task.py upsamples, crops with the box and
thresholds at 0.5 (process_mask order: crop at proto scale then upsample -> here upsample then crop, same mask
up to boundary pixels).

[Unified] as YOLO11 (no pretrained weights, common init, flip + YOLO-paper augmentation only, Adam, common
post-processing). Ground-truth masks are the dataset's instance masks downsampled to proto resolution with
nearest interpolation (Ultralytics rasterises polygons at 1/4 resolution; overlap_mask=False path, one mask
per object). Ultralytics also multiplies the summed loss by the batch size; the framework's v8 criterion does
not, so all four terms keep the Ultralytics relative weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.anchor_free.common import to_raw
from detectors.yolo11.build import YOLO11, make_divisible, SCALES, VARIANT
from detectors.yolo11.modules import Conv

NM, NPR = 32, 256


class Proto(nn.Module):
    """Ultralytics block.Proto."""

    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class _RecordingMatcher:
    """Wraps the TaskAlignedAssigner of the v8 criterion to keep target_gt_idx (the criterion drops it)."""

    def __init__(self, matcher):
        self.matcher, self.calls = matcher, []

    def __call__(self, **kw):
        out = self.matcher(**kw)
        self.calls.append(out[-1])               # target_gt_idx [1, M]
        return out


def crop_mask(masks, boxes):
    """Ultralytics ops.crop_mask: zero outside xyxy boxes. masks [n, h, w], boxes [n, 4] (mask pixels)."""
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def split_seg_targets(targets, batch_size, img_size):
    """{"boxes": [N, 6], "masks": [N, S, S]} -> per-image dicts (boxes xyxy px, labels 0..C-1, masks)."""
    t_all, m_all = targets["boxes"], targets["masks"]
    out = []
    for b in range(batch_size):
        sel = t_all[:, 0] == b
        t, m = t_all[sel], m_all[sel]
        cxcywh = t[:, 2:6] * img_size
        boxes = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 1e-3) & (boxes[:, 3] - boxes[:, 1] > 1e-3)
        out.append({"labels": t[keep, 1].long(), "boxes": boxes[keep], "masks": m[keep]})
    return out


class YOLO11Seg(YOLO11):
    def __init__(self, num_classes, variant=VARIANT):
        super().__init__(num_classes, variant)
        depth, width, max_ch = SCALES[variant]
        ch = lambda c: make_divisible(min(c, max_ch) * width, 8)            # noqa: E731
        chs = [ch(256), ch(512), ch(1024)]
        npr = make_divisible(min(NPR, max_ch) * width, 8)                   # parse_model scales Segment's npr
        c4 = max(chs[0] // 4, NM)
        self.proto = Proto(chs[0], npr, NM)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, NM, 1)) for x in chs)
        self.criterion.matcher = _RecordingMatcher(self.criterion.matcher)
        self.mask_gain = self.criterion.loss_box_weight                      # Ultralytics: loss[1] *= hyp.box

    def _heads(self, x):
        feats = self.features(x)
        out = self.pred_layers([h(f) for h, f in zip(self.cls_heads, feats)],
                               [h(f) for h, f in zip(self.reg_heads, feats)])
        for k in ("anchors", "stride_tensor"):
            out[k] = [a.to(x.device) for a in out[k]]
        B = x.shape[0]
        coef = torch.cat([h(f).permute(0, 2, 3, 1).reshape(B, -1, NM) for h, f in zip(self.cv4, feats)], 1)
        return out, coef, self.proto(feats[0])

    def forward(self, x, targets=None):
        out, coef, proto = self._heads(x)
        cls = torch.cat(out["pred_cls"], 1)
        raw = to_raw(torch.cat(out["pred_box"], 1), cls.sigmoid())
        if targets is None:
            if self.training:                                                # PreciseBN forward
                return raw
            return raw, {"coef": coef, "proto": proto}
        B, S = x.shape[0], x.shape[2]
        tgt = split_seg_targets(targets, B, S)
        self.criterion.matcher.calls = []
        loss_dict, fg_mask, fg_labels = self.criterion(out, tgt)
        loss_mask = self.mask_loss(coef, proto, tgt, fg_mask.view(B, -1), S)
        loss_dict = {**loss_dict, "loss_mask": loss_mask}
        loss_dict["losses"] = loss_dict["losses"] + self.mask_gain * loss_mask
        flat = cls.reshape(-1, self.num_classes)
        self.stats.update(loss_dict, flat.sigmoid().max(-1).values, flat, fg_mask, fg_labels)
        return loss_dict["losses"], raw

    def mask_loss(self, coef, proto, tgt, fg_mask, img_size):
        """Ultralytics v8SegmentationLoss.calculate_segmentation_loss (overlap_mask=False)."""
        calls = iter(self.criterion.matcher.calls)
        _, nm, mh, mw = proto.shape
        total = coef.sum() * 0.0
        for i, t in enumerate(tgt):
            # the criterion calls the matcher only for images with a valid target (same condition)
            if len(t["labels"]) == 0 or t["boxes"].max().item() == 0.0:
                continue
            gt_idx = next(calls)[0]                                          # [M]
            fg = fg_mask[i]
            if not fg.any():
                continue
            idx = gt_idx[fg]
            gt_m = F.interpolate(t["masks"][None].float(), (mh, mw), mode="nearest")[0][idx]   # [n, mh, mw]
            box = t["boxes"][idx]                                            # input px, xyxy
            # [Unified] vseg02: canvas H x W from the target masks (wide 1216 x 384); square: H = W = img_size
            H, W = t["masks"].shape[-2:]
            mxyxy = box * torch.tensor([mw / W, mh / H] * 2, device=box.device)
            marea = ((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) / (W * H)).clamp(min=1e-9)
            pred = (coef[i, fg] @ proto[i].view(nm, -1)).view(-1, mh, mw)
            loss = F.binary_cross_entropy_with_logits(pred, gt_m, reduction="none")
            total = total + (crop_mask(loss, mxyxy).mean(dim=(1, 2)) / marea).sum()
        return total / fg_mask.sum().clamp(min=1)

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        proto = aux["proto"][b]
        nm, mh, mw = proto.shape
        return {"kind": "full", "prob": (aux["coef"][b, rows] @ proto.view(nm, -1)).sigmoid().view(-1, mh, mw)}


def build_model(task, num_classes):
    return YOLO11Seg(num_classes)
