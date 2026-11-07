"""
KITTI-MOTS instance segmentation for the seg task (vseg02, 2026-10-31). UNIFIED_SEG_DATASET=kittimots, UNIFIED_IMAGE_H=384.

Boxes, classes, ignore regions and frame lists come from KITTI_MOTS_VOC (tools/mot/kittimots_to_voc.py, MOT stage):
the box branch is trained on exactly the targets of the MOT detectors. Masks come from the instance PNGs
KITTI_MOTS/instances/<seq>/<frame>.png (uint16, class * 1000 + id, 10000 = ignore; built from KITTI-STEP,
tools/mot/step_to_mots_png.py):
    object  <track_id> v      -> mask = (png == v)            (tight box of the mask = XML box)
    ignore  <difficult> 1     -> mask = ignore pixels inside the XML box (one 8-connected component); written twice
                                 (car, pedestrian) by the converter, so both classes ignore it. Training skips them,
                                 evaluation treats them as ignore ground truth (VOC convention, as the MOT stage).

Output contract = VOCInstanceSeg on the wide canvas W x H = img_size x CANVAS_H (1216 x 384):
    image [3, H, W]; targets {"boxes": [N, 6] normalised by the canvas HEIGHT on both axes (cx, w may exceed 1,
    the MOT-stage convention), "masks": [N, H, W] uint8}.
Augmentation = profile v1.1 with the VOCDetection random-number order (voc.random_scale_translate on the wide canvas),
masks follow with nearest interpolation; an instance is dropped by the detection rule or when its mask becomes empty.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common.datasets.voc import CANVAS_H, VOCDetection, hsv_jitter, letterbox, letterbox_params, to_tensor

IGNORE_ID = 10000


class KittiMotsInstSeg(VOCDetection):
    def __init__(self, root, split, img_size, class_names, train=False, hflip_prob=0.0, max_samples=None, aug=None,
                 instances_root=None):
        super().__init__(root, split, img_size, class_names, train=train, hflip_prob=hflip_prob,
                         max_samples=max_samples, aug=aug)
        if not CANVAS_H:
            raise RuntimeError("KITTI-MOTS seg needs UNIFIED_IMAGE_H=384")
        from common import config as C
        self.canvas_h = CANVAS_H
        self.instances_root = Path(instances_root or C.DATASETS_ROOT / "KITTI_MOTS" / "instances")
        if not self.instances_root.is_dir():
            raise FileNotFoundError(self.instances_root)

    def annotation(self, i):
        """VOCDetection annotation + track ids (IGNORE_ID for ignore regions)."""
        if i in self._ann_cache:
            return self._ann_cache[i]
        tree = ET.parse(self.root / "Annotations" / f"{self.ids[i]}.xml").getroot()
        boxes, labels, difficult, tracks = [], [], [], []
        for obj in tree.iter("object"):
            name = obj.find("name").text.strip().lower()
            if name not in self.cls_to_id:
                continue
            bb = obj.find("bndbox")
            boxes.append([float(bb.find(k).text) - 1.0 for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(self.cls_to_id[name])
            difficult.append(bool(int(obj.find("difficult").text)))
            tracks.append(int(obj.find("track_id").text))
        ann = {"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
               "labels": np.array(labels, dtype=np.int64),
               "ignore": np.array(difficult, dtype=bool),
               "track": np.array(tracks, dtype=np.int64)}
        self._ann_cache[i] = ann
        return ann

    def instance_png(self, i):
        seq, frame = self.ids[i].split("_")
        p = self.instances_root / seq / f"{frame}.png"
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(p)
        return m

    def masks(self, i):
        """bool [N, h, w] in original pixels, row-aligned with annotation(i)['boxes']."""
        ann, png = self.annotation(i), self.instance_png(i)
        out = np.zeros((len(ann["track"]),) + png.shape, bool)
        for k, (v, b) in enumerate(zip(ann["track"], ann["boxes"])):
            if v == IGNORE_ID:
                x0, y0, x1, y1 = (int(round(t)) for t in b)
                out[k, y0:y1 + 1, x0:x1 + 1] = png[y0:y1 + 1, x0:x1 + 1] == IGNORE_ID
            else:
                out[k] = png == v
        return out

    def _letterbox_masks(self, masks):
        n, h, w = masks.shape
        W, H = self.img_size, self.canvas_h
        _, px, py, rw, rh = letterbox_params(w, h, W, H)
        out = np.zeros((n, H, W), np.uint8)
        for k in range(n):
            out[k, py:py + rh, px:px + rw] = cv2.resize(masks[k].astype(np.uint8), (rw, rh),
                                                        interpolation=cv2.INTER_NEAREST)
        return out

    def __getitem__(self, i):
        image = Image.open(self.image_path(i)).convert("RGB")
        ann = self.annotation(i)
        keep = ~ann["ignore"] if self.train else np.ones(len(ann["labels"]), bool)
        boxes, labels = ann["boxes"][keep], ann["labels"][keep]
        masks = self._letterbox_masks(self.masks(i)[keep])
        image, scale, px, py = letterbox(image, self.img_size)
        W, H = image.size
        s, xmax = float(H), W / float(H)
        b = boxes * scale
        if len(b):
            b[:, [0, 2]] += px
            b[:, [1, 3]] += py
        if self.aug:
            arr = hsv_jitter(np.asarray(image), self.aug["hue"], self.aug["saturation"], self.aug["exposure"])
            j = self.aug["jitter"]                        # = voc.random_scale_translate (same draws), masks too
            sc = np.random.uniform(1.0 - j, 1.0 + j)
            tx, ty = np.random.uniform(-j, j, 2) * np.array([W, H])
            m = np.array([[sc, 0, (1.0 - sc) * W / 2.0 + tx], [0, sc, (1.0 - sc) * H / 2.0 + ty]], dtype=np.float32)
            arr = cv2.warpAffine(arr, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            if len(b):
                masks = np.stack([cv2.warpAffine(x, m, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                                  for x in masks], 0)
                b = b.copy()
                b[:, [0, 2]] = b[:, [0, 2]] * sc + m[0, 2]
                b[:, [1, 3]] = b[:, [1, 3]] * sc + m[1, 2]
                area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, W)
                b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, H)
                area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                k = (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2) & (area1 >= 0.25 * np.maximum(area0, 1e-6))
                k &= masks.reshape(len(masks), -1).any(1)
                b, labels, masks = b[k], labels[k], masks[k]
            image = Image.fromarray(arr)
        targets = torch.zeros((len(b), 6), dtype=torch.float32)
        if len(b):
            targets[:, 1] = torch.from_numpy(labels).float()
            targets[:, 2] = torch.from_numpy((b[:, 0] + b[:, 2]) / 2 / s)
            targets[:, 3] = torch.from_numpy((b[:, 1] + b[:, 3]) / 2 / s)
            targets[:, 4] = torch.from_numpy((b[:, 2] - b[:, 0]) / s)
            targets[:, 5] = torch.from_numpy((b[:, 3] - b[:, 1]) / s)
            targets[:, [2, 4]] = targets[:, [2, 4]].clamp(0.0, xmax)
            targets[:, [3, 5]] = targets[:, [3, 5]].clamp(0.0, 1.0)
        mk = torch.from_numpy(np.ascontiguousarray(masks)).reshape(-1, H, W)
        img = to_tensor(image)
        if self.hflip_prob and np.random.random() < self.hflip_prob:
            img = torch.flip(img, [-1])
            targets[:, 2] = xmax - targets[:, 2]
            mk = torch.flip(mk, [-1])
        return i, img, {"boxes": targets, "masks": mk}

    @staticmethod
    def collate_fn(batch):
        idx, images, targets = zip(*batch)
        boxes, masks = [], []
        for b, t in enumerate(targets):
            if t["boxes"].numel():
                bt = t["boxes"].clone()
                bt[:, 0] = b
                boxes.append(bt)
                masks.append(t["masks"])
        H, W = images[0].shape[-2:]
        return list(idx), torch.stack(images, 0), {
            "boxes": torch.cat(boxes, 0) if boxes else torch.zeros((0, 6)),
            "masks": torch.cat(masks, 0) if masks else torch.zeros((0, H, W), dtype=torch.uint8)}
