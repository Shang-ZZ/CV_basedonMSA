"""
TinyVision datasets.

Synthetic dataset (default):
  ShapeDataset — generates circles/rectangles/triangles on synthetic backgrounds.

Real-world datasets (Ext-4):
  COCODataset — COCO-format JSON annotations + image directory.
  VOCDataset  — Pascal VOC XML annotations.

Both real-world loaders produce the same output contract as ShapeDataset
and call the shared build_targets() function.

Target tensors per image (single-scale, default):
  obj_mask:       [G, G]      bool  — True if grid cell is assigned a GT box
  box_targets:    [G, G, 4]   float — (cx_off, cy_off, w_norm, h_norm)
  cls_targets:    [G, G]      long  — class index, -1 if no object
  region_targets: [G, G]      long  — gold P3 region index, -1 if no object

Multi-scale mode (config.multiscale=True):
  Returns a 9-tuple:
    (img,
     obj_p3, box_p3, cls_p3, reg_p3,   # P3-scale targets (G=32)
     obj_p4, box_p4, cls_p4, reg_p4)   # P4-scale targets (G=16)

  P3-scale detection head uses stride=8 (finer, detects smaller objects).
  P4-scale detection head uses stride=16 (coarser, detects larger objects).
  Objects are assigned to scales based on normalized area:
    area < SMALL_THRESHOLD → P3 scale
    area >= SMALL_THRESHOLD → P4 scale
    (or assigned to both, ensuring all objects are covered)

COCO quick start:
  ds = COCODataset(
      img_dir='path/to/coco/train2017',
      ann_file='path/to/annotations/instances_train2017.json',
      config=ModelConfig(num_classes=80),
  )

VOC quick start:
  ds = VOCDataset(
      root='path/to/VOCdevkit',
      year='2012',
      split='train',
      config=ModelConfig(num_classes=20),
  )
"""
import os
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader

from .config import ModelConfig, TrainConfig

CLASSES = ['circle', 'rectangle', 'triangle']

# Area threshold for scale assignment in multi-scale mode:
# boxes with normalized area < 0.05 (5% of image) go to fine P3 scale.
SMALL_AREA_THRESHOLD = 0.05

COLORS = [
    (220,  40,  40),
    ( 40, 180,  40),
    ( 40,  80, 220),
    (200, 160,   0),
    (160,  40, 200),
    (  0, 180, 180),
    (220, 100,   0),
    (140,   0, 100),
]


# ─── Synthetic image generation ───────────────────────────────────────────────

def _iou(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def generate_image(img_size: int = 256, max_objects: int = 3, seed: int = None):
    """
    Generate one synthetic image with 1-3 non-overlapping colored shapes.

    Returns:
      img_np:      (H, W, 3) uint8 numpy array
      annotations: list of {'class': int, 'bbox': [x1,y1,x2,y2]}
    """
    rng    = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    bg  = tuple(int(v) for v in np_rng.randint(210, 245, 3))
    img = Image.new('RGB', (img_size, img_size), bg)
    draw = ImageDraw.Draw(img)

    n_objects    = rng.randint(1, max_objects)
    placed_boxes = []
    annotations  = []

    for _ in range(n_objects):
        cls   = rng.randint(0, 2)
        color = rng.choice(COLORS)
        size  = rng.randint(22, 68)

        for _ in range(20):
            cx = rng.randint(size + 4, img_size - size - 4)
            cy = rng.randint(size + 4, img_size - size - 4)

            if cls == 0:
                x1, y1 = cx - size, cy - size
                x2, y2 = cx + size, cy + size
            elif cls == 1:
                w = rng.randint(20, size * 2)
                h = rng.randint(20, size * 2)
                x1, y1 = cx - w // 2, cy - h // 2
                x2, y2 = cx + w // 2, cy + h // 2
            else:
                x1, y1 = cx - size, cy - size
                x2, y2 = cx + size, cy + size

            bbox = [max(0, x1), max(0, y1),
                    min(img_size, x2), min(img_size, y2)]

            if all(_iou(bbox, pb) < 0.15 for pb in placed_boxes):
                placed_boxes.append(bbox)
                break
        else:
            continue

        if cls == 0:
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif cls == 1:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            pts = [(cx, y1), (x1, y2), (x2, y2)]
            draw.polygon(pts, fill=color)

        annotations.append({'class': cls, 'bbox': bbox})

    return np.array(img, dtype=np.uint8), annotations


# ─── Target assignment ─────────────────────────────────────────────────────────

def build_targets(
    annotations: List[Dict],
    config: ModelConfig,
    grid_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    YOLO-style target assignment + MSA routing supervision.

    For each GT box:
      1. Find the grid cell at the GT center for the given grid_size.
      2. Assign obj_mask, box offsets, class label.
      3. Compute gold_region: which P3 Region the GT center falls into.
         Region index is independent of grid_size (always in P3 space).

    Args:
      annotations:  list of {'class': int, 'bbox': [x1,y1,x2,y2]}
      config:       ModelConfig
      grid_size:    override grid size (for multi-scale; defaults to config.grid_size)

    Returns:
      obj_mask:       [G, G]   bool
      box_targets:    [G, G, 4] float (cx_off, cy_off, w, h) — all normalised
      cls_targets:    [G, G]   long  (-1 if no object)
      region_targets: [G, G]   long  (-1 if no object)
    """
    G   = grid_size if grid_size is not None else config.grid_size
    S   = config.img_size
    p3  = config.p3_size          # 32
    rs  = config.p3_region_size   # 8 (or 4 for 64-region config)
    nrs = config.n_reg_side       # 4 (or 8)

    obj_mask       = torch.zeros(G, G, dtype=torch.bool)
    box_targets    = torch.zeros(G, G, 4)
    cls_targets    = torch.full((G, G), -1, dtype=torch.long)
    region_targets = torch.full((G, G), -1, dtype=torch.long)

    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        cls = ann['class']

        cx = (x1 + x2) / 2.0 / S
        cy = (y1 + y2) / 2.0 / S
        w  = (x2 - x1) / S
        h  = (y2 - y1) / S

        gi = min(int(cx * G), G - 1)
        gj = min(int(cy * G), G - 1)

        obj_mask[gj, gi]    = True
        cls_targets[gj, gi] = cls
        box_targets[gj, gi] = torch.tensor([
            cx * G - gi,
            cy * G - gj,
            w,
            h,
        ])

        # MSA routing supervision (gold P3 region — same for all grid scales)
        cx_p3 = cx * p3
        cy_p3 = cy * p3
        rcol  = min(int(cx_p3 / rs), nrs - 1)
        rrow  = min(int(cy_p3 / rs), nrs - 1)
        region_targets[gj, gi] = rrow * nrs + rcol

    return obj_mask, box_targets, cls_targets, region_targets


def build_targets_multiscale(
    annotations: List[Dict],
    config: ModelConfig,
) -> Tuple:
    """
    Build targets for both P3 (stride-8) and P4 (stride-16) detection scales.

    Scale assignment:
      normalized_area = w * h  (fraction of image area)
      area < SMALL_AREA_THRESHOLD  → assigned to P3 scale (fine)
      area >= SMALL_AREA_THRESHOLD → assigned to P4 scale (coarse)
      If area falls between, assign to BOTH scales for better recall.

    Returns 8 tensors:
      (obj_p3, box_p3, cls_p3, reg_p3,
       obj_p4, box_p4, cls_p4, reg_p4)
    """
    G_p4 = config.grid_size      # 16
    G_p3 = config.grid_size_p3   # 32
    S    = config.img_size

    # Split annotations by scale
    ann_p3, ann_p4 = [], []
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        w = (x2 - x1) / S
        h = (y2 - y1) / S
        area = w * h

        if area < SMALL_AREA_THRESHOLD:
            ann_p3.append(ann)
            ann_p4.append(ann)   # also assign to P4 for recall
        else:
            ann_p4.append(ann)
            # assign large-ish objects to P3 too if they're not too big
            if area < 0.25:
                ann_p3.append(ann)

    tgt_p3 = build_targets(ann_p3, config, grid_size=G_p3)
    tgt_p4 = build_targets(ann_p4, config, grid_size=G_p4)
    return tgt_p3 + tgt_p4   # 8-tuple


# ─── Synthetic dataset ────────────────────────────────────────────────────────

class ShapeDataset(Dataset):
    """
    Deterministic synthetic shape dataset.
    seed = base_seed + idx ensures reproducibility.
    """

    def __init__(self, n_samples: int, config: ModelConfig, base_seed: int = 0):
        self.n_samples  = n_samples
        self.config     = config
        self.base_seed  = base_seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seed = self.base_seed + idx
        img_np, annotations = generate_image(
            img_size=self.config.img_size,
            max_objects=3,
            seed=seed,
        )
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        if self.config.multiscale:
            tgts = build_targets_multiscale(annotations, self.config)
            return (img,) + tgts   # 9-tuple
        else:
            tgts = build_targets(annotations, self.config)
            return (img,) + tgts   # 5-tuple


# ─── COCO dataset (Ext-4) ─────────────────────────────────────────────────────

# Standard 80-class COCO category names (in category_id order after mapping)
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush',
]


class COCODataset(Dataset):
    """
    COCO-format object detection dataset.

    Loads images from `img_dir` and annotations from `ann_file`
    (instances_train2017.json / instances_val2017.json format).

    Does NOT require pycocotools — parses the JSON directly.

    Quick start:
      config = ModelConfig(num_classes=80)
      ds = COCODataset('coco/train2017', 'coco/annotations/instances_train2017.json', config)

    The dataset resizes all images to config.img_size × config.img_size and
    scales bounding boxes accordingly before calling build_targets().
    """

    def __init__(
        self,
        img_dir:   str,
        ann_file:  str,
        config:    ModelConfig,
        min_area:  float = 32.0,    # skip boxes smaller than this many pixels²
        max_dets:  int   = 20,      # max GT boxes per image
    ):
        self.img_dir  = Path(img_dir)
        self.config   = config
        self.min_area = min_area
        self.max_dets = max_dets

        print(f'Loading COCO annotations from {ann_file} ...')
        with open(ann_file) as f:
            data = json.load(f)

        # Build category_id → contiguous index mapping
        cats     = sorted(data['categories'], key=lambda c: c['id'])
        self.cat_map = {c['id']: i for i, c in enumerate(cats)}
        self.class_names = [c['name'] for c in cats]

        # Build image_id → filename lookup
        id2img = {img['id']: img['file_name'] for img in data['images']}
        id2hw  = {img['id']: (img['height'], img['width']) for img in data['images']}

        # Group annotations by image_id; skip crowd and tiny boxes
        from collections import defaultdict
        ann_by_img = defaultdict(list)
        for ann in data['annotations']:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w * h < min_area:
                continue
            ann_by_img[ann['image_id']].append(ann)

        # Keep only images that have at least 1 annotation
        self.samples = []
        for img_id, anns in ann_by_img.items():
            fname = id2img.get(img_id)
            if fname is None:
                continue
            fpath = self.img_dir / fname
            if not fpath.exists():
                continue
            self.samples.append((fpath, id2hw[img_id], anns))

        print(f'  {len(self.samples)} images with annotations.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, (orig_h, orig_w), coco_anns = self.samples[idx]
        S = self.config.img_size

        # Load and resize image
        img_pil = Image.open(fpath).convert('RGB')
        img_pil = img_pil.resize((S, S), Image.BILINEAR)
        img = torch.from_numpy(np.array(img_pil, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        # Convert COCO annotations → standard format, scaled to new size
        scale_x = S / orig_w
        scale_y = S / orig_h
        annotations = []
        for ann in coco_anns[:self.max_dets]:
            x, y, w, h = ann['bbox']
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            cls_idx = self.cat_map.get(ann['category_id'], 0)
            annotations.append({'class': cls_idx, 'bbox': [x1, y1, x2, y2]})

        if self.config.multiscale:
            tgts = build_targets_multiscale(annotations, self.config)
            return (img,) + tgts
        else:
            tgts = build_targets(annotations, self.config)
            return (img,) + tgts


# ─── Pascal VOC dataset (Ext-4) ───────────────────────────────────────────────

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


class VOCDataset(Dataset):
    """
    Pascal VOC object detection dataset.

    Directory structure expected:
      root/
        VOC{year}/
          JPEGImages/     ← images (*.jpg)
          Annotations/    ← XML annotation files
          ImageSets/Main/{split}.txt

    Quick start:
      config = ModelConfig(num_classes=20)
      ds = VOCDataset('path/to/VOCdevkit', year='2012', split='train', config=config)
    """

    def __init__(
        self,
        root:   str,
        year:   str = '2012',
        split:  str = 'train',      # 'train' | 'val' | 'trainval'
        config: ModelConfig = None,
    ):
        self.config = config or ModelConfig(num_classes=20)
        self.cls2idx = {c: i for i, c in enumerate(VOC_CLASSES)}

        voc_root  = Path(root) / f'VOC{year}'
        split_file = voc_root / 'ImageSets' / 'Main' / f'{split}.txt'

        img_ids = split_file.read_text().strip().split('\n')

        self.samples = []
        for img_id in img_ids:
            img_id = img_id.strip()
            if not img_id:
                continue
            img_path = voc_root / 'JPEGImages' / f'{img_id}.jpg'
            ann_path = voc_root / 'Annotations' / f'{img_id}.xml'
            if img_path.exists() and ann_path.exists():
                self.samples.append((img_path, ann_path))

        print(f'VOC{year} {split}: {len(self.samples)} images')

    def _parse_xml(self, ann_path: Path, scale_x: float, scale_y: float) -> List[Dict]:
        """Parse VOC XML annotation file → standard annotation format."""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            if name not in self.cls2idx:
                continue
            diff = obj.find('difficult')
            if diff is not None and int(diff.text) == 1:
                continue  # skip difficult instances

            bndbox = obj.find('bndbox')
            x1 = float(bndbox.find('xmin').text) * scale_x
            y1 = float(bndbox.find('ymin').text) * scale_y
            x2 = float(bndbox.find('xmax').text) * scale_x
            y2 = float(bndbox.find('ymax').text) * scale_y
            annotations.append({
                'class': self.cls2idx[name],
                'bbox':  [x1, y1, x2, y2],
            })
        return annotations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        S = self.config.img_size

        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        img_pil = img_pil.resize((S, S), Image.BILINEAR)
        img = torch.from_numpy(np.array(img_pil, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        scale_x = S / orig_w
        scale_y = S / orig_h
        annotations = self._parse_xml(ann_path, scale_x, scale_y)

        if self.config.multiscale:
            tgts = build_targets_multiscale(annotations, self.config)
            return (img,) + tgts
        else:
            tgts = build_targets(annotations, self.config)
            return (img,) + tgts


# ─── Collate functions ────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate for single-scale mode (5-tuple per sample)."""
    imgs, obj_masks, box_tgts, cls_tgts, region_tgts = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(obj_masks),
        torch.stack(box_tgts),
        torch.stack(cls_tgts),
        torch.stack(region_tgts),
    )


def collate_fn_multiscale(batch):
    """Collate for multi-scale mode (9-tuple per sample)."""
    (imgs,
     obj_p3, box_p3, cls_p3, reg_p3,
     obj_p4, box_p4, cls_p4, reg_p4) = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(obj_p3), torch.stack(box_p3),
        torch.stack(cls_p3), torch.stack(reg_p3),
        torch.stack(obj_p4), torch.stack(box_p4),
        torch.stack(cls_p4), torch.stack(reg_p4),
    )


# ─── DataLoader factories ─────────────────────────────────────────────────────

def get_dataloader(n_samples, config, batch_size, base_seed=0, shuffle=True):
    """Single-scale synthetic shape DataLoader."""
    ds  = ShapeDataset(n_samples, config, base_seed=base_seed)
    cfn = collate_fn_multiscale if config.multiscale else collate_fn
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=cfn, pin_memory=True, num_workers=0,
    )


def get_dataloader_from_dataset(dataset, config, batch_size, shuffle=True):
    """
    DataLoader factory for any dataset instance (ShapeDataset / COCODataset / VOCDataset).
    Automatically selects the appropriate collate_fn.
    """
    cfn = collate_fn_multiscale if config.multiscale else collate_fn
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=cfn, pin_memory=True, num_workers=0,
    )
