"""
TinyVision — Full detection model (Backbone + MSA Neck + Detection Head).

Architecture (single-scale, default):

  Input [B, 3, 256, 256]
      │
  Backbone  (custom CNN / ResNet-18 / MobileNetV2)
      ├─► P3 [B, 128, 32, 32]   fine-grained "documents"
      └─► P4 [B, 256, 16, 16]   detection queries
              │
  MSA Region Routing Neck
      │   Stage 1: pool P3 → 16 (or 64) region vectors
      │   Stage 2: each P4 position scores regions (+ 2D RoPE + softcap)
      │   Stage 3: attend to top-K regions (+ gated residuals)
      │
  Enhanced P4 [B, 256, 16, 16]
      │
  YOLO Detection Head  [B, 16, 16, 8]

Multi-scale extension (config.multiscale=True):
  Enhanced P4  ─────────────────► Head P4 [B, 16, 16, 8]
      │
  FPN top-down path
  Upsample(2×) → lateral_conv → + raw P3
      │
  Enhanced P3 [B, 128, 32, 32]
      │
  Head P3 [B, 32, 32, 8]

Detection output per grid cell:
  obj_conf — raw logit → sigmoid → object probability
  cx_off   — sigmoid → x offset within cell [0,1]
  cy_off   — sigmoid → y offset within cell [0,1]
  w_norm   — sigmoid → normalized width
  h_norm   — sigmoid → normalized height
  cls[0-2] — logits → softmax → class probabilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config import ModelConfig
from .backbone import build_backbone
from .neck import MSARegionRoutingNeck


# ─── Detection Head ───────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    YOLO-style single-scale detection head.

    Input:  feature map [B, in_ch, G, G]
    Output: predictions [B, G, G, 5 + num_classes]
    """
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        out = 5 + num_classes   # obj + cx + cy + w + h + classes

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,      in_ch,      3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch,      in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(in_ch // 2, out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pred(x)                    # [B, out, G, G]
        return x.permute(0, 2, 3, 1)        # [B, G, G, out]


# ─── FPN top-down lateral connection ──────────────────────────────────────────

class FPNTopDown(nn.Module):
    """
    Lightweight FPN top-down path: fuses enhanced P4 back into P3 scale.

    Upsample enhanced_P4 (stride 16) by 2× → merge with raw P3 (stride 8)
    via a lateral 1×1 projection + element-wise addition.

    This gives the P3 detection head access to the MSA-enriched context from P4
    without running a second MSA neck on P3.
    """
    def __init__(self, c4: int, c3: int):
        super().__init__()
        # Project C4 → C3 before adding to raw P3
        self.lateral = nn.Sequential(
            nn.Conv2d(c4, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

    def forward(self, enhanced_p4: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """
        enhanced_p4: [B, C4, H/16, W/16]
        p3:          [B, C3, H/8,  W/8 ]
        returns:     [B, C3, H/8,  W/8 ]
        """
        p4_up = F.interpolate(enhanced_p4, scale_factor=2, mode='nearest')  # [B,C4,H/8,W/8]
        p4_proj = self.lateral(p4_up)   # [B, C3, H/8, W/8]
        return p3 + p4_proj             # residual merge


# ─── TinyVision ───────────────────────────────────────────────────────────────

class TinyVision(nn.Module):
    """
    Tiny YOLO-like detector with MSA Region Routing Neck.

    Supports:
      - Pluggable backbone (custom CNN / ResNet-18 / MobileNetV2)
      - 16 or 64 P3 regions (config.n_regions)
      - Multi-scale detection heads (config.multiscale)
      - Dynamic Top-K routing (config.dynamic_topk)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Backbone (Ext-1: pluggable) ──────────────────────────────────────
        self.backbone = build_backbone(config)

        # ── MSA Region Routing Neck ──────────────────────────────────────────
        self.neck = MSARegionRoutingNeck(config)

        # ── Detection Head(s) ────────────────────────────────────────────────
        # Primary: P4-scale head (always present)
        self.head = DetectionHead(config.c4, config.num_classes)

        # Multi-scale: add P3-scale head + FPN path (Ext-2)
        if config.multiscale:
            self.fpn      = FPNTopDown(config.c4, config.c3)
            self.head_p3  = DetectionHead(config.c3, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def param_count(self) -> Dict[str, int]:
        def count(m): return sum(p.numel() for p in m.parameters())
        d = {
            'backbone': count(self.backbone),
            'neck':     count(self.neck),
            'head_p4':  count(self.head),
            'total':    count(self),
        }
        if self.config.multiscale:
            d['fpn']     = count(self.fpn)
            d['head_p3'] = count(self.head_p3)
        return d

    def forward(
        self,
        images: torch.Tensor,                          # [B, 3, H, W]
        region_targets: Optional[torch.Tensor] = None, # [B, G, G]  (training)
        sparse: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Single-scale returns:
          pred:           [B, G, G, 8]
          routing_scores: [B, N_q, N_r]
          routing_loss:   scalar or None

        Multi-scale (config.multiscale=True) returns additionally:
          pred_p3:        [B, G_p3, G_p3, 8]
        """
        # ── Feature extraction ───────────────────────────────────────────────
        p3, p4 = self.backbone(images)

        # ── MSA Region Routing Neck ──────────────────────────────────────────
        enhanced_p4, routing_scores, routing_loss = self.neck(
            p3, p4,
            region_targets=region_targets,
            sparse=sparse,
        )

        # ── P4 Detection Head ────────────────────────────────────────────────
        pred_p4 = self.head(enhanced_p4)   # [B, G, G, 8]

        out = {
            'pred':           pred_p4,
            'routing_scores': routing_scores,
            'routing_loss':   routing_loss,
        }

        # ── Multi-scale: FPN + P3 Head (Ext-2) ──────────────────────────────
        if self.config.multiscale:
            enhanced_p3 = self.fpn(enhanced_p4, p3)  # [B, C3, H/8, W/8]
            pred_p3     = self.head_p3(enhanced_p3)  # [B, G_p3, G_p3, 8]
            out['pred_p3'] = pred_p3

        return out

    # ── Inference utilities ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.4,
        nms_threshold:  float = 0.4,
        sparse: bool = True,
    ) -> List[Dict]:
        """
        Full inference pipeline with NMS.

        Returns per-image list of dicts:
          {'boxes': [N,4], 'scores': [N], 'classes': [N], 'routing_scores': [N_q, N_r]}
          boxes in (x1, y1, x2, y2) pixel coordinates.

        Multi-scale: merges P3 and P4 predictions before NMS.
        """
        self.eval()
        out = self.forward(images, sparse=sparse)
        routing_scores = out['routing_scores']   # [B, N_q, N_r]

        B = images.shape[0]
        S = self.config.img_size
        results = []

        for b in range(B):
            all_boxes, all_scores, all_cls = [], [], []

            # Decode P4 predictions
            boxes_p4, scores_p4, cls_p4 = _decode_pred(
                out['pred'][b], self.config.grid_size, S
            )
            all_boxes.append(boxes_p4)
            all_scores.append(scores_p4)
            all_cls.append(cls_p4)

            # Decode P3 predictions (multi-scale)
            if self.config.multiscale and 'pred_p3' in out:
                boxes_p3, scores_p3, cls_p3 = _decode_pred(
                    out['pred_p3'][b], self.config.grid_size_p3, S
                )
                all_boxes.append(boxes_p3)
                all_scores.append(scores_p3)
                all_cls.append(cls_p3)

            boxes  = torch.cat(all_boxes,  dim=0)
            scores = torch.cat(all_scores, dim=0)
            cls_idx = torch.cat(all_cls,   dim=0)

            # Confidence threshold
            mask   = scores > conf_threshold
            boxes  = boxes[mask]
            scores = scores[mask]
            cls_idx = cls_idx[mask]

            # NMS
            keep = _nms(boxes, scores, nms_threshold)
            results.append({
                'boxes':          boxes[keep].clamp(0, S),
                'scores':         scores[keep],
                'classes':        cls_idx[keep],
                'routing_scores': routing_scores[b],
            })

        return results


# ─── Decode helper ────────────────────────────────────────────────────────────

def _decode_pred(
    p: torch.Tensor,    # [G, G, out]
    G: int,
    S: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw predictions for one image at one scale."""
    num_classes = p.shape[-1] - 5

    obj_conf = torch.sigmoid(p[..., 0])
    cx_off   = torch.sigmoid(p[..., 1])
    cy_off   = torch.sigmoid(p[..., 2])
    w        = torch.sigmoid(p[..., 3]).clamp(0.01, 1.0)
    h        = torch.sigmoid(p[..., 4]).clamp(0.01, 1.0)
    cls_prob = torch.softmax(p[..., 5:], dim=-1)

    grid_col = torch.arange(G, device=p.device).float()
    grid_row = torch.arange(G, device=p.device).float()
    gy, gx   = torch.meshgrid(grid_row, grid_col, indexing='ij')

    cx_abs = (gx + cx_off) / G
    cy_abs = (gy + cy_off) / G

    x1 = (cx_abs - w / 2) * S
    y1 = (cy_abs - h / 2) * S
    x2 = (cx_abs + w / 2) * S
    y2 = (cy_abs + h / 2) * S

    boxes   = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
    scores  = obj_conf.view(-1)
    cls_idx = cls_prob.view(-1, num_classes).argmax(-1)

    return boxes, scores, cls_idx


# ─── Simple NMS (no torchvision dependency) ───────────────────────────────────

def _box_iou(box, boxes):
    ix1 = torch.max(box[0], boxes[:, 0])
    iy1 = torch.max(box[1], boxes[:, 1])
    ix2 = torch.min(box[2], boxes[:, 2])
    iy2 = torch.min(box[3], boxes[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    a0    = (box[2] - box[0]) * (box[3] - box[1])
    a1    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a0 + a1 - inter + 1e-6)


def _nms(boxes, scores, iou_thr):
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious  = _box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    return torch.tensor(keep, dtype=torch.long)
