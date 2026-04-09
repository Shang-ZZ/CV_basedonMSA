"""
TinyVision configuration.

Quick presets:
  ModelConfig()                       — default: custom backbone, 16 regions, single-scale
  ModelConfig(n_regions=64, n_reg_side=8, pool_size=4)  — 8×8=64 region neck
  ModelConfig(backbone_type='resnet18')                  — ResNet-18 backbone
  ModelConfig(backbone_type='mobilenet_v2')              — MobileNetV2 backbone
  ModelConfig(multiscale=True)        — P3+P4 dual detection heads
  ModelConfig(dynamic_topk=True)      — adaptive top-K routing
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """
    Model architecture.

    Spatial dimensions (default):
      img_size=256, stride=16 → P4 detection grid = 16×16
      P3 (stride 8): 32×32 feature map → divided into 4×4=16 regions
      P4 (stride16): 16×16 feature map → 256 detection query positions

    MSA parameters:
      n_regions=16  → 4×4 spatial regions (each covers 64×64 pixels)
      top_k=4       → each query attends to 4 most relevant regions
      pool_size=8   → region pooling: 8×8 P3 patches → 1 region vector

    Extension flags (v2):
      backbone_type  — 'custom' | 'resnet18' | 'mobilenet_v2'
      multiscale     — add P3-scale detection head alongside P4 head
      dynamic_topk   — entropy-based adaptive top-K routing
      topk_min/max   — range when dynamic_topk=True
      freeze_backbone— freeze pretrained backbone weights during training

    64-region preset:
      ModelConfig(n_regions=64, n_reg_side=8, pool_size=4)
      Requires: p3_size(32) / pool_size(4) == n_reg_side(8) ✓
    """
    img_size:    int = 256
    num_classes: int = 3         # circle=0, rectangle=1, triangle=2

    # Backbone
    backbone_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    backbone_type:     str = 'custom'   # 'custom' | 'resnet18' | 'mobilenet_v2'
    freeze_backbone:   bool = False     # freeze pretrained weights (fine-tuning mode)

    # MSA Neck
    n_regions:    int = 16       # number of spatial "documents" (4×4 grid over P3)
    n_reg_side:   int = 4        # sqrt(n_regions) — regions per side
    pool_size:    int = 8        # spatial pooling size in P3 space (8×8 → 1 vector)
    top_k:        int = 4        # top-K regions selected at inference
    n_heads:      int = 4        # attention heads in MSA neck
    d_neck:       int = 128      # neck projection dimension (= n_heads × head_dim)

    # Detection Head (P4 scale — primary)
    grid_size:    int = 16       # detection grid H = W (stride = img_size / grid_size)
    stride:       int = 16       # effective stride from image to grid

    # Multi-scale detection (v2)
    multiscale:   bool = False   # enable P3+P4 dual detection heads

    # Dynamic Top-K routing (v2)
    dynamic_topk: bool = False   # entropy-based adaptive K per query
    topk_min:     int = 2        # minimum K (used when dynamic_topk=True)
    topk_max:     int = 8        # maximum K (used when dynamic_topk=True)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def p3_size(self):
        """P3 feature map spatial size (stride 8)."""
        return self.img_size // (self.stride // 2)   # 32

    @property
    def p3_region_size(self):
        """Each P3 region's spatial size (in P3 feature pixels)."""
        return self.p3_size // self.n_reg_side        # default: 8

    @property
    def grid_size_p3(self):
        """P3-scale detection grid size (for multiscale head)."""
        return self.img_size // (self.stride // 2)   # 32

    @property
    def n_query(self):
        """Total P4 detection query positions."""
        return self.grid_size * self.grid_size        # 256

    @property
    def c3(self):
        """P3 channels (from backbone)."""
        return self.backbone_channels[2]             # 128

    @property
    def c4(self):
        """P4 channels (from backbone)."""
        return self.backbone_channels[3]             # 256


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size:  int   = 32
    n_epochs:    int   = 30
    lr:          float = 1e-3
    weight_decay: float = 0.0
    lr_step:     int   = 20          # epoch at which LR is multiplied by lr_gamma
    lr_gamma:    float = 0.1

    # Loss weights
    obj_weight:     float = 1.0
    noobj_weight:   float = 0.5      # weight for cells without objects
    box_weight:     float = 2.0
    cls_weight:     float = 1.0
    routing_weight: float = 0.2      # weight for auxiliary routing loss
    p3_weight:      float = 0.5      # relative weight for P3-scale loss (multiscale)

    conf_threshold: float = 0.4
    nms_threshold:  float = 0.4

    n_train: int = 8000
    n_val:   int = 1000
