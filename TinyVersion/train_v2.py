"""
TinyVision v2 — improved model training script.

Improvements over baseline:
  - 64 P3 regions (8×8 grid, pool_size=4) for finer-grained spatial routing
  - Multi-scale detection heads: P3 (32×32) + P4 (16×16)
  - Dynamic Top-K routing at inference (entropy-based adaptive K, range 2-8)
  - Longer warmup (8% of steps) for multi-scale stability

Run:
  python train_v2.py
  (or: E:/Python3.11.0/python.exe train_v2.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tinyvision.config import ModelConfig, TrainConfig
from tinyvision.train import train

model_cfg = ModelConfig(
    # Ext-3: 64 regions (8×8 grid over P3 instead of 4×4)
    n_regions  = 64,
    n_reg_side = 8,
    pool_size  = 4,

    # Ext-2: multi-scale P3 + P4 detection heads
    multiscale = True,

    # Ext-5: dynamic top-K routing at inference
    dynamic_topk = True,
    topk_min     = 2,
    topk_max     = 10,   # up to 10/64 regions selected dynamically

    # Keep custom backbone (fast, fair synthetic-data comparison)
    backbone_type = 'custom',
    num_classes   = 3,
)

train_cfg = TrainConfig(
    batch_size      = 32,
    n_epochs        = 30,
    lr              = 1e-3,
    weight_decay    = 0.0,
    obj_weight      = 1.0,
    noobj_weight    = 0.5,
    box_weight      = 2.0,
    cls_weight      = 1.0,
    routing_weight  = 0.3,   # slightly higher routing weight for 64-region task
    p3_weight       = 0.6,   # P3-scale head contribution to total loss
    n_train         = 8000,
    n_val           = 1000,
)

OUT = 'F:/ShangResearchPaper/exampleTorebuild/Tiny_CV_basedonMSA/model_output/v2'

print('TinyVision v2 — Improved Model')
print('=' * 60)
print(f'Backbone:     {model_cfg.backbone_type}  (custom CNN)')
print(f'Regions:      {model_cfg.n_regions}  ({model_cfg.n_reg_side}x{model_cfg.n_reg_side} grid, pool_size={model_cfg.pool_size})')
print(f'Multi-scale:  P3({model_cfg.grid_size_p3}x{model_cfg.grid_size_p3}) + P4({model_cfg.grid_size}x{model_cfg.grid_size})')
print(f'Dynamic TopK: True  (K in [{model_cfg.topk_min}, {model_cfg.topk_max}])')
print(f'Output dir:   {OUT}')
print()

train(model_cfg=model_cfg, train_cfg=train_cfg, out_dir=OUT)
