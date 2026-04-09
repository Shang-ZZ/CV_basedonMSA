# TinyVision

**A YOLO-like object detector built on Memory Sparse Attention (MSA) ideas.**

Demonstrates how MSA's three core mechanisms (KV compression, neural routing,
sparse attention) transfer from NLP long-context to 2D computer vision.

------

## What This Demonstrates

| MSA concept (NLP)      | TinyVision mapping (CV)                                   |
| ---------------------- | --------------------------------------------------------- |
| Long document sequence | P3 feature map `[B, 128, 32, 32]`                         |
| N documents            | **16 spatial regions** (4×4 grid over P3)                 |
| `_pool_doc_kv()`       | `AvgPool2d(8,8)` compresses each 8×8 P3 block → 1 vector  |
| Query tokens           | P4 detection positions `[B, 256, 16×16]`                  |
| Routing scores Q×K     | Each P4 cell scores all 16 P3 regions                     |
| Top-K doc selection    | **Top-4 region selection** at inference                   |
| Sparse attention       | Cross-attention to top-4 regions only (4× context saving) |
| InfoNCE routing loss   | CE loss: route to the region containing the GT box        |

------

## Task

Detect colored shapes (circle / rectangle / triangle) on synthetic images.

```
Input image (256×256):
  ┌─────────────────┐
  │   [CIRCLE]      │  ← center in region R5 (row=1, col=1)
  │                 │
  │         [RECT]  │  ← center in region R14 (row=3, col=2)
  └─────────────────┘

Detection grid: 16×16 (256 cells, stride=16)
P3 regions:      4×4 (16 regions, each covers 64×64 pixels)

MSA routing for the circle query:
  scores: R5=0.83, R6=0.21, R1=0.18, R9=0.11, ...
  → select top-4: [R5, R6, R1, R9]
  → sparse attention over only these 4 regions (vs 16 full attention)
```

------

## Quick Start

```bash
pip install torch torchvision Pillow numpy tqdm matplotlib
```

```bash
# Architecture overview (no training needed)
python -m tinyvision info

# Smoke test: one forward + backward pass
python -m tinyvision quicktest

# Train (~10 min on GPU, ~40 min on CPU)
python -m tinyvision train

# Demo with routing visualization
python -m tinyvision demo --n 6
```

------

## Three-Stage Inference (MSA in CV)

```
Stage 1 — Feature Extraction + Region Pooling
  P3 [B,128,32,32] → AvgPool(8,8) → 16 region vectors [B,16,128]
  This is identical to MSA's document KV compression.

Stage 2 — Routing: P4 queries score P3 regions
  routing_scores [B, 256_queries, 16_regions]
  Each detection position asks: "which P3 region is most relevant to me?"
  Top-4 selected via argmax.

Stage 3 — Sparse Cross-Attention
  Training:  soft attention over all 16 (for correct gradients)
  Inference: hard top-4 selection → attend to 4/16 regions = 75% memory saved
```

------

## Key Metric: routing_acc

```
routing_acc = (predicted top-1 region == gold region containing GT box)

Epoch  1:  routing_acc ≈ 0.06  (random, 1/16 chance)
Epoch 10:  routing_acc ≈ 0.50
Epoch 30:  routing_acc ≈ 0.80-0.90  ← model learned to route correctly
```

This metric directly shows whether the MSA routing mechanism is learning.

------

## Demo Output

For each test image, two panels are generated:

```
┌──────────────────────┬──────────────────────────┐
│  Detection Output    │  MSA Routing Heatmap      │
│                      │                           │
│  [circle 0.87]       │  R0  R1  R2  R3           │
│    ●                 │  dim dim dim dim           │
│           [rect]     │  R4  R5  R6  R7           │
│            ■         │  dim HOT HOT dim           │
│                      │  R8  R9  R10 R11           │
│                      │  dim dim dim dim           │
│ (dashed=GT, solid=pred)  R12 R13 R14 R15         │
│                      │  dim dim HOT dim           │
└──────────────────────┴──────────────────────────┘
  Hot regions = where the model chose to look
  Cold regions = skipped by sparse routing
```

------

## File Structure

```
tinyvision/
├── config.py     — ModelConfig, TrainConfig
├── dataset.py    — Synthetic shape generation + YOLO targets + MSA region targets
├── backbone.py   — 4-stage CNN → P3 [32×32] and P4 [16×16]
├── neck.py       — MSA Region Routing Neck  ← KEY FILE
│                   pool_regions / compute_routing_scores /
│                   sparse_cross_attn / routing_loss
├── model.py      — TinyVision: backbone + neck + detection head
├── loss.py       — obj_loss + box_loss + cls_loss + routing_loss
├── train.py      — Training loop (tracks routing_acc separately)
├── inference.py  — Detect + routing heatmap visualization
└── __main__.py   — CLI (train/demo/info/quicktest)
```

------

## Architecture (~1.3M parameters)

```
Backbone    ~390K   4 conv stages → P3(128ch) + P4(256ch)
MSA Neck    ~240K   region routing (4 proj layers + FFN + LayerNorm)
Det. Head   ~660K   2 conv layers + 1×1 pred conv
─────────────────
Total      ~1.3M
```

------

## Comparison with MSA-main

|                     | MSA-main              | TinyVision            |
| ------------------- | --------------------- | --------------------- |
| Domain              | NLP (100M tokens)     | CV (256×256 image)    |
| "Documents"         | text documents        | P3 spatial regions    |
| Routing             | cosine similarity Q×K | dot-product Q×K       |
| Sparsity            | top-K docs            | top-K regions         |
| Routing loss        | InfoNCE               | Cross-entropy         |
| Routing supervision | gold document ID      | GT box → region index |
| Scale               | 3B params             | ~1.3M params          |
