"""
TinyVision Inference — Detection + MSA Routing Visualization.

Two key outputs per image:
  1. Detection result: bounding boxes with class labels and confidence scores
  2. Routing heatmap: which P3 regions each part of the image routed to
     This directly visualizes the MSA mechanism working in 2D space.

The routing visualization divides the image into a 4×4 grid overlay.
Regions selected by the detection queries are highlighted in warm colors.
Non-selected regions are shown faintly. This teaches the intuition:
"the model focuses only on the spatial areas likely to contain objects."

Usage:
  python -m tinyvision demo           # run on random generated images
  python -m tinyvision demo --n 8     # show 8 images
"""
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from .config import ModelConfig, TrainConfig
from .model import TinyVision
from .dataset import generate_image, CLASSES

CKPT_DIR = Path('checkpoints')
CLASS_COLORS = ['#e74c3c', '#2ecc71', '#3498db']   # red, green, blue


class TinyVisionInference:

    def __init__(self, model: TinyVision, config: ModelConfig, device: str):
        self.model  = model
        self.config = config
        self.device = device

    @classmethod
    def load(cls, ckpt_path=None, config_path=None, device=None):
        ckpt_path   = ckpt_path   or str(CKPT_DIR / 'best_model.pt')
        config_path = config_path or str(CKPT_DIR / 'config.json')

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(config_path) as f:
            raw = json.load(f)
        # Remove computed properties (they're not constructor args)
        for key in ('p3_size', 'p3_region_size', 'n_query', 'c3', 'c4'):
            raw.pop(key, None)
        cfg = ModelConfig(**raw)

        model = TinyVision(cfg).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"Loaded TinyVision (~{model.param_count()['total']/1e6:.2f}M params)")
        return cls(model, cfg, device)

    def _preprocess(self, img_np: np.ndarray) -> torch.Tensor:
        """numpy (H,W,3) uint8 → tensor (1,3,H,W) float32 in [0,1]"""
        t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def detect(self, img_np: np.ndarray, conf_thr=0.35, nms_thr=0.4, sparse=True):
        """Run detection on a single numpy image."""
        tensor  = self._preprocess(img_np)
        results = self.model.predict(tensor, conf_thr, nms_thr, sparse=sparse)
        return results[0]

    # ── Visualization ─────────────────────────────────────────────────────────

    def visualize(
        self,
        img_np: np.ndarray,
        result: dict,
        annotations=None,
        title: str = '',
        save_path: str = None,
    ):
        """
        Draw:
          Left panel  — detected boxes on the original image
          Right panel — MSA routing heatmap (which 4×4 regions were selected)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title or 'TinyVision — MSA Region Routing Demo', fontsize=13)

        # ── Left: Detection ───────────────────────────────────────────────────
        ax = axes[0]
        ax.imshow(img_np)
        ax.set_title('Detection Output', fontsize=11)
        ax.axis('off')

        boxes   = result['boxes'].cpu()
        scores  = result['scores'].cpu()
        classes = result['classes'].cpu()

        for box, sc, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            color = CLASS_COLORS[cls.item()]
            rect  = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"{CLASSES[cls]}: {sc:.2f}",
                    color=color, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, pad=1))

        # Draw GT boxes if provided
        if annotations:
            for ann in annotations:
                x1, y1, x2, y2 = ann['bbox']
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=1.5, edgecolor='white', facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)
                ax.text(x1, y2 + 10, f"GT:{CLASSES[ann['class']]}",
                        color='white', fontsize=7,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

        # ── Right: Routing heatmap ────────────────────────────────────────────
        ax2 = axes[1]
        ax2.imshow(img_np, alpha=0.5)
        ax2.set_title(
            f'MSA Routing Heatmap\n'
            f'(4×4 = {self.config.n_regions} P3 regions, top-{self.config.top_k} selected per query)',
            fontsize=11
        )
        ax2.axis('off')

        routing_scores = result['routing_scores'].cpu()  # [N_q, N_r]
        G   = self.config.grid_size    # 16
        nrs = self.config.n_reg_side   # 4
        S   = self.config.img_size     # 256

        # For each P3 region, compute average routing attention it received
        # routing_scores: [N_q, N_r] → mean over queries → [N_r]
        attn_weights    = torch.softmax(routing_scores, dim=-1)  # [N_q, N_r]
        region_attn     = attn_weights.mean(dim=0)               # [N_r]  mean attention per region
        region_map      = region_attn.view(nrs, nrs).numpy()     # [4, 4]

        # Normalize to [0,1] for color mapping
        vmin, vmax = region_map.min(), region_map.max()
        if vmax > vmin:
            region_map_norm = (region_map - vmin) / (vmax - vmin)
        else:
            region_map_norm = region_map

        region_px = S // nrs   # 64 pixels per region side

        cmap_warm = LinearSegmentedColormap.from_list(
            'routing', ['#1a1a2e', '#e94560', '#f5a623'], N=256
        )

        for rr in range(nrs):
            for rc in range(nrs):
                intensity = region_map_norm[rr, rc]
                color = cmap_warm(intensity)

                # Region rectangle
                rect = patches.Rectangle(
                    (rc * region_px, rr * region_px),
                    region_px, region_px,
                    linewidth=1.5, edgecolor='white',
                    facecolor=(*color[:3], 0.45),
                )
                ax2.add_patch(rect)

                # Region index and attention value
                region_id = rr * nrs + rc
                ax2.text(
                    rc * region_px + region_px / 2,
                    rr * region_px + region_px / 2,
                    f"R{region_id}\n{intensity:.2f}",
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold',
                )

        # Colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap_warm,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, fraction=0.03, pad=0.04, label='Routing Attention')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"  Saved → {save_path}")
        else:
            plt.show()
        plt.close()

    # ── Demo ──────────────────────────────────────────────────────────────────

    def demo(self, n: int = 4, save_dir: str = 'demo_outputs', seed: int = 42):
        """
        Generate n random images, detect, and visualize routing.

        Saves images to save_dir/.
        """
        Path(save_dir).mkdir(exist_ok=True)
        rng = random.Random(seed)

        print(f"\nRunning demo on {n} synthetic images …")
        print(f"  sparse attention (inference mode): top-{self.config.top_k} "
              f"of {self.config.n_regions} regions\n")

        for i in range(n):
            s = rng.randint(0, 99999)
            img_np, annotations = generate_image(
                self.config.img_size, max_objects=3, seed=s
            )

            result = self.detect(img_np, conf_thr=0.35, nms_thr=0.4, sparse=True)

            n_det = len(result['boxes'])
            n_gt  = len(annotations)
            print(f"  [{i+1}/{n}] GT={n_gt} objects, Detected={n_det}")

            # Show per-region routing attention
            rs   = result['routing_scores'].cpu()
            attn = torch.softmax(rs, dim=-1).mean(dim=0)   # [N_r]
            top4 = torch.topk(attn, 4).indices.tolist()
            print(f"         Top-4 attended regions: {top4}  "
                  f"(covers image quadrants with objects)")

            save_path = f"{save_dir}/result_{i+1:02d}.png"
            self.visualize(
                img_np, result,
                annotations=annotations,
                title=f"Image {i+1}  |  GT={n_gt}  Detected={n_det}  "
                      f"(sparse top-{self.config.top_k} routing)",
                save_path=save_path,
            )

        print(f"\nAll results saved to {Path(save_dir).resolve()}/")
