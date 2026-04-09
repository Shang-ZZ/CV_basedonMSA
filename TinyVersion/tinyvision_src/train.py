"""
Training loop for TinyVision.

Tracks two families of metrics:
  Detection:  obj_loss, box_loss, cls_loss, detection_acc
  Routing:    routing_loss, routing_acc  <- shows MSA learning to select regions

routing_acc = fraction of positive cells where the top-1 predicted region
              matches the gold region.  Starts ~6% (random over 16 regions),
              should rise to 60-90% as training progresses.

Autoresearch optimizations applied:
  2. LR warmup+flat+cosine-cooldown schedule  (per-step, not per-epoch)
  3. Zero weight decay                        (set in TrainConfig)
  5. Muon optimizer for neck 2D weight matrices

Run:  python -m tinyvision train
"""
import os, time, math, json, shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .model import TinyVision
from .dataset import get_dataloader
from .loss import detection_loss, multiscale_detection_loss

CKPT_DIR = Path('checkpoints')
OUT_DIR  = Path('F:/ShangResearchPaper/exampleTorebuild/Tiny_CV_basedonMSA/model_output')


# ─── Muon optimizer (Opt-5) ───────────────────────────────────────────────────
# Adapted from Keller Jordan's Muon implementation.
# Applies Newton-Schulz orthogonalization to the gradient, producing an
# approximate steepest descent step in the spectral norm sense.

def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute the zeroth power (orthogonalization) of G via Newton-Schulz.

    The iteration X_{k+1} = a*X_k + b*(X_k X_k^T)*X_k + c*(X_k X_k^T)^2*X_k
    converges to the orthogonal factor of G's polar decomposition.

    Coefficients (a, b, c) tuned for fast convergence to the unit sphere.
    """
    assert G.ndim == 2, "Muon requires 2D weight matrices"
    a, b, c = 3.4445, -4.7750, 2.0315
    dtype = G.dtype
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz.

    For each 2D weight matrix W:
      1. Accumulate Nesterov momentum on the raw gradient
      2. Orthogonalize the momentum buffer via Newton-Schulz (5 steps)
      3. Scale update by sqrt(max(m, n)) for consistent update magnitude
      4. Apply learning rate

    Empirically: acts as a Newton-like step in spectral/Frobenius norm,
    giving faster convergence than Adam for weight matrices in attention layers.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)

                buf = state['buf']
                buf.mul_(momentum).add_(g)

                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalize: maps update -> approximate steepest descent dir
                update = _zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Scale by sqrt(max_dim) for consistent update magnitude across shapes
                update.mul_(max(update.shape) ** 0.5)

                p.add_(update, alpha=-lr)

        return loss


# ─── Routing accuracy ─────────────────────────────────────────────────────────

def routing_accuracy(routing_scores, region_targets):
    """
    Fraction of positive cells where predicted top-1 region = gold region.
    Measures how well the MSA neck has learned to route.
    """
    B, G, G2 = region_targets.shape
    N_r  = routing_scores.shape[-1]
    rs   = routing_scores.view(B, G, G, N_r)
    pos  = (region_targets >= 0)
    if not pos.any():
        return 0.0
    pred_region = rs[pos].argmax(dim=-1)    # [n_pos]
    gold_region = region_targets[pos]       # [n_pos]
    return (pred_region == gold_region).float().mean().item()


# ─── Detection accuracy (simple) ─────────────────────────────────────────────

def detection_accuracy(pred, obj_mask, cls_targets):
    """
    For positive cells: fraction where the predicted class is correct.
    """
    pos = obj_mask
    if not pos.any():
        return 0.0
    pred_cls = pred[..., 5:][pos].argmax(dim=-1)
    return (pred_cls == cls_targets[pos]).float().mean().item()


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, cfg: TrainConfig):
    model.eval()
    stats = dict(loss=0, obj=0, box=0, cls=0, routing=0,
                 routing_acc=0, det_acc=0, n=0)

    multiscale = model.config.multiscale
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        if multiscale:
            (imgs,
             obj_p3, box_p3, cls_p3, reg_p3,
             obj_p4, box_p4, cls_p4, reg_p4) = batch
            reg_tgt  = reg_p4
            obj_mask = obj_p4
            cls_tgt  = cls_p4
        else:
            imgs, obj_mask, box_tgt, cls_tgt, reg_tgt = batch

        out   = model(imgs, region_targets=reg_tgt, sparse=False)
        pred  = out['pred']
        rs    = out['routing_scores']

        if multiscale:
            losses = multiscale_detection_loss(
                out,
                tgt_p3=(obj_p3, box_p3, cls_p3, reg_p3),
                tgt_p4=(obj_p4, box_p4, cls_p4, reg_p4),
                cfg=cfg,
            )
        else:
            losses = detection_loss(pred, obj_mask, box_tgt, cls_tgt, rs, reg_tgt, cfg)

        stats['loss']    += losses['loss'].item()
        stats['obj']     += losses['obj_loss'].item()
        stats['box']     += losses['box_loss'].item()
        stats['cls']     += losses['cls_loss'].item()
        stats['routing'] += losses['routing_loss'].item()
        stats['routing_acc'] += routing_accuracy(rs, reg_tgt)
        stats['det_acc'] += detection_accuracy(pred, obj_mask, cls_tgt)
        stats['n']       += 1

    n = max(stats.pop('n'), 1)
    model.train()
    return {k: v / n for k, v in stats.items()}


# ─── Main training loop ────────────────────────────────────────────────────────

def train(model_cfg: ModelConfig = None, train_cfg: TrainConfig = None,
          out_dir: str = None):
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    _out_dir  = Path(out_dir) if out_dir else OUT_DIR

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f'Device: {device}')

    # ── Model ──
    model = TinyVision(model_cfg).to(device)
    pc    = model.param_count()
    print(f"TinyVision parameters:")
    print(f"  backbone : {pc['backbone']:>8,}")
    print(f"  neck     : {pc['neck']:>8,}  <- MSA routing")
    print(f"  head_p4  : {pc['head_p4']:>8,}")
    if model_cfg.multiscale:
        print(f"  fpn      : {pc['fpn']:>8,}")
        print(f"  head_p3  : {pc['head_p3']:>8,}")
    print(f"  total    : {pc['total']:>8,}  (~{pc['total']/1e6:.2f}M)")
    print(f"\nMSA config: n_regions={model_cfg.n_regions}  "
          f"top_k={model_cfg.top_k}  pool_size={model_cfg.pool_size}")
    print(f"Grid: {model_cfg.grid_size}x{model_cfg.grid_size}  "
          f"({model_cfg.grid_size**2} detection cells)\n")

    # ── Data ──
    train_loader = get_dataloader(
        train_cfg.n_train, model_cfg, train_cfg.batch_size,
        base_seed=0, shuffle=True
    )
    val_loader   = get_dataloader(
        train_cfg.n_val, model_cfg, train_cfg.batch_size,
        base_seed=100000, shuffle=False
    )

    # ── Optimizer (Opt-5): Muon for neck 2D matrices, Adam for everything else ──
    neck_2d_ids = set()
    neck_2d_params = []
    for n, p in model.neck.named_parameters():
        if p.ndim == 2:
            neck_2d_ids.add(id(p))
            neck_2d_params.append(p)
    other_params = [p for p in model.parameters() if id(p) not in neck_2d_ids]

    muon_lr  = 0.002         # Tuned for small CV model (20x Adam is too aggressive)
    adam_lr  = train_cfg.lr  # 1e-3

    muon_opt = Muon(neck_2d_params, lr=muon_lr, momentum=0.95)
    adam_opt = torch.optim.Adam(
        other_params, lr=adam_lr, weight_decay=train_cfg.weight_decay
    )

    print(f"Optimizers: Muon ({len(neck_2d_params)} neck 2D tensors, lr={muon_lr}) "
          f"+ Adam ({len(other_params)} tensors, lr={adam_lr})")

    # ── LR Schedule (Opt-2): warmup(5%) -> flat -> cosine-cooldown(50%) ──
    total_steps     = train_cfg.n_epochs * len(train_loader)
    warmup_steps    = int(total_steps * 0.05)
    cooldown_start  = int(total_steps * 0.50)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        elif step < cooldown_start:
            return 1.0
        else:
            progress = (step - cooldown_start) / max(total_steps - cooldown_start, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    muon_sched = torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)
    adam_sched = torch.optim.lr_scheduler.LambdaLR(adam_opt, lr_lambda)

    scaler = GradScaler() if device == 'cuda' else None

    CKPT_DIR.mkdir(exist_ok=True)
    best_val_loss = float('inf')
    t0 = time.time()
    global_step = 0
    training_log = []

    # ── Training epochs ────────────────────────────────────────────────────────
    for epoch in range(1, train_cfg.n_epochs + 1):
        model.train()
        epoch_stats = dict(loss=0, obj=0, box=0, cls=0,
                           routing=0, routing_acc=0, det_acc=0, n=0)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{train_cfg.n_epochs}',
                    leave=False, ncols=110)

        for batch in pbar:
            batch = tuple(t.to(device) for t in batch)

            # Unpack based on mode
            if model_cfg.multiscale:
                (imgs,
                 obj_p3, box_p3, cls_p3, reg_p3,
                 obj_p4, box_p4, cls_p4, reg_p4) = batch
                reg_tgt  = reg_p4   # routing supervision comes from P4 targets
                obj_mask = obj_p4
                cls_tgt  = cls_p4
            else:
                imgs, obj_mask, box_tgt, cls_tgt, reg_tgt = batch

            muon_opt.zero_grad()
            adam_opt.zero_grad()

            if scaler:
                with autocast():
                    out = model(imgs, region_targets=reg_tgt, sparse=False)
                    if model_cfg.multiscale:
                        losses = multiscale_detection_loss(
                            out,
                            tgt_p3=(obj_p3, box_p3, cls_p3, reg_p3),
                            tgt_p4=(obj_p4, box_p4, cls_p4, reg_p4),
                            cfg=train_cfg,
                        )
                    else:
                        losses = detection_loss(
                            out['pred'], obj_mask, box_tgt, cls_tgt,
                            out['routing_scores'], reg_tgt, train_cfg
                        )
                scaler.scale(losses['loss']).backward()
                scaler.unscale_(adam_opt)
                scaler.unscale_(muon_opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(adam_opt)
                scaler.step(muon_opt)
                scaler.update()
            else:
                out = model(imgs, region_targets=reg_tgt, sparse=False)
                if model_cfg.multiscale:
                    losses = multiscale_detection_loss(
                        out,
                        tgt_p3=(obj_p3, box_p3, cls_p3, reg_p3),
                        tgt_p4=(obj_p4, box_p4, cls_p4, reg_p4),
                        cfg=train_cfg,
                    )
                else:
                    losses = detection_loss(
                        out['pred'], obj_mask, box_tgt, cls_tgt,
                        out['routing_scores'], reg_tgt, train_cfg
                    )
                losses['loss'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                adam_opt.step()
                muon_opt.step()

            adam_sched.step()
            muon_sched.step()
            global_step += 1

            ra  = routing_accuracy(out['routing_scores'], reg_tgt)
            da  = detection_accuracy(out['pred'], obj_mask, cls_tgt)

            for k in ('loss', 'obj_loss', 'box_loss', 'cls_loss', 'routing_loss'):
                epoch_stats[k.replace('_loss', '')] += losses[k].item()
            epoch_stats['routing_acc'] += ra
            epoch_stats['det_acc']     += da
            epoch_stats['n']           += 1

            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.3f}",
                'rt_acc': f"{ra:.2f}",
            })

        n = max(epoch_stats.pop('n'), 1)
        tr = {k: v / n for k, v in epoch_stats.items()}

        # ── Validation ──
        val = evaluate(model, val_loader, device, train_cfg)

        elapsed = time.time() - t0
        cur_lr  = adam_sched.get_last_lr()[0] * adam_lr
        line = (
            f"[{epoch:3d}/{train_cfg.n_epochs}] "
            f"lr={cur_lr:.2e} | "
            f"train  loss={tr['loss']:.3f}  "
            f"obj={tr['obj']:.3f}  box={tr['box']:.3f}  "
            f"cls={tr['cls']:.3f}  rt={tr['routing']:.3f} | "
            f"route_acc(tr)={tr['routing_acc']:.2f}  "
            f"det_acc(tr)={tr['det_acc']:.2f} | "
            f"val  loss={val['loss']:.3f}  "
            f"route_acc={val['routing_acc']:.2f}  "
            f"det_acc={val['det_acc']:.2f} | "
            f"{elapsed:.0f}s"
        )
        print(line)
        training_log.append(line)

        if val['loss'] < best_val_loss:
            best_val_loss = val['loss']
            torch.save({'model_state': model.state_dict(),
                        'epoch': epoch, 'val_loss': best_val_loss},
                       CKPT_DIR / 'best_model.pt')
            print(f"  Best saved (val_loss={best_val_loss:.4f})")

    # ── Save config ────────────────────────────────────────────────────────────
    import dataclasses
    with open(CKPT_DIR / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(model_cfg), f, indent=2)

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {(CKPT_DIR / 'best_model.pt').resolve()}")

    # ── Copy results to output dir ────────────────────────────────────────────
    _out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CKPT_DIR / 'best_model.pt', _out_dir / 'best_model.pt')
    shutil.copy(CKPT_DIR / 'config.json',   _out_dir / 'config.json')

    log_path = _out_dir / 'training_log.txt'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("TinyVision Training Log (with autoresearch optimizations)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Optimizations: softcap(1) + lr_schedule(2) + zero_wd(3) "
                f"+ 2D-RoPE(4) + Muon(5) + gated-residuals(6)\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model params: {pc['total']:,}  (~{pc['total']/1e6:.2f}M)\n")
        f.write(f"n_regions={model_cfg.n_regions}  top_k={model_cfg.top_k}  "
                f"pool_size={model_cfg.pool_size}\n")
        f.write(f"Total epochs: {train_cfg.n_epochs}  "
                f"Total time: {total_time:.0f}s\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n")
        f.write("=" * 70 + "\n\n")
        for line in training_log:
            f.write(line + "\n")

    print(f"\nResults saved to: {_out_dir}")
    print(f"  best_model.pt")
    print(f"  config.json")
    print(f"  training_log.txt")
