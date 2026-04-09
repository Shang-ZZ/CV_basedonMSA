"""
TinyVision CLI.

  python -m tinyvision train              # train the model
  python -m tinyvision demo               # run demo on random images
  python -m tinyvision demo --n 8         # visualize 8 images
  python -m tinyvision info               # print architecture summary
  python -m tinyvision quicktest          # 5-step smoke test (no checkpoint needed)
"""
import argparse


def cmd_train(args):
    from .train import train
    train()


def cmd_demo(args):
    from .inference import TinyVisionInference
    inf = TinyVisionInference.load()
    inf.demo(n=args.n, save_dir=args.save_dir)


def cmd_info(args):
    from .config import ModelConfig
    from .model import TinyVision

    cfg = ModelConfig()
    model = TinyVision(cfg)
    pc  = model.param_count()

    print('\nTinyVision Architecture (MSA-based YOLO-like detector)')
    print('=' * 60)
    print(f'Input image:  {cfg.img_size}×{cfg.img_size}×3')
    print(f'Detection:    {cfg.grid_size}×{cfg.grid_size} grid  '
          f'({cfg.grid_size**2} cells, stride={cfg.stride})')
    print(f'Classes:      3  (circle / rectangle / triangle)')
    print()
    print('Component       Params    Role')
    print('─' * 60)
    print(f'Backbone      {pc["backbone"]:>8,}   P3[32x32] + P4[16x16]')
    print(f'MSA Neck      {pc["neck"]:>8,}   Region routing (KEY)')
    print(f'Det. Head P4  {pc["head_p4"]:>8,}   YOLO-style prediction')
    print(f'Total         {pc["total"]:>8,}   (~{pc["total"]/1e6:.2f}M)')
    print()
    print('MSA Neck details:')
    print(f'  P3 regions:  {cfg.n_regions}  (4×4 spatial grid over 32×32 P3)')
    print(f'  P4 queries:  {cfg.n_query}  (16×16 detection positions)')
    print(f'  pool_size:   {cfg.pool_size}  '
          f'(compress 8×8 P3 patch → 1 region vector)')
    print(f'  top_k:       {cfg.top_k}  '
          f'(inference: attend to only {cfg.top_k}/{cfg.n_regions} regions)')
    print(f'  n_heads:     {cfg.n_heads}')
    print()
    print('Analogy with MSA-main (NLP):')
    rows = [
        ('long document sequence', 'P3 feature map [B,128,32,32]'),
        ('N documents',            f'{cfg.n_regions} spatial regions (4×4 grid)'),
        ('pool_doc_kv()',          'AvgPool2d(8,8) on P3'),
        ('query tokens',           'P4 positions [B,256,16×16]'),
        ('routing scores Q×K',     'each P4 pos scores 16 P3 regions'),
        ('top-K doc selection',    f'top-{cfg.top_k} region selection'),
        ('sparse attention',       'cross-attn to top-K regions only'),
        ('InfoNCE routing loss',   'CE: route to GT box region'),
    ]
    for nlp, cv in rows:
        print(f'  {nlp:<30} →  {cv}')


def cmd_quicktest(args):
    """Smoke test: one forward + backward pass without checkpoint."""
    import torch
    from .config import ModelConfig, TrainConfig
    from .model import TinyVision
    from .dataset import ShapeDataset
    from .loss import detection_loss

    print('Running quick smoke test …')
    cfg   = ModelConfig()
    tcfg  = TrainConfig()
    model = TinyVision(cfg)
    model.train()

    ds    = ShapeDataset(4, cfg, base_seed=0)
    imgs, obj_mask, box_tgt, cls_tgt, reg_tgt = zip(*[ds[i] for i in range(4)])

    imgs      = torch.stack(imgs)
    obj_mask  = torch.stack(obj_mask)
    box_tgt   = torch.stack(box_tgt)
    cls_tgt   = torch.stack(cls_tgt)
    reg_tgt   = torch.stack(reg_tgt)

    out    = model(imgs, region_targets=reg_tgt, sparse=False)
    losses = detection_loss(out['pred'], obj_mask, box_tgt, cls_tgt,
                            out['routing_scores'], reg_tgt, tcfg)
    losses['loss'].backward()

    print(f'  Forward  OK')
    print(f'  Backward OK')
    print(f'  Loss:    {losses["loss"].item():.4f}')
    print(f'  Routing loss: {losses["routing_loss"].item():.4f}')
    print(f'  pred shape:   {out["pred"].shape}')
    print(f'  routing_scores shape: {out["routing_scores"].shape}')
    print()

    # Check routing correctness
    rs  = out['routing_scores']   # [B, N_q, N_r]
    B, G = 4, cfg.grid_size
    rs_map = rs.view(B, G, G, cfg.n_regions)
    pos    = (reg_tgt >= 0)
    if pos.any():
        pred_r = rs_map[pos].argmax(-1)
        gold_r = reg_tgt[pos]
        acc = (pred_r == gold_r).float().mean().item()
        print(f'  Routing acc (random init, expect ~1/16≈0.06): {acc:.3f}')

    print('\nSmoke test passed OK')


def main():
    parser = argparse.ArgumentParser(
        prog='tinyvision',
        description='TinyVision — MSA-based YOLO-like detector demo',
    )
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('train', help='Train the model')

    p_demo = sub.add_parser('demo', help='Run demo + routing visualization')
    p_demo.add_argument('--n', type=int, default=4, help='Number of images')
    p_demo.add_argument('--save-dir', default='demo_outputs')

    sub.add_parser('info',      help='Print model architecture')
    sub.add_parser('quicktest', help='Smoke test (no checkpoint needed)')

    args = parser.parse_args()

    if   args.command == 'train':     cmd_train(args)
    elif args.command == 'demo':      cmd_demo(args)
    elif args.command == 'info':      cmd_info(args)
    elif args.command == 'quicktest': cmd_quicktest(args)
    else:                             parser.print_help()


if __name__ == '__main__':
    main()
