"""
TinyVision — 实时摄像头目标检测演示
=====================================
功能：
  · 实时读取摄像头（或视频文件）画面
  · 每帧运行 TinyVision v2 检测（圆形 / 矩形 / 三角形）
  · 可叠加 MSA 路由热力图（8×8 网格，颜色深浅表示各区域被关注程度）
  · 支持暂停、保存帧、动态调整置信度
  · 若无摄像头，自动切换为合成图像连续测试模式

运行方式
--------
  cd F:/ShangResearchPaper/exampleTorebuild/Tiny_CV_basedonMSA/TinyVersion

  # 默认摄像头
  E:/Python3.11.0/python.exe video_demo.py

  # 指定摄像头编号
  E:/Python3.11.0/python.exe video_demo.py --source 1

  # 视频文件
  E:/Python3.11.0/python.exe video_demo.py --source path/to/video.mp4

  # 使用合成图像（不需要摄像头）
  E:/Python3.11.0/python.exe video_demo.py --synthetic

  # 启动时显示路由热力图
  E:/Python3.11.0/python.exe video_demo.py --routing

窗口快捷键
----------
  Q / ESC   退出
  R         开/关 MSA 路由热力图叠加层
  SPACE     暂停 / 继续
  S         保存当前帧到 captures/
  +  /  -   置信度阈值 ±0.05

注意
----
模型仅在合成形状（圆形/矩形/三角形）上训练，用于演示 MSA 路由机制。
"""

import sys
import argparse
import time
import datetime
import itertools
from pathlib import Path

import cv2
import numpy as np
import torch

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tinyvision_src.inference import TinyVisionInference
from tinyvision_src.dataset   import CLASSES, generate_image

# ── 常量 ──────────────────────────────────────────────────────────────────────
# 检测框颜色 (BGR)
CLASS_BGR = {
    0: (50,  80, 230),   # circle    — 红
    1: (40, 200,  50),   # rectangle — 绿
    2: (210, 130, 40),   # triangle  — 蓝
}
CLASS_ZH = {0: 'Circle', 1: 'Rectangle', 2: 'Triangle'}

WINDOW_NAME = 'TinyVision  |  Q=退出  R=路由热力图  SPACE=暂停  S=保存'


# ── 绘制函数 ──────────────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray,
               boxes:   torch.Tensor,
               scores:  torch.Tensor,
               classes: torch.Tensor) -> np.ndarray:
    """在帧上绘制检测框、类别标签和置信度。"""
    for box, sc, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = (int(v) for v in box.tolist())
        c   = int(cls.item())
        bgr = CLASS_BGR[c]

        # 框
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

        # 标签背景 + 文字
        label = f"{CLASS_ZH[c]} {sc:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ly = max(y1 - 1, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + baseline), bgr, -1)
        cv2.putText(frame, label, (x1 + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_routing_heatmap(frame: np.ndarray,
                         routing_scores: torch.Tensor,
                         n_reg_side: int,
                         alpha: float = 0.38) -> np.ndarray:
    """
    在帧上叠加 MSA 路由热力图。

    routing_scores : [N_q, N_r] — 检测查询对每个 P3 区域的打分
    n_reg_side     : 每侧区域数（v2 为 8）
    alpha          : 叠加透明度（越大越显著）
    """
    H, W   = frame.shape[:2]
    n      = n_reg_side
    cell_h = H // n
    cell_w = W // n

    # 各区域平均关注度 [N_r]
    with torch.no_grad():
        attn = torch.softmax(routing_scores.float().cpu(), dim=-1).mean(0).numpy()
    grid = attn.reshape(n, n)
    lo, hi = grid.min(), grid.max()
    norm = (grid - lo) / (hi - lo + 1e-8)   # [n, n] in [0,1]

    overlay = frame.copy()

    for row in range(n):
        for col in range(n):
            t  = float(norm[row, col])
            # 颜色：冷色（低关注） → 暖橙（高关注）
            r_ = int(30  + t * 220)
            g_ = int(30  + t * 130)
            b_ = int(50  + t * 20)
            colour = (b_, g_, r_)

            px1, py1 = col * cell_w,       row * cell_h
            px2, py2 = px1 + cell_w - 1,   py1 + cell_h - 1

            cv2.rectangle(overlay, (px1, py1), (px2, py2), colour, -1)
            # 网格线
            cv2.rectangle(frame, (px1, py1), (px2, py2), (180, 180, 180), 1)

            rid = row * n + col
            cx, cy = px1 + cell_w // 2, py1 + cell_h // 2
            cv2.putText(overlay, f"R{rid}",
                        (px1 + 3, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"{t:.2f}",
                        (px1 + 3, cy + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def draw_hud(frame: np.ndarray,
             fps:          float,
             conf_thr:     float,
             show_routing: bool,
             paused:       bool,
             n_det:        int,
             source_label: str) -> np.ndarray:
    """左上角信息栏：FPS / 置信度 / 状态。"""
    lines = []
    if paused:
        lines.append(("[ Paused - SPACE Continue ]", (0, 200, 255)))
    lines += [
        (f"FPS: {fps:5.1f}",                        (255, 255, 255)),
        (f"Number of detections: {n_det}",                          (255, 255, 255)),
        (f"Confidence Level: {conf_thr:.2f}  (+/- Adjust)",       (200, 200, 200)),
        (f"Routing Diagram: {'ON' if show_routing else 'OFF'}  (R Switch)", (200, 200, 200)),
        (f"Source: {source_label}",                    (160, 160, 160)),
    ]
    y = 20
    for text, colour in lines:
        # 黑色描边
        cv2.putText(frame, text, (9, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (9, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 1, cv2.LINE_AA)
        y += 22
    return frame


# ── 帧缩放辅助 ────────────────────────────────────────────────────────────────

def scale_boxes(boxes: torch.Tensor,
                from_size: int,
                to_w: int, to_h: int) -> torch.Tensor:
    """将检测框从 from_size×from_size 坐标系缩放到 to_w×to_h。"""
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone().float()
    out[:, [0, 2]] *= to_w / from_size
    out[:, [1, 3]] *= to_h / from_size
    return out


# ── 合成帧生成器（无摄像头时使用）────────────────────────────────────────────

def synthetic_frame_generator(img_size: int):
    """无限生成合成形状图像，模拟摄像头流。"""
    seed_counter = itertools.count(0)
    while True:
        seed = next(seed_counter) % 100000
        img_np, annotations = generate_image(img_size, max_objects=3, seed=seed)
        # img_np: (H, W, 3) RGB uint8 → BGR for OpenCV
        frame_bgr = img_np[:, :, ::-1].copy()
        yield frame_bgr, annotations
        time.sleep(0.08)   # ~12 fps 合成流


# ── 主循环 ────────────────────────────────────────────────────────────────────

def run(args):
    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("TinyVision 实时视频检测演示")
    print("=" * 55)
    print(f"加载模型: {args.ckpt}")

    inf = TinyVisionInference.load(
        ckpt_path   = args.ckpt,
        config_path = args.cfg,
    )
    cfg = inf.config
    print(f"  参数量: {inf.model.param_count()['total']/1e6:.2f}M  |  "
          f"输入: {cfg.img_size}×{cfg.img_size}  |  "
          f"区域: {cfg.n_regions} ({cfg.n_reg_side}×{cfg.n_reg_side})")
    print()

    # ── 打开视频源 ────────────────────────────────────────────────────────────
    use_synthetic = args.synthetic
    cap = None
    source_label = "Synthesized Image"

    if not use_synthetic:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[警告] 无法打开摄像头/视频源 '{args.source}'，切换为合成图像模式。")
            use_synthetic = True
        else:
            cam_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            source_label = f"Camera {args.source}  ({cam_w} * {cam_h} @ {cam_fps:.0f}fps)"
            print(f"Vodeo Source: {source_label}")

    if use_synthetic:
        print("模式: 合成图像连续测试（每帧随机生成圆/矩/三角）")
        synth_gen = synthetic_frame_generator(cfg.img_size)
        # 合成帧本身就是 cfg.img_size × cfg.img_size
        cam_w = cam_h = cfg.img_size

    print()
    print("快捷键:  Q/ESC=退出  R=路由热力图  SPACE=暂停  S=保存  +/-=置信度")
    print("=" * 55)
    print()

    # ── 状态变量 ──────────────────────────────────────────────────────────────
    conf_thr     = args.conf
    show_routing = args.routing
    paused       = False
    save_dir     = Path(args.save_dir)
    save_count   = 0

    fps_t0       = time.perf_counter()
    fps_frames   = 0
    fps_display  = 0.0

    # 持有上一帧状态（暂停时复用）
    last_frame_bgr   = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
    last_scaled_boxes = torch.zeros(0, 4)
    last_scores      = torch.zeros(0)
    last_classes     = torch.zeros(0, dtype=torch.long)
    last_routing     = None
    last_n_det       = 0

    # ── 主循环 ────────────────────────────────────────────────────────────────
    while True:
        if not paused:
            # 读取帧
            if use_synthetic:
                frame_bgr, _ = next(synth_gen)
                ret = True
            else:
                ret, frame_bgr = cap.read()

            if not ret:
                print("视频流结束。")
                break

            h_disp, w_disp = frame_bgr.shape[:2]

            # 预处理：缩放到模型输入尺寸 → RGB uint8
            frame_model = cv2.resize(frame_bgr, (cfg.img_size, cfg.img_size))
            frame_rgb   = cv2.cvtColor(frame_model, cv2.COLOR_BGR2RGB)

            # 推理
            result = inf.detect(
                frame_rgb,
                conf_thr = conf_thr,
                nms_thr  = args.nms,
                sparse   = True,
            )

            # 将检测框坐标从模型空间（256×256）映射回显示分辨率
            scaled_boxes = scale_boxes(result['boxes'], cfg.img_size, w_disp, h_disp)

            # 缓存（供暂停时复用）
            last_frame_bgr    = frame_bgr
            last_scaled_boxes = scaled_boxes
            last_scores       = result['scores']
            last_classes      = result['classes']
            last_routing      = result.get('routing_scores')
            last_n_det        = len(scaled_boxes)

            # FPS 统计
            fps_frames += 1
            elapsed = time.perf_counter() - fps_t0
            if elapsed >= 0.5:
                fps_display = fps_frames / elapsed
                fps_frames  = 0
                fps_t0      = time.perf_counter()

        # ── 构建显示帧 ────────────────────────────────────────────────────────
        display = last_frame_bgr.copy()
        h_disp, w_disp = display.shape[:2]

        # 路由热力图（叠加在检测框之下，先画）
        if show_routing and last_routing is not None:
            draw_routing_heatmap(display, last_routing,
                                 n_reg_side=cfg.n_reg_side, alpha=0.38)

        # 检测框
        draw_boxes(display, last_scaled_boxes, last_scores, last_classes)

        # HUD
        draw_hud(display, fps_display, conf_thr, show_routing,
                 paused, last_n_det, source_label)

        cv2.imshow(WINDOW_NAME, display)

        # ── 按键处理 ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):           # 退出
            break

        elif key in (ord('r'), ord('R')):              # 路由热力图开关
            show_routing = not show_routing
            print(f"路由热力图: {'开' if show_routing else '关'}")

        elif key == ord(' '):                          # 暂停/继续
            paused = not paused
            print("已暂停。" if paused else "继续。")

        elif key in (ord('+'), ord('=')):              # 提高置信度
            conf_thr = min(conf_thr + 0.05, 0.95)
            print(f"置信度阈值: {conf_thr:.2f}")

        elif key in (ord('-'), ord('_')):              # 降低置信度
            conf_thr = max(conf_thr - 0.05, 0.05)
            print(f"置信度阈值: {conf_thr:.2f}")

        elif key in (ord('s'), ord('S')):              # 保存帧
            save_dir.mkdir(parents=True, exist_ok=True)
            ts    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = save_dir / f"capture_{ts}_{save_count:03d}.png"
            cv2.imwrite(str(fname), display)
            save_count += 1
            print(f"已保存 → {fname}")

    # ── 清理 ──────────────────────────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("演示结束。")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    CKPT = str(ROOT / 'modelOutput' / 'best_model.pt')
    CFG  = str(ROOT / 'modelOutput' / 'config.json')

    parser = argparse.ArgumentParser(
        description='TinyVision Real-time Video Detection Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--source',    default='0',
                        help='摄像头编号（0/1/…）或视频文件路径（默认: 0）')
    parser.add_argument('--synthetic', action='store_true',
                        help='使用合成图像代替摄像头（无需硬件）')
    parser.add_argument('--ckpt',      default=CKPT,
                        help='模型权重路径（默认: modelOutput/best_model.pt）')
    parser.add_argument('--cfg',       default=CFG,
                        help='模型配置路径（默认: modelOutput/config.json）')
    parser.add_argument('--conf',      default=0.35, type=float,
                        help='置信度阈值（默认: 0.35）')
    parser.add_argument('--nms',       default=0.40, type=float,
                        help='NMS IoU 阈值（默认: 0.40）')
    parser.add_argument('--routing',   action='store_true',
                        help='启动时显示路由热力图')
    parser.add_argument('--save-dir',  default='captures',
                        help='保存帧的目录（默认: captures/）')
    args = parser.parse_args()

    run(args)
