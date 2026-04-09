"""
Detection loss + MSA routing loss.

Single-scale loss:
  L = w_obj * obj_loss  +  w_box * box_loss  +  w_cls * cls_loss
    + w_routing * routing_loss

Multi-scale loss (config.multiscale=True):
  L = detection_loss(P4) + p3_weight * detection_loss_no_routing(P3)

  The P3 head has no MSA routing supervision (routing_scores=None),
  so pass routing_scores=None to disable the routing loss term.

obj_loss:     BCEWithLogits for all grid cells (positive + negative)
box_loss:     smooth-L1 on sigmoid-bounded cx/cy/w/h (positive cells only)
cls_loss:     CrossEntropy on class logits (positive cells only)
routing_loss: CrossEntropy on routing_scores vs gold P3 region (positive only)
              skipped when routing_scores=None (P3-scale head)
"""
import torch
import torch.nn.functional as F
from .config import TrainConfig


def detection_loss(
    pred:           torch.Tensor,            # [B, G, G, 5+nc]
    obj_mask:       torch.Tensor,            # [B, G, G]  bool
    box_targets:    torch.Tensor,            # [B, G, G, 4]
    cls_targets:    torch.Tensor,            # [B, G, G]  long
    routing_scores: torch.Tensor | None,     # [B, N_q, N_r] or None
    region_targets: torch.Tensor,            # [B, G, G]  long
    cfg:            TrainConfig,
) -> dict:
    """
    Compute all losses for one detection scale.

    Pass routing_scores=None to skip routing loss (e.g. for the P3 head
    in multi-scale mode which has no MSA routing supervision).
    """
    B, G, _, _ = pred.shape

    pred_obj = pred[..., 0]
    pred_cxy = pred[..., 1:3]
    pred_wh  = pred[..., 3:5]
    pred_cls = pred[..., 5:]

    # ── Objectness loss ───────────────────────────────────────────────────────
    obj_target = obj_mask.float()
    obj_weight = torch.where(obj_mask, 1.0, cfg.noobj_weight)

    obj_loss = F.binary_cross_entropy_with_logits(
        pred_obj, obj_target, weight=obj_weight, reduction='mean'
    )

    # ── Regression + classification losses (positive cells only) ─────────────
    pos = obj_mask

    if pos.any():
        pred_cxy_pos = torch.sigmoid(pred_cxy[pos])
        pred_wh_pos  = torch.sigmoid(pred_wh[pos])
        box_pos  = box_targets[pos]
        tgt_cxy  = box_pos[:, :2]
        tgt_wh   = box_pos[:, 2:]

        box_loss = (
            F.smooth_l1_loss(pred_cxy_pos, tgt_cxy) +
            F.smooth_l1_loss(pred_wh_pos,  tgt_wh)
        )

        cls_loss = F.cross_entropy(pred_cls[pos], cls_targets[pos])

        # Routing loss — skipped when routing_scores is None
        if routing_scores is not None:
            N_r = routing_scores.shape[-1]
            rs  = routing_scores.view(B, G, G, N_r)
            routing_loss = F.cross_entropy(rs[pos], region_targets[pos])
        else:
            routing_loss = pred.new_tensor(0.0)

    else:
        box_loss     = pred.new_tensor(0.0)
        cls_loss     = pred.new_tensor(0.0)
        routing_loss = pred.new_tensor(0.0)

    total = (
        cfg.obj_weight     * obj_loss  +
        cfg.box_weight     * box_loss  +
        cfg.cls_weight     * cls_loss  +
        cfg.routing_weight * routing_loss
    )

    return {
        'loss':         total,
        'obj_loss':     obj_loss,
        'box_loss':     box_loss,
        'cls_loss':     cls_loss,
        'routing_loss': routing_loss,
    }


def multiscale_detection_loss(
    out:          dict,
    tgt_p3:       tuple,   # (obj_p3, box_p3, cls_p3, reg_p3)
    tgt_p4:       tuple,   # (obj_p4, box_p4, cls_p4, reg_p4)
    cfg:          TrainConfig,
) -> dict:
    """
    Combined loss for multi-scale detection (P3 + P4 heads).

    P4: full loss including MSA routing supervision.
    P3: detection loss only (no routing loss; P3 queries bypass the neck).

    Total = loss_p4 + cfg.p3_weight * loss_p3
    """
    obj_p4, box_p4, cls_p4, reg_p4 = tgt_p4
    obj_p3, box_p3, cls_p3, reg_p3 = tgt_p3

    losses_p4 = detection_loss(
        out['pred'],
        obj_p4, box_p4, cls_p4,
        out['routing_scores'],
        reg_p4, cfg,
    )
    losses_p3 = detection_loss(
        out['pred_p3'],
        obj_p3, box_p3, cls_p3,
        None,        # no routing supervision for P3 head
        reg_p3, cfg,
    )

    total = losses_p4['loss'] + cfg.p3_weight * losses_p3['loss']

    return {
        'loss':           total,
        # P4 components
        'obj_loss':       losses_p4['obj_loss'],
        'box_loss':       losses_p4['box_loss'],
        'cls_loss':       losses_p4['cls_loss'],
        'routing_loss':   losses_p4['routing_loss'],
        # P3 components
        'obj_loss_p3':    losses_p3['obj_loss'],
        'box_loss_p3':    losses_p3['box_loss'],
        'cls_loss_p3':    losses_p3['cls_loss'],
    }
