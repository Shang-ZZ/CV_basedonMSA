"""
MSA Region Routing Neck  ─  the educational core of TinyVision.

Adapts the three key MSA ideas from language to 2D vision:

┌─────────────────────────────────────────────────────────────────────┐
│  NLP (MSA-main)            │  CV (this file)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Long document sequence    │  P3 feature map  [B,128,32,32]         │
│  N documents               │  N=16 spatial regions (4×4 grid)       │
│  pool_doc_kv (avg pool)    │  pool_regions (AvgPool2d on P3)        │
│  Query tokens              │  P4 detection positions [B,256,16×16]  │
│  Routing scores Q×K        │  Each P4 pos scores 16 P3 regions      │
│  top_k doc selection       │  top_k=4 region selection at inference │
│  Sparse attention          │  Cross-attn to top-K regions only      │
│  InfoNCE routing loss      │  CE loss: route to GT box's region     │
└─────────────────────────────────────────────────────────────────────┘

Spatial layout (default config):
  Image:  256×256
  P3:      32×32  (stride 8)   divided into 4×4=16 regions of 8×8 each
  P4:      16×16  (stride 16)  = 256 detection query positions

  Each P3 region covers an 8×8 patch in P3 space = 64×64 pixels in the image.
  When a detection query routes to a region, it "zooms into" the fine detail
  of that 64×64 image area -- exactly what's needed for accurate detection.

Training vs Inference:
  Training:  soft attention over ALL 16 regions (correct gradients for LM loss)
             + auxiliary routing loss (gold_region supervision)
  Inference: hard top-K selection -> sparse attention over top-K regions only
             Context reduces from 16 regions to K=4 (4x memory saving in neck)

Autoresearch optimizations applied:
  1. Routing score softcap  -- 12*tanh(s/12) prevents extreme routing logits
  4. 2D RoPE on P4 queries  -- row/col rotary embeddings for spatial awareness
  6. Gated value residuals  -- sigmoid gate on attended values (ResFormer-style)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


class MSARegionRoutingNeck(nn.Module):
    """
    Cross-scale feature fusion via MSA-inspired spatial routing.

    Takes:
      p3: [B, C3, 32, 32]  -- fine-grained feature map  ("documents")
      p4: [B, C4, 16, 16]  -- coarse detection feature  ("queries")

    Returns:
      enhanced_p4: [B, C4, 16, 16]  -- p4 enriched with routed p3 context
      routing_scores: [B, N_q, N_r]  -- raw routing logits (for loss + vis)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        C3 = config.c3          # 128  (P3 channels)
        C4 = config.c4          # 256  (P4 channels)
        D  = config.d_neck      # 128  (neck projection dim)
        H  = config.n_heads     # 4    (attention heads)
        self.head_dim = D // H  # 32

        self.n_regions   = config.n_regions   # 16 or 64
        self.pool_size   = config.pool_size   # 8 or 4  (AvgPool kernel in P3 space)
        self.top_k       = config.top_k       # 4
        self.n_heads     = H
        self.d_neck      = D
        self.scale       = self.head_dim ** -0.5
        self.n_reg_side  = config.n_reg_side  # 4 or 8
        self.softcap     = 12.0               # [Opt-1] routing score softcap

        # [Ext-5] Dynamic Top-K parameters
        self.dynamic_topk = config.dynamic_topk
        self.topk_min     = config.topk_min   # 2
        self.topk_max     = config.topk_max   # 8

        # ── Projection layers ─────────────────────────────────────────────────
        # Query projections (from P4 positions -> neck dim)
        self.q_proj = nn.Linear(C4, D, bias=False)
        # Key/Value projections (from pooled P3 regions -> neck dim)
        self.k_proj = nn.Linear(C3, D, bias=False)
        self.v_proj = nn.Linear(C3, D, bias=False)
        # Output projection (neck dim -> P4 channels, for residual add)
        self.out_proj = nn.Linear(D, C4, bias=False)

        # [Opt-6] Gated value residuals: gate queries -> D space
        self.gate_proj = nn.Linear(C4, D, bias=False)

        # Optional feed-forward after cross-attention (stabilises training)
        self.ffn = nn.Sequential(
            nn.Linear(C4, C4),
            nn.GELU(),
            nn.Linear(C4, C4),
        )
        self.norm1 = nn.LayerNorm(C4)
        self.norm2 = nn.LayerNorm(C4)

    # ── 2D RoPE helpers (Opt-4) ───────────────────────────────────────────────

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate adjacent pairs: (a, b, c, d, ...) -> (-b, a, -d, c, ...)
        Used in RoPE: x_rot = x*cos + rotate_half(x)*sin
        """
        x_even = x[..., 0::2]   # [B, H, N, D//2]
        x_odd  = x[..., 1::2]
        return torch.stack([-x_odd, x_even], dim=-1).reshape(x.shape)

    def _apply_2d_rope(self, Q: torch.Tensor, G: int) -> torch.Tensor:
        """
        Apply 2D rotary position embeddings to multi-head queries.

        CV analogue of MSA's document-level RoPE, but in 2D:
          - Each query has a (row, col) position in the G×G detection grid
          - First Dh//2 of each head: encode row position
          - Last  Dh//2 of each head: encode col position

        Q: [B, H, N_q, Dh]  N_q = G*G
        Returns: Q with 2D rotary embeddings applied
        """
        B, H, N_q, Dh = Q.shape
        half = Dh // 2       # first half = row, second half = col
        freq_dim = half // 2 # number of frequency pairs per spatial dim

        # Row/col indices for each query position
        row_idx = torch.arange(N_q, device=Q.device) // G  # [N_q]
        col_idx = torch.arange(N_q, device=Q.device) % G   # [N_q]

        # Frequency bands (standard RoPE)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, freq_dim, device=Q.device).float() / freq_dim)
        )  # [freq_dim]

        # Angles per query: [N_q, freq_dim]
        row_angles = row_idx.float().unsqueeze(-1) * inv_freq  # [N_q, freq_dim]
        col_angles = col_idx.float().unsqueeze(-1) * inv_freq

        # Repeat each angle for the pair of (cos, sin) slots: [N_q, half]
        row_sin = torch.sin(row_angles).repeat_interleave(2, dim=-1)
        row_cos = torch.cos(row_angles).repeat_interleave(2, dim=-1)
        col_sin = torch.sin(col_angles).repeat_interleave(2, dim=-1)
        col_cos = torch.cos(col_angles).repeat_interleave(2, dim=-1)

        # Broadcast to [1, 1, N_q, half]
        row_sin = row_sin[None, None]
        row_cos = row_cos[None, None]
        col_sin = col_sin[None, None]
        col_cos = col_cos[None, None]

        Q_row = Q[..., :half]   # [B, H, N_q, half]  -- row-position dims
        Q_col = Q[..., half:]   # [B, H, N_q, half]  -- col-position dims

        # RoPE rotation: q' = q*cos + rotate_half(q)*sin
        Q_row = Q_row * row_cos + self._rotate_half(Q_row) * row_sin
        Q_col = Q_col * col_cos + self._rotate_half(Q_col) * col_sin

        return torch.cat([Q_row, Q_col], dim=-1)   # [B, H, N_q, Dh]

    # ── Stage 1: Region Compression ───────────────────────────────────────────

    def pool_regions(self, p3: torch.Tensor) -> torch.Tensor:
        """
        Compress P3 fine-grained features into region-level vectors.

        CV analogue of MSA's _pool_doc_kv():
          Each spatial region of the P3 map = one "document"
          AvgPool2d(kernel=8, stride=8) compresses 8x8 P3 patches -> 1 vector

        p3: [B, C3, 32, 32]
        Returns: pooled_regions [B, 16, C3]
        """
        # AvgPool: [B, C3, 32, 32] -> [B, C3, 4, 4]
        pooled = F.avg_pool2d(p3, kernel_size=self.pool_size, stride=self.pool_size)
        B, C, Hr, Wr = pooled.shape
        # Reshape to sequence: [B, C3, 16] -> [B, 16, C3]
        return pooled.view(B, C, -1).permute(0, 2, 1)   # [B, 16, C3]

    # ── Stage 2: Routing Scores ───────────────────────────────────────────────

    def compute_routing_scores(
        self,
        queries: torch.Tensor,          # [B, N_q, C4]
        pooled_regions: torch.Tensor,   # [B, N_r, C3]
        G: int,                         # detection grid side length
    ) -> torch.Tensor:
        """
        For each P4 detection query, compute relevance to each P3 region.

        CV analogue of MSA's _compute_routing_scores():
          Multi-head dot-product: Q [B,H,N_q,Dh] x K [B,H,Dh,N_r] -> scores

        Optimizations applied:
          [Opt-1] Softcap: 12*tanh(s/12) -- stabilises large routing logits
          [Opt-4] 2D RoPE on Q           -- spatial position encoding

        Returns: routing_scores [B, N_q, N_r]
        """
        B, N_q, _ = queries.shape
        N_r = pooled_regions.shape[1]
        H, Dh = self.n_heads, self.head_dim

        # Project to neck dimension
        Q = self.q_proj(queries)           # [B, N_q, D]
        K = self.k_proj(pooled_regions)    # [B, N_r, D]

        # Reshape for multi-head: [B, H, N_*, Dh]
        Q = Q.view(B, N_q, H, Dh).permute(0, 2, 1, 3)   # [B, H, N_q, Dh]
        K = K.view(B, N_r, H, Dh).permute(0, 2, 1, 3)   # [B, H, N_r, Dh]

        # [Opt-4] Apply 2D RoPE to Q (spatial position awareness)
        Q = self._apply_2d_rope(Q, G)   # [B, H, N_q, Dh]

        # [B, H, N_q, N_r]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # [Opt-1] Routing score softcap: prevents extreme logit values
        sc = self.softcap
        scores = sc * torch.tanh(scores / sc)

        # Max over heads -> [B, N_q, N_r]
        scores = scores.max(dim=1).values
        return scores

    # ── Dynamic Top-K (Ext-5) ────────────────────────────────────────────────

    def _adaptive_k(self, routing_scores: torch.Tensor) -> int:
        """
        Compute an adaptive K for the current batch using routing entropy.

        High routing entropy → queries are uncertain → use larger K (more context).
        Low  routing entropy → queries are confident → use smaller K (efficiency).

        routing_scores: [B, N_q, N_r]
        Returns: a single int K in [topk_min, topk_max] for the whole batch.

        We use batch-mean entropy to get a single K (allows fixed tensor shapes).
        Per-query variable K would break batching; this approximates it cheaply.
        """
        with torch.no_grad():
            probs = F.softmax(routing_scores.detach(), dim=-1)  # [B, N_q, N_r]
            # Shannon entropy: H = -sum(p * log(p))  in [0, log(N_r)]
            H = -(probs * (probs + 1e-8).log()).sum(dim=-1)     # [B, N_q]
            H_norm = (H / math.log(self.n_regions)).clamp(0, 1) # [B, N_q] in [0,1]
            mean_H = H_norm.mean().item()
        # Linear mapping: H=0 → topk_min, H=1 → topk_max
        k_float = self.topk_min + mean_H * (self.topk_max - self.topk_min)
        return max(self.topk_min, min(self.topk_max, int(round(k_float))))

    # ── Stage 3: Sparse Cross-Attention ───────────────────────────────────────

    def sparse_cross_attn(
        self,
        queries: torch.Tensor,          # [B, N_q, C4]
        pooled_regions: torch.Tensor,   # [B, N_r, C3]
        routing_scores: torch.Tensor,   # [B, N_q, N_r]
        sparse: bool = False,
    ) -> torch.Tensor:
        """
        Cross-attention from P4 queries to (selected) P3 regions.

        Training (sparse=False):
          Soft attention over all 16 regions (weighted by routing_scores).
          Gradients flow correctly through the router.

        Inference (sparse=True):
          Hard top-K selection: each query attends to only K=4 regions.
          Memory: O(N_q x K) instead of O(N_q x N_r).
          This is the actual sparse attention benefit of MSA.

        [Opt-6] Gated value residuals (ResFormer-style):
          gate = sigmoid(gate_proj(queries))  -- input-dependent scaling
          attended = attended * gate           -- suppresses irrelevant regions

        Returns: attended [B, N_q, C4]
        """
        B, N_q, C4 = queries.shape
        N_r = pooled_regions.shape[1]
        H, Dh = self.n_heads, self.head_dim

        V = self.v_proj(pooled_regions)    # [B, N_r, D]

        if not sparse:
            # ── Soft attention (training) ──────────────────────────────────
            attn_weights = F.softmax(routing_scores, dim=-1)   # [B, N_q, N_r]
            # V: [B, N_r, D] -> weighted sum -> [B, N_q, D]
            attended = torch.bmm(attn_weights, V)              # [B, N_q, D]

        else:
            # ── Hard top-K sparse attention (inference) ────────────────────
            # [Ext-5] Adaptive K: entropy-based, or fixed config top_k
            top_k = (self._adaptive_k(routing_scores)
                     if self.dynamic_topk else self.top_k)
            top_k = min(top_k, N_r)
            topk_scores, topk_idx = torch.topk(routing_scores, k=top_k, dim=-1)
            # topk_idx:    [B, N_q, K]
            # topk_scores: [B, N_q, K]

            # Gather the top-K region values for each query
            # V: [B, N_r, D] -> expand -> [B, N_q, N_r, D] -> gather -> [B, N_q, K, D]
            idx_exp = topk_idx.unsqueeze(-1).expand(B, N_q, top_k, self.d_neck)
            V_exp   = V.unsqueeze(1).expand(B, N_q, N_r, self.d_neck)
            V_topk  = V_exp.gather(dim=2, index=idx_exp)   # [B, N_q, K, D]

            # Sparse attention weights over only K regions
            attn_weights = F.softmax(topk_scores, dim=-1)          # [B, N_q, K]
            attended = (attn_weights.unsqueeze(-1) * V_topk).sum(dim=2)  # [B, N_q, D]

        # [Opt-6] Gated value residuals: input-dependent gating
        gate = torch.sigmoid(self.gate_proj(queries))   # [B, N_q, D]
        attended = attended * gate

        out = self.out_proj(attended)    # [B, N_q, C4]
        return out

    # ── Routing Loss ──────────────────────────────────────────────────────────

    @staticmethod
    def routing_loss(
        routing_scores: torch.Tensor,   # [B, N_q, N_r]
        region_targets: torch.Tensor,   # [B, G, G]  long, gold region (-1=no obj)
    ) -> torch.Tensor:
        """
        CV analogue of MSA's InfoNCE routing loss.

        For each positive grid cell (obj_mask=True), the routing scores should
        assign the highest value to the region containing the GT box center.

        Loss = CE( routing_scores[positive_cells], gold_region_ids )
        """
        B, N_q, N_r = routing_scores.shape
        # Flatten grid to query sequence
        flat_targets = region_targets.view(B, -1)       # [B, N_q]
        flat_scores  = routing_scores                   # [B, N_q, N_r]

        # Positive mask: cells that have a GT region assignment
        pos_mask = (flat_targets >= 0)                  # [B, N_q]

        if not pos_mask.any():
            return routing_scores.new_tensor(0.0)

        pos_scores  = flat_scores[pos_mask]             # [n_pos, N_r]
        pos_targets = flat_targets[pos_mask]            # [n_pos]

        return F.cross_entropy(pos_scores, pos_targets)

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        p3: torch.Tensor,                              # [B, C3, 32, 32]
        p4: torch.Tensor,                              # [B, C4, 16, 16]
        region_targets: Optional[torch.Tensor] = None, # [B, G, G] for training
        sparse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        MSA routing neck forward pass.

        Returns:
          enhanced_p4:    [B, C4, 16, 16]  -- p4 enriched with selected p3 context
          routing_scores: [B, N_q, N_r]    -- for loss computation and visualization
          routing_loss:   scalar or None
        """
        B, C4, G, _ = p4.shape
        N_q = G * G   # 256

        # ── Stage 1: Compress P3 into region vectors ("document pooling") ──
        pooled_regions = self.pool_regions(p3)          # [B, 16, C3]

        # ── Flatten P4 to query sequence ────────────────────────────────────
        # [B, C4, G, G] -> [B, G*G, C4]
        queries = p4.view(B, C4, N_q).permute(0, 2, 1)  # [B, 256, C4]

        # ── Stage 2: Compute routing scores (with 2D RoPE + softcap) ────────
        routing_scores = self.compute_routing_scores(queries, pooled_regions, G)
        # routing_scores: [B, N_q, N_r]

        # ── Stage 3: Sparse (or soft) cross-attention (with gated residuals) ─
        attended = self.sparse_cross_attn(
            queries, pooled_regions, routing_scores, sparse=sparse
        )   # [B, N_q, C4]

        # Residual + LayerNorm (pre-norm style)
        queries = self.norm1(queries + attended)

        # FFN
        queries = self.norm2(queries + self.ffn(queries))

        # Reshape back to spatial: [B, N_q, C4] -> [B, C4, G, G]
        enhanced_p4 = queries.permute(0, 2, 1).view(B, C4, G, G)

        # ── Routing loss (training only) ─────────────────────────────────────
        r_loss = None
        if region_targets is not None and self.training:
            r_loss = self.routing_loss(routing_scores, region_targets)

        return enhanced_p4, routing_scores, r_loss
