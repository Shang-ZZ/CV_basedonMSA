"""
TinyVision backbone — pluggable feature extractor.

All backbones produce two feature map levels:
  P3: [B, C3, H/8,  W/8 ]   stride-8  — fine-grained spatial detail
  P4: [B, C4, H/16, W/16]   stride-16 — semantic detection features

P3 is the "document memory" for the MSA neck.
P4 is the "query sequence" that routes into P3.

Available backbones (set via ModelConfig.backbone_type):
  'custom'       — lightweight 4-stage CNN (default, ~1.17M params, no extra deps)
  'resnet18'     — torchvision ResNet-18 (layer2=P3, layer3=P4)
  'mobilenet_v2' — torchvision MobileNetV2 (features[0:7]=P3, [7:14]=P4)

For pretrained backbones, adaptor 1×1 convs project backbone channels to the
config's C3/C4, so the MSA Neck always receives the same dimensions regardless
of which backbone is used.
"""
import torch
import torch.nn as nn
from .config import ModelConfig


# ─── Custom lightweight backbone (original) ──────────────────────────────────

def _conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvStage(nn.Module):
    """Two conv-BN-ReLU layers followed by MaxPool(2)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(in_ch,  out_ch),
            _conv_bn_relu(out_ch, out_ch),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CustomBackbone(nn.Module):
    """
    Lightweight 4-stage CNN backbone (~1.17M params).

    Input:  [B, 3,   H,    W   ]
    Stage1: [B, 32,  H/2,  W/2 ]
    Stage2: [B, 64,  H/4,  W/4 ]
    Stage3: [B, 128, H/8,  W/8 ]  <- P3  (stride 8)
    Stage4: [B, 256, H/16, W/16]  <- P4  (stride 16)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        ch = config.backbone_channels        # [32, 64, 128, 256]
        self.stage1 = ConvStage(3,     ch[0])
        self.stage2 = ConvStage(ch[0], ch[1])
        self.stage3 = ConvStage(ch[1], ch[2])  # → P3
        self.stage4 = ConvStage(ch[2], ch[3])  # → P4

    def forward(self, x):
        x  = self.stage1(x)
        x  = self.stage2(x)
        p3 = self.stage3(x)    # [B, 128, H/8,  W/8 ]
        p4 = self.stage4(p3)   # [B, 256, H/16, W/16]
        return p3, p4


# ─── ResNet-18 backbone ───────────────────────────────────────────────────────

class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone using torchvision.

    Feature extraction:
      layer2 output → P3  (128 ch, stride 8)
      layer3 output → P4  (256 ch, stride 16)

    ResNet-18 channels (128, 256) happen to match the default config.c3/c4,
    so adaptor convs are identity by default. Set different backbone_channels
    in ModelConfig to use adaptors.

    Set ModelConfig.freeze_backbone=True to freeze encoder weights and only
    train the adaptors + MSA Neck + Detection Head (standard fine-tuning).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        try:
            import torchvision.models as tv
        except ImportError:
            raise ImportError(
                "torchvision is required for backbone_type='resnet18'.\n"
                "Install: pip install torchvision"
            )

        resnet = tv.resnet18(weights=None)

        # Stem: conv1(stride 2) → bn1 → relu → maxpool(stride 2) = stride 4
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 64ch, stride 4
        self.layer2 = resnet.layer2   # 128ch, stride 8   ← P3
        self.layer3 = resnet.layer3   # 256ch, stride 16  ← P4

        # Adaptor convs: project resnet channels to config.c3 / config.c4
        # (1×1 convs, BN, ReLU — used when config channels differ from ResNet defaults)
        self.adapt_p3 = (nn.Identity() if config.c3 == 128 else
                         nn.Sequential(nn.Conv2d(128, config.c3, 1, bias=False),
                                       nn.BatchNorm2d(config.c3), nn.ReLU(inplace=True)))
        self.adapt_p4 = (nn.Identity() if config.c4 == 256 else
                         nn.Sequential(nn.Conv2d(256, config.c4, 1, bias=False),
                                       nn.BatchNorm2d(config.c4), nn.ReLU(inplace=True)))

        if config.freeze_backbone:
            for p in list(self.stem.parameters()) + \
                     list(self.layer1.parameters()) + \
                     list(self.layer2.parameters()) + \
                     list(self.layer3.parameters()):
                p.requires_grad_(False)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        p3 = self.layer2(x)             # [B, 128, H/8,  W/8 ]
        p4 = self.layer3(p3)            # [B, 256, H/16, W/16]
        return self.adapt_p3(p3), self.adapt_p4(p4)


# ─── MobileNetV2 backbone ─────────────────────────────────────────────────────

class MobileNetBackbone(nn.Module):
    """
    MobileNetV2 backbone using torchvision.

    Feature extraction (MobileNetV2 inverted residual layout):
      features[0:7]  → P3  (32ch at stride 8)
      features[7:14] → P4  (96ch at stride 16)

    Adaptor convs project 32→C3 and 96→C4 (default C3=128, C4=256).

    MobileNetV2 is ~3.4M params total (used portion ~1M), much lighter than
    ResNet-18 at ~11M. Suitable for mobile/edge deployment scenarios.
    """
    # Channels at each feature level boundary
    _P3_CH = 32    # features[6] output
    _P4_CH = 96    # features[13] output

    def __init__(self, config: ModelConfig):
        super().__init__()
        try:
            import torchvision.models as tv
        except ImportError:
            raise ImportError(
                "torchvision is required for backbone_type='mobilenet_v2'.\n"
                "Install: pip install torchvision"
            )

        mv2      = tv.mobilenet_v2(weights=None)
        features = mv2.features

        # features[0..6]:  stride 2→2→4→4→8→8→8 = stride 8, 32ch output
        self.p3_features = nn.Sequential(*features[:7])
        # features[7..13]: stride 16→16→16→16→16→16→16, 96ch output
        self.p4_features = nn.Sequential(*features[7:14])

        # Adaptor convs: 32 → C3 and 96 → C4
        self.adapt_p3 = nn.Sequential(
            nn.Conv2d(self._P3_CH, config.c3, 1, bias=False),
            nn.BatchNorm2d(config.c3), nn.ReLU(inplace=True),
        )
        self.adapt_p4 = nn.Sequential(
            nn.Conv2d(self._P4_CH, config.c4, 1, bias=False),
            nn.BatchNorm2d(config.c4), nn.ReLU(inplace=True),
        )

        if config.freeze_backbone:
            for p in list(self.p3_features.parameters()) + \
                     list(self.p4_features.parameters()):
                p.requires_grad_(False)

    def forward(self, x):
        p3_raw = self.p3_features(x)    # [B, 32, H/8,  W/8 ]
        p4_raw = self.p4_features(p3_raw)  # [B, 96, H/16, W/16]
        return self.adapt_p3(p3_raw), self.adapt_p4(p4_raw)


# ─── Keep original name as alias for backward compatibility ───────────────────
Backbone = CustomBackbone


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_backbone(config: ModelConfig) -> nn.Module:
    """
    Instantiate the backbone specified by config.backbone_type.

    Returns a module with signature: forward(x) -> (p3, p4)
    """
    _registry = {
        'custom':       CustomBackbone,
        'resnet18':     ResNetBackbone,
        'mobilenet_v2': MobileNetBackbone,
    }
    if config.backbone_type not in _registry:
        raise ValueError(
            f"Unknown backbone_type='{config.backbone_type}'. "
            f"Choose from: {list(_registry.keys())}"
        )
    return _registry[config.backbone_type](config)
