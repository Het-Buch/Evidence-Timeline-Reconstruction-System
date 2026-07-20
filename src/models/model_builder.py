"""
Model Builder
=============
Four model types:

1. VideoLevelModel    ← PRIMARY CNN classifiers (improved with temporal attention pooling)
   Input:  (B, N, 3, H, W) — B videos, N frames each
   Step 1: CNN backbone extracts per-frame features → (B, N, feat_dim)
   Step 2: Temporal ATTENTION pooling → (B, feat_dim)   [was mean — now learned]
   Step 3: Classify → (B, num_classes)

2. VideoModel3D       ← NEW: True 3D video models pretrained on Kinetics-400
   Backbones: r2plus1d_18, r3d_18, s3d, swin3d_t
   Input:  (B, N, 3, H, W) → internally permuted to (B, 3, N, H, W)
   Captures spatiotemporal motion patterns implicitly — best for anomaly detection.

3. CNNLSTM            ← Temporal localization (per-frame binary + class)
4. CNNTransformer     ← Temporal localization (per-frame binary + class)

Key fix for overfitting:
  OLD: feats.mean(dim=1)  — creates a unique "fingerprint" per video → memorization
  NEW: TemporalAttentionPool — learns which frames carry discriminative evidence
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.models.video as tvm_video

log = logging.getLogger(__name__)


# ── Temporal Attention Pooling ────────────────────────────────
class TemporalAttentionPool(nn.Module):
    """
    Replace mean pooling with learned attention over video frames.

    Instead of treating all frames equally (mean pooling), this module
    learns a scalar attention weight per frame. Frames containing the
    key anomaly event get higher weight; background/redundant frames
    get downweighted.

    This is critical for anomaly detection: an 'Assault' video has only
    a few frames that actually show the assault — mean pooling dilutes
    that signal with dozens of 'walking normally' frames.
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, feat_dim) — B videos, N frames each
        Returns:
            (B, feat_dim) — attention-weighted pooled video feature
        """
        w = self.attn(x)                    # (B, N, 1)
        w = torch.softmax(w, dim=1)         # (B, N, 1)
        return (x * w).sum(dim=1)           # (B, feat_dim)


# ── Classifier Head ──────────────────────────────────────────
class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ════════════════════════════════════════════════════════════
# Model 1: VideoLevelModel — PRIMARY CNN CLASSIFIER
#          Now with temporal attention pooling instead of mean
# ════════════════════════════════════════════════════════════
class VideoLevelModel(nn.Module):
    """
    Processes a video (stack of N frames) as one sample.

    Architecture:
      For each frame: CNN backbone → feature vector
      Across N frames: temporal attention pooling → single video feature
      → Classifier head → class logits

    Fix over original: replaced mean pooling with TemporalAttentionPool.
    Mean pooling was creating a unique 'fingerprint' per video, making
    memorization trivial (93% train / 30% val). Attention pooling forces
    the model to identify WHICH frames matter, not just average everything.
    """

    def __init__(self, backbone_name: str, num_classes: int,
                 pretrained: bool = True, freeze_backbone: bool = False,
                 dropout: float = 0.5, full_freeze: bool = False,
                 simple_head: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self._full_freeze  = full_freeze
        self._simple_head  = simple_head
        self.backbone, self.feat_dim = self._build_backbone(backbone_name, pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if not full_freeze:
                self._unfreeze_last_block()

        if simple_head:
            # Linear probe: mean-pool frames → single linear layer.
            # ~21K params for 14 classes vs 1.18M for full head.
            # With 15K training samples → 714 samples/param → no overfit.
            self.temporal_pool = None
            self.classifier    = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feat_dim, num_classes),
            )
        else:
            self.temporal_pool = TemporalAttentionPool(self.feat_dim)
            self.classifier    = ClassifierHead(self.feat_dim, num_classes, dropout)

    def _build_backbone(self, name: str, pretrained: bool):
        try:
            return self._build_backbone_inner(name, pretrained)
        except Exception as e:
            if pretrained and ("urlopen" in str(e) or "getaddrinfo" in str(e)
                               or "URLError" in type(e).__name__
                               or "gaierror" in str(e)):
                log.warning(
                    f"[model_builder] Cannot download pretrained weights for '{name}' "
                    f"(no internet). Training from scratch (random init). "
                    f"To fix: download weights manually — see README or ask for URLs."
                )
                return self._build_backbone_inner(name, pretrained=False)
            raise

    def _build_backbone_inner(self, name: str, pretrained: bool):
        w = "DEFAULT" if pretrained else None

        if name == "resnet50":
            m = tvm.resnet50(weights=w)
            f = m.fc.in_features
            m.fc = nn.Identity()
            return m, f

        elif name == "efficientnet_b3":
            m = tvm.efficientnet_b3(weights=w)
            f = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, f

        elif name == "mobilenet_v3_large":
            m = tvm.mobilenet_v3_large(weights=w)
            f = m.classifier[0].in_features
            m.classifier = nn.Identity()
            return m, f

        elif name == "densenet121":
            m = tvm.densenet121(weights=w)
            f = m.classifier.in_features
            m.classifier = nn.Identity()
            return m, f

        elif name == "convnext_tiny":
            m = tvm.convnext_tiny(weights=w)
            f = m.classifier[2].in_features
            m.classifier = nn.Identity()
            return m, f

        elif name == "convnext_small":
            m = tvm.convnext_small(weights=w)
            f = m.classifier[2].in_features
            m.classifier = nn.Identity()
            return m, f

        elif name == "vit_b_16":
            m = tvm.vit_b_16(weights=w)
            f = m.heads.head.in_features
            m.heads = nn.Identity()
            return m, f

        elif name == "swin_t":
            m = tvm.swin_t(weights=w)
            f = m.head.in_features
            m.head = nn.Identity()
            return m, f

        elif name == "swin_s":
            m = tvm.swin_s(weights=w)
            f = m.head.in_features
            m.head = nn.Identity()
            return m, f

        # Legacy backbones kept for compatibility (not recommended — weak on small data)
        elif name == "vgg16":
            m = tvm.vgg16(weights=w)
            f = 512
            m.classifier = nn.Identity()
            m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            return m, f

        elif name == "inception_v3":
            m = tvm.inception_v3(weights=w, aux_logits=True)
            f = m.fc.in_features
            m.fc = nn.Identity()
            m.AuxLogits = None
            m.aux_logits = False
            return m, f

        else:
            raise ValueError(f"Unknown backbone: {name}")

    def _unfreeze_last_block(self):
        """Unfreeze last layer block even when freeze_backbone=True."""
        last_blocks = {
            "resnet50":          "layer4",
            "efficientnet_b3":   "features.8",
            "mobilenet_v3_large":"features.16",
            "densenet121":       "features.denseblock4",
            "convnext_tiny":     "features.7",
            "convnext_small":    "features.7",
            "vit_b_16":          "encoder.layers.encoder_layer_11",
            "swin_t":            "layers.3",
            "swin_s":            "layers.3",
            "vgg16":             "features.28",
            "inception_v3":      "Mixed_7c",
        }
        key = last_blocks.get(self.backbone_name, "")
        for name, param in self.backbone.named_parameters():
            if key and key in name:
                param.requires_grad = True

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a single frame (B, 3, H, W) → (B, feat_dim)."""
        out = self.backbone(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 4:
            out = out.flatten(1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3, H, W) — B videos, N frames each
        Returns:
            logits: (B, num_classes)
        """
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)
        feats  = self._extract_features(x_flat)   # (B*N, feat_dim)
        feats  = feats.view(B, N, -1)             # (B, N, feat_dim)

        if self._simple_head:
            video_feat = feats.mean(dim=1)         # (B, feat_dim) — simple mean pool
        else:
            video_feat = self.temporal_pool(feats) # (B, feat_dim) — attention pool

        return self.classifier(video_feat)

    def extract_video_features(self, x: torch.Tensor) -> torch.Tensor:
        """For GradCAM / XAI — returns attention-pooled video features."""
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)
        feats  = self._extract_features(x_flat).view(B, N, -1)
        return self.temporal_pool(feats)


# ════════════════════════════════════════════════════════════
# Model 2: VideoModel3D — TRUE 3D VIDEO MODELS (NEW)
#          Pretrained on Kinetics-400/600
# ════════════════════════════════════════════════════════════
class VideoModel3D(nn.Module):
    """
    True 3D video understanding models from torchvision.models.video.
    Pretrained on Kinetics-400 — the largest public action recognition dataset.

    These models IMPLICITLY learn motion patterns via 3D or (2+1)D convolutions,
    which is far more powerful than applying a 2D CNN frame-by-frame for tasks
    like anomaly detection where motion IS the signal.

    Supported backbones:
      - r2plus1d_18: R(2+1)D-18 — factorized 3D convolutions, fast + accurate
      - r3d_18:      R3D-18     — pure 3D convolutions, slightly stronger
      - s3d:         S3D        — lightweight (8M params), great for small datasets
      - swin3d_t:    Video Swin Tiny — transformer, best accuracy among small models

    Input:  (B, N, 3, H, W) — automatically permuted to (B, 3, N, H, W)
    Output: (B, num_classes)
    """

    # (factory_fn, feature_dim, classifier_attr_to_replace)
    _CONFIGS = {
        "r2plus1d_18": ("r2plus1d_18", 512,  "fc"),
        "r3d_18":      ("r3d_18",      512,  "fc"),
        "s3d":         ("s3d",         1024, "classifier"),
        "swin3d_t":    ("swin3d_t",    768,  "head"),
    }

    def __init__(self, backbone_name: str, num_classes: int,
                 pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        assert backbone_name in self._CONFIGS, \
            f"Unknown 3D backbone: {backbone_name}. Choose from {list(self._CONFIGS)}"

        model_name, self.feat_dim, cls_attr = self._CONFIGS[backbone_name]

        def _load(use_pretrained):
            w = "DEFAULT" if use_pretrained else None
            return getattr(tvm_video, model_name)(weights=w)

        try:
            backbone = _load(pretrained)
        except Exception as e:
            if pretrained and ("urlopen" in str(e) or "getaddrinfo" in str(e)
                               or "URLError" in type(e).__name__
                               or "gaierror" in str(e)):
                log.warning(
                    f"[model_builder] Cannot download pretrained weights for '{backbone_name}' "
                    f"(no internet). Training from scratch (random init)."
                )
                backbone = _load(False)
            else:
                raise

        # Replace the final classifier with Identity so we get raw features
        setattr(backbone, cls_attr, nn.Identity())
        self.backbone = backbone
        self.backbone_name = backbone_name

        self.classifier = ClassifierHead(self.feat_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3, H, W) — standard video tensor from VideoLevelDataset
        Returns:
            logits: (B, num_classes)
        """
        # Permute to torchvision video format: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        feat = self.backbone(x)

        # Handle tuple outputs (some models return InceptionOutputs)
        if isinstance(feat, tuple):
            feat = feat[0]

        # Flatten any extra spatial/temporal dims: (B, D, 1, 1, 1) → (B, D)
        if feat.dim() > 2:
            feat = feat.flatten(1)

        return self.classifier(feat)

    def extract_video_features(self, x: torch.Tensor) -> torch.Tensor:
        """For GradCAM / XAI."""
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.backbone(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.dim() > 2:
            feat = feat.flatten(1)
        return feat


# ════════════════════════════════════════════════════════════
# Model 3: CNN-LSTM — Temporal localization
# ════════════════════════════════════════════════════════════
class CNNLSTM(nn.Module):
    """
    CNN-LSTM for temporal anomaly localization.
    Input:  (B, T, 3, H, W)
    Output: frame_logits (B, T, 2), class_logits (B, C)
    """

    def __init__(self, num_classes: int, hidden_size: int = 512,
                 num_layers: int = 2, dropout: float = 0.4,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        backbone = tvm.resnet50(weights="DEFAULT" if pretrained else None)
        self.feat_dim = backbone.fc.in_features   # 2048
        backbone.fc   = nn.Identity()
        for name, p in backbone.named_parameters():
            if "layer4" not in name and "layer3" not in name:
                p.requires_grad = False
        self.cnn = backbone

        self.feat_proj = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_out = hidden_size * 2

        self.frame_head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.class_head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats  = self.cnn(x_flat)
        if isinstance(feats, tuple): feats = feats[0]
        if feats.dim() == 4: feats = feats.mean(dim=[2, 3])
        feats  = feats.view(B, T, -1)
        feats  = self.feat_proj(feats.view(B * T, -1)).view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        frame_logits = self.frame_head(lstm_out)
        class_logits = self.class_head(lstm_out.mean(dim=1))
        return frame_logits, class_logits


# ════════════════════════════════════════════════════════════
# Model 4: CNN-Transformer — Temporal localization
# ════════════════════════════════════════════════════════════
class CNNTransformer(nn.Module):
    """
    CNN-Transformer for temporal anomaly localization.
    Input:  (B, T, 3, H, W)
    Output: frame_logits (B, T, 2), class_logits (B, C)
    """

    def __init__(self, num_classes: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.d_model     = d_model

        backbone = tvm.efficientnet_b3(weights="DEFAULT" if pretrained else None)
        self.feat_dim = backbone.classifier[1].in_features   # 1536
        backbone.classifier = nn.Identity()
        for name, p in backbone.named_parameters():
            if "features.7" not in name and "features.8" not in name:
                p.requires_grad = False
        self.cnn = backbone

        self.feat_proj = nn.Sequential(
            nn.Linear(self.feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.pos_encoding = nn.Embedding(512, d_model)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.frame_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 2),
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats  = self.cnn(x_flat)
        if isinstance(feats, tuple): feats = feats[0]
        if feats.dim() == 4: feats = feats.mean(dim=[2, 3])
        feats  = feats.view(B, T, -1)
        feats  = self.feat_proj(feats.view(B * T, -1)).view(B, T, -1)
        pos    = torch.arange(T, device=x.device).unsqueeze(0)
        feats  = feats + self.pos_encoding(pos)
        cls    = self.cls_token.expand(B, -1, -1)
        feats  = torch.cat([cls, feats], dim=1)
        out    = self.transformer(feats)
        frame_logits = self.frame_head(out[:, 1:, :])
        class_logits = self.class_head(out[:, 0, :])
        return frame_logits, class_logits


# ── Factory functions ────────────────────────────────────────
_3D_BACKBONES = {"r2plus1d_18", "r3d_18", "s3d", "swin3d_t"}


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    name     = model_cfg["name"]
    backbone = model_cfg.get("backbone", "")
    dropout  = float(model_cfg.get("dropout", 0.5))

    if name == "CNNLSTM":
        return CNNLSTM(
            num_classes=num_classes,
            hidden_size=model_cfg.get("hidden_size", 512),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=dropout,
            pretrained=model_cfg.get("pretrained", True),
        )
    elif name == "CNNTransformer":
        return CNNTransformer(
            num_classes=num_classes,
            d_model=model_cfg.get("d_model", 512),
            nhead=model_cfg.get("nhead", 8),
            num_layers=model_cfg.get("num_layers", 4),
            dropout=dropout,
            pretrained=model_cfg.get("pretrained", True),
        )
    elif backbone in _3D_BACKBONES:
        return VideoModel3D(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            dropout=dropout,
        )
    else:
        return VideoLevelModel(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            dropout=dropout,
            full_freeze=model_cfg.get("full_freeze", False),
            simple_head=model_cfg.get("simple_head", False),
        )


def load_model(model_cfg: dict, num_classes: int,
               checkpoint_path: str, device: str = "cpu") -> nn.Module:
    model = build_model(model_cfg, num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def is_temporal_model(model_cfg: dict) -> bool:
    return model_cfg["name"] in ("CNNLSTM", "CNNTransformer")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    num_classes = 14
    x_vid = torch.randn(2, 16, 3, 224, 224)

    print("=== 2D VideoLevelModel (with temporal attention) ===")
    for name, backbone in [
        ("ResNet50",       "resnet50"),
        ("EfficientNetB3", "efficientnet_b3"),
        ("ConvNeXtTiny",   "convnext_tiny"),
        ("SwinT",          "swin_t"),
    ]:
        cfg = {"name": name, "backbone": backbone,
               "pretrained": False, "freeze_backbone": False}
        m   = build_model(cfg, num_classes)
        out = m(x_vid)
        print(f"  [OK] {name:20s} → {out.shape}  params={count_parameters(m):,}")

    print("\n=== 3D VideoModel3D (Kinetics pretrained) ===")
    for backbone in ["r2plus1d_18", "s3d"]:
        cfg = {"name": backbone, "backbone": backbone, "pretrained": False}
        m   = build_model(cfg, num_classes)
        out = m(x_vid)
        print(f"  [OK] {backbone:20s} → {out.shape}  params={count_parameters(m):,}")

    print("\n=== Temporal Models ===")
    for name in ["CNNLSTM", "CNNTransformer"]:
        cfg = {"name": name, "pretrained": False}
        m   = build_model(cfg, num_classes)
        fl, cl = m(x_vid)
        print(f"  [OK] {name:20s} → frame={fl.shape} class={cl.shape}  params={count_parameters(m):,}")
