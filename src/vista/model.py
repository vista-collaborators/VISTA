import os
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn

from .utils import extract_state_dict, strip_state_prefix, torch_load


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_DINOV3_ROOT = REPO_ROOT / "third_party" / "dinov3"


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        hidden_dim = max(1, int(feature_dim) // 2)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)


def _load_dinov3_direct(repo_dir: str, model_name: str, weights: str = "") -> nn.Module:
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from dinov3.hub import backbones

    builder = getattr(backbones, model_name)
    kwargs: dict[str, Any] = {}
    if weights:
        kwargs["weights"] = weights
    else:
        kwargs["pretrained"] = False
    return builder(**kwargs)


def _load_dinov3_installed(model_name: str, weights: str = "") -> nn.Module:
    from dinov3.hub import backbones

    builder = getattr(backbones, model_name)
    kwargs: dict[str, Any] = {}
    if weights:
        kwargs["weights"] = weights
    else:
        kwargs["pretrained"] = False
    return builder(**kwargs)


def load_dinov3_backbone(
    hub_repo: str,
    hub_model: str,
    weights: str = "",
) -> nn.Module:
    if weights and not os.path.isfile(weights):
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")

    errors: List[str] = []

    def try_build(with_weights: bool) -> nn.Module:
        weight_arg = weights if with_weights else ""
        candidate_repos = []
        if hub_repo:
            candidate_repos.append(hub_repo)
        candidate_repos.append(str(VENDORED_DINOV3_ROOT))

        for repo in candidate_repos:
            if not repo or not os.path.isdir(repo):
                continue
            try:
                kwargs: dict[str, Any] = {"source": "local"}
                if weight_arg:
                    kwargs["weights"] = weight_arg
                else:
                    kwargs["pretrained"] = False
                return torch.hub.load(repo, hub_model, **kwargs)
            except Exception as exc:
                errors.append(f"torch.hub local repo failed for {repo}: {exc}")
            try:
                return _load_dinov3_direct(repo, hub_model, weights=weight_arg)
            except Exception as exc:
                errors.append(f"direct local repo import failed for {repo}: {exc}")
        try:
            return _load_dinov3_installed(hub_model, weights=weight_arg)
        except Exception as exc:
            errors.append(f"installed dinov3 import failed: {exc}")
        raise RuntimeError("; ".join(errors[-3:]))

    try:
        return try_build(with_weights=bool(weights))
    except Exception:
        if not weights:
            raise RuntimeError(
                "failed to build DINOv3. Run scripts/install_dinov3.sh, "
                "pip install -e custom/dinov3, or pass --dinov3_hub_repo custom/dinov3. "
                f"Details: {'; '.join(errors[-3:])}"
            )

    try:
        backbone = try_build(with_weights=False)
    except Exception as exc:
        raise RuntimeError(
            "failed to build a DINOv3 architecture for manual weight loading. "
            "Run scripts/install_dinov3.sh, pip install -e custom/dinov3, "
            "or pass --dinov3_hub_repo custom/dinov3."
        ) from exc

    checkpoint = torch_load(weights, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    current = backbone.state_dict()
    cleaned = {}
    skipped_shape = 0
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        new_key = strip_state_prefix(key)
        if new_key in current and value.shape != current[new_key].shape:
            skipped_shape += 1
            continue
        cleaned[new_key] = value
    msg = backbone.load_state_dict(cleaned, strict=False)
    print(
        "Loaded DINOv3 weights manually | "
        f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} "
        f"skipped_shape={skipped_shape}"
    )
    return backbone


class Dinov3MultiSliceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_slices: int,
        adaptation: str,
        dinov3_hub_repo: str,
        dinov3_hub_model: str = "dinov3_vitl16",
        dinov3_weights: str = "",
        dinov3_feature_mode: str = "avg_patch",
        fusion_layers: int = 2,
        fusion_dropout: float = 0.1,
        head_hidden_dims: Sequence[int] = (512, 256),
        head_dropouts: Sequence[float] = (0.5, 0.3),
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_slices = int(num_slices)
        self.adaptation = str(adaptation).strip().lower()
        self.dinov3_feature_mode = str(dinov3_feature_mode).strip().lower()
        if self.adaptation not in {"lp", "finetune"}:
            raise ValueError("adaptation must be lp or finetune")

        self.backbone = load_dinov3_backbone(
            hub_repo=dinov3_hub_repo,
            hub_model=dinov3_hub_model,
            weights=dinov3_weights,
        )
        self.feature_dim = self._infer_feature_dim(self.backbone)
        self.slice_pos_embed = nn.Parameter(
            torch.randn(1, self.num_slices, self.feature_dim) * 0.02,
            requires_grad=True,
        )
        n_head = 12 if self.feature_dim % 12 == 0 else (8 if self.feature_dim % 8 == 0 else 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=n_head,
            dim_feedforward=self.feature_dim * 4,
            dropout=float(fusion_dropout),
            batch_first=True,
            activation="gelu",
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(fusion_layers),
        )
        self.attention_pool = AttentionPooling(self.feature_dim)
        self.fusion_norm = nn.LayerNorm(self.feature_dim)
        self.classifier = self._build_classifier_head(
            self.feature_dim,
            [int(x) for x in head_hidden_dims],
            [float(x) for x in head_dropouts],
            self.num_classes,
        )

        if self.adaptation == "lp":
            self._freeze_backbone()
        else:
            self._enable_finetune()

    @staticmethod
    def _infer_feature_dim(backbone: nn.Module) -> int:
        for attr in ("embed_dim", "num_features"):
            value = getattr(backbone, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        head = getattr(backbone, "head", None)
        if head is not None and hasattr(head, "in_features"):
            return int(head.in_features)
        raise RuntimeError("cannot infer DINOv3 feature dimension")

    @staticmethod
    def _build_classifier_head(
        feature_dim: int,
        hidden_dims: List[int],
        dropouts: List[float],
        out_dim: int,
    ) -> nn.Module:
        if len(hidden_dims) == 0:
            return nn.Linear(feature_dim, out_dim)
        if len(dropouts) == 1 and len(hidden_dims) > 1:
            dropouts = [dropouts[0] for _ in hidden_dims]
        if len(dropouts) != len(hidden_dims):
            raise ValueError("head_dropouts length must match head_hidden_dims length")

        layers: List[nn.Module] = []
        in_dim = int(feature_dim)
        for hidden_dim, dropout in zip(hidden_dims, dropouts):
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(out_dim)))
        return nn.Sequential(*layers)

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _enable_finetune(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.adaptation == "lp":
            self.backbone.eval()
        return self

    def _extract_single_features(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "forward_features"):
            out = self.backbone(x)
            return out if out.ndim != 1 else out.unsqueeze(0)

        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            first_tensor = next((v for v in out.values() if torch.is_tensor(v)), None)
            if first_tensor is None:
                raise RuntimeError("DINOv3 forward_features dict contains no tensor")
            if self.dinov3_feature_mode == "avg_patch":
                for key in ("x_norm_patchtokens", "x_prenorm"):
                    value = out.get(key)
                    if torch.is_tensor(value):
                        out = value.mean(dim=1) if value.ndim == 3 else value
                        break
                else:
                    out = out.get("x_norm_clstoken", first_tensor)
            else:
                out = out.get("x_norm_clstoken", first_tensor)
        if isinstance(out, (list, tuple)):
            if not out:
                raise RuntimeError("empty output returned by DINOv3 forward_features")
            out = out[0]
        if not torch.is_tensor(out):
            raise RuntimeError("DINOv3 feature output is not a tensor")
        if out.ndim == 3:
            out = out[:, 0, :] if out.shape[1] == 1 else out.mean(dim=1)
        elif out.ndim == 1:
            out = out.unsqueeze(0)
        return out

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, n_slices, channels, height, width = pixel_values.shape
        if n_slices != self.num_slices:
            if n_slices > self.num_slices:
                pixel_values = pixel_values[:, : self.num_slices]
            else:
                pad_size = self.num_slices - n_slices
                pad = pixel_values[:, -1:].expand(-1, pad_size, -1, -1, -1)
                pixel_values = torch.cat([pixel_values, pad], dim=1)
            n_slices = self.num_slices

        flattened = pixel_values.reshape(batch_size * n_slices, channels, height, width)
        if self.adaptation == "lp":
            with torch.no_grad():
                features_flat = self._extract_single_features(flattened)
        else:
            features_flat = self._extract_single_features(flattened)
        features = features_flat.reshape(batch_size, n_slices, -1)
        fused_seq = self.fusion_transformer(features + self.slice_pos_embed)
        fused_features, _ = self.attention_pool(fused_seq)
        fused_features = self.fusion_norm(fused_features)
        return self.classifier(fused_features)


def build_model_from_config(config: dict[str, Any]) -> Dinov3MultiSliceClassifier:
    return Dinov3MultiSliceClassifier(
        num_classes=int(config["num_classes"]),
        num_slices=len(config["slice_indices"]),
        adaptation=config.get("adaptation", "finetune"),
        dinov3_hub_repo=config.get("dinov3_hub_repo", ""),
        dinov3_hub_model=config.get("dinov3_hub_model", "dinov3_vitl16"),
        dinov3_weights=config.get("dinov3_weights", ""),
        dinov3_feature_mode=config.get("dinov3_feature_mode", "avg_patch"),
        fusion_layers=int(config.get("fusion_layers", 2)),
        fusion_dropout=float(config.get("fusion_dropout", 0.1)),
        head_hidden_dims=config.get("head_hidden_dims", [512, 256]),
        head_dropouts=config.get("head_dropouts", [0.5, 0.3]),
    )
