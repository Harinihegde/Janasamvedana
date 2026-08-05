"""CSRNet: dilated-convolution crowd-density estimator (Li et al., CVPR 2018).

Used as Detector's dense-crowd fallback in place of the edge-density heuristic
when a pretrained checkpoint is supplied. The architecture and state_dict key
names match the reference implementation exactly
(github.com/leeyeehoo/CSRNet-pytorch) so its official ShanghaiTech-pretrained
checkpoints load unmodified - verified against the ShanghaiA checkpoint
(state_dict shapes match, best_prec1=65.9 vs. the repo's documented MAE 66.4).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# CSRNet is fully convolutional (any input size works); this only bounds
# inference cost on very large source frames.
MAX_SIDE = 768
# Smaller cap for the always-on per-frame pass (Detector.csrnet_density),
# which runs on every sampled frame rather than just the rare YOLO-fallback
# case - roughly (768/384)^2 = 4x less compute per frame to keep full-dataset
# extraction tractable.
ALWAYS_ON_MAX_SIDE = 384


def _make_layers(cfg: list, in_channels: int = 3, dilation: bool = False) -> nn.Sequential:
    d_rate = 2 if dilation else 1
    layers: list[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate),
                nn.ReLU(inplace=True),
            ]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """VGG16-frontend + dilated-conv backend density-map regressor."""

    FRONTEND_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
    BACKEND_CFG = [512, 512, 512, 256, 128, 64]

    def __init__(self):
        super().__init__()
        self.frontend = _make_layers(self.FRONTEND_CFG)
        self.backend = _make_layers(self.BACKEND_CFG, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)


def default_device() -> str:
    """Best available torch device for CSRNet inference."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_csrnet(weights_path: str, device: str | None = None) -> CSRNet:
    """Load a reference-implementation CSRNet checkpoint.

    Accepts either a bare state_dict or the reference repo's full training
    checkpoint dict (``{"state_dict":..., "optimizer":..., "epoch":...}``).

    Uses ``weights_only=False``: the reference checkpoints bundle optimizer
    state that the restricted unpickler rejects. Only pass a checkpoint you
    trust - this executes arbitrary pickle content, like any ``torch.load``
    of a full training checkpoint.
    """
    device = device or default_device()
    model = CSRNet()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def _density_map(model: CSRNet, frame_bgr: np.ndarray, max_side: int = MAX_SIDE) -> torch.Tensor:
    """Run CSRNet and return its raw predicted density map (1, 1, H', W')."""
    device = next(model.parameters()).device
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return model(tensor)


def estimate_count(model: CSRNet, frame_bgr: np.ndarray) -> float:
    """Estimate a head count for one BGR frame via its predicted density map.

    Sums the predicted density map - the standard CSRNet counting readout
    (the network is trained so the density map integrates to the head count).
    """
    return float(_density_map(model, frame_bgr).sum().item())


def estimate_density(
    model: CSRNet, frame_bgr: np.ndarray, max_side: int = MAX_SIDE
) -> tuple[float, float]:
    """Return ``(count, peak_density)`` for one BGR frame.

    ``count`` is the same whole-frame headcount estimate as
    :func:`estimate_count`. ``peak_density`` is the single highest-value cell
    in the density map (before any downstream normalisation) - a crush/
    hotspot signal distinct from the whole-frame count: a tightly packed
    cluster in one corner of an otherwise sparse frame raises ``peak_density``
    without necessarily raising ``count`` much, which is exactly the
    "packed subgroup within a bigger scene" case a single count can miss.
    """
    density = _density_map(model, frame_bgr, max_side=max_side)
    return float(density.sum().item()), float(density.max().item())
