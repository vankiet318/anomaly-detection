"""Inference PatchCore từ checkpoint đã train trong notebooks/defect-detection.ipynb."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


CHECKPOINT_DIR = Path(__file__).parent / "notebooks" / "checkpoints"
RESULTS_PATH = CHECKPOINT_DIR / "results.csv"


def list_categories(checkpoint_dir=CHECKPOINT_DIR):
    return sorted(p.stem for p in Path(checkpoint_dir).glob("*.pt"))


class FeatureExtractor(nn.Module):
    """
    WideResNet50 cắt tới layer3.

    Kết quả giống hệt forward hook trên layer2/layer3 lúc train, nhưng bỏ layer4 + fc
    nên nhẹ hơn nhiều (RAM và thời gian chạy trên CPU).
    """

    def __init__(self):
        super().__init__()

        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, images):
        layer2 = self.layer2(self.stem(images))
        layer3 = self.layer3(layer2)

        return layer2, layer3


def load_feature_extractor(device):
    extractor = FeatureExtractor().to(device).eval()

    for param in extractor.parameters():
        param.requires_grad = False

    return extractor


def extract_features(images, extractor):
    """
    images: [B, 3, H, W]

    return:
        patches: [B, h*w, C] (L2-normalized)
        feature_size: (h, w)
    """
    layer2, layer3 = extractor(images)

    layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
    combined = torch.cat([layer2, layer3], dim=1)

    B, C, H, W = combined.shape

    patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
    patches = F.normalize(patches, p=2, dim=-1)

    return patches, (H, W)


@dataclass
class PatchCoreModel:
    category: str
    memory_bank: torch.Tensor  # [N, D], float32
    threshold: float
    metrics: dict
    image_size: tuple
    transform: transforms.Compose

    @classmethod
    def load(cls, category, device, checkpoint_dir=CHECKPOINT_DIR):
        checkpoint = torch.load(Path(checkpoint_dir) / f"{category}.pt", map_location="cpu")
        config = checkpoint["config"]
        image_size = tuple(config["image_size"])

        return cls(
            category=checkpoint["category"],
            memory_bank=checkpoint["memory_bank"].float().to(device),
            threshold=float(checkpoint["threshold"]),
            metrics=checkpoint["metrics"],
            image_size=image_size,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=config["mean"], std=config["std"]),
            ]),
        )


@dataclass
class Prediction:
    score: float
    anomaly_map: np.ndarray  # [H, W] theo output_size


@torch.no_grad()
def predict(image: Image.Image, model: PatchCoreModel, extractor, device, output_size=None):
    """output_size: (width, height) của anomaly map, mặc định bằng kích thước ảnh."""
    image = image.convert("RGB")
    width, height = output_size or image.size
    batch = model.transform(image).unsqueeze(0).to(device)

    patches, (h, w) = extract_features(batch, extractor)

    # Khoảng cách tới normal patch gần nhất: [h*w]
    nearest = torch.cdist(patches[0], model.memory_bank).min(dim=1).values

    # Upsample để hiển thị chồng lên ảnh
    anomaly_map = F.interpolate(
        nearest.reshape(1, 1, h, w),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return Prediction(score=nearest.max().item(), anomaly_map=anomaly_map.cpu().numpy())
