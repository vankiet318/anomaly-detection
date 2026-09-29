"""Tạo ảnh heatmap / vùng lỗi từ anomaly map."""

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageFilter


DEFECT_COLOR = (255, 40, 40)


def normalize_map(anomaly_map, threshold, mode):
    """
    mode = "threshold": thang màu cố định quanh ngưỡng -> ảnh tốt trông "mát", ảnh lỗi trông "nóng"
    mode = "image": min-max theo từng ảnh -> luôn thấy vùng bất thường nhất, kể cả ảnh tốt
    """
    if mode == "image":
        low, high = anomaly_map.min(), anomaly_map.max()
    else:
        low, high = threshold * 0.5, threshold * 1.5

    return np.clip((anomaly_map - low) / max(high - low, 1e-8), 0, 1)


def heatmap_overlay(image, anomaly_map, threshold, mode, alpha):
    normalized = normalize_map(anomaly_map, threshold, mode)
    heatmap = (colormaps["jet"](normalized)[..., :3] * 255).astype(np.uint8)

    return Image.blend(image.convert("RGB"), Image.fromarray(heatmap), alpha)


def defect_overlay(image, anomaly_map, threshold, fill_alpha=0.35, border_width=5):
    """Tô vùng có anomaly score >= threshold và vẽ viền quanh vùng đó."""
    mask = Image.fromarray(((anomaly_map >= threshold) * 255).astype(np.uint8))

    base = image.convert("RGB")
    color = Image.new("RGB", base.size, DEFECT_COLOR)

    filled = Image.composite(Image.blend(base, color, fill_alpha), base, mask)

    # Viền = mask trừ đi bản co lại của nó
    eroded = mask.filter(ImageFilter.MinFilter(border_width))
    border = Image.fromarray(((np.array(mask) > 0) & (np.array(eroded) == 0)).astype(np.uint8) * 255)

    return Image.composite(color, filled, border)


def defect_area_ratio(anomaly_map, threshold):
    return float((anomaly_map >= threshold).mean())
