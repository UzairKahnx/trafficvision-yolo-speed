"""Sanity-check the exported ONNX model on validation images.

Reports IoU@0.5 precision/recall per class and writes a contact sheet of
predictions to assets/predictions_grid.jpg for the README.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.detector import YoloOnnxDetector, annotate

VAL_IMAGES = ROOT / "data/raw/val/images"
VAL_LABELS = ROOT / "data/raw/val/labels"
OUT = ROOT / "assets" / "predictions_grid.jpg"


def read_gt(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append(
            (
                cls,
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            )
        )
    return boxes


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    detector = YoloOnnxDetector()
    images = sorted(VAL_IMAGES.glob("*.jpg"))[:120]
    if not images:
        raise SystemExit("No validation images found. Run scripts/download_dataset.py first.")

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    previews: list[np.ndarray] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        detections = detector.detect(image, conf_threshold=0.25, iou_threshold=0.45)
        gt = read_gt(VAL_LABELS / (image_path.stem + ".txt"), w, h)

        used: set[int] = set()
        for det in detections:
            label = det.label
            best_iou, best_index = 0.0, -1
            for index, box in enumerate(gt):
                if index in used or detector.names.get(box[0]) != label:
                    continue
                iou = box_iou((det.x1, det.y1, det.x2, det.y2), box[1:])
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_iou >= 0.5:
                stats[label]["tp"] += 1
                used.add(best_index)
            else:
                stats[label]["fp"] += 1
        for index, box in enumerate(gt):
            if index not in used:
                stats[detector.names.get(box[0], str(box[0]))]["fn"] += 1

        if len(previews) < 9:
            previews.append(annotate(image, detections))

    print(f"Evaluated {len(images)} validation images (IoU threshold 0.5, conf 0.25)\n")
    print(f"{'class':<8}{'TP':>6}{'FP':>6}{'FN':>6}{'precision':>12}{'recall':>10}{'F1':>8}")
    for label in ["car", "bus", "truck", "van"]:
        s = stats[label]
        precision = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else math.nan
        recall = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else math.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
        print(
            f"{label:<8}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}"
            f"{precision:>12.3f}{recall:>10.3f}{f1:>8.3f}"
        )

    tiles = []
    for preview in previews:
        small = cv2.resize(preview, (320, 320))
        tiles.append(small)
    while len(tiles) < 9:
        tiles.append(np.full((320, 320, 3), 255, dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[row : row + 3]) for row in range(0, 9, 3)])
    OUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT), grid)
    print(f"\nPrediction contact sheet: {OUT}")


if __name__ == "__main__":
    main()
