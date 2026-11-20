"""Lightweight ONNX inference for the fine-tuned YOLOv8 vehicle detector.

The training pipeline (see scripts/train.py) exports the fine-tuned model to
ONNX.  Running inference with onnxruntime keeps the Streamlit deployment small
(no PyTorch / CUDA wheels needed in the cloud).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "vehicle_detector.onnx"
NAMES_PATH = Path(__file__).resolve().parent.parent / "models" / "class_names.json"

CLASS_COLORS = {
    "car": (52, 152, 219),
    "bus": (155, 89, 182),
    "truck": (230, 126, 34),
    "van": (46, 204, 113),
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        names_path: Path = NAMES_PATH,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run scripts/train.py first.")

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.imgsz = int(input_shape[2]) if input_shape[2] > 0 else 640

        if names_path.exists():
            self.names: dict[int, str] = {int(k): v for k, v in json.loads(names_path.read_text()).items()}
        else:
            self.names = {0: "bus", 1: "car", 2: "truck", 3: "van"}

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        dw, dh = (self.imgsz - new_w) // 2, (self.imgsz - new_h) // 2
        canvas[dh : dh + new_h, dw : dw + new_w] = resized
        return canvas, scale, dw, dh

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[Detection]:
        """Detect vehicles in a BGR image and return detections in image pixels."""
        if image is None or image.size == 0:
            return []
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        iou = self.iou_threshold if iou_threshold is None else iou_threshold

        canvas, scale, dw, dh = self._letterbox(image)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None, ...])

        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        if output.ndim == 3:
            # Ultralytics exports either [1, num_classes+4, anchors] or [1, anchors, num_classes+4]
            output = output[0]
            if output.shape[0] < output.shape[1]:
                output = output.T
        preds = np.asarray(output, dtype=np.float32)

        boxes = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        mask = confidences >= conf
        boxes, class_ids, confidences = boxes[mask], class_ids[mask], confidences[mask]
        if len(boxes) == 0:
            return []

        xywh = []
        for cx, cy, bw, bh in boxes:
            x1 = (cx - bw / 2 - dw) / scale
            y1 = (cy - bh / 2 - dh) / scale
            x2 = (cx + bw / 2 - dw) / scale
            y2 = (cy + bh / 2 - dh) / scale
            xywh.append([x1, y1, x2 - x1, y2 - y1])
        xywh = np.asarray(xywh, dtype=np.float32)

        keep = cv2.dnn.NMSBoxes(xywh.tolist(), confidences.tolist(), conf, iou)
        if len(keep) == 0:
            return []
        keep = np.asarray(keep).flatten()

        h, w = image.shape[:2]
        detections: list[Detection] = []
        for idx in keep:
            x, y, bw, bh = xywh[idx]
            detections.append(
                Detection(
                    class_id=int(class_ids[idx]),
                    label=self.names.get(int(class_ids[idx]), str(class_ids[idx])),
                    confidence=float(confidences[idx]),
                    x1=float(np.clip(x, 0, w)),
                    y1=float(np.clip(y, 0, h)),
                    x2=float(np.clip(x + bw, 0, w)),
                    y2=float(np.clip(y + bh, 0, h)),
                )
            )
        return sorted(detections, key=lambda d: d.confidence, reverse=True)


def annotate(
    image: np.ndarray,
    detections: list[Detection],
    extra_labels: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of a BGR image."""
    output = image.copy()
    extra_labels = extra_labels or {}
    for det in detections:
        color = CLASS_COLORS.get(det.label, (255, 255, 0))
        x1, y1 = int(round(det.x1)), int(round(det.y1))
        x2, y2 = int(round(det.x2)), int(round(det.y2))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        caption = f"{det.label} {det.confidence:.2f}"
        extra = extra_labels.get(id(det))
        if extra:
            caption += f" | {extra}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(output, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(output, caption, (x1 + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output
