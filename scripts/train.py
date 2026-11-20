"""Fine-tune YOLOv8n on the Roboflow-exported CCTV vehicle dataset.

Usage (from the project root, with the training venv active):
    python scripts/train.py --epochs 25 --imgsz 416 --batch 8

The script trains, validates, and exports the best checkpoint to ONNX so the
Streamlit app can run without PyTorch.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/data.yaml")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset config not found: {args.data}. Run scripts/download_dataset.py first.")

    model = YOLO(args.weights)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,  # required on Windows / safe everywhere
        patience=0,  # always complete the requested number of epochs
        project=str(ROOT / "runs"),
        name="vehicle_detector",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    print("\n--- Training complete ---")

    metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
    )
    summary = {
        "map50_95": round(float(metrics.box.map), 4),
        "map50": round(float(metrics.box.map50), 4),
        "per_class_map50_95": [round(float(x), 4) for x in metrics.box.maps],
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "dataset": "Roboflow UA-DETRAC CCTV vehicles (YOLOv8 export, 4 classes)",
    }
    print("Validation:", json.dumps(summary, indent=2))

    exported = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, opset=17)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    shutil.copy2(exported, model_dir / "vehicle_detector.onnx")
    (model_dir / "class_names.json").write_text(
        json.dumps({str(i): name for i, name in model.names.items()}, indent=2),
        encoding="utf-8",
    )
    (model_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Exported ONNX model to {model_dir / 'vehicle_detector.onnx'}")
    print(f"Training results: {getattr(results, 'save_dir', ROOT / 'runs')}")


if __name__ == "__main__":
    main()
