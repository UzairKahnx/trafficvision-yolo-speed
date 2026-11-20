"""Train YOLOv8n on the TrafficVision vehicle dataset (Colab-friendly).

Run from this folder after uploading `dataset.zip`:

    pip install -r requirements.txt
    python train_colab.py

The script:
  1. extracts `dataset.zip` (if not already extracted),
  2. writes a `data.yaml` with an absolute dataset path,
  3. fine-tunes `yolov8n.pt` for 25 epochs (GPU if available, else CPU),
  4. validates and exports `vehicle_detector.onnx`, `class_names.json`,
     and `training_summary.json` into an `output/` folder.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"


def prepare_dataset() -> Path:
    """Extract the zip if needed and return the absolute data.yaml path."""
    if not (DATASET / "train/images").exists():
        zpath = ROOT / "dataset.zip"
        if not zpath.exists():
            raise SystemExit(f"Missing {zpath}. Upload dataset.zip next to this script.")
        print(f"Extracting {zpath.name} ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(ROOT)

    n_train = len(list((DATASET / "train/images").glob("*.jpg")))
    n_val = len(list((DATASET / "val/images").glob("*.jpg")))
    print(f"Dataset: {n_train} train images, {n_val} val images")

    data_yaml = ROOT / "data.yaml"
    data_yaml.write_text(
        f"path: {DATASET.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 4\n"
        "names: [bus, car, truck, van]\n",
        encoding="utf-8",
    )
    return data_yaml


def main() -> None:
    try:
        import torch

        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    device = 0 if use_gpu else "cpu"
    batch = 16 if use_gpu else 8
    print(f"Training on device={device} with batch={batch}")

    data_yaml = prepare_dataset()
    weights = ROOT / "yolov8n.pt"
    if not weights.exists():
        weights = Path("yolov8n.pt")  # Ultralytics downloads it automatically

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=25,
        imgsz=416,
        batch=batch,
        device=device,
        workers=0,
        patience=0,
        project=str(ROOT / "runs"),
        name="vehicle_detector",
        exist_ok=True,
    )
    print("\n--- Training complete ---")

    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=416,
        batch=batch,
        device=device,
        workers=0,
    )
    summary = {
        "map50_95": round(float(metrics.box.map), 4),
        "map50": round(float(metrics.box.map50), 4),
        "per_class_map50_95": [round(float(x), 4) for x in metrics.box.maps],
        "epochs": 25,
        "imgsz": 416,
        "dataset": "Roboflow UA-DETRAC CCTV vehicles (YOLOv8 export, 4 classes)",
    }
    print("Validation:", json.dumps(summary, indent=2))

    onnx_file = model.export(format="onnx", imgsz=416, dynamic=False, opset=17)
    OUTPUT.mkdir(exist_ok=True)
    shutil.copy2(onnx_file, OUTPUT / "vehicle_detector.onnx")
    (OUTPUT / "class_names.json").write_text(
        json.dumps({str(i): name for i, name in model.names.items()}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checkpoints = sorted(ROOT.rglob("best.pt"))
    if checkpoints:
        shutil.copy2(checkpoints[-1], OUTPUT / "best.pt")
    else:
        print("Warning: best.pt not found; ONNX export is still complete.")
    print(f"\nSaved outputs to {OUTPUT}:")
    for f in sorted(OUTPUT.iterdir()):
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
