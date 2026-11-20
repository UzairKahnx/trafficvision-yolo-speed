# TrafficVision — Google Colab Training Package

This folder contains everything you need to train the YOLOv8n vehicle detector yourself on Google Colab.

## What is inside

| File | Purpose |
| --- | --- |
| `train_colab.ipynb` | **Recommended.** Upload this notebook to Colab and run all cells. |
| `train_colab.py` | Same training logic as a plain Python script (alternative to the notebook). |
| `dataset.zip` | The labelled dataset in YOLOv8 format (749 images + labels, ~39 MB). |
| `yolov8n.pt` | Pretrained COCO weights (6.5 MB) so no download is needed. Ultralytics will re-download it if missing. |
| `data.yaml` | Dataset config with the **corrected** class order (`0=bus, 1=car, 2=truck, 3=van`). |
| `requirements.txt` | Python packages the training needs. |

## How to train (notebook — easiest)

1. Open <https://colab.research.google.com/> and choose **File → Upload notebook**.
2. Select `train_colab.ipynb` from this folder.
3. In the notebook's file browser (left sidebar), upload **`dataset.zip`** and **`yolov8n.pt`** into the same folder as the notebook (`/content/` by default).
4. (Optional) Enable **GPU**: *Runtime → Change runtime type → T4 GPU*. Training is much faster, but CPU also works.
5. Run all cells (**Runtime → Run all**).
6. Training runs for **25 epochs** at 416 px. At the end the notebook downloads these files for you:
   - `vehicle_detector.onnx` — the trained model used by the Streamlit app
   - `class_names.json` — class id → name mapping
   - `training_summary.json` — mAP50 / mAP50-95 metrics
   - `runs/vehicle_detector/weights/best.pt` — the raw PyTorch checkpoint

## What to send back to me

After training finishes, give me the **metrics** printed in the last cell (or paste `training_summary.json`) and upload `vehicle_detector.onnx` + `class_names.json` into the project's `models/` folder (or just attach them here). I will then finish the app, evaluate it, and push everything to GitHub + Streamlit Cloud.

## Known dataset note

The upstream `data.yaml` listed the class names in the wrong order. We verified against the label distribution (class `1` is 82% of boxes = `car`) and use `0=bus, 1=car, 2=truck, 3=van`. Both `data.yaml` files here already use the corrected order — do **not** change it.
