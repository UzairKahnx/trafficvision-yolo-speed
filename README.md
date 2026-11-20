# 🚗 TrafficVision — YOLO Vehicle Detection & Speed Estimation

**🔴 Live demo:** [https://trafficvision-yolo-speed-o2eyxmetrbbdxydilvzcjb.streamlit.app/](https://trafficvision-yolo-speed-o2eyxmetrbbdxydilvzcjb.streamlit.app/)

A small end-to-end computer-vision product:

1. **Dataset** — Roboflow-exported CCTV traffic images (UA-DETRAC) in YOLOv8 format, 4 classes: `car`, `bus`, `truck`, `van`.
2. **Training** — `yolov8n.pt` fine-tuned with Ultralytics for 25 epochs on CPU.
3. **Evaluation** — mAP on a held-out split plus a prediction contact sheet.
4. **App** — Streamlit UI with image upload, confidence slider, bounding boxes, detection table, a video mode that tracks vehicles and estimates speed in km/h, and a **Test your data** tab for batch-testing uploaded images/ZIPs (with optional YOLO-label accuracy scoring and CSV downloads).
5. **Deployment** — ready for Streamlit Community Cloud (ONNX inference keeps the cloud dependency list small).

---

## 1. Use case and dataset

**Problem:** detect road vehicles in fixed-camera traffic footage and estimate their speed.

**Dataset:** [`thundarstrom/traffic-vehicle-detection`](https://huggingface.co/datasets/thundarstrom/traffic-vehicle-detection) on Hugging Face, originally exported from **Roboflow** (`UA-DETRAC-10K-SAMPLE`, 21 Sep 2025). Images are 640×640 with YOLOv8 normalized labels. The upstream `data.yaml` had the class names in the wrong order; we verified against the label distribution (class `1` is 82% of boxes, i.e. `car`) and use `0=bus, 1=car, 2=truck, 3=van`. The original export contains 23,319 images; this project downloads a reproducible 600-image train / 150-image validation subset so training works on a normal CPU.

`scripts/download_dataset.py` fetches images + labels and writes `data/raw/data.yaml`.

---

## 2. Setup

```bash
# Training environment (CPU PyTorch)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt

# App-only environment (what Streamlit Cloud installs)
pip install -r requirements.txt
```

> The ONNX model is committed in `models/`, so you do **not** need to train to run the app.

---

## 3. Train

```bash
python scripts/download_dataset.py --train 600 --val 150
python scripts/train.py --epochs 25 --imgsz 416 --batch 8
```

`train.py` fine-tunes `yolov8n.pt`, validates, exports `models/vehicle_detector.onnx`, and saves `models/training_summary.json`.

### Training on Google Colab

The `colab_training/` folder is a ready-to-upload package containing the dataset zip, a notebook, pretrained weights, and instructions. Upload `train_colab.ipynb` + `dataset.zip` to Colab and run all cells; it trains for 25 epochs and downloads the ONNX model, class names, and metrics. The files returned from that run are already installed in `models/`.

### Results

| Metric | Value |
| --- | --- |
| mAP50-95 (val) | **0.6141** |
| mAP50 (val) | **0.7875** |
| Epochs | 25 |
| Input size | 416×416 |
| Classes | car, bus, truck, van |

Per-class mAP50-95: bus 0.8955, car 0.5583, truck 0.5167, van 0.4858. Training was run with the `colab_training/` package (see below).

Run `python scripts/evaluate.py` for per-class IoU@0.5 precision/recall and a prediction contact sheet at `assets/predictions_grid.jpg`.

---

## 4. Run the app

```bash
streamlit run streamlit_app.py
```

**Image tab:** upload an image (or use the bundled sample), adjust the confidence slider, see boxes + labels and a table of detections with confidence scores.

**Video tab:** upload a short video. The app samples frames, tracks each vehicle, and estimates speed from pixel displacement using the vehicle's apparent width and a typical physical length as scale. Overspeeding relative to the configurable limit is flagged. The estimate is a monocular approximation — a calibrated camera and a known reference distance give more accurate results.

**Test your data tab:** upload multiple images or one `.zip` archive (up to 200 images) to batch-test the detector. If the archive also contains YOLO `.txt` labels, the tab reports per-class precision / recall / F1 at IoU 0.5. Detection and accuracy results can be downloaded as CSV files, and annotated samples are shown as a gallery.

---

## 5. Deploy to Streamlit Community Cloud

```bash
git init
git add .
git commit -m "TrafficVision: YOLOv8 vehicle detection and speed estimation"
git remote add origin https://github.com/UzairKahnx/trafficvision-yolo-speed.git
git branch -M main
git push -u origin main
```

Then on [share.streamlit.io](https://share.streamlit.io):

- **Repository:** `UzairKahnx/trafficvision-yolo-speed`
- **Branch:** `main`
- **Main file:** `streamlit_app.py`
- **Dependencies:** `requirements.txt` (auto-detected)

The committed `models/vehicle_detector.onnx` (~12.7 MB) is loaded at startup; no secrets are required.

---

## Project layout

```
.
├── streamlit_app.py          # Streamlit entry point
├── app/
│   ├── detector.py           # ONNX YOLO inference + drawing
│   └── tracker.py            # greedy tracker + speed estimation
├── scripts/
│   ├── download_dataset.py   # HF/Roboflow dataset subset downloader
│   ├── train.py              # Ultralytics fine-tuning + ONNX export
│   └── evaluate.py           # precision/recall + prediction grid
├── models/                   # exported ONNX model + metadata
├── assets/                   # sample image + prediction grid
├── requirements.txt          # cloud/app dependencies
└── requirements-train.txt    # training-only dependencies
```
