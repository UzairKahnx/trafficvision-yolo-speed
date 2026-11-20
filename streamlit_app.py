"""TrafficVision — YOLO vehicle detection and speed estimation.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from app.detector import YoloOnnxDetector, annotate
from app.tracker import SpeedTracker

ROOT = Path(__file__).resolve().parent
SAMPLE_IMAGE = ROOT / "assets" / "sample_traffic.jpg"
SAMPLE_VIDEO = ROOT / "assets" / "sample_traffic.mp4"

st.set_page_config(page_title="TrafficVision", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO vehicle detector…")
def load_detector() -> YoloOnnxDetector:
    return YoloOnnxDetector()


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_gif(frames_rgb: list[np.ndarray], duration_ms: int) -> io.BytesIO:
    """Build an animated preview from at most 40 annotated frames."""
    stride = max(1, len(frames_rgb) // 40)
    images = [Image.fromarray(frame) for frame in frames_rgb[::stride]]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(duration_ms, 20),
        loop=0,
    )
    buffer.seek(0)
    return buffer


st.title("🚗 TrafficVision — Vehicle Detection & Speed Estimation")
st.markdown(
    "Fine-tuned **YOLOv8n** (Roboflow / UA-DETRAC CCTV vehicles) detects "
    "`car`, `bus`, `truck` and `van`. Upload an image for labelled detections, "
    "or a short video to track vehicles and estimate their speed."
)

with st.sidebar:
    st.header("⚙️ Settings")
    try:
        detector = load_detector()
    except Exception as exc:  # noqa: BLE001 - surface setup problems in the UI
        st.error(f"Could not load model: {exc}")
        st.info("Run `python scripts/train.py` or add `models/vehicle_detector.onnx`.")
        st.stop()

    st.success("Model ready")
    st.caption(
        f"Input {detector.imgsz}×{detector.imgsz} · "
        f"classes: {', '.join(detector.names.values())}"
    )
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
    speed_limit = st.number_input("Speed limit (km/h)", 10, 200, 60, 5)
    st.caption(
        "Speed uses each vehicle's apparent width and typical length "
        "(car 4.5 m, van 5 m, bus 12 m, truck 8.5 m) as a scale reference."
    )

image_tab, video_tab = st.tabs(["📷 Image detection", "🎬 Video speed estimation"])

# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------
with image_tab:
    st.subheader("Detect vehicles in a single image")
    st.caption(
        "A single still image cannot reveal true speed — use the **Video** tab "
        "for tracking-based km/h estimates."
    )
    uploaded_image = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="image_upload"
    )
    use_sample = st.button("Try the bundled sample image", use_container_width=True)

    image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
    if use_sample and SAMPLE_IMAGE.exists():
        image_bytes = SAMPLE_IMAGE.read_bytes()
        st.caption("Using bundled sample image")

    if image_bytes:
        image_bgr = decode_image(image_bytes)
        if image_bgr is None:
            st.error("Could not decode that image. Try a JPG or PNG file.")
        else:
            with st.spinner("Running YOLO inference…"):
                detections = detector.detect(image_bgr, confidence, iou_threshold)
            annotated = annotate(image_bgr, detections)

            left, right = st.columns([3, 2])
            with left:
                st.image(to_rgb(annotated), use_container_width=True, caption="Detected vehicles")
            with right:
                st.metric("Vehicles detected", len(detections))
                if detections:
                    rows = [
                        {
                            "Class": det.label,
                            "Confidence": f"{det.confidence:.3f}",
                            "Box (x1,y1,x2,y2)": (
                                f"{det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f}"
                            ),
                        }
                        for det in detections
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.bar_chart(pd.DataFrame(rows)["Class"].value_counts())
                else:
                    st.info("No vehicles above the confidence threshold. Try lowering it.")
    else:
        st.info("Upload an image or use the bundled sample to see detections.")

# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------
with video_tab:
    st.subheader("Track vehicles and estimate speed")
    st.caption(
        "A fixed camera works best. The tracker links detections across frames "
        "and converts pixel motion to km/h using the vehicle's apparent size."
    )
    uploaded_video = st.file_uploader(
        "Choose a short video", type=["mp4", "mov", "avi", "mkv", "webm"], key="video_upload"
    )
    use_sample_video = st.button("Try the bundled sample video", use_container_width=True)
    frame_step = st.slider("Process every Nth frame", 1, 10, 3)
    max_frames = st.slider("Maximum frames to process", 30, 300, 90, 30)

    video_bytes = uploaded_video.getvalue() if uploaded_video is not None else None
    video_name = uploaded_video.name if uploaded_video is not None else "sample_traffic.mp4"
    if use_sample_video and SAMPLE_VIDEO.exists():
        video_bytes = SAMPLE_VIDEO.read_bytes()
        video_name = "sample_traffic.mp4"
        st.caption("Using bundled sample video")

    if video_bytes is not None:
        suffix = Path(video_name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(video_bytes)
            temp_path = Path(temp.name)

        capture = cv2.VideoCapture(str(temp_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled: list[tuple[float, np.ndarray]] = []
        frame_index = 0
        while len(sampled) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                sampled.append((frame_index / fps, frame))
            frame_index += 1
        capture.release()
        temp_path.unlink(missing_ok=True)

        if not sampled:
            st.error("Could not read any frames from that video.")
        else:
            st.caption(f"Video: {total_frames} frames @ {fps:.1f} fps · analysing {len(sampled)} frames")
            tracker = SpeedTracker()
            annotated_frames: list[np.ndarray] = []
            progress = st.progress(0.0)
            with st.spinner("Detecting, tracking and measuring speed…"):
                for position, (timestamp, frame) in enumerate(sampled):
                    detections = detector.detect(frame, confidence, iou_threshold)
                    tracks = tracker.update(detections, timestamp)
                    extra = tracker.labels_for(detections, timestamp)
                    annotated_frames.append(to_rgb(annotate(frame, detections, extra)))
                    progress.progress((position + 1) / len(sampled))

            preview_col, stats_col = st.columns([3, 2])
            with preview_col:
                duration_ms = int(1000 * frame_step / fps)
                st.image(make_gif(annotated_frames, duration_ms), use_container_width=True)
                picked = st.slider("Inspect a processed frame", 0, len(annotated_frames) - 1, 0)
                st.image(annotated_frames[picked], use_container_width=True)

            with stats_col:
                st.metric("Tracks found", len(tracker.tracks))
                finished = [
                    track
                    for track in tracker.tracks
                    if len(track.points) >= 3 and track.mean_speed_kmh > 0
                ]
                if finished:
                    rows = []
                    for track in finished:
                        overspeed = track.max_speed_kmh > speed_limit
                        rows.append(
                            {
                                "ID": track.track_id,
                                "Class": track.label,
                                "Avg km/h": round(track.mean_speed_kmh, 1),
                                "Max km/h": round(track.max_speed_kmh, 1),
                                "Frames": len(track.points),
                                "Speeding": "⚠️" if overspeed else "✅",
                            }
                        )
                    speed_df = pd.DataFrame(rows).sort_values("Max km/h", ascending=False)
                    st.dataframe(speed_df, use_container_width=True, hide_index=True)
                    fastest = speed_df.iloc[0]
                    st.metric("Fastest track", f"{fastest['Max km/h']:.0f} km/h")
                    if (speed_df["Speeding"] == "⚠️").any():
                        st.warning(f"At least one vehicle exceeded {speed_limit} km/h.")
                else:
                    st.info("No track yet has enough frames for a reliable speed estimate.")
    else:
        st.info("Upload a short traffic video to see tracking and speed estimates.")
