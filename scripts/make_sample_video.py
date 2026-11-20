"""Build a short demo clip from one camera sequence in the downloaded dataset.

This gives the Streamlit video tab a bundled sample so it can be tried without
uploading a file.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/raw/train/images"
OUTPUT = ROOT / "assets" / "sample_traffic.mp4"
FRAME_RE = re.compile(r"img(\d+)")


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    return int(match.group(1)) if match else 0


def main() -> None:
    sequences: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in IMAGES.glob("*.jpg"):
        sequence = path.name.split("_img")[0]
        number = frame_number(path)
        sequences[sequence].setdefault(number, path)

    if not sequences:
        raise SystemExit("No images found. Run scripts/download_dataset.py first.")

    best = max(sequences, key=lambda key: len(sequences[key]))
    frames = [sequences[best][number] for number in sorted(sequences[best])][:80]
    if len(frames) < 10:
        raise SystemExit(f"Sequence {best} has only {len(frames)} frames; need at least 10.")

    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    OUTPUT.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise SystemExit("Could not create sample video (codec unavailable).")
    for path in frames:
        image = cv2.imread(str(path))
        if image is not None:
            writer.write(image)
    writer.release()
    print(f"Wrote {len(frames)} frames from {best} to {OUTPUT}")


if __name__ == "__main__":
    main()
