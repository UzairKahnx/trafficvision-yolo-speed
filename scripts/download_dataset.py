"""Download a small, reproducible subset of a Roboflow-exported YOLO dataset.

Source: https://huggingface.co/datasets/thundarstrom/traffic-vehicle-detection
Original export: Roboflow Universe (UA-DETRAC CCTV vehicles), YOLOv8 format.
Classes: car, bus, truck, van.

Only a subset is downloaded so that training is feasible on a CPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = "thundarstrom/traffic-vehicle-detection"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
TIMEOUT = 60


def list_files(split: str, folder: str, limit: int) -> list[str]:
    url = f"{API}/{split}/{folder}"
    response = requests.get(url, params={"recursive": "false", "limit": limit}, timeout=TIMEOUT)
    response.raise_for_status()
    paths = [
        item["path"]
        for item in response.json()
        if item.get("type") == "file" and item["path"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return paths[:limit]


def label_path(image_path: str) -> str:
    return image_path.rsplit(".", 1)[0].rsplit("/images/", 1)[0] + "/labels/" + Path(image_path).stem + ".txt"


def fetch_one(rel_path: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return "cached"
    response = requests.get(f"{RESOLVE}/{rel_path}", params={"download": "true"}, timeout=TIMEOUT)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=600, help="number of training images")
    parser.add_argument("--val", type=int, default=150, help="number of validation images")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    root: Path = args.output
    jobs: list[tuple[str, Path]] = []

    for split, count in (("train", args.train), ("test", args.val)):
        images = list_files(split, "images", count)
        print(f"[{split}] found {len(images)} images; requesting {count}")
        if len(images) < count:
            print(f"WARNING: only {len(images)} images available for {split}", file=sys.stderr)
        for image_rel in images[:count]:
            name = Path(image_rel).name
            target_split = "val" if split == "test" else "train"
            jobs.append((image_rel, root / target_split / "images" / name))
            jobs.append((label_path(image_rel), root / target_split / "labels" / (Path(name).stem + ".txt")))

    print(f"Downloading {len(jobs)} files with {args.workers} workers ...")
    downloaded = cached = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, rel, dst): rel for rel, dst in jobs}
        for future in as_completed(futures):
            rel = futures[future]
            try:
                result = future.result()
                downloaded += result == "downloaded"
                cached += result == "cached"
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append(f"{rel}: {exc}")
    print(f"Done: {downloaded} downloaded, {cached} cached, {len(failures)} failed")
    for failure in failures[:10]:
        print("  FAILED", failure, file=sys.stderr)

    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: val/images\n"
        "nc: 4\n"
        "# Corrected order: id 1 is car (82% of boxes), id 0 is bus.\n"
        "names: [bus, car, truck, van]\n",
        encoding="utf-8",
    )

    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    sample = next((root / "val" / "images").glob("*.jpg"), None)
    if sample:
        shutil.copy2(sample, assets / "sample_traffic.jpg")
        print(f"Sample image saved to {assets / 'sample_traffic.jpg'}")

    print(f"Dataset ready in {root.resolve()}")


if __name__ == "__main__":
    main()
