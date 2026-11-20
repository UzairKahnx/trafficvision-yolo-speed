"""Simple vehicle tracking and monocular speed estimation.

The detector gives a box per frame.  A greedy nearest-centre tracker links
boxes into tracks.  Speed is estimated from pixel displacement divided by
elapsed time, with a scale derived from each vehicle's typical physical length
and its apparent box width (a practical single-camera approximation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

from .detector import Detection

# Typical lengths (metres) used to convert pixels -> metres.
TYPICAL_LENGTH_M = {
    "car": 4.5,
    "van": 5.0,
    "bus": 12.0,
    "truck": 8.5,
}
DEFAULT_LENGTH_M = 4.5


@dataclass
class TrackPoint:
    t: float
    cx: float
    cy: float
    width: float
    height: float


@dataclass
class Track:
    track_id: int
    class_id: int
    label: str
    points: list[TrackPoint] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    missed: int = 0

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def speed_kmh(self) -> float:
        return self.speeds[-1] if self.speeds else 0.0

    @property
    def mean_speed_kmh(self) -> float:
        return sum(self.speeds) / len(self.speeds) if self.speeds else 0.0

    @property
    def max_speed_kmh(self) -> float:
        return max(self.speeds) if self.speeds else 0.0

    @property
    def bearing_deg(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[0], self.points[-1]
        return math.degrees(math.atan2(b.cy - a.cy, b.cx - a.cx))


class SpeedTracker:
    def __init__(
        self,
        max_age_s: float = 1.2,
        max_distance_ratio: float = 1.6,
        min_history: int = 3,
        window: int = 10,
    ) -> None:
        self.max_age_s = max_age_s
        self.max_distance_ratio = max_distance_ratio
        self.min_history = min_history
        self.window = window
        self.tracks: list[Track] = []
        self._next_id = 1

    def _predict(self, track: Track, t: float) -> tuple[float, float]:
        if len(track.points) < 2:
            return track.last.cx, track.last.cy
        a, b = track.points[-2], track.points[-1]
        dt = max(b.t - a.t, 1e-3)
        vx, vy = (b.cx - a.cx) / dt, (b.cy - a.cy) / dt
        elapsed = min(t - b.t, self.max_age_s)
        return b.cx + vx * elapsed, b.cy + vy * elapsed

    def _speed(self, track: Track) -> float | None:
        points = track.points[-self.window :]
        if len(points) < self.min_history:
            return None
        span = points[-1].t - points[0].t
        if span < 0.12:
            return None
        dx = points[-1].cx - points[0].cx
        dy = points[-1].cy - points[0].cy
        distance_px = math.hypot(dx, dy)
        if distance_px < 1.0:
            return track.speeds[-1] if track.speeds else 0.0

        length_m = TYPICAL_LENGTH_M.get(track.label, DEFAULT_LENGTH_M)
        width_px = float(median([p.width for p in points]))
        meters_per_px = length_m / max(width_px, 8.0)
        speed_kmh = distance_px * meters_per_px / span * 3.6
        if not math.isfinite(speed_kmh) or speed_kmh < 0 or speed_kmh > 220:
            return None

        # Exponential smoothing keeps the readout stable without heavy lag.
        previous = track.speeds[-1] if track.speeds else speed_kmh
        return 0.6 * speed_kmh + 0.4 * previous

    def update(self, detections: list[Detection], t: float) -> list[Track]:
        """Associate detections to tracks at timestamp ``t`` (seconds)."""
        # Greedy association by normalised centre distance.
        candidates: list[tuple[float, Track, int, Detection]] = []
        for track in self.tracks:
            if t - track.last.t > self.max_age_s:
                continue
            px, py = self._predict(track, t)
            diag = max(math.hypot(track.last.width, track.last.height), 20.0)
            for det_index, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue
                dist = math.hypot(det.center[0] - px, det.center[1] - py)
                cost = dist / diag
                if cost <= self.max_distance_ratio:
                    candidates.append((cost, track, det_index, det))
        candidates.sort(key=lambda item: item[0])

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, track, det_index, det in candidates:
            if id(track) in used_tracks or det_index in used_dets:
                continue
            track.points.append(
                TrackPoint(t, det.center[0], det.center[1], det.width, det.height)
            )
            track.missed = 0
            speed = self._speed(track)
            if speed is not None:
                track.speeds.append(speed)
            used_tracks.add(id(track))
            used_dets.add(det_index)

        for track in self.tracks:
            if id(track) not in used_tracks:
                track.missed += 1

        for index, det in enumerate(detections):
            if index in used_dets:
                continue
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    label=det.label,
                    points=[TrackPoint(t, det.center[0], det.center[1], det.width, det.height)],
                )
            )
            self._next_id += 1

        self.tracks = [
            track for track in self.tracks if t - track.last.t <= self.max_age_s
        ]
        return self.tracks

    def labels_for(self, detections: list[Detection], t: float) -> dict[int, str]:
        """Map detection identity to a display string such as ``ID 3 · 42 km/h``."""
        annotations: dict[int, str] = {}
        for track in self.tracks:
            if abs(track.last.t - t) > 1e-6:
                continue
            for det in detections:
                if det.class_id != track.class_id:
                    continue
                if math.hypot(det.center[0] - track.last.cx, det.center[1] - track.last.cy) > 5:
                    continue
                if track.speed_kmh > 0:
                    annotations[id(det)] = f"ID {track.track_id} | {track.speed_kmh:.0f} km/h"
                else:
                    annotations[id(det)] = f"ID {track.track_id}"
                break
        return annotations
