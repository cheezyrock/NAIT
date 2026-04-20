"""Track catalog for classroom demos.

Each track definition includes explicit wall segments plus a recommended spawn
position and heading for stable demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from simulator_core import Track, build_rect_track


@dataclass(frozen=True)
class TrackDefinition:
    name: str
    description: str
    spawn: tuple[float, float]
    heading_deg: float
    track: Track
    start_line: tuple[tuple[float, float], tuple[float, float]]


def _rectangle() -> Track:
    return build_rect_track(width=14.0, height=8.0)


def _chicane() -> Track:
    segments = [
        ((0.0, 0.0), (16.0, 0.0)),
        ((16.0, 0.0), (16.0, 8.0)),
        ((16.0, 8.0), (0.0, 8.0)),
        ((0.0, 8.0), (0.0, 0.0)),
        ((6.0, 0.0), (6.0, 5.0)),
        ((10.0, 3.0), (10.0, 8.0)),
    ]
    return Track.from_segments(segments)


def _slalom() -> Track:
    segments = [
        ((0.0, 0.0), (18.0, 0.0)),
        ((18.0, 0.0), (18.0, 8.0)),
        ((18.0, 8.0), (0.0, 8.0)),
        ((0.0, 8.0), (0.0, 0.0)),
        ((4.0, 2.5), (4.0, 8.0)),
        ((8.0, 0.0), (8.0, 5.5)),
        ((12.0, 2.5), (12.0, 8.0)),
    ]
    return Track.from_segments(segments)


def _pinch_turn() -> Track:
    segments = [
        ((0.0, 0.0), (15.0, 0.0)),
        ((15.0, 0.0), (15.0, 9.0)),
        ((15.0, 9.0), (0.0, 9.0)),
        ((0.0, 9.0), (0.0, 0.0)),
        ((5.0, 0.0), (5.0, 6.0)),
        ((5.0, 6.0), (9.0, 6.0)),
        ((9.0, 3.0), (9.0, 9.0)),
    ]
    return Track.from_segments(segments)


def _nascar_oval() -> Track:
    """Asymmetric oval to feel closer to a stock-car style speedway."""
    outer_cx, outer_cy = 10.0, 5.0
    inner_cx, inner_cy = 9.7, 5.0
    points = 56

    def outer_radius(angle: float) -> tuple[float, float]:
        x_scale = 8.5 if math.cos(angle) < 0 else 7.3
        y_scale = 4.4 if math.sin(angle) >= 0 else 4.0
        return x_scale, y_scale

    def inner_radius(angle: float) -> tuple[float, float]:
        x_scale = 5.5 if math.cos(angle) < 0 else 4.5
        y_scale = 2.6 if math.sin(angle) >= 0 else 2.3
        return x_scale, y_scale

    outer_points = []
    inner_points = []
    for idx in range(points):
        angle = (idx / points) * math.tau
        out_rx, out_ry = outer_radius(angle)
        in_rx, in_ry = inner_radius(angle)
        outer_points.append((outer_cx + math.cos(angle) * out_rx, outer_cy + math.sin(angle) * out_ry))
        inner_points.append((inner_cx + math.cos(angle) * in_rx, inner_cy + math.sin(angle) * in_ry))

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for idx in range(points):
        segments.append((outer_points[idx], outer_points[(idx + 1) % points]))
        segments.append((inner_points[idx], inner_points[(idx + 1) % points]))
    return Track.from_segments(segments)


def _figure8() -> Track:
    """Road-like looping figure-eight with explicit lane walls."""
    points = 72
    cx, cy = 9.0, 5.0
    outer_points = []
    inner_points = []
    for idx in range(points):
        t = (idx / points) * math.tau
        x = cx + math.sin(t) * 6.2
        y = cy + math.sin(t * 2.0) * 2.8
        dx = 6.2 * math.cos(t)
        dy = 5.6 * math.cos(t * 2.0)
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        width = 1.15
        outer_points.append((x + nx * width, y + ny * width))
        inner_points.append((x - nx * width, y - ny * width))

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for idx in range(points):
        segments.append((outer_points[idx], outer_points[(idx + 1) % points]))
        segments.append((inner_points[idx], inner_points[(idx + 1) % points]))
    return Track.from_segments(segments)


TRACK_LIBRARY: dict[str, dict] = {
    "nascar_oval": {
        "description": "Default asymmetrical speedway with a marked start/finish line.",
        "spawn": (2.8, 4.8),
        "heading": 0.0,
        "builder": _nascar_oval,
        "start_line": ((2.3, 3.1), (2.3, 6.6)),
    },
    "rectangle": {
        "description": "Baseline corridor for first concepts.",
        "spawn": (2.0, 4.0),
        "heading": 0.0,
        "builder": _rectangle,
        "start_line": ((2.0, 2.0), (2.0, 6.0)),
    },
    "chicane": {
        "description": "Requires steering changes left then right.",
        "spawn": (2.0, 4.0),
        "heading": 0.0,
        "builder": _chicane,
        "start_line": ((2.0, 2.0), (2.0, 6.0)),
    },
    "slalom": {
        "description": "Alternating obstacles test sensor balance.",
        "spawn": (1.8, 1.5),
        "heading": 20.0,
        "builder": _slalom,
        "start_line": ((1.8, 0.7), (1.8, 2.6)),
    },
    "pinch_turn": {
        "description": "Narrow pinch then cornering challenge.",
        "spawn": (1.7, 1.7),
        "heading": 10.0,
        "builder": _pinch_turn,
        "start_line": ((1.7, 0.9), (1.7, 2.8)),
    },
    "figure8": {
        "description": "Looping figure-eight road for more advanced control demos.",
        "spawn": (2.8, 5.0),
        "heading": 15.0,
        "builder": _figure8,
        "start_line": ((2.4, 4.2), (2.4, 5.8)),
    },
}


def track_names() -> list[str]:
    return list(TRACK_LIBRARY.keys())


def build_track_definition(name: str) -> TrackDefinition:
    if name not in TRACK_LIBRARY:
        raise ValueError(f"Unknown track name: {name}")
    data = TRACK_LIBRARY[name]
    return TrackDefinition(
        name=name,
        description=data["description"],
        spawn=data["spawn"],
        heading_deg=data["heading"],
        track=data["builder"](),
        start_line=data["start_line"],
    )
