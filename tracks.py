"""Track catalog for classroom demos.

Each track definition includes explicit wall segments plus a recommended spawn
position and heading for stable demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulator_core import Track, build_rect_track


@dataclass(frozen=True)
class TrackDefinition:
    name: str
    description: str
    spawn: tuple[float, float]
    heading_deg: float
    track: Track


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


TRACK_LIBRARY: dict[str, dict] = {
    "rectangle": {
        "description": "Baseline corridor for first concepts.",
        "spawn": (2.0, 4.0),
        "heading": 0.0,
        "builder": _rectangle,
    },
    "chicane": {
        "description": "Requires steering changes left then right.",
        "spawn": (2.0, 4.0),
        "heading": 0.0,
        "builder": _chicane,
    },
    "slalom": {
        "description": "Alternating obstacles test sensor balance.",
        "spawn": (1.8, 1.5),
        "heading": 20.0,
        "builder": _slalom,
    },
    "pinch_turn": {
        "description": "Narrow pinch then cornering challenge.",
        "spawn": (1.7, 1.7),
        "heading": 10.0,
        "builder": _pinch_turn,
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
    )
