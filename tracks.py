"""Track catalog for the teaching app.

Tracks are built as explicit boundary line segments to stay compatible with
`simulator_core.Track`.
"""

from __future__ import annotations

from simulator_core import Track, build_rect_track


def build_chicane_track() -> Track:
    """A simple custom corridor with a bend/chicane-like center shift."""
    segments = [
        # Outer-ish walls
        ((0.0, 0.0), (14.0, 0.0)),
        ((14.0, 0.0), (14.0, 8.0)),
        ((14.0, 8.0), (0.0, 8.0)),
        ((0.0, 8.0), (0.0, 0.0)),
        # Interior blocking wall to force steering change
        ((6.5, 0.0), (6.5, 5.0)),
        ((8.0, 3.0), (8.0, 8.0)),
    ]
    return Track.from_segments(segments)


TRACK_BUILDERS = {
    "rectangle": lambda: build_rect_track(width=14.0, height=8.0),
    "chicane": build_chicane_track,
}


def build_track(name: str) -> Track:
    if name not in TRACK_BUILDERS:
        raise ValueError(f"Unknown track name: {name}")
    return TRACK_BUILDERS[name]()
