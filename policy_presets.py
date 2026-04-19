"""Built-in policy presets for guided classroom demos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyPreset:
    name: str
    sensor_angles_deg: list[float]
    weights: list[float]
    bias: float
    note: str


PRESETS: dict[str, PolicyPreset] = {
    "bad": PolicyPreset(
        name="Bad policy",
        sensor_angles_deg=[-60, -30, 0, 30, 60],
        weights=[-0.8, -0.6, 0.0, 0.6, 0.8],
        bias=0.0,
        note="Signs are flipped: danger on left causes left turn (toward crash).",
    ),
    "decent": PolicyPreset(
        name="Decent policy",
        sensor_angles_deg=[-60, -30, 0, 30, 60],
        weights=[0.8, 0.45, 0.0, -0.45, -0.8],
        bias=0.0,
        note="Mostly steers away from walls but can under-react in tight turns.",
    ),
    "good": PolicyPreset(
        name="Good policy",
        sensor_angles_deg=[-60, -30, 0, 30, 60],
        weights=[1.25, 0.8, 0.05, -0.8, -1.25],
        bias=0.0,
        note="Balanced and strong side response for stable wall avoidance.",
    ),
}
