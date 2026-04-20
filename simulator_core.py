"""Core simulation primitives for a teachable neural-network driving demo.

This module intentionally avoids UI/network concerns so it can be reused for
CLI demos, notebooks, a web front-end, or a classroom API service.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


Vec2 = tuple[float, float]
Segment = tuple[Vec2, Vec2]


@dataclass
class LinearPolicy:
    """Single-layer policy with no hidden nodes.

    steering = tanh(bias + sum(sensor[i] * weight[i]))
    """

    sensor_angles_deg: list[float]
    weights: list[float]
    bias: float = 0.0

    def __post_init__(self) -> None:
        if len(self.sensor_angles_deg) == 0:
            raise ValueError("At least one sensor angle is required")
        if len(self.sensor_angles_deg) != len(self.weights):
            raise ValueError("sensor_angles_deg and weights must be the same length")

    @property
    def sensor_count(self) -> int:
        return len(self.sensor_angles_deg)

    def forward(self, sensor_values: Iterable[float]) -> float:
        sensor_values_list = list(sensor_values)
        if len(sensor_values_list) != self.sensor_count:
            raise ValueError("sensor_values length must match sensor count")

        steering_raw = self.bias
        for value, weight in zip(sensor_values_list, self.weights, strict=True):
            steering_raw += value * weight

        return math.tanh(steering_raw)

    def to_dict(self) -> dict:
        return {
            "sensorAnglesDeg": self.sensor_angles_deg,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinearPolicy":
        return cls(
            sensor_angles_deg=list(data["sensorAnglesDeg"]),
            weights=list(data["weights"]),
            bias=float(data.get("bias", 0.0)),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "LinearPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class NeuralPolicy:
    """Tiny teachable neural-network policy with optional output heads."""

    sensor_angles_deg: list[float]
    hidden_weights: list[list[float]]
    hidden_biases: list[float]
    steering_weights: list[float]
    steering_bias: float = 0.0
    accel_weights: list[float] | None = None
    accel_bias: float = 0.0
    brake_weights: list[float] | None = None
    brake_bias: float = 0.0

    def __post_init__(self) -> None:
        sensor_count = len(self.sensor_angles_deg)
        if sensor_count == 0:
            raise ValueError("At least one sensor angle is required")
        if len(self.hidden_weights) == 0:
            raise ValueError("At least one hidden node is required")
        if len(self.hidden_weights) != len(self.hidden_biases):
            raise ValueError("hidden_weights and hidden_biases must be same length")
        for row in self.hidden_weights:
            if len(row) != sensor_count:
                raise ValueError("Each hidden layer row must match sensor count")
        hidden_count = len(self.hidden_weights)
        if len(self.steering_weights) != hidden_count:
            raise ValueError("steering_weights length must match hidden size")
        if self.accel_weights is not None and len(self.accel_weights) != hidden_count:
            raise ValueError("accel_weights length must match hidden size")
        if self.brake_weights is not None and len(self.brake_weights) != hidden_count:
            raise ValueError("brake_weights length must match hidden size")

    @property
    def sensor_count(self) -> int:
        return len(self.sensor_angles_deg)

    @property
    def hidden_count(self) -> int:
        return len(self.hidden_weights)

    def forward_with_activations(self, sensor_values: Iterable[float]) -> tuple[dict[str, list[float]], dict[str, float]]:
        sensor_values_list = list(sensor_values)
        if len(sensor_values_list) != self.sensor_count:
            raise ValueError("sensor_values length must match sensor count")

        hidden: list[float] = []
        for row, bias in zip(self.hidden_weights, self.hidden_biases, strict=True):
            total = bias + sum(value * weight for value, weight in zip(sensor_values_list, row, strict=True))
            hidden.append(math.tanh(total))

        steering_raw = self.steering_bias + sum(value * weight for value, weight in zip(hidden, self.steering_weights, strict=True))
        steering = math.tanh(steering_raw)

        accel = 0.0
        if self.accel_weights is not None:
            accel_raw = self.accel_bias + sum(value * weight for value, weight in zip(hidden, self.accel_weights, strict=True))
            accel = (math.tanh(accel_raw) + 1.0) / 2.0

        brake = 0.0
        if self.brake_weights is not None:
            brake_raw = self.brake_bias + sum(value * weight for value, weight in zip(hidden, self.brake_weights, strict=True))
            brake = (math.tanh(brake_raw) + 1.0) / 2.0

        activations = {
            "sensor": sensor_values_list,
            "hidden": hidden,
        }
        outputs = {"steering": steering, "accel": accel, "brake": brake}
        return activations, outputs

    def forward(self, sensor_values: Iterable[float]) -> dict[str, float]:
        _, outputs = self.forward_with_activations(sensor_values)
        return outputs


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _cross(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def distance_to_segment(point: Vec2, segment: Segment) -> float:
    """Shortest distance from point to line segment."""
    (x1, y1), (x2, y2) = segment
    px, py = point

    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = clamp(t, 0.0, 1.0)

    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def ray_segment_intersection_distance(ray_origin: Vec2, ray_dir: Vec2, segment: Segment) -> float | None:
    """Distance from ray origin to segment intersection, or None if no hit."""
    p = ray_origin
    r = ray_dir
    q = segment[0]
    s = _sub(segment[1], segment[0])

    denom = _cross(r, s)
    if abs(denom) < 1e-9:
        return None

    q_minus_p = _sub(q, p)
    t = _cross(q_minus_p, s) / denom
    u = _cross(q_minus_p, r) / denom

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


@dataclass
class Track:
    """Track represented explicitly as boundary line segments."""

    wall_segments: list[Segment]

    @classmethod
    def from_segments(cls, segments: Iterable[Segment]) -> "Track":
        return cls(list(segments))


@dataclass
class SensorArray:
    sensor_angles_deg: list[float]
    max_range: float

    def read_with_distances(self, position: Vec2, heading_deg: float, track: Track) -> list[tuple[float, float | None]]:
        """Return (normalized_reading, hit_distance) for each sensor."""
        scan: list[tuple[float, float | None]] = []
        for rel_angle in self.sensor_angles_deg:
            angle = math.radians(heading_deg + rel_angle)
            direction = (math.cos(angle), math.sin(angle))

            hit_distance: float | None = None
            for segment in track.wall_segments:
                d = ray_segment_intersection_distance(position, direction, segment)
                if d is None:
                    continue
                if hit_distance is None or d < hit_distance:
                    hit_distance = d

            if hit_distance is None:
                scan.append((0.0, None))
            elif hit_distance > self.max_range:
                scan.append((0.0, hit_distance))
            else:
                scan.append((clamp(1.0 - (hit_distance / self.max_range), 0.0, 1.0), hit_distance))

        return scan

    def read(self, position: Vec2, heading_deg: float, track: Track) -> list[float]:
        return [reading for reading, _ in self.read_with_distances(position, heading_deg, track)]


@dataclass
class CarAgent:
    position: Vec2
    heading_deg: float
    speed: float
    max_turn_rate_deg: float
    collision_radius: float = 0.1
    alive: bool = True
    time_alive: float = 0.0
    distance_traveled: float = 0.0
    path_trace: list[Vec2] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path_trace.append(self.position)

    def step(self, steering: float, dt: float) -> None:
        if not self.alive:
            return

        steering = clamp(steering, -1.0, 1.0)
        self.heading_deg += steering * self.max_turn_rate_deg * dt

        angle = math.radians(self.heading_deg)
        dx = math.cos(angle) * self.speed * dt
        dy = math.sin(angle) * self.speed * dt

        new_pos = (self.position[0] + dx, self.position[1] + dy)
        self.distance_traveled += math.hypot(new_pos[0] - self.position[0], new_pos[1] - self.position[1])
        self.position = new_pos
        self.path_trace.append(self.position)
        self.time_alive += dt


@dataclass
class SimulationResult:
    survived_seconds: float
    distance_traveled: float
    cause_of_death: Literal["collision", "timeout", "alive"]
    path_trace: list[Vec2]


@dataclass
class Simulator:
    track: Track
    policy: LinearPolicy | NeuralPolicy
    agent: CarAgent
    sensor_array: SensorArray

    def run(self, max_steps: int, dt: float) -> SimulationResult:
        cause: Literal["collision", "timeout", "alive"] = "alive"

        for _ in range(max_steps):
            if not self.agent.alive:
                cause = "collision"
                break

            sensor_values = self.sensor_array.read(
                position=self.agent.position,
                heading_deg=self.agent.heading_deg,
                track=self.track,
            )
            policy_output = self.policy.forward(sensor_values)
            steering = policy_output["steering"] if isinstance(policy_output, dict) else policy_output
            self.agent.step(steering=steering, dt=dt)

            if self._is_colliding(self.agent.position):
                self.agent.alive = False
                cause = "collision"
                break
        else:
            cause = "timeout"

        return SimulationResult(
            survived_seconds=self.agent.time_alive,
            distance_traveled=self.agent.distance_traveled,
            cause_of_death=cause,
            path_trace=list(self.agent.path_trace),
        )

    def _is_colliding(self, position: Vec2) -> bool:
        return any(
            distance_to_segment(position, segment) <= self.agent.collision_radius
            for segment in self.track.wall_segments
        )


def build_rect_track(width: float = 10.0, height: float = 6.0) -> Track:
    """Simple default track: a rectangular corridor boundary."""
    segments: list[Segment] = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    return Track.from_segments(segments)
