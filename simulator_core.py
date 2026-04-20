"""Core simulation primitives for a teachable neural-network driving demo.

This module intentionally avoids UI/network concerns so it can be reused for
CLI demos, notebooks, a web front-end, or a classroom API service.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


Vec2 = tuple[float, float]
Segment = tuple[Vec2, Vec2]


@dataclass
class PolicyStep:
    raw_outputs: dict[str, float]
    actions: dict[str, float]
    layer_activations: list[list[float]]


@dataclass
class NeuralPolicy:
    """Configurable fully-connected network policy.

    Layer configuration examples:
    - [inputs, outputs]
    - [inputs, hidden, outputs]
    - [inputs, hidden1, hidden2, outputs]
    """

    sensor_angles_deg: list[float]
    action_names: list[str]
    layer_sizes: list[int]
    weights: list[list[list[float]]]
    biases: list[list[float]]
    hidden_activation: Literal["tanh", "relu"] = "tanh"
    output_activation: Literal["tanh", "sigmoid", "linear"] = "tanh"

    def __post_init__(self) -> None:
        if len(self.sensor_angles_deg) == 0:
            raise ValueError("At least one sensor angle is required")
        if len(self.action_names) == 0:
            raise ValueError("At least one action output is required")
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output")
        if self.layer_sizes[0] != len(self.sensor_angles_deg):
            raise ValueError("Input layer must match sensor count")
        if self.layer_sizes[-1] != len(self.action_names):
            raise ValueError("Output layer must match action count")
        if len(self.weights) != len(self.layer_sizes) - 1:
            raise ValueError("weights must have one matrix per layer transition")
        if len(self.biases) != len(self.layer_sizes) - 1:
            raise ValueError("biases must have one vector per layer transition")

        for layer_index, (matrix, bias_vec) in enumerate(zip(self.weights, self.biases, strict=True)):
            expected_out = self.layer_sizes[layer_index + 1]
            expected_in = self.layer_sizes[layer_index]
            if len(matrix) != expected_out:
                raise ValueError(f"weights[{layer_index}] row count mismatch")
            if len(bias_vec) != expected_out:
                raise ValueError(f"biases[{layer_index}] size mismatch")
            for row in matrix:
                if len(row) != expected_in:
                    raise ValueError(f"weights[{layer_index}] column count mismatch")

    @property
    def sensor_count(self) -> int:
        return len(self.sensor_angles_deg)

    def clone(self) -> "NeuralPolicy":
        return NeuralPolicy(
            sensor_angles_deg=list(self.sensor_angles_deg),
            action_names=list(self.action_names),
            layer_sizes=list(self.layer_sizes),
            weights=[[list(row) for row in layer] for layer in self.weights],
            biases=[list(layer_bias) for layer_bias in self.biases],
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
        )

    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        for layer in self.weights:
            for row in layer:
                for idx, value in enumerate(row):
                    if random.random() < mutation_rate:
                        row[idx] = value + random.uniform(-mutation_strength, mutation_strength)
        for bias_layer in self.biases:
            for idx, value in enumerate(bias_layer):
                if random.random() < mutation_rate:
                    bias_layer[idx] = value + random.uniform(-mutation_strength, mutation_strength)

    def forward(self, sensor_values: Iterable[float]) -> PolicyStep:
        input_values = list(sensor_values)
        if len(input_values) != self.sensor_count:
            raise ValueError("sensor_values length must match sensor count")

        activations = input_values
        layer_activations: list[list[float]] = [list(activations)]

        for layer_index, (matrix, bias_vec) in enumerate(zip(self.weights, self.biases, strict=True)):
            is_output_layer = layer_index == len(self.weights) - 1
            next_values: list[float] = []
            for neuron_weights, bias in zip(matrix, bias_vec, strict=True):
                total = bias
                for value, weight in zip(activations, neuron_weights, strict=True):
                    total += value * weight
                if is_output_layer:
                    next_values.append(self._activate_output(total))
                else:
                    next_values.append(self._activate_hidden(total))
            activations = next_values
            layer_activations.append(list(activations))

        raw_outputs = {
            action_name: activations[idx]
            for idx, action_name in enumerate(self.action_names)
        }

        actions = {
            "steering": clamp(raw_outputs.get("steering", 0.0), -1.0, 1.0),
            "throttle": clamp(raw_outputs.get("throttle", 0.0), 0.0, 1.0),
            "brake": clamp(raw_outputs.get("brake", 0.0), 0.0, 1.0),
        }
        return PolicyStep(raw_outputs=raw_outputs, actions=actions, layer_activations=layer_activations)

    def _activate_hidden(self, value: float) -> float:
        if self.hidden_activation == "relu":
            return max(0.0, value)
        return math.tanh(value)

    def _activate_output(self, value: float) -> float:
        if self.output_activation == "linear":
            return value
        if self.output_activation == "sigmoid":
            return 1.0 / (1.0 + math.exp(-value))
        return math.tanh(value)

    def to_dict(self) -> dict:
        return {
            "sensorAnglesDeg": self.sensor_angles_deg,
            "actionNames": self.action_names,
            "layerSizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
            "hiddenActivation": self.hidden_activation,
            "outputActivation": self.output_activation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NeuralPolicy":
        return cls(
            sensor_angles_deg=list(data["sensorAnglesDeg"]),
            action_names=list(data["actionNames"]),
            layer_sizes=list(data["layerSizes"]),
            weights=[[[float(v) for v in row] for row in layer] for layer in data["weights"]],
            biases=[[float(v) for v in layer] for layer in data["biases"]],
            hidden_activation=data.get("hiddenActivation", "tanh"),
            output_activation=data.get("outputActivation", "tanh"),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NeuralPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class VehicleDynamicsConfig:
    speed_mode: Literal["constant", "dynamic"] = "constant"
    constant_speed: float = 2.0
    min_speed: float = 0.0
    max_speed: float = 6.0
    acceleration_rate: float = 2.3
    brake_rate: float = 3.2
    drag: float = 0.25


@dataclass
class ActionCommand:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0


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
    max_turn_rate_deg: float
    dynamics: VehicleDynamicsConfig
    speed: float = 0.0
    collision_radius: float = 0.1
    alive: bool = True
    time_alive: float = 0.0
    distance_traveled: float = 0.0
    last_acceleration: float = 0.0
    last_brake_force: float = 0.0
    path_trace: list[Vec2] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dynamics.speed_mode == "constant":
            self.speed = self.dynamics.constant_speed
        else:
            self.speed = clamp(self.speed, self.dynamics.min_speed, self.dynamics.max_speed)
        self.path_trace.append(self.position)

    def step(self, command: ActionCommand, dt: float) -> None:
        if not self.alive:
            return

        steering = clamp(command.steering, -1.0, 1.0)
        self.heading_deg += steering * self.max_turn_rate_deg * dt

        if self.dynamics.speed_mode == "constant":
            self.last_acceleration = 0.0
            self.last_brake_force = 0.0
            self.speed = self.dynamics.constant_speed
        else:
            throttle = clamp(command.throttle, 0.0, 1.0)
            brake = clamp(command.brake, 0.0, 1.0)
            accel_force = throttle * self.dynamics.acceleration_rate
            brake_force = brake * self.dynamics.brake_rate
            drag_force = self.speed * self.dynamics.drag
            net = accel_force - brake_force - drag_force
            self.speed = clamp(self.speed + net * dt, self.dynamics.min_speed, self.dynamics.max_speed)
            self.last_acceleration = accel_force
            self.last_brake_force = brake_force

        angle = math.radians(self.heading_deg)
        dx = math.cos(angle) * self.speed * dt
        dy = math.sin(angle) * self.speed * dt

        new_pos = (self.position[0] + dx, self.position[1] + dy)
        self.distance_traveled += math.hypot(new_pos[0] - self.position[0], new_pos[1] - self.position[1])
        self.position = new_pos
        self.path_trace.append(self.position)
        self.time_alive += dt


@dataclass
class StepSnapshot:
    sensor_readings: list[float]
    sensor_hit_distances: list[float | None]
    layer_activations: list[list[float]]
    raw_outputs: dict[str, float]
    actions: dict[str, float]
    speed: float
    acceleration: float
    brake_force: float
    heading_deg: float


@dataclass
class SimulationResult:
    survived_seconds: float
    distance_traveled: float
    cause_of_death: Literal["collision", "timeout", "alive"]
    path_trace: list[Vec2]
    final_snapshot: StepSnapshot | None


@dataclass
class Simulator:
    track: Track
    policy: NeuralPolicy
    agent: CarAgent
    sensor_array: SensorArray
    latest_snapshot: StepSnapshot | None = None

    def run(self, max_steps: int, dt: float) -> SimulationResult:
        cause: Literal["collision", "timeout", "alive"] = "alive"

        for _ in range(max_steps):
            if not self.agent.alive:
                cause = "collision"
                break

            snapshot = self.step(dt)
            if self._is_colliding(self.agent.position):
                self.agent.alive = False
                cause = "collision"
                self.latest_snapshot = snapshot
                break
        else:
            cause = "timeout"

        return SimulationResult(
            survived_seconds=self.agent.time_alive,
            distance_traveled=self.agent.distance_traveled,
            cause_of_death=cause,
            path_trace=list(self.agent.path_trace),
            final_snapshot=self.latest_snapshot,
        )

    def step(self, dt: float) -> StepSnapshot:
        scan = self.sensor_array.read_with_distances(
            position=self.agent.position,
            heading_deg=self.agent.heading_deg,
            track=self.track,
        )
        sensor_values = [reading for reading, _ in scan]
        step = self.policy.forward(sensor_values)
        command = ActionCommand(
            steering=step.actions["steering"],
            throttle=step.actions["throttle"],
            brake=step.actions["brake"],
        )
        self.agent.step(command=command, dt=dt)

        snapshot = StepSnapshot(
            sensor_readings=sensor_values,
            sensor_hit_distances=[dist for _, dist in scan],
            layer_activations=step.layer_activations,
            raw_outputs=step.raw_outputs,
            actions=step.actions,
            speed=self.agent.speed,
            acceleration=self.agent.last_acceleration,
            brake_force=self.agent.last_brake_force,
            heading_deg=self.agent.heading_deg,
        )
        self.latest_snapshot = snapshot
        return snapshot

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


def random_policy(
    sensor_angles_deg: list[float],
    action_names: list[str],
    hidden_layers: list[int] | None = None,
) -> NeuralPolicy:
    hidden_layers = hidden_layers or []
    layer_sizes = [len(sensor_angles_deg), *hidden_layers, len(action_names)]
    weights: list[list[list[float]]] = []
    biases: list[list[float]] = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
        weights.append([[random.uniform(-1.0, 1.0) for _ in range(in_size)] for _ in range(out_size)])
        biases.append([random.uniform(-0.2, 0.2) for _ in range(out_size)])
    return NeuralPolicy(
        sensor_angles_deg=sensor_angles_deg,
        action_names=action_names,
        layer_sizes=layer_sizes,
        weights=weights,
        biases=biases,
    )
