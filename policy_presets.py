"""Built-in policy presets for guided classroom demos."""

from __future__ import annotations

from dataclasses import dataclass

from simulator_core import NeuralPolicy


@dataclass(frozen=True)
class PolicyPreset:
    name: str
    note: str
    sensor_angles_deg: list[float]
    action_names: list[str]
    layer_sizes: list[int]
    weights: list[list[list[float]]]
    biases: list[list[float]]

    def build_policy(self) -> NeuralPolicy:
        return NeuralPolicy(
            sensor_angles_deg=list(self.sensor_angles_deg),
            action_names=list(self.action_names),
            layer_sizes=list(self.layer_sizes),
            weights=[[list(row) for row in layer] for layer in self.weights],
            biases=[list(layer_bias) for layer_bias in self.biases],
            hidden_activation="tanh",
            output_activation="tanh",
        )


PRESETS: dict[str, PolicyPreset] = {
    "steering_basic": PolicyPreset(
        name="Steering only (no hidden)",
        note="Linear steering behavior for first lessons.",
        sensor_angles_deg=[-60, -30, 0, 30, 60],
        action_names=["steering"],
        layer_sizes=[5, 1],
        weights=[[[1.2, 0.8, 0.05, -0.8, -1.2]]],
        biases=[[0.0]],
    ),
    "steering_hidden": PolicyPreset(
        name="Steering with hidden layer",
        note="Shows intermediate activations from a hidden layer.",
        sensor_angles_deg=[-60, -30, 0, 30, 60],
        action_names=["steering"],
        layer_sizes=[5, 4, 1],
        weights=[
            [
                [0.9, 0.4, 0.0, -0.4, -0.9],
                [0.6, 0.2, 0.0, -0.2, -0.6],
                [-0.8, -0.35, 0.0, 0.35, 0.8],
                [0.2, -0.1, 0.0, 0.1, -0.2],
            ],
            [[0.9, 0.8, -0.7, 0.3]],
        ],
        biases=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0],
        ],
    ),
    "speed_control": PolicyPreset(
        name="Steering + throttle + brake",
        note="Multi-output policy for speed lessons and braking demos.",
        sensor_angles_deg=[-75, -40, -15, 0, 15, 40, 75],
        action_names=["steering", "throttle", "brake"],
        layer_sizes=[7, 5, 3],
        weights=[
            [
                [0.8, 0.6, 0.3, 0.0, -0.3, -0.6, -0.8],
                [0.5, 0.4, 0.2, 0.0, -0.2, -0.4, -0.5],
                [-0.6, -0.3, 0.0, 0.0, 0.0, 0.3, 0.6],
                [0.2, 0.25, 0.35, 0.45, 0.35, 0.25, 0.2],
                [0.0, 0.0, 0.2, 0.4, 0.2, 0.0, 0.0],
            ],
            [
                [0.9, 0.6, -0.6, 0.0, 0.0],
                [0.1, 0.2, 0.0, 1.1, 0.8],
                [0.0, 0.0, 0.2, -1.2, -1.0],
            ],
        ],
        biases=[
            [0.0, 0.0, 0.0, -0.4, -0.5],
            [0.0, 0.1, -0.15],
        ],
    ),
}
