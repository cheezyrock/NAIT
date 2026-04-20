"""First-class teaching scenarios for simulator setup."""

from __future__ import annotations

from dataclasses import dataclass

from policy_presets import PRESETS
from simulator_core import NeuralPolicy, VehicleDynamicsConfig


@dataclass(frozen=True)
class ScenarioConfig:
    key: str
    label: str
    description: str
    track_name: str
    policy_preset: str
    dynamics: VehicleDynamicsConfig

    def build_policy(self) -> NeuralPolicy:
        return PRESETS[self.policy_preset].build_policy()


SCENARIOS: dict[str, ScenarioConfig] = {
    "steering_constant": ScenarioConfig(
        key="steering_constant",
        label="Steering-only constant speed",
        description="Baseline lesson: agent only controls steering at fixed speed.",
        track_name="rectangle",
        policy_preset="steering_hidden",
        dynamics=VehicleDynamicsConfig(
            speed_mode="constant",
            constant_speed=2.0,
            min_speed=0.0,
            max_speed=2.0,
            acceleration_rate=0.0,
            brake_rate=0.0,
            drag=0.0,
        ),
    ),
    "speed_wall": ScenarioConfig(
        key="speed_wall",
        label="Speed control wall test",
        description="Straight-line speed lesson with throttle/brake enabled.",
        track_name="rectangle",
        policy_preset="speed_control",
        dynamics=VehicleDynamicsConfig(
            speed_mode="dynamic",
            constant_speed=0.0,
            min_speed=0.0,
            max_speed=5.0,
            acceleration_rate=2.8,
            brake_rate=4.2,
            drag=0.35,
        ),
    ),
}


def scenario_names() -> list[str]:
    return list(SCENARIOS.keys())


def get_scenario(key: str) -> ScenarioConfig:
    return SCENARIOS[key]
