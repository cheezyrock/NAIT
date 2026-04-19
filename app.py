"""Runnable entrypoint for the NAIT simulation core.

Usage:
    python app.py
"""

from __future__ import annotations

from simulator_core import CarAgent, LinearPolicy, SensorArray, Simulator, build_rect_track


def build_default_simulator() -> Simulator:
    """Create a simulator instance with sensible defaults."""
    track = build_rect_track(width=14.0, height=8.0)

    sensor_angles = [-60.0, -30.0, 0.0, 30.0, 60.0]
    sensor_array = SensorArray(sensor_angles_deg=sensor_angles, max_range=3.5)

    policy = LinearPolicy(
        sensor_angles_deg=sensor_angles,
        weights=[0.8, 0.45, 0.0, -0.45, -0.8],
        bias=0.0,
    )

    agent = CarAgent(
        position=(2.0, 4.0),
        heading_deg=0.0,
        speed=2.0,
        max_turn_rate_deg=120.0,
    )

    return Simulator(track=track, policy=policy, agent=agent, sensor_array=sensor_array)


def main() -> None:
    simulator = build_default_simulator()
    result = simulator.run(max_steps=2_000, dt=0.05)

    print("Simulation complete")
    print(f"Distance: {result.distance_traveled:.3f}")
    print(f"Survival time: {result.survived_seconds:.3f}")
    print(f"Crash reason: {result.cause_of_death}")


if __name__ == "__main__":
    main()
