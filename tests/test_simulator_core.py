import tempfile
import unittest
from pathlib import Path

from simulator_core import (
    CarAgent,
    NeuralPolicy,
    SensorArray,
    Simulator,
    Track,
    VehicleDynamicsConfig,
    build_rect_track,
)


class NeuralPolicyTests(unittest.TestCase):
    def test_forward_shapes_and_tanh_bounds(self):
        policy = NeuralPolicy(
            sensor_angles_deg=[-45, 0, 45],
            action_names=["steering", "throttle", "brake"],
            layer_sizes=[3, 4, 3],
            weights=[
                [
                    [1.0, 0.0, -1.0],
                    [0.2, 0.1, -0.2],
                    [-0.4, 0.0, 0.4],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [1.0, 0.2, -0.3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, -1.0],
                ],
            ],
            biases=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        step = policy.forward([1.0, 0.0, 0.0])
        self.assertTrue(-1.0 <= step.actions["steering"] <= 1.0)
        self.assertTrue(0.0 <= step.actions["throttle"] <= 1.0)
        self.assertTrue(0.0 <= step.actions["brake"] <= 1.0)

    def test_json_round_trip(self):
        policy = NeuralPolicy(
            sensor_angles_deg=[-60, -30, 0, 30, 60],
            action_names=["steering"],
            layer_sizes=[5, 1],
            weights=[[[1, 2, 3, 4, 5]]],
            biases=[[0.5]],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "policy.json"
            policy.save_json(path)
            loaded = NeuralPolicy.load_json(path)

        self.assertEqual(policy.sensor_angles_deg, loaded.sensor_angles_deg)
        self.assertEqual(policy.layer_sizes, loaded.layer_sizes)
        self.assertEqual(policy.weights, loaded.weights)
        self.assertEqual(policy.biases, loaded.biases)


class SensorTests(unittest.TestCase):
    def test_sensor_reads_high_when_wall_is_close(self):
        track = Track.from_segments([((2.0, -1.0), (2.0, 1.0))])
        sensors = SensorArray(sensor_angles_deg=[0.0], max_range=4.0)
        readings = sensors.read(position=(0.0, 0.0), heading_deg=0.0, track=track)
        self.assertEqual(len(readings), 1)
        self.assertAlmostEqual(readings[0], 0.5, places=6)


class SimulationTests(unittest.TestCase):
    def test_simulation_runs_and_returns_metrics(self):
        track = build_rect_track(width=10.0, height=6.0)
        policy = NeuralPolicy(
            sensor_angles_deg=[-60, -30, 0, 30, 60],
            action_names=["steering"],
            layer_sizes=[5, 1],
            weights=[[[1.2, 0.8, 0.0, -0.8, -1.2]]],
            biases=[[0.0]],
        )
        sensors = SensorArray(sensor_angles_deg=policy.sensor_angles_deg, max_range=3.0)
        agent = CarAgent(
            position=(5.0, 3.0),
            heading_deg=0.0,
            max_turn_rate_deg=120.0,
            dynamics=VehicleDynamicsConfig(speed_mode="constant", constant_speed=1.5),
        )

        sim = Simulator(track=track, policy=policy, agent=agent, sensor_array=sensors)
        result = sim.run(max_steps=100, dt=0.1)

        self.assertGreater(result.survived_seconds, 0.0)
        self.assertGreater(result.distance_traveled, 0.0)
        self.assertIn(result.cause_of_death, {"collision", "timeout", "alive"})
        self.assertGreater(len(result.path_trace), 1)

    def test_dynamic_speed_uses_throttle_and_brake(self):
        track = build_rect_track(width=20.0, height=8.0)
        policy = NeuralPolicy(
            sensor_angles_deg=[0.0],
            action_names=["steering", "throttle", "brake"],
            layer_sizes=[1, 3],
            weights=[[[0.0], [1.0], [-1.0]]],
            biases=[[0.0, 0.8, -0.2]],
        )
        sensors = SensorArray(sensor_angles_deg=[0.0], max_range=8.0)
        agent = CarAgent(
            position=(2.0, 4.0),
            heading_deg=0.0,
            max_turn_rate_deg=120.0,
            dynamics=VehicleDynamicsConfig(speed_mode="dynamic", max_speed=5.0),
        )
        sim = Simulator(track=track, policy=policy, agent=agent, sensor_array=sensors)

        for _ in range(10):
            sim.step(dt=0.1)

        self.assertGreater(agent.speed, 0.0)


if __name__ == "__main__":
    unittest.main()
