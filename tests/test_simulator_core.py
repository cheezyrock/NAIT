import tempfile
import unittest
from pathlib import Path

from simulator_core import (
    CarAgent,
    LinearPolicy,
    SensorArray,
    Simulator,
    Track,
    build_rect_track,
)


class LinearPolicyTests(unittest.TestCase):
    def test_forward_shapes_and_tanh_bounds(self):
        policy = LinearPolicy(sensor_angles_deg=[-45, 0, 45], weights=[1.0, 0.0, -1.0], bias=0.0)
        output = policy.forward([1.0, 0.0, 0.0])
        self.assertTrue(-1.0 <= output <= 1.0)
        self.assertGreater(output, 0.0)

    def test_json_round_trip(self):
        policy = LinearPolicy(sensor_angles_deg=[-60, -30, 0, 30, 60], weights=[1, 2, 3, 4, 5], bias=0.5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "policy.json"
            policy.save_json(path)
            loaded = LinearPolicy.load_json(path)

        self.assertEqual(policy.sensor_angles_deg, loaded.sensor_angles_deg)
        self.assertEqual(policy.weights, loaded.weights)
        self.assertEqual(policy.bias, loaded.bias)


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
        policy = LinearPolicy(
            sensor_angles_deg=[-60, -30, 0, 30, 60],
            weights=[1.2, 0.8, 0.0, -0.8, -1.2],
            bias=0.0,
        )
        sensors = SensorArray(sensor_angles_deg=policy.sensor_angles_deg, max_range=3.0)
        agent = CarAgent(position=(5.0, 3.0), heading_deg=0.0, speed=1.5, max_turn_rate_deg=120.0)

        sim = Simulator(track=track, policy=policy, agent=agent, sensor_array=sensors)
        result = sim.run(max_steps=100, dt=0.1)

        self.assertGreater(result.survived_seconds, 0.0)
        self.assertGreater(result.distance_traveled, 0.0)
        self.assertIn(result.cause_of_death, {"collision", "timeout", "alive"})
        self.assertGreater(len(result.path_trace), 1)

    def test_collision_detected_when_wall_is_crossed_between_steps(self):
        track = Track.from_segments([((1.0, -2.0), (1.0, 2.0))])
        policy = LinearPolicy(sensor_angles_deg=[0.0], weights=[0.0], bias=0.0)
        sensors = SensorArray(sensor_angles_deg=policy.sensor_angles_deg, max_range=4.0)
        agent = CarAgent(position=(0.0, 0.0), heading_deg=0.0, speed=5.0, max_turn_rate_deg=0.0, collision_radius=0.1)

        sim = Simulator(track=track, policy=policy, agent=agent, sensor_array=sensors)
        result = sim.run(max_steps=1, dt=0.5)

        self.assertEqual(result.cause_of_death, "collision")
        self.assertFalse(agent.alive)


if __name__ == "__main__":
    unittest.main()
