# NAIT Teaching AI Simulator (Core)

A minimal Python simulation core for teaching high-school students neural-network basics with a policy they can manually tune.

## What is implemented

- **Policy**: single-layer linear NN (no hidden layer), `N` sensors -> 1 steering output.
- **Forward equation**: `steering = tanh(bias + sum(sensor[i] * weight[i]))`.
- **Sensors**: configurable sensor angles and max range; ray/segment intersections against track boundaries.
- **Track**: explicit line-segment walls so any track can be evaluated with the same policy.
- **Agent**: constant-speed car with steering-only control.
- **Simulator output**: survival time, distance traveled, cause of death, and path trace.
- **Policy I/O**: JSON save/load helpers.

## Quick start

```python
from simulator_core import (
    LinearPolicy,
    SensorArray,
    CarAgent,
    Simulator,
    build_rect_track,
)

policy = LinearPolicy(
    sensor_angles_deg=[-60, -30, 0, 30, 60],
    # left sensors positive, right sensors negative for "steer away from danger"
    weights=[1.2, 0.8, 0.0, -0.8, -1.2],
    bias=0.0,
)

track = build_rect_track(width=12.0, height=8.0)
sensors = SensorArray(sensor_angles_deg=policy.sensor_angles_deg, max_range=3.5)
agent = CarAgent(position=(2.0, 4.0), heading_deg=0.0, speed=2.0, max_turn_rate_deg=120.0)

sim = Simulator(track=track, policy=policy, agent=agent, sensor_array=sensors)
result = sim.run(max_steps=400, dt=0.05)
print(result)
```

## Notes

This repository intentionally keeps the simulation core independent from any future UI, QR join flow, or REST API layer.
