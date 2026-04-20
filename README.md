# NAIT Teaching AI Simulator

A classroom-focused local app for teaching how a **configurable neural policy network** drives using ray sensors.

## Highlights

- Fully configurable feed-forward policy network (no framework dependencies).
- Supports any layer layout such as `[inputs, outputs]`, `[inputs, hidden, outputs]`, and deeper stacks.
- Multi-output actions with configurable outputs (`steering`, `throttle`, `brake`).
- Vehicle dynamics now support:
  - constant-speed mode (teaching baseline)
  - dynamic speed with acceleration, braking, drag, and max speed limits.
- Runtime visibility data exposed for UI/teaching overlays:
  - sensor readings + hit distances
  - per-layer activations
  - raw output values and final action values
  - speed, acceleration force, and braking force.
- First-class teaching scenarios:
  - steering-only constant-speed driving
  - speed-control wall lesson with throttle/brake.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Controls

- `Space`: pause/resume
- `R`: reset episode
- `T` / `Y`: next / previous track
- `M`: cycle teaching scenario
- `Esc`: quit

## Attribution

See `CREDITS.md` for asset and license notes.
