# NAIT Teaching AI Simulator

A classroom-focused desktop app for teaching how a **configurable neural policy network** controls a simulated car.

## Teaching-focused architecture

- Fully configurable feed-forward policy (`NeuralPolicy`) implemented manually in Python (no TensorFlow/PyTorch).
- Arbitrary layer layouts supported:
  - `[inputs, outputs]`
  - `[inputs, hidden, outputs]`
  - `[inputs, hidden1, hidden2, outputs]`
- Multi-output policy actions:
  - `steering`
  - `throttle`
  - `brake`
- Simulator supports both:
  - **constant-speed steering mode**
  - **dynamic speed mode** with acceleration, braking, drag, and max speed constraints.

## Visibility-first runtime data

The app exposes the values needed for teaching and debugging:

- sensor readings + hit distances
- per-layer activations
- raw network outputs
- final bounded action outputs
- current speed
- acceleration and braking forces

## Included instructional scenarios

- **Steering-only constant speed**
- **Speed-control wall lesson** (steering + throttle + brake)

You can also open the in-app Options panel and reconfigure:

- sensor count/spread/range
- hidden layer sizes
- enabled outputs (throttle/brake)
- speed mode and dynamics parameters
- evolutionary training settings

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
- `N`: single-step one frame
- `T` / `Y`: next / previous track
- `M`: cycle scenarios
- `G`: queue training batch
- `C`: continuous training
- `O`: open/close options modal
- Right-click: context menu

## Attribution

See `CREDITS.md` for asset and license notes.
