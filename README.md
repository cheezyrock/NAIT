# NAIT Teaching AI Simulator

A projector-friendly local teaching app for explaining how a **single-layer neural network policy** maps sensor readings to steering.

## What this version focuses on

- Classroom polish and legibility (large visual stage + clearer hierarchy).
- Explicit teaching overlays that show how `sensor * weight + bias` creates steering.
- Guided demo flow with built-in policy presets (`bad`, `decent`, `good`).
- Manual tuning controls suitable for live demos (no training algorithm yet).

## Core model (unchanged)

The simulation remains powered by `simulator_core.py` with no hidden layers:

```text
steering = tanh(bias + Σ(sensor[i] * weight[i]))
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

`app.py` is the runnable entrypoint. It constructs a default rectangular track, configures sensors and a linear policy, runs the simulator, and prints distance, survival time, and crash reason to the console.

## Teacher controls (right panel)

### Runtime
- **Run / Pause / Reset**
- **Slow-Mo** (20% speed)
- **Step** (single simulation step)
- **Track** (cycles through demonstration tracks)
- **Demo UI** (larger text, less clutter)

### Policy/demo
- **Bad / Decent / Good** preset buttons with on-screen explanation.
- **Load / Save** policy JSON with `LinearPolicy` format compatibility.

### Numeric controls (exact editing still available)
- Sensor angles CSV
- Weights CSV
- Bias
- Sensor max range
- Speed
- Max turn rate

Use **Apply** after editing numeric fields.

## Tracks

At least 4 built-in tracks are included, each with a tuned spawn point and heading:
- `rectangle`
- `chicane`
- `slalom`
- `pinch_turn`

## Teaching overlays included

- Vehicle shape + heading arrow (instead of plain point marker).
- Strongly visible sensor rays.
- Neural-network mini diagram:
  - sensor input nodes
  - weighted connection lines
  - single output node (steering)
  - explicit bias term and `tanh` output mapping
- Gauges/bars for steering and speed.
- Left/front/right danger interpretation from sensor groupings.

## Architecture notes

- **Simulation engine (authoritative, UI-free):** `simulator_core.py`
  - Physics, sensing, policy math, collision, metrics.
- **Track/demo content:** `tracks.py`
  - Track library, builders, spawn points, headings.
- **Preset teaching content:** `policy_presets.py`
  - Built-in policies + classroom explanation notes.
- **Rendering + controls + teaching overlays:** `app.py`
  - Pygame stage, control panel, NN diagram, gauges, demo/presentation behavior.

## Intentionally out of scope (for now)

- Hidden layers
- Automated training algorithms
- REST/QR/web/multiplayer

This keeps the app focused on manual tuning and explainability for short classroom sessions.
