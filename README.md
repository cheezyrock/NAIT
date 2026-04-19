# NAIT Teaching AI Simulator

A minimal local Python app for teaching high-school students how a small neural-network policy maps sensor inputs to steering behavior.

## Features

- **Authoritative simulation core** in `simulator_core.py`.
- **Single-layer policy** (no hidden layer):
  - `steering = tanh(bias + sum(sensor[i] * weight[i]))`
- **Interactive desktop UI** (pygame) in `app.py`.
- **Manual controls** for classroom tuning:
  - sensor count
  - sensor angle list
  - sensor max range
  - per-sensor weights
  - bias
  - car speed
  - max turn rate
  - run / pause / reset
- **Policy JSON load/save** using the existing `LinearPolicy` format.
- **Track switching** between:
  - rectangular track
  - custom chicane track
- Live metrics display:
  - sensor readings
  - steering output
  - survival time
  - distance traveled
  - crash reason

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

This starts a local desktop window (no REST/QR/web mode).

## How to use in class

1. Click text fields in the right panel to edit values.
2. Use `Apply` to rebuild the policy/agent with edited parameters.
3. Use `Run`, `Pause`, and `Reset` for simulation control.
4. Use `Track` to switch between the rectangle and chicane tracks.
5. Use `Save` / `Load` with the policy JSON path field (default `policy.json`).

## Architecture notes

- **Simulation ownership (reusable without UI):**
  - `simulator_core.py`
    - `LinearPolicy`: no-hidden-layer policy, JSON serialization helpers
    - `SensorArray`: ray/segment sensing and normalization
    - `CarAgent`: car state and movement
    - `Track`: explicit wall segments
    - `Simulator`: episode loop + result object
- **Visualization and local controls:**
  - `app.py`
    - pygame rendering for track/car/sensors
    - text-field + button controls for manual tuning
    - runtime loop and on-screen metrics
- **Track catalog / extension point:**
  - `tracks.py`
    - central registry of named track builders for easy future additions

## Notes

- This version intentionally excludes hidden layers, training algorithms, REST APIs, QR joining, and multiplayer.
- The simulator core remains importable and usable independently from the UI.
