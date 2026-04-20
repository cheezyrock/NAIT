# NAIT Teaching AI Simulator

A projector-friendly local teaching app for explaining how a **single-layer neural network policy** maps sensor readings to steering while students watch a live driving simulation.

## What this version focuses on

- Real-time simulation loop in a classroom UI (track, vehicle, sensors, path trace).
- Fast in-app training controls (quick/deep cycles with configurable evolution rules in code).
- Neural-flow visualization showing sensor activation, weighted links, and steering output.
- Multi-track testing so students can train on one track and immediately try others.
- Architecture is already compatible with a future prefab-based track builder + save/load flow.

## Core model

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

## Interactive controls

- `Space` pause/resume
- `R` reset current run
- `N` single-step one frame
- `T` / `Y` switch to next/previous built-in track
- `1 / 2 / 3` load bad/decent/good policy presets
- `G` quick training pass (`20` cycles)
- `H` deeper training pass (`60` cycles)
- `C` toggle continuous training (runs 1-cycle mini-rounds repeatedly)
- `[` / `]` decrease/increase speed
- `-` / `=` decrease/increase max turn rate
- `<` / `>` decrease/increase sensor count
- `;` / `'` decrease/increase sensor spread
- `S` show/hide optional speed controls in UI
- `A` show/hide acceleration model behavior
- `M` simple mode toggle (hides neural math panel by default)

All key features also have clickable mouse buttons in the right-side control panel.

## Tracks

Built-in demos include:
- `oval_racetrack` (default)
- `rectangle`
- `chicane`
- `slalom`
- `pinch_turn`

## Architecture notes

- **Simulation engine (authoritative, UI-free):** `simulator_core.py`
- **Track/demo content:** `tracks.py`
- **Preset teaching content:** `policy_presets.py`
- **Rendering + controls + training UI:** `app.py`

## Planned later (not implemented yet)

- Prefab track builder with snapping parts
- Save/load custom track files from the builder workflow
