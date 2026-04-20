# NAIT Teaching AI Simulator

A classroom-focused local app for teaching how a **single-layer neural policy** drives using ray sensors.

## Highlights

- Cleaner simulation view with reduced visual clutter.
- Attempt-history traces so students can watch training evolution over many failures.
- Automatic restart after failure (default: 3 seconds, configurable).
- Context menu + options modal (manual numeric entry for settings).
- Updated figure-eight style track that avoids center-wall self-collision in 2D.
- Start/finish lines span lane walls (track-width crossing).

## Core model

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

## Controls

- `Space`: pause/resume
- `R`: reset current attempt
- `N`: single-step one frame
- `T` / `Y`: next / previous track
- `G`: queue training batch
- `C`: toggle continuous training
- `O`: open/close options modal
- Right-click: open context menu

## Attribution

See `CREDITS.md` for asset and license notes.
