"""Interactive teaching application for neural-network driving demos.

Usage:
    python app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from scenarios import get_scenario, scenario_names
from simulator_core import CarAgent, NeuralPolicy, SensorArray, Simulator
from tracks import build_track_definition, track_names

WINDOW_SIZE = (1280, 760)
BG = (16, 20, 27)
PANEL = (26, 32, 42)
TEXT = (236, 241, 250)
MUTED = (157, 168, 188)
ACCENT = (102, 186, 255)
GREEN = (82, 212, 153)
RED = (255, 115, 115)


@dataclass
class SimConfig:
    sensor_max_range: float = 4.0
    max_turn_rate_deg: float = 120.0
    dt: float = 0.05
    auto_restart_delay: float = 2.5


class TeachingApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("NAIT Neural Driving Trainer")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("inter,arial", 18)
        self.small = pygame.font.SysFont("inter,arial", 15)
        self.big = pygame.font.SysFont("inter,arial", 24)

        self.config = SimConfig()
        self.track_names = track_names()
        self.scenario_keys = scenario_names()
        self.scenario_idx = 0
        self.track_idx = 0

        self.running = True
        self.paused = False
        self.restart_countdown = 0.0
        self.status = "Ready"

        self._apply_scenario(self.scenario_keys[self.scenario_idx])

    def _apply_scenario(self, scenario_key: str) -> None:
        self.scenario = get_scenario(scenario_key)
        if self.scenario.track_name in self.track_names:
            self.track_idx = self.track_names.index(self.scenario.track_name)
        self.track_def = build_track_definition(self.track_names[self.track_idx])
        self.policy: NeuralPolicy = self.scenario.build_policy()
        self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)
        self.agent = CarAgent(
            position=self.track_def.spawn,
            heading_deg=self.track_def.heading_deg,
            max_turn_rate_deg=self.config.max_turn_rate_deg,
            dynamics=self.scenario.dynamics,
        )
        self.simulator = Simulator(self.track_def.track, self.policy, self.agent, self.sensor_array)
        self.status = f"Scenario loaded: {self.scenario.label}"

    def _reset_episode(self) -> None:
        self.agent = CarAgent(
            position=self.track_def.spawn,
            heading_deg=self.track_def.heading_deg,
            max_turn_rate_deg=self.config.max_turn_rate_deg,
            dynamics=self.scenario.dynamics,
        )
        self.simulator = Simulator(self.track_def.track, self.policy, self.agent, self.sensor_array)
        self.restart_countdown = 0.0

    def run(self) -> None:
        while self.running:
            self._handle_events()
            if not self.paused:
                self._tick_simulation(self.config.dt)
            self._render()
            self.clock.tick(60)
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_episode()
                    self.status = "Episode reset"
                elif event.key == pygame.K_t:
                    self.track_idx = (self.track_idx + 1) % len(self.track_names)
                    self.track_def = build_track_definition(self.track_names[self.track_idx])
                    self._reset_episode()
                    self.status = f"Track changed: {self.track_def.name}"
                elif event.key == pygame.K_y:
                    self.track_idx = (self.track_idx - 1) % len(self.track_names)
                    self.track_def = build_track_definition(self.track_names[self.track_idx])
                    self._reset_episode()
                    self.status = f"Track changed: {self.track_def.name}"
                elif event.key == pygame.K_m:
                    self.scenario_idx = (self.scenario_idx + 1) % len(self.scenario_keys)
                    self._apply_scenario(self.scenario_keys[self.scenario_idx])

    def _tick_simulation(self, dt: float) -> None:
        if not self.agent.alive:
            self.restart_countdown -= dt
            if self.restart_countdown <= 0:
                self._reset_episode()
            return

        self.simulator.step(dt)
        if self.simulator._is_colliding(self.agent.position):
            self.agent.alive = False
            self.restart_countdown = self.config.auto_restart_delay
            self.status = f"Collision. Restart in {self.config.auto_restart_delay:.1f}s"

    def _world_bounds(self) -> tuple[float, float, float, float]:
        xs = [pt[0] for seg in self.track_def.track.wall_segments for pt in seg]
        ys = [pt[1] for seg in self.track_def.track.wall_segments for pt in seg]
        return min(xs), max(xs), min(ys), max(ys)

    def _to_screen(self, p: tuple[float, float], rect: pygame.Rect) -> tuple[int, int]:
        min_x, max_x, min_y, max_y = self._world_bounds()
        world_w, world_h = max_x - min_x, max_y - min_y
        scale = min((rect.width - 30) / world_w, (rect.height - 30) / world_h)
        x = rect.left + 15 + (p[0] - min_x) * scale
        y = rect.bottom - 15 - (p[1] - min_y) * scale
        return int(x), int(y)

    def _render(self) -> None:
        self.screen.fill(BG)
        track_rect = pygame.Rect(14, 70, 900, 676)
        hud_rect = pygame.Rect(14, 12, 1252, 50)
        side_rect = pygame.Rect(926, 70, 340, 676)

        pygame.draw.rect(self.screen, PANEL, hud_rect, border_radius=12)
        pygame.draw.rect(self.screen, (12, 16, 22), track_rect, border_radius=12)
        pygame.draw.rect(self.screen, PANEL, side_rect, border_radius=12)

        self._draw_track(track_rect)
        self._draw_hud(hud_rect)
        self._draw_side(side_rect)
        pygame.display.flip()

    def _draw_hud(self, rect: pygame.Rect) -> None:
        text = (
            f"Scenario: {self.scenario.label}  |  Track: {self.track_def.name}  |  "
            f"Mode: {self.scenario.dynamics.speed_mode}  |  Time: {self.agent.time_alive:.2f}s"
        )
        self.screen.blit(self.big.render(text, True, TEXT), (rect.x + 14, rect.y + 12))

    def _draw_track(self, rect: pygame.Rect) -> None:
        for seg in self.track_def.track.wall_segments:
            pygame.draw.line(self.screen, (220, 231, 248), self._to_screen(seg[0], rect), self._to_screen(seg[1], rect), 3)

        if len(self.agent.path_trace) > 2:
            pts = [self._to_screen(p, rect) for p in self.agent.path_trace[-350:]]
            pygame.draw.lines(self.screen, ACCENT, False, pts, 2)

        car = self._to_screen(self.agent.position, rect)
        pygame.draw.circle(self.screen, GREEN if self.agent.alive else RED, car, 9)
        angle = math.radians(self.agent.heading_deg)
        nose = (int(car[0] + math.cos(angle) * 17), int(car[1] - math.sin(angle) * 17))
        pygame.draw.line(self.screen, (255, 217, 107), car, nose, 3)

        snapshot = self.simulator.latest_snapshot
        if snapshot is None:
            return
        for rel_angle, hit in zip(self.policy.sensor_angles_deg, snapshot.sensor_hit_distances, strict=True):
            a = math.radians(self.agent.heading_deg + rel_angle)
            length = self.config.sensor_max_range if hit is None else min(hit, self.config.sensor_max_range)
            end_w = (self.agent.position[0] + math.cos(a) * length, self.agent.position[1] + math.sin(a) * length)
            pygame.draw.line(self.screen, (94, 155, 235), car, self._to_screen(end_w, rect), 1)

    def _draw_side(self, rect: pygame.Rect) -> None:
        x, y = rect.x + 14, rect.y + 14
        rows = [
            f"Status: {self.status}",
            f"Alive: {self.agent.alive}",
            f"Speed: {self.agent.speed:.2f}",
            f"Acceleration force: {self.agent.last_acceleration:.2f}",
            f"Brake force: {self.agent.last_brake_force:.2f}",
            f"Distance: {self.agent.distance_traveled:.2f}",
            "",
            "Keys: SPACE pause, R reset, T/Y track, M scenario",
        ]
        for row in rows:
            color = MUTED if row.startswith("Status") or row.startswith("Keys") else TEXT
            self.screen.blit(self.small.render(row, True, color), (x, y))
            y += 20

        snapshot = self.simulator.latest_snapshot
        if snapshot is None:
            return

        y += 8
        self.screen.blit(self.font.render("Action outputs", True, TEXT), (x, y))
        y += 24
        for name, value in snapshot.actions.items():
            self.screen.blit(self.small.render(f"{name}: {value:+.3f}", True, TEXT), (x, y))
            y += 18

        y += 8
        self.screen.blit(self.font.render("Raw policy outputs", True, TEXT), (x, y))
        y += 24
        for name, value in snapshot.raw_outputs.items():
            self.screen.blit(self.small.render(f"{name}: {value:+.3f}", True, TEXT), (x, y))
            y += 18

        y += 8
        self.screen.blit(self.font.render("Layer activations", True, TEXT), (x, y))
        y += 24
        for idx, values in enumerate(snapshot.layer_activations):
            msg = f"L{idx} [{', '.join(f'{v:+.2f}' for v in values[:6])}]"
            self.screen.blit(self.small.render(msg, True, MUTED), (x, y))
            y += 18
            if y > rect.bottom - 80:
                break

        y += 8
        self.screen.blit(self.font.render("Sensors", True, TEXT), (x, y))
        y += 24
        for idx, (reading, hit) in enumerate(zip(snapshot.sensor_readings, snapshot.sensor_hit_distances, strict=True)):
            hit_text = "none" if hit is None else f"{hit:.2f}"
            self.screen.blit(
                self.small.render(f"s{idx}: read={reading:.2f} hit={hit_text}", True, MUTED),
                (x, y),
            )
            y += 18
            if y > rect.bottom - 20:
                break


def main() -> None:
    TeachingApp().run()


if __name__ == "__main__":
    main()
