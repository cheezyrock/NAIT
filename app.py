from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from simulator_core import CarAgent, LinearPolicy, SensorArray, Simulator
from tracks import TRACK_BUILDERS, build_track

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 720
PANEL_WIDTH = 380
WORLD_MARGIN = 30


@dataclass
class TextField:
    label: str
    value: str
    rect: pygame.Rect


@dataclass
class Button:
    label: str
    rect: pygame.Rect


class TeachingApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("NAIT Neural Net Teaching App")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)

        self.active_field: str | None = None
        self.status_message = "Ready"
        self.track_names = list(TRACK_BUILDERS.keys())
        self.track_index = 0

        self.fields = self._make_fields()
        self.buttons = self._make_buttons()

        self.sensor_values: list[float] = []
        self.latest_steering = 0.0
        self.sim_result = None
        self.running = False

        self._reset_simulation()

    def _make_fields(self) -> dict[str, TextField]:
        x = WINDOW_WIDTH - PANEL_WIDTH + 18
        w = PANEL_WIDTH - 36
        y = 30
        h = 30
        spacing = 42

        def f(key: str, label: str, value: str, row: int) -> tuple[str, TextField]:
            return key, TextField(label=label, value=value, rect=pygame.Rect(x, y + row * spacing, w, h))

        return dict(
            [
                f("policy_path", "Policy JSON Path", "policy.json", 0),
                f("sensor_count", "Sensor Count", "5", 1),
                f("sensor_angles", "Sensor Angles (deg csv)", "-60,-30,0,30,60", 2),
                f("sensor_max_range", "Sensor Max Range", "3.5", 3),
                f("weights", "Weights (csv)", "1.2,0.8,0,-0.8,-1.2", 4),
                f("bias", "Bias", "0.0", 5),
                f("speed", "Car Speed", "2.0", 6),
                f("max_turn_rate", "Max Turn Rate Deg/s", "120.0", 7),
            ]
        )

    def _make_buttons(self) -> dict[str, Button]:
        x = WINDOW_WIDTH - PANEL_WIDTH + 18
        y = 380
        w = 108
        h = 34
        gap = 12
        rows = [
            ("run", "Run", 0, 0),
            ("pause", "Pause", 1, 0),
            ("reset", "Reset", 2, 0),
            ("apply", "Apply", 0, 1),
            ("load", "Load", 1, 1),
            ("save", "Save", 2, 1),
            ("track", "Track", 0, 2),
        ]
        out: dict[str, Button] = {}
        for key, label, col, row in rows:
            out[key] = Button(
                label=label,
                rect=pygame.Rect(x + col * (w + gap), y + row * (h + gap), w, h),
            )
        return out

    def _parse_csv_floats(self, text: str) -> list[float]:
        raw = [part.strip() for part in text.split(",") if part.strip()]
        return [float(x) for x in raw]

    def _build_symmetric_angles(self, count: int) -> list[float]:
        if count <= 1:
            return [0.0]
        left = -60.0
        right = 60.0
        step = (right - left) / (count - 1)
        return [left + i * step for i in range(count)]

    def _read_config(self) -> tuple[LinearPolicy, SensorArray, CarAgent]:
        sensor_count = int(self.fields["sensor_count"].value)
        angles = self._parse_csv_floats(self.fields["sensor_angles"].value)
        if len(angles) != sensor_count:
            angles = self._build_symmetric_angles(sensor_count)
            self.fields["sensor_angles"].value = ",".join(f"{a:.1f}" for a in angles)

        weights = self._parse_csv_floats(self.fields["weights"].value)
        if len(weights) != sensor_count:
            weights = [0.0] * sensor_count
            self.fields["weights"].value = ",".join("0" for _ in weights)

        bias = float(self.fields["bias"].value)
        speed = float(self.fields["speed"].value)
        max_turn_rate = float(self.fields["max_turn_rate"].value)
        sensor_max_range = float(self.fields["sensor_max_range"].value)

        policy = LinearPolicy(sensor_angles_deg=angles, weights=weights, bias=bias)
        sensors = SensorArray(sensor_angles_deg=angles, max_range=sensor_max_range)
        # Spawn inside both demo tracks.
        agent = CarAgent(position=(2.0, 4.0), heading_deg=0.0, speed=speed, max_turn_rate_deg=max_turn_rate)
        return policy, sensors, agent

    def _reset_simulation(self) -> None:
        self.track = build_track(self.track_names[self.track_index])
        self.policy, self.sensor_array, self.agent = self._read_config()
        self.simulator = Simulator(
            track=self.track,
            policy=self.policy,
            agent=self.agent,
            sensor_array=self.sensor_array,
        )
        self.running = False
        self.sensor_values = self.sensor_array.read(self.agent.position, self.agent.heading_deg, self.track)
        self.latest_steering = self.policy.forward(self.sensor_values)
        self.sim_result = None

    def _apply_config(self) -> None:
        try:
            self._reset_simulation()
            self.status_message = "Config applied"
        except Exception as exc:  # Parse/validation errors shown in UI.
            self.status_message = f"Config error: {exc}"

    def _load_policy(self) -> None:
        path = self.fields["policy_path"].value
        try:
            policy = LinearPolicy.load_json(path)
            self.fields["sensor_count"].value = str(policy.sensor_count)
            self.fields["sensor_angles"].value = ",".join(str(x) for x in policy.sensor_angles_deg)
            self.fields["weights"].value = ",".join(str(x) for x in policy.weights)
            self.fields["bias"].value = str(policy.bias)
            self._reset_simulation()
            self.status_message = f"Loaded policy: {path}"
        except Exception as exc:
            self.status_message = f"Load failed: {exc}"

    def _save_policy(self) -> None:
        path = self.fields["policy_path"].value
        try:
            policy, _, _ = self._read_config()
            policy.save_json(path)
            self.status_message = f"Saved policy: {path}"
        except Exception as exc:
            self.status_message = f"Save failed: {exc}"

    def _switch_track(self) -> None:
        self.track_index = (self.track_index + 1) % len(self.track_names)
        self._reset_simulation()
        self.status_message = f"Track: {self.track_names[self.track_index]}"

    def _handle_button(self, name: str) -> None:
        if name == "run":
            self.running = True
            self.status_message = "Running"
        elif name == "pause":
            self.running = False
            self.status_message = "Paused"
        elif name == "reset":
            self._reset_simulation()
            self.status_message = "Reset"
        elif name == "apply":
            self._apply_config()
        elif name == "load":
            self._load_policy()
        elif name == "save":
            self._save_policy()
        elif name == "track":
            self._switch_track()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.active_field = None
                for key, field in self.fields.items():
                    if field.rect.collidepoint(event.pos):
                        self.active_field = key
                for key, button in self.buttons.items():
                    if button.rect.collidepoint(event.pos):
                        self._handle_button(key)
            if event.type == pygame.KEYDOWN and self.active_field is not None:
                field = self.fields[self.active_field]
                if event.key == pygame.K_BACKSPACE:
                    field.value = field.value[:-1]
                elif event.key == pygame.K_RETURN:
                    self.active_field = None
                else:
                    if event.unicode and event.unicode.isprintable():
                        field.value += event.unicode
        return True

    def _step_simulation(self, dt: float) -> None:
        if not self.running:
            return
        if not self.agent.alive:
            return

        self.sensor_values = self.sensor_array.read(self.agent.position, self.agent.heading_deg, self.track)
        self.latest_steering = self.policy.forward(self.sensor_values)
        self.agent.step(self.latest_steering, dt)
        if self.simulator._is_colliding(self.agent.position):
            self.agent.alive = False
            self.running = False
            self.sim_result = self.simulator.run(max_steps=0, dt=dt)
            self.status_message = "Crashed"

    def _world_bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for seg in self.track.wall_segments for p in seg]
        ys = [p[1] for seg in self.track.wall_segments for p in seg]
        return min(xs), min(ys), max(xs), max(ys)

    def _world_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        min_x, min_y, max_x, max_y = self._world_bounds()
        world_w = max_x - min_x
        world_h = max_y - min_y

        draw_w = WINDOW_WIDTH - PANEL_WIDTH - 2 * WORLD_MARGIN
        draw_h = WINDOW_HEIGHT - 2 * WORLD_MARGIN
        scale = min(draw_w / max(world_w, 1e-6), draw_h / max(world_h, 1e-6))

        x = WORLD_MARGIN + (point[0] - min_x) * scale
        y = WINDOW_HEIGHT - WORLD_MARGIN - (point[1] - min_y) * scale
        return int(x), int(y)

    def _draw_world(self) -> None:
        # track
        for a, b in self.track.wall_segments:
            pygame.draw.line(self.screen, (200, 200, 200), self._world_to_screen(a), self._world_to_screen(b), 3)

        # trail
        if len(self.agent.path_trace) > 1:
            trail = [self._world_to_screen(p) for p in self.agent.path_trace]
            pygame.draw.lines(self.screen, (90, 180, 255), False, trail, 2)

        # sensors
        self.sensor_values = self.sensor_array.read(self.agent.position, self.agent.heading_deg, self.track)
        for angle_deg, reading in zip(self.policy.sensor_angles_deg, self.sensor_values, strict=True):
            world_angle = math.radians(self.agent.heading_deg + angle_deg)
            distance = self.sensor_array.max_range * (1.0 - reading)
            end = (
                self.agent.position[0] + math.cos(world_angle) * distance,
                self.agent.position[1] + math.sin(world_angle) * distance,
            )
            color = (255, int(220 * (1.0 - reading)), int(220 * (1.0 - reading)))
            pygame.draw.line(self.screen, color, self._world_to_screen(self.agent.position), self._world_to_screen(end), 1)

        # car body
        car_pos = self._world_to_screen(self.agent.position)
        pygame.draw.circle(self.screen, (255, 220, 80), car_pos, 8)

        # heading vector
        heading_rad = math.radians(self.agent.heading_deg)
        nose = (
            self.agent.position[0] + math.cos(heading_rad) * 0.8,
            self.agent.position[1] + math.sin(heading_rad) * 0.8,
        )
        pygame.draw.line(self.screen, (255, 100, 100), car_pos, self._world_to_screen(nose), 3)

    def _draw_panel(self) -> None:
        panel_x = WINDOW_WIDTH - PANEL_WIDTH
        pygame.draw.rect(self.screen, (30, 30, 36), pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT))

        title = self.font.render("Manual Policy Controls", True, (240, 240, 240))
        self.screen.blit(title, (panel_x + 18, 6))

        for key, field in self.fields.items():
            label = self.small_font.render(field.label, True, (200, 200, 200))
            self.screen.blit(label, (field.rect.x, field.rect.y - 16))

            color = (70, 70, 90) if self.active_field != key else (95, 95, 130)
            pygame.draw.rect(self.screen, color, field.rect, border_radius=4)
            pygame.draw.rect(self.screen, (140, 140, 160), field.rect, 1, border_radius=4)

            txt = self.small_font.render(field.value, True, (235, 235, 235))
            self.screen.blit(txt, (field.rect.x + 8, field.rect.y + 7))

        for button in self.buttons.values():
            pygame.draw.rect(self.screen, (60, 100, 130), button.rect, border_radius=4)
            txt = self.small_font.render(button.label, True, (255, 255, 255))
            self.screen.blit(txt, (button.rect.x + 12, button.rect.y + 9))

        info_y = 520
        rows = [
            f"Track: {self.track_names[self.track_index]}",
            f"Steering: {self.latest_steering:+.3f}",
            f"Sensors: {[round(v, 3) for v in self.sensor_values]}",
            f"Alive: {self.agent.alive}",
            f"Survival s: {self.agent.time_alive:.2f}",
            f"Distance: {self.agent.distance_traveled:.2f}",
            f"Crash reason: {'collision' if not self.agent.alive else 'n/a'}",
            f"Status: {self.status_message}",
        ]
        for i, row in enumerate(rows):
            surf = self.small_font.render(row, True, (230, 230, 230))
            self.screen.blit(surf, (panel_x + 18, info_y + i * 22))

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            running = self._handle_events()
            self._step_simulation(dt)

            self.screen.fill((16, 16, 20))
            self._draw_world()
            self._draw_panel()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    TeachingApp().run()
