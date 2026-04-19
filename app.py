from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from policy_presets import PRESETS
from simulator_core import CarAgent, LinearPolicy, SensorArray, Simulator, clamp
from tracks import build_track_definition, track_names

WINDOW_WIDTH = 1360
WINDOW_HEIGHT = 800
PANEL_WIDTH = 430
STAGE_BG = (18, 24, 36)


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: str


@dataclass
class NumberField:
    key: str
    label: str
    value: str
    rect: pygame.Rect


class TeachingApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Neural Net Car Lab - Classroom Demo")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.demo_mode = False
        self.active_field: str | None = None
        self.status = "Ready"
        self.slow_motion = False
        self.single_step_requested = False
        self.running = False

        self.track_list = track_names()
        self.track_idx = 0
        self.track_def = build_track_definition(self.track_list[self.track_idx])
        self.preset_key = "decent"
        self.preset_note = PRESETS[self.preset_key].note

        self.fields = self._init_fields()
        self.buttons = self._init_buttons()

        self.sensor_values: list[float] = []
        self.latest_steering = 0.0

        self._apply_preset(self.preset_key)
        self._reset_simulation()

    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        return pygame.font.SysFont("arial", size + (4 if self.demo_mode else 0), bold=bold)

    def _init_fields(self) -> dict[str, NumberField]:
        x = WINDOW_WIDTH - PANEL_WIDTH + 20
        y0 = 120
        row_h = 47
        w = PANEL_WIDTH - 40
        h = 34

        specs = [
            ("sensor_angles", "Sensor Angles (csv)", "-60,-30,0,30,60"),
            ("weights", "Weights (csv)", "0.8,0.45,0,-0.45,-0.8"),
            ("bias", "Bias", "0.0"),
            ("sensor_max_range", "Sensor Range", "3.5"),
            ("speed", "Speed", "2.0"),
            ("max_turn_rate", "Max Turn Rate", "120.0"),
            ("policy_path", "Policy Path", "policy.json"),
        ]
        fields = {}
        for i, (key, label, value) in enumerate(specs):
            fields[key] = NumberField(key=key, label=label, value=value, rect=pygame.Rect(x, y0 + i * row_h, w, h))
        return fields

    def _init_buttons(self) -> list[Button]:
        x = WINDOW_WIDTH - PANEL_WIDTH + 20
        y = 470
        w = 125
        h = 36
        gap = 12
        btn = []

        def add(label: str, action: str, col: int, row: int) -> None:
            btn.append(Button(label, pygame.Rect(x + col * (w + gap), y + row * (h + gap), w, h), action))

        add("Run", "run", 0, 0)
        add("Pause", "pause", 1, 0)
        add("Reset", "reset", 2, 0)
        add("Apply", "apply", 0, 1)
        add("Load", "load", 1, 1)
        add("Save", "save", 2, 1)
        add("Track", "track", 0, 2)
        add("Slow-Mo", "slow", 1, 2)
        add("Step", "step", 2, 2)
        add("Demo UI", "demo", 0, 3)
        add("Bad", "preset_bad", 1, 3)
        add("Decent", "preset_decent", 2, 3)
        add("Good", "preset_good", 1, 4)

        return btn

    def _parse_csv(self, text: str) -> list[float]:
        return [float(p.strip()) for p in text.split(",") if p.strip()]

    def _validate_and_build(self) -> tuple[LinearPolicy, SensorArray, CarAgent]:
        angles = self._parse_csv(self.fields["sensor_angles"].value)
        weights = self._parse_csv(self.fields["weights"].value)
        if len(angles) == 0:
            raise ValueError("Need at least one sensor angle")
        if len(angles) != len(weights):
            raise ValueError("Weights count must match sensor count")

        bias = float(self.fields["bias"].value)
        speed = float(self.fields["speed"].value)
        max_turn_rate = float(self.fields["max_turn_rate"].value)
        sensor_max_range = float(self.fields["sensor_max_range"].value)

        if sensor_max_range <= 0.0:
            raise ValueError("Sensor range must be > 0")
        if speed <= 0.0:
            raise ValueError("Speed must be > 0")
        if max_turn_rate <= 0.0:
            raise ValueError("Max turn rate must be > 0")

        policy = LinearPolicy(sensor_angles_deg=angles, weights=weights, bias=bias)
        sensors = SensorArray(sensor_angles_deg=angles, max_range=sensor_max_range)
        spawn = self.track_def.spawn
        agent = CarAgent(
            position=spawn,
            heading_deg=self.track_def.heading_deg,
            speed=speed,
            max_turn_rate_deg=max_turn_rate,
        )
        return policy, sensors, agent

    def _reset_simulation(self) -> None:
        self.policy, self.sensor_array, self.agent = self._validate_and_build()
        self.simulator = Simulator(self.track_def.track, self.policy, self.agent, self.sensor_array)
        self.running = False
        self.sensor_values = self.sensor_array.read(self.agent.position, self.agent.heading_deg, self.track_def.track)
        self.latest_steering = self.policy.forward(self.sensor_values)

    def _apply_preset(self, key: str) -> None:
        preset = PRESETS[key]
        self.preset_key = key
        self.preset_note = preset.note
        self.fields["sensor_angles"].value = ",".join(str(v) for v in preset.sensor_angles_deg)
        self.fields["weights"].value = ",".join(str(v) for v in preset.weights)
        self.fields["bias"].value = str(preset.bias)
        self.status = f"Preset loaded: {preset.name}"

    def _do_action(self, action: str) -> None:
        try:
            if action == "run":
                self.running = True
                self.status = "Running"
            elif action == "pause":
                self.running = False
                self.status = "Paused"
            elif action == "reset":
                self._reset_simulation()
                self.status = "Simulation reset"
            elif action == "apply":
                self._reset_simulation()
                self.status = "Applied numeric controls"
            elif action == "save":
                self.policy.save_json(self.fields["policy_path"].value)
                self.status = "Policy saved"
            elif action == "load":
                loaded = LinearPolicy.load_json(self.fields["policy_path"].value)
                self.fields["sensor_angles"].value = ",".join(str(v) for v in loaded.sensor_angles_deg)
                self.fields["weights"].value = ",".join(str(v) for v in loaded.weights)
                self.fields["bias"].value = str(loaded.bias)
                self._reset_simulation()
                self.status = "Policy loaded"
            elif action == "track":
                self.track_idx = (self.track_idx + 1) % len(self.track_list)
                self.track_def = build_track_definition(self.track_list[self.track_idx])
                self._reset_simulation()
                self.status = f"Track: {self.track_def.name}"
            elif action == "slow":
                self.slow_motion = not self.slow_motion
                self.status = f"Slow motion: {self.slow_motion}"
            elif action == "step":
                self.single_step_requested = True
                self.status = "Single step"
            elif action == "demo":
                self.demo_mode = not self.demo_mode
                self.status = f"Demo mode: {self.demo_mode}"
            elif action.startswith("preset_"):
                key = action.split("_", 1)[1]
                self._apply_preset(key)
                self._reset_simulation()
        except Exception as exc:
            self.status = f"Validation error: {exc}"

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.active_field = None
                for k, f in self.fields.items():
                    if f.rect.collidepoint(event.pos):
                        self.active_field = k
                for b in self.buttons:
                    if b.rect.collidepoint(event.pos):
                        self._do_action(b.action)
            if event.type == pygame.KEYDOWN:
                if self.active_field:
                    field = self.fields[self.active_field]
                    if event.key == pygame.K_BACKSPACE:
                        field.value = field.value[:-1]
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.active_field = None
                    elif event.unicode and event.unicode.isprintable():
                        field.value += event.unicode
                else:
                    if event.key == pygame.K_SPACE:
                        self.running = not self.running
                    if event.key == pygame.K_n:
                        self.single_step_requested = True
        return True

    def _step(self, dt: float) -> None:
        effective_dt = dt * (0.2 if self.slow_motion else 1.0)
        should_step = self.running or self.single_step_requested
        if not should_step or not self.agent.alive:
            return
        self.single_step_requested = False

        self.sensor_values = self.sensor_array.read(self.agent.position, self.agent.heading_deg, self.track_def.track)
        self.latest_steering = self.policy.forward(self.sensor_values)
        self.agent.step(self.latest_steering, effective_dt)

        if self.simulator._is_colliding(self.agent.position):
            self.agent.alive = False
            self.running = False
            self.status = "Crash: collision"

    def _world_bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for seg in self.track_def.track.wall_segments for p in seg]
        ys = [p[1] for seg in self.track_def.track.wall_segments for p in seg]
        return min(xs), min(ys), max(xs), max(ys)

    def _to_screen(self, p: tuple[float, float]) -> tuple[int, int]:
        sx0, sy0, sx1, sy1 = 30, 30, WINDOW_WIDTH - PANEL_WIDTH - 30, WINDOW_HEIGHT - 30
        min_x, min_y, max_x, max_y = self._world_bounds()
        scale = min((sx1 - sx0) / max(max_x - min_x, 1e-6), (sy1 - sy0) / max(max_y - min_y, 1e-6))
        x = sx0 + (p[0] - min_x) * scale
        y = sy1 - (p[1] - min_y) * scale
        return int(x), int(y)

    def _draw_stage(self) -> None:
        stage_rect = pygame.Rect(12, 12, WINDOW_WIDTH - PANEL_WIDTH - 24, WINDOW_HEIGHT - 24)
        pygame.draw.rect(self.screen, STAGE_BG, stage_rect, border_radius=10)
        pygame.draw.rect(self.screen, (72, 102, 138), stage_rect, 2, border_radius=10)

        for a, b in self.track_def.track.wall_segments:
            pygame.draw.line(self.screen, (200, 220, 240), self._to_screen(a), self._to_screen(b), 4)

        if len(self.agent.path_trace) > 1:
            trail = [self._to_screen(p) for p in self.agent.path_trace]
            pygame.draw.lines(self.screen, (88, 170, 255), False, trail, 3)

        # Sensor rays with strong color coding.
        self.sensor_values = self.sensor_array.read(self.agent.position, self.agent.heading_deg, self.track_def.track)
        for angle_deg, value in zip(self.policy.sensor_angles_deg, self.sensor_values, strict=True):
            world_angle = math.radians(self.agent.heading_deg + angle_deg)
            d = self.sensor_array.max_range * (1 - value)
            end = (self.agent.position[0] + math.cos(world_angle) * d, self.agent.position[1] + math.sin(world_angle) * d)
            ray_color = (255, int(200 * (1 - value)), int(40 + 120 * (1 - value)))
            pygame.draw.line(self.screen, ray_color, self._to_screen(self.agent.position), self._to_screen(end), 2)

        # Vehicle shape (triangle + tail) instead of plain circle.
        h = math.radians(self.agent.heading_deg)
        nose = (self.agent.position[0] + math.cos(h) * 0.7, self.agent.position[1] + math.sin(h) * 0.7)
        left = (self.agent.position[0] + math.cos(h + 2.45) * 0.45, self.agent.position[1] + math.sin(h + 2.45) * 0.45)
        right = (self.agent.position[0] + math.cos(h - 2.45) * 0.45, self.agent.position[1] + math.sin(h - 2.45) * 0.45)
        tail = (self.agent.position[0] - math.cos(h) * 0.55, self.agent.position[1] - math.sin(h) * 0.55)
        pygame.draw.polygon(self.screen, (255, 220, 80), [self._to_screen(nose), self._to_screen(left), self._to_screen(tail), self._to_screen(right)])
        pygame.draw.line(self.screen, (255, 120, 80), self._to_screen(self.agent.position), self._to_screen(nose), 3)

        title = self._font(28, bold=True).render("Teaching Stage", True, (242, 246, 255))
        self.screen.blit(title, (28, 20))
        subtitle = self._font(18).render(f"Track: {self.track_def.name}  |  {self.track_def.description}", True, (200, 220, 238))
        self.screen.blit(subtitle, (28, 54))

    def _danger_labels(self) -> tuple[str, str, str]:
        angles = self.policy.sensor_angles_deg
        vals = self.sensor_values
        left_vals = [v for a, v in zip(angles, vals, strict=True) if a < -10]
        right_vals = [v for a, v in zip(angles, vals, strict=True) if a > 10]
        front_vals = [v for a, v in zip(angles, vals, strict=True) if -15 <= a <= 15]

        def level(vs: list[float]) -> str:
            if not vs:
                return "n/a"
            m = sum(vs) / len(vs)
            if m > 0.66:
                return "HIGH"
            if m > 0.33:
                return "MED"
            return "LOW"

        return level(left_vals), level(front_vals), level(right_vals)

    def _draw_gauge(self, x: int, y: int, w: int, label: str, value01: float, color: tuple[int, int, int]) -> None:
        label_s = self._font(16, bold=True).render(label, True, (235, 235, 245))
        self.screen.blit(label_s, (x, y))
        bar = pygame.Rect(x, y + 20, w, 16)
        pygame.draw.rect(self.screen, (55, 58, 70), bar, border_radius=5)
        fill = pygame.Rect(x, y + 20, int(w * clamp(value01, 0.0, 1.0)), 16)
        pygame.draw.rect(self.screen, color, fill, border_radius=5)

    def _draw_nn_overlay(self, x: int, y: int, w: int) -> None:
        pygame.draw.rect(self.screen, (32, 34, 44), pygame.Rect(x, y, w, 210), border_radius=8)
        pygame.draw.rect(self.screen, (90, 100, 124), pygame.Rect(x, y, w, 210), 1, border_radius=8)
        title = self._font(18, bold=True).render("Neural Network View (N -> 1)", True, (245, 245, 255))
        self.screen.blit(title, (x + 12, y + 8))

        in_x = x + 55
        out_x = x + w - 70
        top = y + 45
        spacing = min(28, 130 / max(len(self.sensor_values), 1))
        output_pos = (out_x, y + 110)

        steering_raw = self.policy.bias
        for i, (sv, wt) in enumerate(zip(self.sensor_values, self.policy.weights, strict=True)):
            py = int(top + i * spacing)
            contrib = sv * wt
            steering_raw += contrib
            c = (70, 180, 255) if contrib >= 0 else (255, 130, 110)
            thickness = 1 + int(min(6, abs(contrib) * 6))
            pygame.draw.line(self.screen, c, (in_x, py), output_pos, thickness)
            pygame.draw.circle(self.screen, (220, 220, 255), (in_x, py), 7)
            txt = self._font(12).render(f"s{i}:{sv:.2f}", True, (220, 230, 240))
            self.screen.blit(txt, (in_x - 48, py - 8))

        pygame.draw.circle(self.screen, (255, 236, 130), output_pos, 12)
        out_txt = self._font(12, bold=True).render(f"steer {self.latest_steering:+.2f}", True, (245, 245, 245))
        self.screen.blit(out_txt, (output_pos[0] - 35, output_pos[1] + 16))

        eq = self._font(13).render(f"raw = bias({self.policy.bias:+.2f}) + sum(sensor*weight)", True, (215, 215, 228))
        self.screen.blit(eq, (x + 12, y + 172))
        eq2 = self._font(13).render(f"raw={steering_raw:+.2f}  -> tanh(raw)={self.latest_steering:+.2f}", True, (215, 215, 228))
        self.screen.blit(eq2, (x + 12, y + 190))

    def _draw_panel(self) -> None:
        panel = pygame.Rect(WINDOW_WIDTH - PANEL_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, (22, 23, 30), panel)
        pygame.draw.line(self.screen, (68, 78, 102), (panel.x, 0), (panel.x, WINDOW_HEIGHT), 2)

        self.screen.blit(self._font(24, bold=True).render("Control + Explain", True, (245, 245, 250)), (panel.x + 20, 20))
        self.screen.blit(self._font(14).render(self.preset_note, True, (195, 205, 220)), (panel.x + 20, 56))

        for key, field in self.fields.items():
            label = self._font(13).render(field.label, True, (205, 215, 228))
            self.screen.blit(label, (field.rect.x, field.rect.y - 16))
            box_col = (90, 95, 122) if self.active_field == key else (62, 68, 86)
            pygame.draw.rect(self.screen, box_col, field.rect, border_radius=6)
            pygame.draw.rect(self.screen, (140, 145, 168), field.rect, 1, border_radius=6)
            val = self._font(14).render(field.value, True, (240, 240, 250))
            self.screen.blit(val, (field.rect.x + 8, field.rect.y + 8))

        for b in self.buttons:
            bg = (55, 110, 140)
            if b.action.startswith("preset_") and b.action.endswith(self.preset_key):
                bg = (75, 148, 100)
            if b.action == "slow" and self.slow_motion:
                bg = (132, 116, 70)
            pygame.draw.rect(self.screen, bg, b.rect, border_radius=6)
            txt = self._font(14, bold=True).render(b.label, True, (250, 250, 250))
            self.screen.blit(txt, (b.rect.x + 12, b.rect.y + 9))

        self._draw_nn_overlay(panel.x + 16, 650 - (70 if self.demo_mode else 0), PANEL_WIDTH - 32)

        steering_val = (self.latest_steering + 1.0) / 2.0
        self._draw_gauge(panel.x + 20, 612 - (70 if self.demo_mode else 0), PANEL_WIDTH - 40, "Steering", steering_val, (130, 210, 255))
        speed_norm = min(1.0, self.agent.speed / 4.0)
        self._draw_gauge(panel.x + 20, 566 - (70 if self.demo_mode else 0), PANEL_WIDTH - 40, "Speed", speed_norm, (150, 220, 150))

        left, front, right = self._danger_labels()
        warn = self._font(16, bold=True).render(f"Danger L/F/R: {left} | {front} | {right}", True, (255, 220, 180))
        self.screen.blit(warn, (panel.x + 20, 536 - (70 if self.demo_mode else 0)))

        info_lines = [
            f"Alive: {self.agent.alive}",
            f"Time: {self.agent.time_alive:.2f}s",
            f"Distance: {self.agent.distance_traveled:.2f}",
            f"Crash reason: {'collision' if not self.agent.alive else 'n/a'}",
            f"Mode: {'DEMO' if self.demo_mode else 'standard'}  |  Slow-mo: {self.slow_motion}",
            f"Status: {self.status}",
        ]
        y0 = 730 - (70 if self.demo_mode else 0)
        for i, line in enumerate(info_lines):
            self.screen.blit(self._font(14).render(line, True, (225, 225, 232)), (panel.x + 20, y0 + i * 18))

    def run(self) -> None:
        active = True
        while active:
            dt = self.clock.tick(60) / 1000.0
            active = self._handle_events()
            self._step(dt)

            self.screen.fill((10, 12, 16))
            self._draw_stage()
            self._draw_panel()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    TeachingApp().run()
