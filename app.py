"""Interactive teaching application for neural-network driving demos.

Usage:
    python app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from policy_presets import PRESETS
from scenarios import get_scenario, scenario_names
from simulator_core import CarAgent, NeuralPolicy, SensorArray, Simulator, VehicleDynamicsConfig, random_policy
from tracks import build_track_definition, track_names

WINDOW_SIZE = (1320, 780)
BG = (16, 20, 27)
PANEL = (26, 32, 42)
TEXT = (236, 241, 250)
MUTED = (157, 168, 188)
ACCENT = (102, 186, 255)
GREEN = (82, 212, 153)
RED = (255, 115, 115)


@dataclass
class TrainRules:
    cycles: int = 20
    population: int = 18
    elite_ratio: float = 0.3
    mutation_rate: float = 0.35
    mutation_strength: float = 0.35
    max_steps: int = 260
    dt: float = 0.06


@dataclass
class SimConfig:
    sensor_max_range: float = 3.8
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
        self.track_def = build_track_definition(self.track_names[self.track_idx])

        self.config = SimConfig()
        self.training_rules = TrainRules()
        self.sensor_spread_deg = 120.0
        self.hidden_layers: list[int] = [4]
        self.enabled_actions = ["steering"]

        self.scenario_keys = scenario_names()
        self.scenario_idx = 0
        self.scenario = get_scenario(self.scenario_keys[self.scenario_idx])
        self.dynamics = VehicleDynamicsConfig(**self.scenario.dynamics.__dict__)

        self.policy = PRESETS[self.scenario.policy_preset].build_policy()
        self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)
        self.agent = self._fresh_agent()
        self.simulator = self._build_simulator()

        self.running = True
        self.paused = False
        self.step_once = False
        self.continuous_training = False
        self.training_requested_cycles = 0
        self.training_population: list[NeuralPolicy] | None = None
        self.total_training_cycles = 0

        self.status = "Ready"
        self.buttons: list[UIButton] = []
        self.context_buttons: list[UIButton] = []
        self.context_menu_open = False

        self.last_fitness = 0.0
        self.attempts_total = 0
        self.attempt_history: list[AttemptRecord] = []
        self.max_attempt_history = 150
        self.restart_countdown = 0.0
        self.lap_count = 0
        self.start_line_cooldown = 0.0

        self.options_open = False
        self.active_input: str | None = None
        self.input_fields: list[InputField] = []

    def _load_remote_asset(self, url: str, local_name: str) -> pygame.Surface | None:
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)
        path = assets_dir / local_name
        if not path.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    path.write_bytes(resp.read())
            except Exception:
                return None
        try:
            return pygame.image.load(str(path)).convert_alpha()
        except Exception:
            return None

    def _fresh_agent(self) -> CarAgent:
        return CarAgent(
            position=self.track_def.spawn,
            heading_deg=self.track_def.heading_deg,
            max_turn_rate_deg=self.config.max_turn_rate_deg,
            dynamics=self.dynamics,
        )

    def _build_simulator(self) -> Simulator:
        return Simulator(self.track_def.track, self.policy, self.agent, self.sensor_array)

    def _fitness(self, simulator: Simulator) -> float:
        alive_bonus = 2.0 if simulator.agent.alive else 0.0
        speed_bonus = simulator.agent.speed * 0.8
        return simulator.agent.distance_traveled + simulator.agent.time_alive + alive_bonus + speed_bonus

    def _reseed_training_population(self) -> None:
        base = self.policy
        self.training_population = [base.clone() for _ in range(self.training_rules.population)]
        for indiv in self.training_population[1:]:
            indiv.mutate(self.training_rules.mutation_rate, self.training_rules.mutation_strength)

    def _evaluate_policy(self, policy: NeuralPolicy) -> float:
        sensors = SensorArray(list(policy.sensor_angles_deg), self.config.sensor_max_range)
        agent = CarAgent(
            position=self.track_def.spawn,
            heading_deg=self.track_def.heading_deg,
            max_turn_rate_deg=self.config.max_turn_rate_deg,
            dynamics=self.dynamics,
        )
        sim = Simulator(self.track_def.track, policy, agent, sensors)
        sim.run(max_steps=self.training_rules.max_steps, dt=self.training_rules.dt)
        return self._fitness(sim)

    def _train_one_cycle(self) -> None:
        if self.training_requested_cycles <= 0:
            return
        if self.training_population is None or len(self.training_population) != self.training_rules.population:
            self._reseed_training_population()
        assert self.training_population is not None

        scored = [(self._evaluate_policy(candidate), candidate) for candidate in self.training_population]
        scored.sort(key=lambda item: item[0], reverse=True)

        elite_count = max(2, int(self.training_rules.population * self.training_rules.elite_ratio))
        elites = [candidate for _, candidate in scored[:elite_count]]

        self.policy = elites[0].clone()
        self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)

        next_generation = [elite.clone() for elite in elites]
        while len(next_generation) < self.training_rules.population:
            parent = random.choice(elites).clone()
            parent.mutate(self.training_rules.mutation_rate, self.training_rules.mutation_strength)
            next_generation.append(parent)

        self.training_population = next_generation
        self.training_requested_cycles -= 1
        self.total_training_cycles += 1
        self.status = f"Training cycle complete. best={scored[0][0]:.2f}"

    def _archive_attempt(self, cause: str) -> None:
        self.attempts_total += 1
        self.attempt_history.append(AttemptRecord(list(self.agent.path_trace), cause, self.total_training_cycles))
        if len(self.attempt_history) > self.max_attempt_history:
            self.attempt_history = self.attempt_history[-self.max_attempt_history :]

    def _reset_episode(self, caused_by: str = "manual") -> None:
        self._archive_attempt(caused_by)
        self.agent = self._fresh_agent()
        self.simulator = self._build_simulator()
        self.last_fitness = 0.0
        self.start_line_cooldown = 0.0
        self.restart_countdown = 0.0

    def _safe_set_float(self, value: str, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, float(value)))

    def _safe_set_int(self, value: str, min_v: int, max_v: int) -> int:
        return max(min_v, min(max_v, int(float(value))))

    def _parse_hidden_layers(self, text: str) -> list[int]:
        if not text.strip():
            return []
        parts = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        layers = [max(1, int(part)) for part in parts]
        return layers[:4]

    def _build_policy_with_current_config(self) -> None:
        sensor_angles = list(self.sensor_array.sensor_angles_deg)
        if "steering" not in self.enabled_actions:
            self.enabled_actions.insert(0, "steering")
        self.policy = random_policy(
            sensor_angles_deg=sensor_angles,
            action_names=list(self.enabled_actions),
            hidden_layers=list(self.hidden_layers),
        )
        self.status = "Policy rebuilt with new architecture"
        self._reseed_training_population()
        self._reset_episode("config")

    def _set_sensor_layout(self, sensor_count: int, spread_deg: float) -> None:
        self.sensor_spread_deg = spread_deg
        if sensor_count == 1:
            angles = [0.0]
        else:
            step = spread_deg / (sensor_count - 1)
            angles = [-(spread_deg / 2.0) + step * idx for idx in range(sensor_count)]
        self.sensor_array = SensorArray(angles, self.config.sensor_max_range)
        self._build_policy_with_current_config()

    def _apply_scenario(self, scenario_key: str) -> None:
        self.scenario = get_scenario(scenario_key)
        if self.scenario.track_name in self.track_names:
            self.track_idx = self.track_names.index(self.scenario.track_name)
            self.track_def = build_track_definition(self.track_names[self.track_idx])

        self.policy = PRESETS[self.scenario.policy_preset].build_policy()
        self.enabled_actions = list(self.policy.action_names)
        self.hidden_layers = self.policy.layer_sizes[1:-1]
        self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)
        self.dynamics = VehicleDynamicsConfig(**self.scenario.dynamics.__dict__)
        self._reseed_training_population()
        self._reset_episode("scenario")
        self.status = f"Scenario loaded: {self.scenario.label}"

    def _apply_options(self) -> None:
        values = {field.key: field.value.strip() for field in self.input_fields}
        try:
            self.config.max_turn_rate_deg = self._safe_set_float(values["turn_rate"], 20.0, 320.0)
            self.config.sensor_max_range = self._safe_set_float(values["sensor_range"], 0.8, 10.0)
            self.config.auto_restart_delay = self._safe_set_float(values["restart_delay"], 0.2, 20.0)
            self.training_rules.cycles = self._safe_set_int(values["cycles"], 1, 200)
            self.training_rules.population = self._safe_set_int(values["population"], 4, 96)
            self.training_rules.mutation_strength = self._safe_set_float(values["mutation_strength"], 0.01, 1.5)
            self.training_rules.max_steps = self._safe_set_int(values["max_steps"], 20, 4000)
            self.training_rules.mutation_rate = self._safe_set_float(values["mutation_rate"], 0.01, 1.0)
            sensors = self._safe_set_int(values["sensor_count"], 3, 14)
            spread = self._safe_set_float(values["sensor_spread"], 30.0, 175.0)

            self.hidden_layers = self._parse_hidden_layers(values["hidden_layers"])
            enable_throttle = self._safe_set_int(values["enable_throttle"], 0, 1) == 1
            enable_brake = self._safe_set_int(values["enable_brake"], 0, 1) == 1
            self.enabled_actions = ["steering"]
            if enable_throttle:
                self.enabled_actions.append("throttle")
            if enable_brake:
                self.enabled_actions.append("brake")

            mode_dynamic = self._safe_set_int(values["speed_dynamic"], 0, 1) == 1
            self.dynamics.speed_mode = "dynamic" if mode_dynamic else "constant"
            self.dynamics.constant_speed = self._safe_set_float(values["constant_speed"], 0.1, 7.0)
            self.dynamics.max_speed = self._safe_set_float(values["max_speed"], 0.5, 15.0)
            self.dynamics.acceleration_rate = self._safe_set_float(values["accel_rate"], 0.0, 10.0)
            self.dynamics.brake_rate = self._safe_set_float(values["brake_rate"], 0.0, 12.0)
            self.dynamics.drag = self._safe_set_float(values["drag"], 0.0, 3.0)

            self._set_sensor_layout(sensors, spread)
            self.sensor_array.max_range = self.config.sensor_max_range
            self.options_open = False
            self.input_fields = []
            self.status = "Options applied"
        except (ValueError, KeyError):
            self.status = "Invalid setting value"

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
                    self._reset_episode("track")
                elif event.key == pygame.K_g:
                    self.training_requested_cycles += max(1, self.training_rules.cycles)
                elif event.key == pygame.K_c:
                    self.continuous_training = not self.continuous_training
                elif event.key == pygame.K_o:
                    self.options_open = not self.options_open
                    if self.options_open:
                        self.input_fields = []
                elif event.key == pygame.K_n:
                    self.step_once = True
                elif event.key == pygame.K_m:
                    self.scenario_idx = (self.scenario_idx + 1) % len(self.scenario_keys)
                    self._apply_scenario(self.scenario_keys[self.scenario_idx])
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    self.context_menu_open = True
                    self._build_context_menu(event.pos)
                elif event.button == 1:
                    if self.options_open and self._handle_options_click(event.pos):
                        continue
                    if self.context_menu_open and self._handle_context_click(event.pos):
                        continue
                    self._handle_button_click(event.pos)

    def _handle_text_input(self, event: pygame.event.Event) -> None:
        if self.active_input is None:
            if event.key == pygame.K_ESCAPE:
                self.options_open = False
            return
        field = next((f for f in self.input_fields if f.key == self.active_input), None)
        if field is None:
            return
        if event.key == pygame.K_RETURN:
            self._apply_options()
        elif event.key == pygame.K_ESCAPE:
            self.options_open = False
        elif event.key == pygame.K_BACKSPACE:
            field.value = field.value[:-1]
        elif event.unicode and event.unicode in "0123456789.-,":
            field.value += event.unicode

    def _build_context_menu(self, pos: tuple[int, int]) -> None:
        x, y = pos
        menu = pygame.Rect(x, y, 200, 126)
        self.context_buttons = [
            UIButton("Open Options", "options", pygame.Rect(menu.x + 8, menu.y + 8, menu.w - 16, 24)),
            UIButton("Reset Attempt", "reset", pygame.Rect(menu.x + 8, menu.y + 36, menu.w - 16, 24)),
            UIButton("Queue Train", "train", pygame.Rect(menu.x + 8, menu.y + 64, menu.w - 16, 24)),
            UIButton("Next Scenario", "scenario", pygame.Rect(menu.x + 8, menu.y + 92, menu.w - 16, 24)),
        ]

    def _handle_context_click(self, pos: tuple[int, int]) -> bool:
        for btn in self.context_buttons:
            if btn.rect.collidepoint(pos):
                if btn.action == "options":
                    self.options_open = True
                    self.input_fields = []
                elif btn.action == "reset":
                    self._reset_episode("manual")
                elif btn.action == "train":
                    self.training_requested_cycles += max(1, self.training_rules.cycles)
                elif btn.action == "scenario":
                    self.scenario_idx = (self.scenario_idx + 1) % len(self.scenario_keys)
                    self._apply_scenario(self.scenario_keys[self.scenario_idx])
                self.context_menu_open = False
                return True
        self.context_menu_open = False
        return False

    def _handle_button_click(self, pos: tuple[int, int]) -> None:
        for btn in self.buttons:
            if btn.rect.collidepoint(pos):
                if btn.action == "pause":
                    self.paused = not self.paused
                elif btn.action == "reset":
                    self._reset_episode("manual")
                elif btn.action == "train":
                    self.training_requested_cycles += max(1, self.training_rules.cycles)
                elif btn.action == "options":
                    self.options_open = True
                    self.input_fields = []
                elif btn.action == "continuous":
                    self.continuous_training = not self.continuous_training
                elif btn.action == "scenario":
                    self.scenario_idx = (self.scenario_idx + 1) % len(self.scenario_keys)
                    self._apply_scenario(self.scenario_keys[self.scenario_idx])
                break

    def _handle_options_click(self, pos: tuple[int, int]) -> bool:
        for field in self.input_fields:
            if field.rect.collidepoint(pos):
                self.active_input = field.key
                return True
        for btn in self.buttons:
            if btn.action in {"apply_options", "close_options"} and btn.rect.collidepoint(pos):
                if btn.action == "apply_options":
                    self._apply_options()
                else:
                    self.options_open = False
                return True
        return False

    def _tick_simulation(self, dt: float) -> None:
        if not self.agent.alive:
            self.restart_countdown -= dt
            if self.restart_countdown <= 0:
                self._reset_episode()
            return

        prev_position = self.agent.position
        self.simulator.step(dt)
        self.last_fitness = self._fitness(self.simulator)

        self.start_line_cooldown = max(0.0, self.start_line_cooldown - dt)
        if self.start_line_cooldown <= 0.0 and self._segments_intersect(
            prev_position,
            self.agent.position,
            self.track_def.start_line[0],
            self.track_def.start_line[1],
        ):
            self.lap_count += 1
            self.start_line_cooldown = 1.2

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
        track_rect = pygame.Rect(14, 70, 920, 686)
        hud_rect = pygame.Rect(14, 12, 1292, 50)
        side_rect = pygame.Rect(946, 70, 360, 686)

        pygame.draw.rect(self.screen, PANEL, hud_rect, border_radius=12)
        pygame.draw.rect(self.screen, (12, 16, 22), track_rect, border_radius=12)
        pygame.draw.rect(self.screen, PANEL, side_rect, border_radius=12)

        self._draw_track(track_rect)
        self._draw_hud(hud_rect)
        self._draw_side(side_rect)
        pygame.display.flip()

    def _draw_hud(self, rect: pygame.Rect) -> None:
        text = (
            f"Track {self.track_def.name}  |  Scenario {self.scenario.label}  |  Attempts {self.attempts_total}  |  "
            f"Laps {self.lap_count}  |  Fitness {self.last_fitness:.2f}  |  Training {self.total_training_cycles}"
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
        snapshot = self.simulator.latest_snapshot

        for line in self._wrap_text(f"Status: {self.status}", rect.width - 28):
            self.screen.blit(self.font.render(line, True, MUTED), (x, y))
            y += 20
        y += 6

        lines = [
            f"Alive: {self.agent.alive}",
            f"Speed mode: {self.dynamics.speed_mode}",
            f"Speed: {self.agent.speed:.2f}",
            f"Acceleration: {self.agent.last_acceleration:.2f}",
            f"Braking: {self.agent.last_brake_force:.2f}",
            f"Enabled outputs: {', '.join(self.enabled_actions)}",
            f"Hidden layers: {self.hidden_layers or 'none'}",
            f"Queued training: {self.training_requested_cycles}",
        ]
        for t in lines:
            self.screen.blit(self.small.render(t, True, TEXT), (x, y))
            y += 19

        if snapshot is not None:
            y += 8
            self.screen.blit(self.small.render("Final action outputs", True, ACCENT), (x, y))
            y += 18
            for action_name in self.enabled_actions:
                self.screen.blit(self.small.render(f"{action_name}: {snapshot.actions.get(action_name, 0.0):+.3f}", True, TEXT), (x, y))
                y += 18

            y += 6
            self.screen.blit(self.small.render("Raw network outputs", True, ACCENT), (x, y))
            y += 18
            for action_name in self.policy.action_names:
                self.screen.blit(self.small.render(f"{action_name}: {snapshot.raw_outputs.get(action_name, 0.0):+.3f}", True, TEXT), (x, y))
                y += 18

            y += 6
            self.screen.blit(self.small.render("Layer activations", True, ACCENT), (x, y))
            y += 18
            for idx, layer in enumerate(snapshot.layer_activations):
                preview = ", ".join(f"{v:+.2f}" for v in layer[:6])
                suffix = "..." if len(layer) > 6 else ""
                self.screen.blit(self.small.render(f"L{idx}: [{preview}{suffix}]", True, MUTED), (x, y))
                y += 16
                if y > rect.bottom - 220:
                    break

            y += 6
            self.screen.blit(self.small.render("Sensor readings", True, ACCENT), (x, y))
            y += 18
            for idx, (reading, hit) in enumerate(zip(snapshot.sensor_readings, snapshot.sensor_hit_distances, strict=True)):
                self.screen.blit(
                    self.small.render(f"S{idx}: {reading:.2f} hit={'none' if hit is None else f'{hit:.2f}'}", True, MUTED),
                    (x, y),
                )
                y += 16
                if y > rect.bottom - 140:
                    break

        self.buttons = []
        by = rect.bottom - 122
        self._draw_button("Pause/Run", "pause", x, by, 110)
        self._draw_button("Restart", "reset", x + 116, by, 110)
        self._draw_button("Train", "train", x + 232, by, 110)
        by += 34
        self._draw_button("Continuous", "continuous", x, by, 110)
        self._draw_button("Scenario", "scenario", x + 116, by, 110)
        self._draw_button("Options", "options", x + 232, by, 110)
        self.screen.blit(self.small.render("Keys: O options, G train, C continuous, M scenario", True, MUTED), (x, rect.bottom - 20))

    def _draw_button(self, label: str, action: str, x: int, y: int, w: int) -> None:
        rect = pygame.Rect(x, y, w, 30)
        self.buttons.append(UIButton(label, action, rect))
        active = action == "continuous" and self.continuous_training
        color = (48, 115, 84) if active else (60, 74, 95)
        pygame.draw.rect(self.screen, color, rect, border_radius=7)
        pygame.draw.rect(self.screen, (115, 131, 159), rect, 1, border_radius=7)
        self.screen.blit(self.small.render(label, True, TEXT), (x + 10, y + 8))

    def _draw_context_menu(self) -> None:
        if not self.context_buttons:
            return
        bounds = self.context_buttons[0].rect.unionall([b.rect for b in self.context_buttons])
        bg = bounds.inflate(8, 8)
        pygame.draw.rect(self.screen, (18, 24, 33), bg, border_radius=6)
        pygame.draw.rect(self.screen, (95, 108, 133), bg, 1, border_radius=6)
        for btn in self.context_buttons:
            pygame.draw.rect(self.screen, (53, 65, 83), btn.rect, border_radius=4)
            self.screen.blit(self.small.render(btn.label, True, TEXT), (btn.rect.x + 8, btn.rect.y + 5))

    def _draw_options_modal(self) -> None:
        modal = pygame.Rect(170, 65, 980, 650)
        pygame.draw.rect(self.screen, (10, 14, 20), modal, border_radius=12)
        pygame.draw.rect(self.screen, (114, 129, 157), modal, 1, border_radius=12)
        self.screen.blit(self.big.render("Options", True, TEXT), (modal.x + 20, modal.y + 16))

        field_defs = [
            ("turn_rate", "Turn rate deg/s", f"{self.config.max_turn_rate_deg:.1f}"),
            ("sensor_range", "Sensor range", f"{self.config.sensor_max_range:.2f}"),
            ("sensor_count", "Sensor count", str(self.policy.sensor_count)),
            ("sensor_spread", "Sensor spread deg", f"{self.sensor_spread_deg:.0f}"),
            ("hidden_layers", "Hidden layers (csv)", ",".join(str(v) for v in self.hidden_layers)),
            ("enable_throttle", "Enable throttle (0/1)", "1" if "throttle" in self.enabled_actions else "0"),
            ("enable_brake", "Enable brake (0/1)", "1" if "brake" in self.enabled_actions else "0"),
            ("speed_dynamic", "Dynamic speed mode (0/1)", "1" if self.dynamics.speed_mode == "dynamic" else "0"),
            ("constant_speed", "Constant speed", f"{self.dynamics.constant_speed:.2f}"),
            ("max_speed", "Max speed", f"{self.dynamics.max_speed:.2f}"),
            ("accel_rate", "Acceleration rate", f"{self.dynamics.acceleration_rate:.2f}"),
            ("brake_rate", "Brake rate", f"{self.dynamics.brake_rate:.2f}"),
            ("drag", "Drag", f"{self.dynamics.drag:.2f}"),
            ("restart_delay", "Auto restart sec", f"{self.config.auto_restart_delay:.1f}"),
            ("cycles", "Training cycles", str(self.training_rules.cycles)),
            ("population", "Population", str(self.training_rules.population)),
            ("mutation_rate", "Mutation rate", f"{self.training_rules.mutation_rate:.2f}"),
            ("mutation_strength", "Mutation strength", f"{self.training_rules.mutation_strength:.2f}"),
            ("max_steps", "Max steps", str(self.training_rules.max_steps)),
        ]

        if not self.input_fields:
            self.input_fields = []
            for i, (k, l, v) in enumerate(field_defs):
                row, col = divmod(i, 2)
                rect = pygame.Rect(modal.x + 24 + col * 468, modal.y + 70 + row * 52, 430, 34)
                self.input_fields.append(InputField(k, l, v, rect))
            self.active_input = self.input_fields[0].key

        for field in self.input_fields:
            self.screen.blit(self.small.render(field.label, True, MUTED), (field.rect.x, field.rect.y - 16))
            active = field.key == self.active_input
            pygame.draw.rect(self.screen, (27, 35, 48), field.rect, border_radius=6)
            pygame.draw.rect(self.screen, ACCENT if active else (95, 108, 133), field.rect, 1, border_radius=6)
            self.screen.blit(self.font.render(field.value, True, TEXT), (field.rect.x + 8, field.rect.y + 8))

        self.buttons.append(UIButton("Apply", "apply_options", pygame.Rect(modal.right - 230, modal.bottom - 52, 95, 34)))
        self.buttons.append(UIButton("Close", "close_options", pygame.Rect(modal.right - 122, modal.bottom - 52, 95, 34)))
        for b in self.buttons[-2:]:
            pygame.draw.rect(self.screen, (57, 70, 90), b.rect, border_radius=7)
            self.screen.blit(self.small.render(b.label, True, TEXT), (b.rect.x + 25, b.rect.y + 9))

    def _wrap_text(self, text: str, max_width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if self.font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines


def main() -> None:
    TeachingApp().run()


if __name__ == "__main__":
    main()
