"""Interactive teaching application for neural-network driving demos.

Usage:
    python app.py
"""

from __future__ import annotations

import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pygame

from policy_presets import PRESETS
from simulator_core import CarAgent, NeuralPolicy, SensorArray, Simulator
from tracks import TrackDefinition, build_track_definition, track_names

WINDOW_SIZE = (1280, 760)
BG = (16, 20, 27)
PANEL = (26, 32, 42)
TEXT = (236, 241, 250)
MUTED = (157, 168, 188)
ACCENT = (102, 186, 255)
GREEN = (82, 212, 153)
RED = (255, 115, 115)

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ASSET_SOURCES = {
    "car.png": "",
    "road.png": "",
}
TEXTURE_INPUT_KEYS = {
    "car_texture_path": "car",
    "road_texture_path": "road",
    "wall_texture_path": "wall",
    "panel_texture_path": "panel",
}

@dataclass
class TrainRules:
    cycles: int = 20
    population: int = 14
    elite_ratio: float = 0.3
    mutation_rate: float = 0.35
    mutation_strength: float = 0.35
    max_steps: int = 260
    dt: float = 0.06


@dataclass
class SimConfig:
    sensor_max_range: float = 3.5
    speed: float = 2.0
    max_turn_rate_deg: float = 120.0
    dt: float = 0.05
    auto_restart_delay: float = 3.0
    step_multiplier: float = 0.25


@dataclass
class UIButton:
    label: str
    action: str
    rect: pygame.Rect


@dataclass
class AttemptRecord:
    path: list[tuple[float, float]]
    cause: str
    cycle: int


@dataclass
class InputField:
    key: str
    label: str
    value: str
    rect: pygame.Rect


@dataclass
class ToggleField:
    key: str
    label: str
    value: bool
    rect: pygame.Rect


@dataclass
class StatChip:
    key: str
    label: str
    value: str
    rect: pygame.Rect


class TeachingApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("NAIT Neural Driving Trainer")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("inter,arial", 18)
        self.small = pygame.font.SysFont("inter,arial", 15)
        self.big = pygame.font.SysFont("inter,arial", 24)

        self.car_sprite: pygame.Surface | None = None
        self.road_texture: pygame.Surface | None = None
        self.wall_texture: pygame.Surface | None = None
        self.panel_texture: pygame.Surface | None = None
        self._build_embedded_assets()

        self.track_names = track_names()
        self.track_idx = 0
        self.track_def = build_track_definition(self.track_names[self.track_idx])

        self.policy = self._policy_from_preset("decent")
        self.policy.accel_weights = None
        self.policy.brake_weights = None
        self.training_rules = TrainRules()
        self.config = SimConfig()
        self.sensor_spread_deg = 120.0
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
        self.best_fitness_seen = float("-inf")
        self.learning_rate = 0.25

        self.status = "Ready"
        self.buttons: list[UIButton] = []
        self.context_buttons: list[UIButton] = []
        self.context_menu_open = False

        self.last_sensor_values = [0.0] * self.policy.sensor_count
        self.last_sensor_hits: list[float | None] = [None] * self.policy.sensor_count
        self.last_steering = 0.0
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
        self.toggle_fields: list[ToggleField] = []
        self.stat_chips: list[StatChip] = []
        self.stat_help_key: str | None = None
        self.nn_activations: dict[str, list[float]] = {"sensor": [], "hidden": []}
        self.nn_outputs: dict[str, float] = {"steering": 0.0, "accel": 0.0, "brake": 0.0}
        self.phase = "sense"
        self.section_help_key: str | None = None
        self.nn_hitboxes: list[tuple[pygame.Rect, str, dict[str, str]]] = []
        self.nn_detail: tuple[str, dict[str, str]] | None = None

    def _build_embedded_assets(self) -> None:
        road = pygame.Surface((192, 192), pygame.SRCALPHA)
        for y in range(192):
            for x in range(192):
                noise = ((x * 17 + y * 29) % 23) - 11
                c = max(35, min(70, 52 + noise))
                road.set_at((x, y), (c, c, c, 255))
        for y in range(0, 192, 24):
            for x in range(0, 192, 24):
                for dy in range(4):
                    for dx in range(4):
                        road.set_at((x + dx, y + dy), (72, 72, 72, 255))

        wall = pygame.Surface((192, 192), pygame.SRCALPHA)
        for y in range(192):
            for x in range(192):
                mortar = (x % 32 in (0, 1, 2)) or (y % 16 in (0, 1))
                base = 130 if mortar else 110 + ((x // 7 + y // 5) % 2) * 12
                wall.set_at((x, y), (base, base, base + 6, 255))

        car = pygame.Surface((96, 160), pygame.SRCALPHA)
        cx, cy = 48, 80
        for y in range(160):
            for x in range(96):
                nx = (x - cx) / 32
                ny = (y - cy) / 66
                if nx * nx + ny * ny <= 1.0:
                    car.set_at((x, y), (42, 181, 125, 255))
        for y in range(48, 76):
            for x in range(30, 66):
                if ((x - 48) / 20) ** 2 + ((y - 62) / 14) ** 2 <= 1.0:
                    car.set_at((x, y), (145, 195, 235, 235))
        for y in range(90, 114):
            for x in range(32, 64):
                if ((x - 48) / 18) ** 2 + ((y - 102) / 12) ** 2 <= 1.0:
                    car.set_at((x, y), (125, 175, 215, 235))
        for y0 in (44, 116):
            for x0 in (18, 70):
                for y in range(y0, y0 + 22):
                    for x in range(x0, x0 + 10):
                        car.set_at((x, y), (25, 25, 25, 255))

        self.road_texture = road
        self.wall_texture = wall
        self.car_sprite = car

    def _blit_segment_texture(
        self,
        texture: pygame.Surface | None,
        start: tuple[int, int],
        end: tuple[int, int],
        width: int,
    ) -> None:
        if texture is None:
            return
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(1, int(math.hypot(dx, dy)))
        angle = -math.degrees(math.atan2(dy, dx))
        strip = pygame.transform.smoothscale(texture, (length, width))
        strip = pygame.transform.rotate(strip, angle)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        self.screen.blit(strip, strip.get_rect(center=mid))


    def _download_asset(self, filename: str, url: str) -> Path | None:
        if not url:
            return None
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        target = ASSET_DIR / filename
        if target.exists():
            return target
        try:
            urllib.request.urlretrieve(url, target)
            return target
        except Exception:
            return None

    def _load_art_assets(self) -> None:
        loaded: dict[str, pygame.Surface] = {}
        for filename, url in ASSET_SOURCES.items():
            path = self._download_asset(filename, url)
            if path is None:
                continue
            try:
                loaded[filename] = pygame.image.load(path.as_posix()).convert_alpha()
            except Exception:
                continue

        if "car.png" in loaded:
            self.car_sprite = loaded["car.png"]
        if "road.png" in loaded:
            self.road_texture = loaded["road.png"]
            self.wall_texture = self.road_texture.copy()
            self.wall_texture.fill((120, 120, 130, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self.panel_texture = self.road_texture.copy()
            self.panel_texture.fill((70, 80, 100, 255), special_flags=pygame.BLEND_RGBA_MULT)

    def _set_texture_from_file(self, texture_kind: str, source_path: str) -> bool:
        path = Path(source_path).expanduser()
        if not path.exists() or not path.is_file():
            return False
        try:
            surface = pygame.image.load(path.as_posix()).convert_alpha()
        except Exception:
            return False

        if texture_kind == "car":
            self.car_sprite = surface
            return True
        if texture_kind == "road":
            self.road_texture = surface
            return True
        if texture_kind == "wall":
            self.wall_texture = surface
            return True
        if texture_kind == "panel":
            self.panel_texture = surface
            return True
        return False

    def _blit_segment_texture(
        self,
        texture: pygame.Surface | None,
        start: tuple[int, int],
        end: tuple[int, int],
        width: int,
    ) -> None:
        if texture is None:
            return
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(1, int(math.hypot(dx, dy)))
        angle = -math.degrees(math.atan2(dy, dx))
        strip = pygame.transform.smoothscale(texture, (length, width))
        strip = pygame.transform.rotate(strip, angle)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        self.screen.blit(strip, strip.get_rect(center=mid))

    def _policy_from_preset(self, preset_name: str) -> NeuralPolicy:
        preset = PRESETS[preset_name]
        sensor_angles = list(preset.sensor_angles_deg)
        hidden_count = 6
        return NeuralPolicy(
            sensor_angles_deg=sensor_angles,
            hidden_weights=[
                [random.uniform(-1.0, 1.0) for _ in sensor_angles]
                for _ in range(hidden_count)
            ],
            hidden_biases=[random.uniform(-0.5, 0.5) for _ in range(hidden_count)],
            steering_weights=[random.uniform(-1.0, 1.0) for _ in range(hidden_count)],
            steering_bias=float(preset.bias),
            accel_weights=[random.uniform(-0.7, 0.7) for _ in range(hidden_count)],
            brake_weights=[random.uniform(-0.7, 0.7) for _ in range(hidden_count)],
        )

    def _fresh_agent(self) -> CarAgent:
        return CarAgent(self.track_def.spawn, self.track_def.heading_deg, self.config.speed, self.config.max_turn_rate_deg)

    def _build_simulator(self) -> Simulator:
        return Simulator(self.track_def.track, self.policy, self.agent, self.sensor_array)

    def _fitness(self, simulator: Simulator) -> float:
        return simulator.agent.distance_traveled + simulator.agent.time_alive + (1.0 if simulator.agent.alive else 0.0)

    def _clone_policy(self, source: NeuralPolicy) -> NeuralPolicy:
        return NeuralPolicy(
            sensor_angles_deg=list(source.sensor_angles_deg),
            hidden_weights=[list(row) for row in source.hidden_weights],
            hidden_biases=list(source.hidden_biases),
            steering_weights=list(source.steering_weights),
            steering_bias=source.steering_bias,
            accel_weights=None if source.accel_weights is None else list(source.accel_weights),
            accel_bias=source.accel_bias,
            brake_weights=None if source.brake_weights is None else list(source.brake_weights),
            brake_bias=source.brake_bias,
        )

    def _reseed_training_population(self) -> None:
        base = self.policy
        self.training_population = [
            self._clone_policy(base)
            for _ in range(self.training_rules.population)
        ]
        for indiv in self.training_population[1:]:
            self._mutate(indiv)

    def _mutate_list(self, values: list[float]) -> None:
        for idx, value in enumerate(values):
            if random.random() < self.training_rules.mutation_rate:
                values[idx] = value + random.uniform(-self.training_rules.mutation_strength, self.training_rules.mutation_strength)

    def _mutate(self, policy: NeuralPolicy) -> None:
        for row in policy.hidden_weights:
            self._mutate_list(row)
        self._mutate_list(policy.hidden_biases)
        self._mutate_list(policy.steering_weights)
        if policy.accel_weights is not None:
            self._mutate_list(policy.accel_weights)
        if policy.brake_weights is not None:
            self._mutate_list(policy.brake_weights)
        if random.random() < self.training_rules.mutation_rate:
            policy.steering_bias += random.uniform(-self.training_rules.mutation_strength, self.training_rules.mutation_strength)

    def _evaluate_policy(self, policy: NeuralPolicy) -> float:
        sensors = SensorArray(list(policy.sensor_angles_deg), self.config.sensor_max_range)
        agent = CarAgent(self.track_def.spawn, self.track_def.heading_deg, self.config.speed, self.config.max_turn_rate_deg)
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

        self.policy = self._clone_policy(elites[0])
        self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)

        next_generation = [self._clone_policy(e) for e in elites]
        while len(next_generation) < self.training_rules.population:
            parent = random.choice(elites)
            child = self._clone_policy(parent)
            self._mutate(child)
            next_generation.append(child)

        self.training_population = next_generation
        self.training_requested_cycles -= 1
        self.total_training_cycles += 1
        self.best_fitness_seen = max(self.best_fitness_seen, scored[0][0])
        self.status = f"Training cycle complete. best={scored[0][0]:.2f}"

    def _archive_attempt(self, cause: str) -> None:
        self.attempts_total += 1
        fitness = self.agent.distance_traveled + self.agent.time_alive
        improved = fitness > self.best_fitness_seen
        if improved:
            self.best_fitness_seen = fitness
        else:
            # Keep learning even after failures: slightly mutate for next attempt.
            self._mutate(self.policy)
        self.status = f"Attempt {self.attempts_total}: {cause} | score {fitness:.2f} | best {self.best_fitness_seen:.2f}"
        self.attempt_history.append(AttemptRecord(list(self.agent.path_trace), cause, self.total_training_cycles))
        if len(self.attempt_history) > self.max_attempt_history:
            self.attempt_history = self.attempt_history[-self.max_attempt_history :]

    def _reset_episode(self, caused_by: str = "manual") -> None:
        self._archive_attempt(caused_by)
        self.agent = self._fresh_agent()
        self.simulator = self._build_simulator()
        self.last_sensor_values = [0.0] * self.policy.sensor_count
        self.last_sensor_hits = [None] * self.policy.sensor_count
        self.last_steering = 0.0
        self.last_fitness = 0.0
        self.start_line_cooldown = 0.0
        self.restart_countdown = 0.0

    def _safe_set_float(self, value: str, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, float(value)))

    def _safe_set_int(self, value: str, min_v: int, max_v: int) -> int:
        return max(min_v, min(max_v, int(float(value))))

    def _set_sensor_layout(self, sensor_count: int, spread_deg: float) -> None:
        self.sensor_spread_deg = spread_deg
        if sensor_count == 1:
            angles = [0.0]
        else:
            step = spread_deg / (sensor_count - 1)
            angles = [-(spread_deg / 2.0) + step * idx for idx in range(sensor_count)]
        hidden_count = max(4, self.policy.hidden_count)
        self.policy = NeuralPolicy(
            sensor_angles_deg=angles,
            hidden_weights=[[random.uniform(-0.5, 0.5) for _ in angles] for _ in range(hidden_count)],
            hidden_biases=[0.0 for _ in range(hidden_count)],
            steering_weights=[random.uniform(-0.5, 0.5) for _ in range(hidden_count)],
            accel_weights=None if self.policy.accel_weights is None else [random.uniform(-0.4, 0.4) for _ in range(hidden_count)],
            brake_weights=None if self.policy.brake_weights is None else [random.uniform(-0.4, 0.4) for _ in range(hidden_count)],
        )
        self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)
        self.status = f"Sensors updated: {sensor_count}"
        self._reset_episode("config")

    def _apply_options(self) -> None:
        values = {field.key: field.value.strip() for field in self.input_fields}
        toggles = {toggle.key: toggle.value for toggle in self.toggle_fields}
        try:
            self.config.speed = self._safe_set_float(values["speed"], 0.2, 5.5)
            self.config.max_turn_rate_deg = self._safe_set_float(values["turn_rate"], 20.0, 320.0)
            self.config.sensor_max_range = self._safe_set_float(values["sensor_range"], 0.8, 10.0)
            self.config.auto_restart_delay = self._safe_set_float(values["restart_delay"], 0.2, 20.0)
            self.training_rules.cycles = self._safe_set_int(values["cycles"], 1, 200)
            self.training_rules.population = self._safe_set_int(values["population"], 4, 96)
            self.training_rules.mutation_strength = self._safe_set_float(values["mutation_strength"], 0.01, 1.5)
            self.training_rules.max_steps = self._safe_set_int(values["max_steps"], 20, 4000)
            sensors = self._safe_set_int(values["sensor_count"], 3, 12)
            spread = self._safe_set_float(values["sensor_spread"], 30.0, 175.0)
            self._set_sensor_layout(sensors, spread)
            use_speed_heads = toggles.get("speed_control_heads", False)
            if use_speed_heads and self.policy.accel_weights is None:
                self.policy.accel_weights = [random.uniform(-0.3, 0.3) for _ in range(self.policy.hidden_count)]
                self.policy.accel_bias = 0.0
            if use_speed_heads and self.policy.brake_weights is None:
                self.policy.brake_weights = [random.uniform(-0.3, 0.3) for _ in range(self.policy.hidden_count)]
                self.policy.brake_bias = 0.0
            if not use_speed_heads:
                self.policy.accel_weights = None
                self.policy.brake_weights = None
            self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)

            linked = []
            for input_key, texture_kind in TEXTURE_INPUT_KEYS.items():
                raw_value = values.get(input_key, "")
                if not raw_value:
                    continue
                if self._set_texture_from_file(texture_kind, raw_value):
                    linked.append(texture_kind)

            self._reseed_training_population()
            self.options_open = False
            self.input_fields = []
            self.toggle_fields = []
            if linked:
                self.status = f"Options applied | linked textures: {', '.join(linked)}"
            else:
                self.status = "Options applied"
        except (ValueError, KeyError):
            self.status = "Invalid setting value"

    def run(self) -> None:
        while self.running:
            self._handle_events()
            if self.continuous_training:
                self.training_requested_cycles += 1
            if self.training_requested_cycles > 0:
                self._train_one_cycle()
            if not self.paused or self.step_once:
                self._tick_simulation(self.config.dt * self.config.step_multiplier)
                self.step_once = False
            self._render()
            self.clock.tick(60)
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.options_open:
                    self._handle_text_input(event)
                    continue
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_episode("manual")
                elif event.key == pygame.K_t:
                    self.track_idx = (self.track_idx + 1) % len(self.track_names)
                    self.track_def = build_track_definition(self.track_names[self.track_idx])
                    self._reset_episode("track")
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
                    if not self.options_open:
                        self.input_fields = []
                        self.toggle_fields = []
                elif event.key == pygame.K_n:
                    self.step_once = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    self.context_menu_open = True
                    self._build_context_menu(event.pos)
                elif event.button == 1:
                    if self.stat_help_key is not None:
                        self.stat_help_key = None
                        continue
                    if self.section_help_key is not None:
                        self.section_help_key = None
                        continue
                    if self.nn_detail is not None:
                        self.nn_detail = None
                        continue
                    if self.options_open and self._handle_options_click(event.pos):
                        continue
                    if self._handle_stat_click(event.pos):
                        continue
                    if self._handle_section_help_click(event.pos):
                        continue
                    if self._handle_nn_click(event.pos):
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
        elif event.unicode and event.unicode in "0123456789.-":
            field.value += event.unicode

    def _build_context_menu(self, pos: tuple[int, int]) -> None:
        x, y = pos
        menu = pygame.Rect(x, y, 170, 98)
        self.context_buttons = [
            UIButton("Open Options", "options", pygame.Rect(menu.x + 8, menu.y + 8, menu.w - 16, 24)),
            UIButton("Reset Attempt", "reset", pygame.Rect(menu.x + 8, menu.y + 36, menu.w - 16, 24)),
            UIButton("Train Batch", "train", pygame.Rect(menu.x + 8, menu.y + 64, menu.w - 16, 24)),
        ]

    def _handle_context_click(self, pos: tuple[int, int]) -> bool:
        for btn in self.context_buttons:
            if btn.rect.collidepoint(pos):
                if btn.action == "options":
                    self.options_open = True
                elif btn.action == "reset":
                    self._reset_episode("manual")
                elif btn.action == "train":
                    self.training_requested_cycles += max(1, self.training_rules.cycles)
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
                    self.toggle_fields = []
                elif btn.action == "continuous":
                    self.continuous_training = not self.continuous_training
                elif btn.action == "track_next":
                    self.track_idx = (self.track_idx + 1) % len(self.track_names)
                    self.track_def = build_track_definition(self.track_names[self.track_idx])
                    self._reset_episode("track")
                elif btn.action == "track_prev":
                    self.track_idx = (self.track_idx - 1) % len(self.track_names)
                    self.track_def = build_track_definition(self.track_names[self.track_idx])
                    self._reset_episode("track")
                elif btn.action == "reset_all":
                    self._reset_all_state()
                elif btn.action == "step_once":
                    self.step_once = True
                    self.paused = True
                elif btn.action == "step_speed_up":
                    rates = [0.25 * (2 ** i) for i in range(8)]
                    current_idx = rates.index(self.config.step_multiplier) if self.config.step_multiplier in rates else 0
                    self.config.step_multiplier = rates[min(len(rates) - 1, current_idx + 1)]
                elif btn.action == "step_speed_down":
                    rates = [0.25 * (2 ** i) for i in range(8)]
                    current_idx = rates.index(self.config.step_multiplier) if self.config.step_multiplier in rates else 0
                    self.config.step_multiplier = rates[max(0, current_idx - 1)]
                break

    def _handle_nn_click(self, pos: tuple[int, int]) -> bool:
        for rect, title, details in self.nn_hitboxes:
            if rect.collidepoint(pos):
                self.nn_detail = (title, details)
                return True
        return False

    def _reset_all_state(self) -> None:
        self.track_idx = 0
        self.track_def = build_track_definition(self.track_names[self.track_idx])
        self.training_population = None
        self.training_requested_cycles = 0
        self.total_training_cycles = 0
        self.attempts_total = 0
        self.attempt_history = []
        self.best_fitness_seen = float("-inf")
        self.lap_count = 0
        self.paused = False
        self.continuous_training = False
        self.policy = self._policy_from_preset("decent")
        self.policy.accel_weights = None
        self.policy.brake_weights = None
        self.sensor_spread_deg = 120.0
        self.sensor_array = SensorArray(list(self.policy.sensor_angles_deg), self.config.sensor_max_range)
        self.status = "Application reset to defaults"
        self._reset_episode("fresh-start")

    def _handle_stat_click(self, pos: tuple[int, int]) -> bool:
        for chip in self.stat_chips:
            if chip.rect.collidepoint(pos):
                self.stat_help_key = chip.key
                return True
        return False

    def _handle_section_help_click(self, pos: tuple[int, int]) -> bool:
        for btn in self.buttons:
            if btn.action.startswith("help_") and btn.rect.collidepoint(pos):
                self.section_help_key = btn.action.removeprefix("help_")
                return True
        return False

    def _handle_options_click(self, pos: tuple[int, int]) -> bool:
        for field in self.input_fields:
            if field.rect.collidepoint(pos):
                self.active_input = field.key
                return True
        for toggle in self.toggle_fields:
            if toggle.rect.collidepoint(pos):
                toggle.value = not toggle.value
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
            self.phase = "recover"
            self.restart_countdown -= dt
            if self.restart_countdown <= 0:
                self._reset_episode("collision")
            return

        self.phase = "sense"
        scan = self.sensor_array.read_with_distances(self.agent.position, self.agent.heading_deg, self.track_def.track)
        self.last_sensor_values = [reading for reading, _ in scan]
        self.last_sensor_hits = [dist for _, dist in scan]
        self.phase = "think"
        self.nn_activations, self.nn_outputs = self.policy.forward_with_activations(self.last_sensor_values)
        self.last_steering = self.nn_outputs["steering"]
        accel = self.nn_outputs["accel"] if self.policy.accel_weights is not None else 0.0
        brake = self.nn_outputs["brake"] if self.policy.brake_weights is not None else 0.0
        target_speed = self.config.speed * (1.0 + accel * 0.65 - brake * 0.75)
        self.agent.speed += (target_speed - self.agent.speed) * min(1.0, dt * 4.0)
        self.agent.speed = max(0.2, min(6.0, self.agent.speed))
        prev_position = self.agent.position
        self.phase = "act"
        self.agent.step(self.last_steering, dt)
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

        if self.simulator._path_collides(prev_position, self.agent.position):
            self.agent.alive = False
            self.phase = "learn"
            self.restart_countdown = self.config.auto_restart_delay
            self.status = f"Collision. Restart in {self.config.auto_restart_delay:.1f}s"

    def _segments_intersect(
        self,
        a1: tuple[float, float],
        a2: tuple[float, float],
        b1: tuple[float, float],
        b2: tuple[float, float],
    ) -> bool:
        def ccw(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> bool:
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

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
        side_rect = pygame.Rect(926, 70, 340, 676)
        self.buttons = []
        self.nn_hitboxes = []

        self._draw_track(track_rect)
        self._draw_side(side_rect)
        pygame.display.flip()

    def _draw_hud(self, rect: pygame.Rect) -> None:
        return

    def _draw_track(self, rect: pygame.Rect) -> None:
        if self.road_texture:
            tile = pygame.transform.smoothscale(self.road_texture, (180, 180))
            for tx in range(rect.x, rect.right, 180):
                for ty in range(rect.y, rect.bottom, 180):
                    self.screen.blit(tile, (tx, ty))
        else:
            pygame.draw.rect(self.screen, (20, 24, 32), rect, border_radius=12)

        for seg in self.track_def.track.wall_segments:
            p1 = self._to_screen(seg[0], rect)
            p2 = self._to_screen(seg[1], rect)
            self._blit_segment_texture(self.wall_texture, p1, p2, 18)
            if self.wall_texture is None:
                pygame.draw.line(self.screen, (220, 231, 248), p1, p2, 3)

        start_a = self._to_screen(self.track_def.start_line[0], rect)
        start_b = self._to_screen(self.track_def.start_line[1], rect)
        pygame.draw.line(self.screen, (255, 255, 255), start_a, start_b, 5)

        if len(self.agent.path_trace) > 2:
            pts = [self._to_screen(p, rect) for p in self.agent.path_trace[-350:]]
            pygame.draw.lines(self.screen, ACCENT, False, pts, 2)

        car = self._to_screen(self.agent.position, rect)
        if self.car_sprite:
            sprite = pygame.transform.rotozoom(self.car_sprite, self.agent.heading_deg - 90.0, 0.34)
            self.screen.blit(sprite, sprite.get_rect(center=car))
        else:
            self._draw_car_vector(car)
        if not self.agent.alive:
            x_size = 16
            pygame.draw.line(self.screen, RED, (car[0] - x_size, car[1] - x_size), (car[0] + x_size, car[1] + x_size), 5)
            pygame.draw.line(self.screen, RED, (car[0] - x_size, car[1] + x_size), (car[0] + x_size, car[1] - x_size), 5)
        angle = math.radians(self.agent.heading_deg)
        nose = (int(car[0] + math.cos(angle) * 17), int(car[1] - math.sin(angle) * 17))
        pygame.draw.line(self.screen, (255, 217, 107), car, nose, 3)

        for rel_angle, _, hit in zip(self.policy.sensor_angles_deg, self.last_sensor_values, self.last_sensor_hits, strict=True):
            a = math.radians(self.agent.heading_deg + rel_angle)
            length = self.config.sensor_max_range if hit is None else min(hit, self.config.sensor_max_range)
            end_w = (self.agent.position[0] + math.cos(a) * length, self.agent.position[1] + math.sin(a) * length)
            pygame.draw.line(self.screen, (94, 155, 235), car, self._to_screen(end_w, rect), 1)

    def _draw_side(self, rect: pygame.Rect) -> None:
        if self.panel_texture is not None:
            panel = pygame.transform.smoothscale(self.panel_texture, (rect.width, rect.height))
            self.screen.blit(panel, rect.topleft)
        y = rect.y + 20
        lines = [
            f"Track: {self.track_def.name}",
            f"Status: {self.status}",
            f"Phase: {self.phase}",
            f"Alive: {self.agent.alive}",
            f"Speed: {self.agent.speed:.2f}",
            f"Fitness: {self.last_fitness:.2f}",
            "Controls: SPACE pause, R reset, T/Y track, G train",
        ]
        for line in lines:
            self.screen.blit(self.small.render(line, True, TEXT), (rect.x + 12, y))
            y += 24

    def _draw_help_button(self, action: str, label: str, x: int, y: int) -> None:
        return


    def _draw_button(self, label: str, action: str, x: int, y: int, w: int) -> None:
        return


    def _draw_context_menu(self) -> None:
        return


    def _draw_car_vector(self, center: tuple[int, int]) -> None:
        return


    def _draw_nn_graph(self, rect: pygame.Rect) -> None:
        return


    def _draw_stat_help_modal(self) -> None:
        return


    def _draw_section_help_modal(self) -> None:
        return


    def _draw_options_modal(self) -> None:
        return


    def _draw_nn_detail_modal(self) -> None:
        return


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
