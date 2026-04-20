"""Interactive teaching application for neural-network driving demos.

Usage:
    python app.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from policy_presets import PRESETS
from simulator_core import CarAgent, LinearPolicy, SensorArray, Simulator
from tracks import TrackDefinition, build_track_definition, track_names

WINDOW_SIZE = (1280, 760)
BG = (19, 22, 28)
PANEL_BG = (28, 34, 44)
TEXT = (232, 238, 246)
MUTED = (141, 152, 170)
GREEN = (66, 211, 146)
YELLOW = (255, 213, 79)
RED = (255, 107, 107)
BLUE = (102, 178, 255)


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
class TrainSummary:
    best_fitness: float
    avg_fitness: float
    cycles: int


@dataclass
class SimConfig:
    sensor_max_range: float = 3.5
    speed: float = 2.0
    max_turn_rate_deg: float = 120.0
    dt: float = 0.05
    enable_acceleration_model: bool = False
    auto_turn_limit: bool = True


@dataclass
class UIButton:
    label: str
    action: str
    rect: pygame.Rect


class TeachingApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("NAIT Neural Network Teaching Simulator")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()

        self.big_font = pygame.font.SysFont("consolas", 26)
        self.font = pygame.font.SysFont("consolas", 19)
        self.small = pygame.font.SysFont("consolas", 16)

        self.track_names = track_names()
        self.track_idx = 0
        self.track_def = build_track_definition(self.track_names[self.track_idx])

        self.policy = self._policy_from_preset("decent")
        self.config = SimConfig()
        self.sensor_array = SensorArray(
            sensor_angles_deg=list(self.policy.sensor_angles_deg),
            max_range=self.config.sensor_max_range,
        )

        self.agent = self._fresh_agent()
        self.simulator = self._build_simulator()

        self.running = True
        self.paused = False
        self.step_once = False
        self.training_summary: TrainSummary | None = None
        self.total_training_cycles = 0
        self.continuous_training = False
        self.last_steering = 0.0
        self.last_sensor_values = [0.0 for _ in self.policy.sensor_angles_deg]
        self.last_sensor_hits: list[float | None] = [None for _ in self.policy.sensor_angles_deg]
        self.last_fitness = 0.0
        self.last_reaction_time = 0.0
        self.status = "Ready"
        self.sensor_spread_deg = 120.0
        self.buttons: list[UIButton] = []
        self.training_rules = TrainRules()
        self.training_options_open = False
        self.settings_open = False
        self.confirm_reset_all_open = False
        self.training_requested_cycles = 0
        self.training_progress_cycle = 0
        self.training_population: list[LinearPolicy] | None = None
        self.training_cycle_best = 0.0
        self.training_cycle_avg = 0.0
        self.lap_count = 0
        self.last_position = self.agent.position
        self.start_line_cooldown = 0.0

    def _policy_from_preset(self, preset_name: str) -> LinearPolicy:
        preset = PRESETS[preset_name]
        return LinearPolicy(
            sensor_angles_deg=list(preset.sensor_angles_deg),
            weights=list(preset.weights),
            bias=preset.bias,
        )

    def _fresh_agent(self) -> CarAgent:
        return CarAgent(
            position=self.track_def.spawn,
            heading_deg=self.track_def.heading_deg,
            speed=self.config.speed,
            max_turn_rate_deg=self.config.max_turn_rate_deg,
        )

    def _build_simulator(self) -> Simulator:
        return Simulator(
            track=self.track_def.track,
            policy=self.policy,
            agent=self.agent,
            sensor_array=self.sensor_array,
        )

    def reset_episode(self) -> None:
        self.agent = self._fresh_agent()
        self.simulator = self._build_simulator()
        self.last_sensor_values = [0.0 for _ in self.policy.sensor_angles_deg]
        self.last_steering = 0.0
        self.last_fitness = 0.0
        self.last_position = self.agent.position
        self.start_line_cooldown = 0.0
        self.status = f"Restarted on track: {self.track_def.name}"

    def next_track(self) -> None:
        self.track_idx = (self.track_idx + 1) % len(self.track_names)
        self.track_def = build_track_definition(self.track_names[self.track_idx])
        self.reset_episode()
        self.status = f"Switched to track: {self.track_def.name}"

    def prev_track(self) -> None:
        self.track_idx = (self.track_idx - 1) % len(self.track_names)
        self.track_def = build_track_definition(self.track_names[self.track_idx])
        self.reset_episode()
        self.status = f"Switched to track: {self.track_def.name}"

    def apply_preset(self, preset_name: str) -> None:
        self.policy = self._policy_from_preset(preset_name)
        self.sensor_array = SensorArray(
            sensor_angles_deg=list(self.policy.sensor_angles_deg),
            max_range=self.config.sensor_max_range,
        )
        self.reset_episode()
        self.training_summary = None
        self.status = f"Loaded preset: {preset_name}"
        self._reseed_training_population()

    def _fitness(self, simulator: Simulator) -> float:
        safety_bonus = 1.0 if simulator.agent.alive else 0.0
        return simulator.agent.distance_traveled + simulator.agent.time_alive + safety_bonus

    def _evaluate_policy(self, policy: LinearPolicy, track_def: TrackDefinition, rules: TrainRules) -> float:
        sensors = SensorArray(sensor_angles_deg=list(policy.sensor_angles_deg), max_range=self.config.sensor_max_range)
        agent = CarAgent(
            position=track_def.spawn,
            heading_deg=track_def.heading_deg,
            speed=self.config.speed,
            max_turn_rate_deg=self.config.max_turn_rate_deg,
        )
        sim = Simulator(track=track_def.track, policy=policy, agent=agent, sensor_array=sensors)
        sim.run(max_steps=rules.max_steps, dt=rules.dt)
        return self._fitness(sim)

    def _reseed_training_population(self) -> None:
        base = self.policy
        self.training_population = [
            LinearPolicy(sensor_angles_deg=list(base.sensor_angles_deg), weights=list(base.weights), bias=base.bias)
            for _ in range(self.training_rules.population)
        ]
        for indiv in self.training_population[1:]:
            self._mutate(indiv, self.training_rules)

    def _queue_training(self, cycles: int) -> None:
        if self.training_population is None or len(self.training_population) != self.training_rules.population:
            self._reseed_training_population()
        self.training_requested_cycles += cycles
        self.status = f"Training queued (+{cycles}). Pending cycles: {self.training_requested_cycles}"

    def train_current_track(self, rules: TrainRules) -> None:
        base = self.policy
        population: list[LinearPolicy] = [
            LinearPolicy(sensor_angles_deg=list(base.sensor_angles_deg), weights=list(base.weights), bias=base.bias)
            for _ in range(rules.population)
        ]

        for indiv in population[1:]:
            self._mutate(indiv, rules)

        for _ in range(rules.cycles):
            scored = [
                (self._evaluate_policy(candidate, self.track_def, rules), candidate)
                for candidate in population
            ]
            scored.sort(key=lambda item: item[0], reverse=True)

            elite_count = max(2, int(rules.population * rules.elite_ratio))
            elites = [candidate for _, candidate in scored[:elite_count]]

            next_generation: list[LinearPolicy] = []
            next_generation.extend(
                LinearPolicy(
                    sensor_angles_deg=list(elite.sensor_angles_deg),
                    weights=list(elite.weights),
                    bias=elite.bias,
                )
                for elite in elites
            )

            while len(next_generation) < rules.population:
                parent = random.choice(elites)
                child = LinearPolicy(
                    sensor_angles_deg=list(parent.sensor_angles_deg),
                    weights=list(parent.weights),
                    bias=parent.bias,
                )
                self._mutate(child, rules)
                next_generation.append(child)

            population = next_generation

        final_scored = [(self._evaluate_policy(candidate, self.track_def, rules), candidate) for candidate in population]
        final_scored.sort(key=lambda item: item[0], reverse=True)

        avg = sum(score for score, _ in final_scored) / len(final_scored)
        self.policy = final_scored[0][1]
        self.training_summary = TrainSummary(best_fitness=final_scored[0][0], avg_fitness=avg, cycles=rules.cycles)
        self.total_training_cycles += rules.cycles
        self.sensor_array = SensorArray(
            sensor_angles_deg=list(self.policy.sensor_angles_deg),
            max_range=self.config.sensor_max_range,
        )
        self.reset_episode()
        self.status = f"Training complete: best={final_scored[0][0]:.2f}, avg={avg:.2f}"
        self._reseed_training_population()

    def _train_one_cycle(self) -> None:
        if self.training_requested_cycles <= 0:
            return
        if self.training_population is None:
            self._reseed_training_population()
        assert self.training_population is not None

        scored = [
            (self._evaluate_policy(candidate, self.track_def, self.training_rules), candidate)
            for candidate in self.training_population
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        self.training_cycle_best = scored[0][0]
        self.training_cycle_avg = sum(score for score, _ in scored) / len(scored)

        elite_count = max(2, int(self.training_rules.population * self.training_rules.elite_ratio))
        elites = [candidate for _, candidate in scored[:elite_count]]

        self.policy = LinearPolicy(
            sensor_angles_deg=list(scored[0][1].sensor_angles_deg),
            weights=list(scored[0][1].weights),
            bias=scored[0][1].bias,
        )
        self.sensor_array = SensorArray(
            sensor_angles_deg=list(self.policy.sensor_angles_deg),
            max_range=self.config.sensor_max_range,
        )

        next_generation: list[LinearPolicy] = [
            LinearPolicy(sensor_angles_deg=list(elite.sensor_angles_deg), weights=list(elite.weights), bias=elite.bias)
            for elite in elites
        ]
        while len(next_generation) < self.training_rules.population:
            parent = random.choice(elites)
            child = LinearPolicy(
                sensor_angles_deg=list(parent.sensor_angles_deg),
                weights=list(parent.weights),
                bias=parent.bias,
            )
            self._mutate(child, self.training_rules)
            next_generation.append(child)

        self.training_population = next_generation
        self.training_progress_cycle += 1
        self.training_requested_cycles -= 1
        self.total_training_cycles += 1
        self.training_summary = TrainSummary(
            best_fitness=self.training_cycle_best,
            avg_fitness=self.training_cycle_avg,
            cycles=self.training_progress_cycle,
        )
        self.status = (
            f"Training cycle {self.training_progress_cycle} | "
            f"best={self.training_cycle_best:.2f} avg={self.training_cycle_avg:.2f}"
        )

    def _mutate(self, policy: LinearPolicy, rules: TrainRules) -> None:
        for idx, weight in enumerate(policy.weights):
            if random.random() < rules.mutation_rate:
                policy.weights[idx] = weight + random.uniform(-rules.mutation_strength, rules.mutation_strength)
        if random.random() < rules.mutation_rate:
            policy.bias += random.uniform(-rules.mutation_strength, rules.mutation_strength)

    def run(self) -> None:
        while self.running:
            self._handle_events()
            if self.continuous_training:
                self.training_requested_cycles += 1
            if self.training_requested_cycles > 0:
                self._train_one_cycle()
            if not self.paused or self.step_once:
                self._tick_simulation(self.config.dt)
                self.step_once = False
            self._render()
            self.clock.tick(60)

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.status = "Paused" if self.paused else "Running"
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_x:
                    self.confirm_reset_all_open = True
                elif event.key == pygame.K_n:
                    self.step_once = True
                elif event.key == pygame.K_t:
                    self.next_track()
                elif event.key == pygame.K_y:
                    self.prev_track()
                elif event.key == pygame.K_1:
                    self.apply_preset("bad")
                elif event.key == pygame.K_2:
                    self.apply_preset("decent")
                elif event.key == pygame.K_3:
                    self.apply_preset("good")
                elif event.key == pygame.K_LEFTBRACKET:
                    self.config.speed = max(0.5, self.config.speed - 0.1)
                    self.reset_episode()
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.config.speed = min(4.5, self.config.speed + 0.1)
                    self.reset_episode()
                elif event.key == pygame.K_MINUS:
                    self.config.max_turn_rate_deg = max(20.0, self.config.max_turn_rate_deg - 10.0)
                    self.reset_episode()
                elif event.key == pygame.K_EQUALS:
                    self.config.max_turn_rate_deg = min(280.0, self.config.max_turn_rate_deg + 10.0)
                    self.reset_episode()
                elif event.key == pygame.K_g:
                    self._queue_training(max(1, self.training_rules.cycles))
                elif event.key == pygame.K_h:
                    self._queue_training(max(2, self.training_rules.cycles * 3))
                elif event.key == pygame.K_c:
                    self.continuous_training = not self.continuous_training
                    self.status = "Continuous training ON" if self.continuous_training else "Continuous training OFF"
                elif event.key == pygame.K_m:
                    self.training_options_open = not self.training_options_open
                elif event.key == pygame.K_a:
                    self.config.enable_acceleration_model = not self.config.enable_acceleration_model
                elif event.key == pygame.K_COMMA:
                    self._set_sensor_layout(max(3, self.policy.sensor_count - 1), self.sensor_spread_deg)
                elif event.key == pygame.K_PERIOD:
                    self._set_sensor_layout(min(11, self.policy.sensor_count + 1), self.sensor_spread_deg)
                elif event.key == pygame.K_SEMICOLON:
                    self._set_sensor_layout(self.policy.sensor_count, max(40.0, self.sensor_spread_deg - 10.0))
                elif event.key == pygame.K_QUOTE:
                    self._set_sensor_layout(self.policy.sensor_count, min(170.0, self.sensor_spread_deg + 10.0))
                elif event.key == pygame.K_9:
                    self.settings_open = not self.settings_open
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

    def _set_sensor_layout(self, sensor_count: int, spread_deg: float) -> None:
        self.sensor_spread_deg = spread_deg
        center = 0.0
        if sensor_count == 1:
            angles = [0.0]
        else:
            step = spread_deg / (sensor_count - 1)
            angles = [center - (spread_deg / 2.0) + step * idx for idx in range(sensor_count)]
        self.policy = LinearPolicy(sensor_angles_deg=angles, weights=[0.0 for _ in angles], bias=0.0)
        self.sensor_array = SensorArray(sensor_angles_deg=list(self.policy.sensor_angles_deg), max_range=self.config.sensor_max_range)
        self.last_sensor_values = [0.0 for _ in angles]
        self.last_sensor_hits = [None for _ in angles]
        self.status = f"Sensors updated: {sensor_count} over {spread_deg:.0f}°"
        self.reset_episode()

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.rect.collidepoint(pos):
                self._run_action(button.action)
                return

    def _run_action(self, action: str) -> None:
        if action == "pause":
            self.paused = not self.paused
        elif action == "reset":
            self.reset_episode()
            self.lap_count = 0
        elif action == "reset_all":
            self.confirm_reset_all_open = True
        elif action == "confirm_reset_yes":
            self.total_training_cycles = 0
            self.training_progress_cycle = 0
            self.training_requested_cycles = 0
            self.training_summary = None
            self.lap_count = 0
            self.policy = self._policy_from_preset("decent")
            self.sensor_array = SensorArray(
                sensor_angles_deg=list(self.policy.sensor_angles_deg),
                max_range=self.config.sensor_max_range,
            )
            self._reseed_training_population()
            self.reset_episode()
            self.confirm_reset_all_open = False
            self.status = "All training data reset."
        elif action == "confirm_reset_no":
            self.confirm_reset_all_open = False
        elif action == "step":
            self.step_once = True
        elif action == "next_track":
            self.next_track()
        elif action == "prev_track":
            self.prev_track()
        elif action == "quick_train":
            self._queue_training(max(1, self.training_rules.cycles))
        elif action == "deep_train":
            self._queue_training(max(2, self.training_rules.cycles * 3))
        elif action == "continuous":
            self.continuous_training = not self.continuous_training
        elif action == "sensor_plus":
            self._set_sensor_layout(min(11, self.policy.sensor_count + 1), self.sensor_spread_deg)
        elif action == "sensor_minus":
            self._set_sensor_layout(max(3, self.policy.sensor_count - 1), self.sensor_spread_deg)
        elif action == "spread_plus":
            self._set_sensor_layout(self.policy.sensor_count, min(170.0, self.sensor_spread_deg + 10.0))
        elif action == "spread_minus":
            self._set_sensor_layout(self.policy.sensor_count, max(40.0, self.sensor_spread_deg - 10.0))
        elif action == "toggle_train_options":
            self.training_options_open = not self.training_options_open
        elif action == "toggle_settings":
            self.settings_open = not self.settings_open
        elif action == "toggle_accel":
            self.config.enable_acceleration_model = not self.config.enable_acceleration_model
        elif action == "toggle_auto_turn":
            self.config.auto_turn_limit = not self.config.auto_turn_limit
        elif action == "cfg_cycles_down":
            self.training_rules.cycles = max(1, self.training_rules.cycles - 1)
        elif action == "cfg_cycles_up":
            self.training_rules.cycles = min(200, self.training_rules.cycles + 1)
        elif action == "cfg_pop_down":
            self.training_rules.population = max(4, self.training_rules.population - 1)
            self._reseed_training_population()
        elif action == "cfg_pop_up":
            self.training_rules.population = min(64, self.training_rules.population + 1)
            self._reseed_training_population()
        elif action == "cfg_mut_down":
            self.training_rules.mutation_strength = max(0.02, self.training_rules.mutation_strength - 0.02)
        elif action == "cfg_mut_up":
            self.training_rules.mutation_strength = min(1.0, self.training_rules.mutation_strength + 0.02)
        elif action == "cfg_steps_down":
            self.training_rules.max_steps = max(40, self.training_rules.max_steps - 20)
        elif action == "cfg_steps_up":
            self.training_rules.max_steps = min(1200, self.training_rules.max_steps + 20)
        elif action == "speed_up":
            self.config.speed = min(4.5, self.config.speed + 0.1)
            self.reset_episode()
        elif action == "speed_down":
            self.config.speed = max(0.5, self.config.speed - 0.1)
            self.reset_episode()
        elif action == "turn_up":
            self.config.max_turn_rate_deg = min(280.0, self.config.max_turn_rate_deg + 10.0)
            self.reset_episode()
        elif action == "turn_down":
            self.config.max_turn_rate_deg = max(20.0, self.config.max_turn_rate_deg - 10.0)
            self.reset_episode()

    def _tick_simulation(self, dt: float) -> None:
        if not self.agent.alive:
            return

        scan = self.sensor_array.read_with_distances(
            position=self.agent.position,
            heading_deg=self.agent.heading_deg,
            track=self.track_def.track,
        )
        self.last_sensor_values = [reading for reading, _ in scan]
        self.last_sensor_hits = [hit for _, hit in scan]
        self.last_steering = self.policy.forward(self.last_sensor_values)
        front_distance = self.last_sensor_hits[len(self.last_sensor_hits) // 2]
        if front_distance is not None and self.agent.speed > 0:
            self.last_reaction_time = front_distance / self.agent.speed
        else:
            self.last_reaction_time = 0.0
        if self.config.enable_acceleration_model:
            target_speed = self.config.speed
            if front_distance is not None and self.last_reaction_time < 1.25:
                target_speed = max(0.5, self.config.speed * 0.45)
            elif abs(self.last_steering) < 0.2:
                target_speed = min(4.8, self.config.speed * 1.08)
            self.agent.speed += (target_speed - self.agent.speed) * 0.12
        if self.config.auto_turn_limit:
            steer = self.last_steering
        else:
            steer = max(-0.35, min(0.35, self.last_steering))
        prev_position = self.agent.position
        self.agent.step(steer, dt)
        self.last_fitness = self._fitness(self.simulator)
        self._update_lap_counter(prev_position, self.agent.position, dt)

        if self.simulator._is_colliding(self.agent.position):
            self.agent.alive = False
            self.status = "Collision: press R to reset"

    def _update_lap_counter(self, prev_pos: tuple[float, float], current_pos: tuple[float, float], dt: float) -> None:
        self.start_line_cooldown = max(0.0, self.start_line_cooldown - dt)
        if self.start_line_cooldown > 0.0:
            return
        if self._segments_intersect(prev_pos, current_pos, self.track_def.start_line[0], self.track_def.start_line[1]):
            self.lap_count += 1
            self.start_line_cooldown = 1.25
            self.status = f"Lap {self.lap_count} complete"

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

    def _to_screen(self, p: tuple[float, float], view_rect: pygame.Rect) -> tuple[int, int]:
        xs = [pt[0] for segment in self.track_def.track.wall_segments for pt in segment]
        ys = [pt[1] for segment in self.track_def.track.wall_segments for pt in segment]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        world_w = max_x - min_x
        world_h = max_y - min_y
        scale = min((view_rect.width - 20) / world_w, (view_rect.height - 20) / world_h)

        x = view_rect.left + 10 + (p[0] - min_x) * scale
        y = view_rect.bottom - 10 - (p[1] - min_y) * scale
        return int(x), int(y)

    def _render(self) -> None:
        self.screen.fill(BG)

        sim_rect = pygame.Rect(16, 16, 820, 728)
        panel_rect = pygame.Rect(850, 16, 414, 728)
        pygame.draw.rect(self.screen, (14, 17, 23), sim_rect, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=10)

        self._draw_track_and_agent(sim_rect)
        self._draw_panel(panel_rect)
        if self.confirm_reset_all_open:
            self._draw_reset_confirmation()

        pygame.display.flip()

    def _draw_track_and_agent(self, sim_rect: pygame.Rect) -> None:
        for segment in self.track_def.track.wall_segments:
            a = self._to_screen(segment[0], sim_rect)
            b = self._to_screen(segment[1], sim_rect)
            pygame.draw.line(self.screen, (227, 236, 255), a, b, 3)
        start_a = self._to_screen(self.track_def.start_line[0], sim_rect)
        start_b = self._to_screen(self.track_def.start_line[1], sim_rect)
        pygame.draw.line(self.screen, (255, 255, 255), start_a, start_b, 4)
        pygame.draw.line(self.screen, (15, 15, 15), ((start_a[0] + start_b[0]) // 2, start_a[1]), ((start_a[0] + start_b[0]) // 2, start_b[1]), 2)

        if len(self.agent.path_trace) > 2:
            trace = [self._to_screen(p, sim_rect) for p in self.agent.path_trace[-250:]]
            pygame.draw.lines(self.screen, BLUE, False, trace, 2)

        car = self._to_screen(self.agent.position, sim_rect)
        pygame.draw.circle(self.screen, GREEN if self.agent.alive else RED, car, 8)

        heading_rad = math.radians(self.agent.heading_deg)
        nose = (int(car[0] + math.cos(heading_rad) * 16), int(car[1] - math.sin(heading_rad) * 16))
        pygame.draw.line(self.screen, YELLOW, car, nose, 3)

        for rel_angle, reading, hit_distance in zip(
            self.policy.sensor_angles_deg,
            self.last_sensor_values,
            self.last_sensor_hits,
            strict=True,
        ):
            angle = math.radians(self.agent.heading_deg + rel_angle)
            if hit_distance is None:
                length = self.config.sensor_max_range
            else:
                length = min(self.config.sensor_max_range, hit_distance)
            world_end = (
                self.agent.position[0] + math.cos(angle) * length,
                self.agent.position[1] + math.sin(angle) * length,
            )
            end = self._to_screen(world_end, sim_rect)
            color = (120, 180, 255) if reading < 0.5 else (255, 155, 86)
            pygame.draw.line(self.screen, color, car, end, 2)

    def _draw_panel(self, panel_rect: pygame.Rect) -> None:
        x = panel_rect.left + 16
        y = panel_rect.top + 12

        def line(text: str, color: tuple[int, int, int] = TEXT, dy: int = 26) -> None:
            nonlocal y
            self.screen.blit(self.font.render(text, True, color), (x, y))
            y += dy

        self.screen.blit(self.big_font.render("Seminar Controls", True, TEXT), (x, y))
        y += 36

        line(f"Track: {self.track_def.name}")
        line(f"Status: {self.status}", MUTED)
        line(f"Alive: {self.agent.alive}")
        line(f"Steering: {self.last_steering:+.3f}")
        line(f"Speed: {self.agent.speed:.2f}")
        line(f"Reaction time est: {self.last_reaction_time:.2f}s")
        line(f"Distance: {self.agent.distance_traveled:.2f}")
        line(f"Fitness: {self.last_fitness:.2f}")
        line(f"Training cycles total: {self.total_training_cycles}")
        line(f"Queued training cycles: {self.training_requested_cycles}")
        line(f"Laps: {self.lap_count}")
        line(f"Sensors: {self.policy.sensor_count} spread {self.sensor_spread_deg:.0f}°")
        line(
            f"Train opts: cycles={self.training_rules.cycles}, pop={self.training_rules.population}, "
            f"mut={self.training_rules.mutation_strength:.2f}, steps={self.training_rules.max_steps}",
            MUTED,
            dy=20,
        )
        line(
            f"Settings: auto-throttle={self.config.enable_acceleration_model} auto-turn={self.config.auto_turn_limit}",
            MUTED,
            dy=20,
        )
        y += 10

        line("Controls", YELLOW)
        line("Buttons are grouped: Drive, Training, Options, Settings.", MUTED)
        line("C: continuous train, G/H: queue training, X: reset all.", MUTED)
        line("T/Y tracks. </> sensors. ;/' spread. A: auto-throttle.", MUTED)
        y += 12

        y = self._draw_buttons(x, y, panel_rect.width - 32)
        y += 8

        self._draw_nn_graph(x, y, panel_rect.width - 32, 225)
        y += 240

        if self.training_summary:
            line("Last Training Summary", YELLOW)
            line(f"Cycles: {self.training_summary.cycles}", MUTED)
            line(f"Best fitness: {self.training_summary.best_fitness:.2f}", MUTED)
            line(f"Avg fitness:  {self.training_summary.avg_fitness:.2f}", MUTED)

    def _draw_buttons(self, x: int, y: int, width: int) -> int:
        self.buttons = []

        def add(label: str, action: str, col: int, row: int, cols: int = 3) -> None:
            gap = 8
            button_w = (width - (cols - 1) * gap) // cols
            button_h = 30
            rect = pygame.Rect(x + col * (button_w + gap), y + row * (button_h + gap), button_w, button_h)
            self.buttons.append(UIButton(label=label, action=action, rect=rect))

        add("Pause/Run", "pause", 0, 0)
        add("Restart", "reset", 1, 0)
        add("Step", "step", 2, 0)
        add("Prev Track", "prev_track", 0, 1)
        add("Next Track", "next_track", 1, 1)
        add("Train Batch", "quick_train", 2, 1)
        add("Train Boost", "deep_train", 0, 2)
        add("Continuous", "continuous", 1, 2)
        add("Train Options", "toggle_train_options", 2, 2)
        add("Sensors -", "sensor_minus", 0, 3)
        add("Sensors +", "sensor_plus", 1, 3)
        add("Auto Throttle", "toggle_accel", 2, 3)
        add("Spread -", "spread_minus", 0, 4)
        add("Spread +", "spread_plus", 1, 4)
        add("Settings", "toggle_settings", 2, 4)
        rows = 5
        if self.training_options_open:
            base_row = rows
            add("Cycles -", "cfg_cycles_down", 0, base_row)
            add("Cycles +", "cfg_cycles_up", 1, base_row)
            add("Pop -", "cfg_pop_down", 0, base_row + 1)
            add("Pop +", "cfg_pop_up", 1, base_row + 1)
            add("Mut -", "cfg_mut_down", 0, base_row + 2)
            add("Mut +", "cfg_mut_up", 1, base_row + 2)
            add("Steps -", "cfg_steps_down", 0, base_row + 3)
            add("Steps +", "cfg_steps_up", 1, base_row + 3)
            rows += 4
        if self.settings_open:
            base_row = rows
            add("Auto Turn", "toggle_auto_turn", 0, base_row)
            add("Speed -", "speed_down", 0, base_row + 1)
            add("Speed +", "speed_up", 1, base_row + 1)
            add("Turn +", "turn_up", 2, base_row + 1)
            add("Turn -", "turn_down", 2, base_row + 2)
            add("Reset All", "reset_all", 2, base_row)
            rows += 3

        for button in self.buttons:
            active = button.action == "continuous" and self.continuous_training
            color = (48, 120, 86) if active else (50, 63, 84)
            pygame.draw.rect(self.screen, color, button.rect, border_radius=6)
            pygame.draw.rect(self.screen, (112, 128, 153), button.rect, 1, border_radius=6)
            label = self.small.render(button.label, True, TEXT)
            self.screen.blit(label, (button.rect.x + 8, button.rect.y + 7))

        return y + rows * 38

    def _draw_reset_confirmation(self) -> None:
        w, h = 520, 180
        rect = pygame.Rect((WINDOW_SIZE[0] - w) // 2, (WINDOW_SIZE[1] - h) // 2, w, h)
        pygame.draw.rect(self.screen, (9, 12, 18), rect, border_radius=10)
        pygame.draw.rect(self.screen, (112, 128, 153), rect, 1, border_radius=10)
        title = self.font.render("Are you sure you want to delete all training data?", True, TEXT)
        self.screen.blit(title, (rect.x + 24, rect.y + 34))

        yes_rect = pygame.Rect(rect.x + 90, rect.y + 104, 140, 42)
        no_rect = pygame.Rect(rect.x + 290, rect.y + 104, 140, 42)
        self.buttons.append(UIButton(label="Yes, Reset", action="confirm_reset_yes", rect=yes_rect))
        self.buttons.append(UIButton(label="No, Keep", action="confirm_reset_no", rect=no_rect))
        for button in self.buttons[-2:]:
            pygame.draw.rect(self.screen, (50, 63, 84), button.rect, border_radius=6)
            pygame.draw.rect(self.screen, (112, 128, 153), button.rect, 1, border_radius=6)
            self.screen.blit(self.small.render(button.label, True, TEXT), (button.rect.x + 14, button.rect.y + 11))

    def _draw_nn_graph(self, x: int, y: int, width: int, height: int) -> None:
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (18, 22, 30), rect, border_radius=8)
        pygame.draw.rect(self.screen, (56, 67, 86), rect, 1, border_radius=8)

        title = self.small.render("Neural Flow: sensor * weight + bias -> tanh", True, TEXT)
        self.screen.blit(title, (x + 10, y + 8))

        input_x = x + 42
        output_x = x + width - 55
        top = y + 42
        spacing = max(26, int((height - 90) / max(1, len(self.policy.weights))))

        out_node = (output_x, y + height // 2)
        pygame.draw.circle(self.screen, (130, 184, 255), out_node, 18)

        for i, (sensor, weight) in enumerate(zip(self.last_sensor_values, self.policy.weights, strict=True)):
            node = (input_x, top + i * spacing)
            intensity = int(60 + min(1.0, abs(sensor)) * 195)
            pygame.draw.circle(self.screen, (intensity, 110, 230), node, 12)

            strength = min(1.0, abs(sensor * weight))
            line_color = (255, int(200 - strength * 120), int(200 - strength * 120)) if weight < 0 else (
                int(160 - strength * 100),
                245,
                int(160 - strength * 100),
            )
            pygame.draw.line(self.screen, line_color, node, out_node, max(1, int(1 + strength * 4)))

            label = self.small.render(f"s{i}:{sensor:.2f} w:{weight:+.2f}", True, TEXT)
            self.screen.blit(label, (node[0] + 18, node[1] - 8))

        out_label = self.small.render(f"steer={self.last_steering:+.2f}  b={self.policy.bias:+.2f}", True, TEXT)
        self.screen.blit(out_label, (out_node[0] - 105, out_node[1] + 24))


def main() -> None:
    app = TeachingApp()
    app.run()


if __name__ == "__main__":
    main()
