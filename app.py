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
    cycles: int = 40
    population: int = 20
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
        self.last_steering = 0.0
        self.last_sensor_values = [0.0 for _ in self.policy.sensor_angles_deg]
        self.last_fitness = 0.0
        self.status = "Ready"

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
        self.status = f"Reset on track: {self.track_def.name}"

    def next_track(self) -> None:
        self.track_idx = (self.track_idx + 1) % len(self.track_names)
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
        self.sensor_array = SensorArray(
            sensor_angles_deg=list(self.policy.sensor_angles_deg),
            max_range=self.config.sensor_max_range,
        )
        self.reset_episode()
        self.status = f"Training complete: best={final_scored[0][0]:.2f}, avg={avg:.2f}"

    def _mutate(self, policy: LinearPolicy, rules: TrainRules) -> None:
        for idx, weight in enumerate(policy.weights):
            if random.random() < rules.mutation_rate:
                policy.weights[idx] = weight + random.uniform(-rules.mutation_strength, rules.mutation_strength)
        if random.random() < rules.mutation_rate:
            policy.bias += random.uniform(-rules.mutation_strength, rules.mutation_strength)

    def run(self) -> None:
        while self.running:
            self._handle_events()
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
                elif event.key == pygame.K_n:
                    self.step_once = True
                elif event.key == pygame.K_t:
                    self.next_track()
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
                    self.train_current_track(TrainRules(cycles=20, population=18))
                elif event.key == pygame.K_h:
                    self.train_current_track(TrainRules(cycles=60, population=28, mutation_strength=0.3))

    def _tick_simulation(self, dt: float) -> None:
        if not self.agent.alive:
            return

        self.last_sensor_values = self.sensor_array.read(
            position=self.agent.position,
            heading_deg=self.agent.heading_deg,
            track=self.track_def.track,
        )
        self.last_steering = self.policy.forward(self.last_sensor_values)
        self.agent.step(self.last_steering, dt)
        self.last_fitness = self._fitness(self.simulator)

        if self.simulator._is_colliding(self.agent.position):
            self.agent.alive = False
            self.status = "Collision: press R to reset"

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

        pygame.display.flip()

    def _draw_track_and_agent(self, sim_rect: pygame.Rect) -> None:
        for segment in self.track_def.track.wall_segments:
            a = self._to_screen(segment[0], sim_rect)
            b = self._to_screen(segment[1], sim_rect)
            pygame.draw.line(self.screen, (227, 236, 255), a, b, 3)

        if len(self.agent.path_trace) > 2:
            trace = [self._to_screen(p, sim_rect) for p in self.agent.path_trace[-250:]]
            pygame.draw.lines(self.screen, BLUE, False, trace, 2)

        car = self._to_screen(self.agent.position, sim_rect)
        pygame.draw.circle(self.screen, GREEN if self.agent.alive else RED, car, 8)

        heading_rad = math.radians(self.agent.heading_deg)
        nose = (int(car[0] + math.cos(heading_rad) * 16), int(car[1] - math.sin(heading_rad) * 16))
        pygame.draw.line(self.screen, YELLOW, car, nose, 3)

        for rel_angle, reading in zip(self.policy.sensor_angles_deg, self.last_sensor_values, strict=True):
            angle = math.radians(self.agent.heading_deg + rel_angle)
            length = self.config.sensor_max_range * (0.12 + 0.88 * (1.0 - reading))
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
        line(f"Distance: {self.agent.distance_traveled:.2f}")
        line(f"Fitness: {self.last_fitness:.2f}")
        y += 10

        line("Keys", YELLOW)
        line("Space pause/resume, R reset", MUTED)
        line("N single-step, T next track", MUTED)
        line("1/2/3 presets bad/decent/good", MUTED)
        line("G quick-train (20), H deep-train (60)", MUTED)
        line("[/] speed, -/= turn-rate", MUTED)
        y += 12

        line("Custom Training Rules", YELLOW)
        line("Currently wired to two fast presets:", MUTED)
        line("G: cycles=20 pop=18 mutate=0.35", MUTED)
        line("H: cycles=60 pop=28 mutate=0.35", MUTED)
        y += 14

        self._draw_nn_graph(x, y, panel_rect.width - 32, 225)
        y += 240

        if self.training_summary:
            line("Last Training Summary", YELLOW)
            line(f"Cycles: {self.training_summary.cycles}", MUTED)
            line(f"Best fitness: {self.training_summary.best_fitness:.2f}", MUTED)
            line(f"Avg fitness:  {self.training_summary.avg_fitness:.2f}", MUTED)

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
