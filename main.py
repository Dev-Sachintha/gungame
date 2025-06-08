import pygame
import numpy as np
import random
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pickle
from collections import deque
import heapq
import json
import os
import sqlite3

# Initialize Pygame
pygame.init()
pygame.font.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
GRID_CELL_SIZE = 40

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
BROWN = (139, 69, 19)
DARK_GREEN = (0, 100, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GRAY = (64, 64, 64)
LIGHT_GREEN = (144, 238, 144)
SAND = (194, 178, 128)
FOREST_GREEN = (34, 139, 34)
CYAN = (0, 255, 255)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
BRONZE = (205, 127, 50)


class GameState(Enum):
    START_MENU = 1
    PLAYING = 2
    GET_PLAYER_NAME = 3
    GAME_OVER = 4
    LEADERBOARD = 5


class EntityState(Enum):
    IDLE, PATROLLING, PURSUING, ATTACKING, TAKING_COVER = range(5)


@dataclass
class Vector2:
    x: float
    y: float

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return (
            Vector2(self.x / scalar, self.y / scalar) if scalar != 0 else Vector2(0, 0)
        )

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        return self / mag if mag != 0 else Vector2(0, 0)

    def distance_to(self, other):
        return (self - other).magnitude()


class Leaderboard:
    def __init__(self, db_name="leaderboard.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY, name TEXT, score INTEGER, waves INTEGER, 
                kills INTEGER, accuracy REAL, time REAL, timestamp INTEGER
            )"""
        )
        self.conn.commit()

    def add_score(self, name, score, waves, kills, accuracy, time):
        timestamp = int(pygame.time.get_ticks() / 1000)
        self.cursor.execute(
            """INSERT INTO scores(name, score, waves, kills, accuracy, time, timestamp)
                               VALUES(?,?,?,?,?,?,?)""",
            (name, score, waves, kills, accuracy, time, timestamp),
        )
        self.cursor.execute(
            "DELETE FROM scores WHERE id NOT IN (SELECT id FROM scores ORDER BY score DESC, timestamp DESC LIMIT 10)"
        )
        self.conn.commit()

    def get_top_scores(self, count=10):
        self.cursor.execute(
            "SELECT name, score, waves FROM scores ORDER BY score DESC, timestamp DESC LIMIT ?",
            (count,),
        )
        return [
            {"name": r[0], "score": r[1], "waves": r[2]} for r in self.cursor.fetchall()
        ]

    def is_high_score(self, score):
        count = self.cursor.execute("SELECT COUNT(id) FROM scores").fetchone()[0]
        if count < 10:
            return True
        min_score = self.cursor.execute("SELECT MIN(score) FROM scores").fetchone()[0]
        return score > min_score

    def close(self):
        if self.conn:
            self.conn.close()


class AudioManager:
    def __init__(self):
        self.music_volume = 0.3
        self.sfx_volume = 0.5
        self.sound_effects = {}
        self._create_sound_effects()
        self._load_background_music()

    def _create_sound_effects(self):
        sfx_data = {
            "gunshot": self._create_gunshot_sound,
            "explosion": self._create_explosion_sound,
            "reload": self._create_reload_sound,
            "pickup": self._create_pickup_sound,
        }
        for name, creator in sfx_data.items():
            sound = pygame.mixer.Sound(buffer=creator())
            sound.set_volume(
                self.sfx_volume
                * (0.8 if name == "reload" else 0.6 if name == "pickup" else 1.0)
            )
            self.sound_effects[name] = sound

    def _create_gunshot_sound(self):
        sr, dur, s = 22050, 0.15, int(22050 * 0.15)
        n, e, t = (
            np.random.uniform(-1, 1, s),
            np.exp(-np.linspace(0, 8, s)),
            np.linspace(0, dur, s),
        )
        lf = 0.3 * np.sin(2 * np.pi * 80 * t) * e
        snd = np.clip((n * e + lf) * 0.5, -1, 1)
        return np.column_stack(((snd * 32767).astype(np.int16),) * 2).tobytes()

    def _create_explosion_sound(self):
        sr, dur, s = 22050, 0.8, int(22050 * 0.8)
        n, e, t = (
            np.random.uniform(-1, 1, s),
            np.exp(-np.linspace(0, 4, s)),
            np.linspace(0, dur, s),
        )
        r = 0.4 * np.sin(2 * np.pi * 40 * t) * e + 0.2 * np.sin(2 * np.pi * 60 * t) * e
        snd = np.clip((n * e * 0.6 + r) * 0.7, -1, 1)
        return np.column_stack(((snd * 32767).astype(np.int16),) * 2).tobytes()

    def _create_reload_sound(self):
        sr, dur, s, c = 22050, 0.6, int(22050 * 0.6), np.zeros(int(22050 * 0.6))
        for t in [0.1, 0.25, 0.4, 0.55]:
            st, d_ = int(t * sr), int(0.05 * sr)
            if st + d_ < s:
                c[st : st + d_] += (
                    np.random.uniform(-1, 1, d_) * np.exp(-np.linspace(0, 20, d_)) * 0.5
                )
        return np.column_stack(((c * 32767).astype(np.int16),) * 2).tobytes()

    def _create_pickup_sound(self):
        sr, dur, s, t = (
            22050,
            0.3,
            int(22050 * 0.3),
            np.linspace(0, 0.3, int(22050 * 0.3)),
        )
        f = 200 + 300 * t / dur
        snd = np.sin(2 * np.pi * f * t) * np.exp(-np.linspace(0, 3, s)) * 0.4
        return np.column_stack(((snd * 32767).astype(np.int16),) * 2).tobytes()

    def _load_background_music(self):
        self._create_background_music()

    def _create_background_music(self):
        self.temp_music_file = None
        try:
            from scipy import signal
            import tempfile, wave

            sr, dur, s, t = (
                22050,
                30,
                int(22050 * 30),
                np.linspace(0, 30, int(22050 * 30)),
            )
            d = (
                0.1 * np.sin(2 * np.pi * 80 * t) + 0.05 * np.sin(2 * np.pi * 120 * t)
            ) * (1 + 0.03 * np.sin(2 * np.pi * 0.1 * t))
            b, a = signal.butter(4, 0.1)
            tex = signal.filtfilt(b, a, 0.02 * np.random.uniform(-1, 1, s))
            m = np.clip(d + tex, -1, 1)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                self.temp_music_file = f.name
            with wave.open(self.temp_music_file, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(
                    np.column_stack(((m * 32767).astype(np.int16),) * 2).tobytes()
                )
        except (ImportError, FileNotFoundError):
            print("Warning: 'scipy' not found. Music generation will be skipped.")

    def play_background_music(self):
        if self.temp_music_file and os.path.exists(self.temp_music_file):
            try:
                pygame.mixer.music.load(self.temp_music_file)
                pygame.mixer.music.set_volume(self.music_volume)
                pygame.mixer.music.play(-1)
            except pygame.error:
                print("Could not load background music")

    def play_sound(self, name):
        if name in self.sound_effects:
            self.sound_effects[name].play()

    def cleanup(self):
        if hasattr(self, "temp_music_file") and self.temp_music_file:
            try:
                os.unlink(self.temp_music_file)
            except:
                pass


# === FIX APPLIED HERE: Replaced invalid list comprehension with a proper while loop ===
class AStar:
    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @classmethod
    def find_path(cls, grid, start, goal):
        rows, cols = len(grid), len(grid[0])
        if not (
            0 <= start[0] < cols
            and 0 <= start[1] < rows
            and 0 <= goal[0] < cols
            and 0 <= goal[1] < rows
            and grid[start[1]][start[0]] == 0
            and grid[goal[1]][goal[0]] == 0
        ):
            return []

        open_set, came_from, g_score = [(0, start)], {}, {start: 0}
        f_score = {start: cls.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (-1, -1),
                (-1, 1),
                (1, 1),
                (1, -1),
            ]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows)
                    or grid[neighbor[1]][neighbor[0]] == 1
                ):
                    continue

                tentative_g = g_score[current] + (1.414 if dx != 0 and dy != 0 else 1)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + cls.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []


class AdaptiveDifficulty:
    def __init__(self):
        self.stats = {
            "accuracy": [],
            "survival_time": [],
            "damage_taken": [],
            "enemies_killed": [],
        }
        self.level, self.rate = 1.0, 0.1

    def update_stats(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.stats:
                self.stats[k].append(v)
                if len(self.stats[k]) > 10:
                    self.stats[k].pop(0)

    def calculate_difficulty(self):
        if not self.stats["accuracy"]:
            return self.level
        p = (
            np.mean(self.stats["accuracy"][-5:]) * 0.6
            + min(np.mean(self.stats["survival_time"][-5:]) / 60, 1.0) * 0.4
        )
        if p > 0.7:
            self.level = min(self.level + self.rate, 3.0)
        elif p < 0.3:
            self.level = max(self.level - self.rate, 0.5)
        print(f"New Difficulty: {self.level:.2f}")
        return self.level


class Weapon:
    def __init__(self, n, d, r, fr, ac):
        (
            self.name,
            self.damage,
            self.range,
            self.fire_rate,
            self.ammo_capacity,
            self.current_ammo,
            self.last_shot_time,
        ) = (n, d, r, fr, ac, ac, 0)

    def can_shoot(self, t):
        return (t - self.last_shot_time) >= (
            1000 / self.fire_rate
        ) and self.current_ammo > 0

    def shoot(self, t):
        if self.can_shoot(t):
            self.current_ammo -= 1
            self.last_shot_time = t
            return True
        return False

    def reload(self):
        self.current_ammo = self.ammo_capacity


class Graphics:
    @staticmethod
    def create_background(w, h):
        bg = pygame.Surface((w, h))
        bg.fill(DARK_GREEN)
        for _ in range(200):
            pygame.draw.rect(
                bg,
                random.choice([FOREST_GREEN, SAND]),
                (
                    random.randint(0, w),
                    random.randint(0, h),
                    random.randint(40, 80),
                    random.randint(40, 80),
                ),
            )
        for _ in range(100):
            pygame.draw.circle(
                bg,
                DARK_GRAY,
                (random.randint(0, w), random.randint(0, h)),
                random.randint(2, 5),
            )
        return bg

    @staticmethod
    def draw_player(s, x, y, sz=20):
        pygame.draw.circle(s, (85, 107, 47), (int(x), int(y)), sz)
        pygame.draw.circle(s, (255, 220, 177), (int(x), int(y - sz // 2)), sz // 3)
        pygame.draw.circle(s, (64, 64, 64), (int(x), int(y - sz // 2)), sz // 3 + 2, 2)

    @staticmethod
    def draw_enemy(s, x, y, sz=20, st=EntityState.IDLE):
        color = {
            EntityState.ATTACKING: (139, 0, 0),
            EntityState.PURSUING: (139, 69, 19),
        }.get(st, (105, 105, 105))
        pygame.draw.circle(s, color, (int(x), int(y)), sz)
        pygame.draw.circle(s, (255, 220, 177), (int(x), int(y - sz // 2)), sz // 3)
        pygame.draw.circle(s, (40, 40, 40), (int(x), int(y - sz // 2)), sz // 3 + 1, 2)

    @staticmethod
    def draw_obstacle(s, o):
        m, b = {
            "wall": (GRAY, DARK_GRAY),
            "bunker": (DARK_GRAY, BLACK),
            "sandbag": (SAND, BROWN),
            "crate": (BROWN, BLACK),
        }[o["type"]]
        pygame.draw.rect(s, m, o["rect"])
        pygame.draw.rect(s, b, o["rect"], 3)

    @staticmethod
    def draw_explosion(s, x, y, sz=30, f=0, mf=10):
        r = sz * (f / mf)
        c = (*random.choice([YELLOW, ORANGE, RED]), max(0, 255 - (f / mf) * 255))
        surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, c, (r, r), r)
        s.blit(surf, (x - r, y - r))

    @staticmethod
    def draw_ammo_pack(s, x, y, sz=24):
        r = pygame.Rect(x - sz / 2, y - sz / 2, sz, sz)
        pygame.draw.rect(s, DARK_GRAY, r, border_radius=4)
        pygame.draw.rect(s, YELLOW, r.inflate(-4, -4), border_radius=4)
        for i in range(3):
            pygame.draw.rect(
                s, BLACK, (r.centerx - 2 + (i - 1) * 7, r.centery - 5, 4, 10)
            )


class AmmoPack:
    def __init__(self, x, y):
        self.position, self.size, self.rect = (
            Vector2(x, y),
            24,
            pygame.Rect(x - 12, y - 12, 24, 24),
        )

    def draw(self, s):
        Graphics.draw_ammo_pack(s, self.position.x, self.position.y, self.size)


class Entity:
    def __init__(self, x, y, h=100, sz=20):
        (
            self.position,
            self.health,
            self.max_health,
            self.alive,
            self.velocity,
            self.size,
        ) = (Vector2(x, y), h, h, True, Vector2(0, 0), sz)
        self.weapon, self.last_damager = Weapon("Rifle", 10, 300, 5, 30), None

    def take_damage(self, d, damager: "Entity"):
        self.health -= d
        self.last_damager = damager
        self.alive = self.health > 0

    def update(self, dt, obs):
        new_pos = self.position + self.velocity * dt
        r = pygame.Rect(
            new_pos.x - self.size / 2, new_pos.y - self.size / 2, self.size, self.size
        )
        if not any(o["rect"].colliderect(r) for o in obs):
            self.position = new_pos
        self.position.x = max(self.size, min(self.position.x, SCREEN_WIDTH - self.size))
        self.position.y = max(
            self.size, min(self.position.y, SCREEN_HEIGHT - self.size)
        )

    def draw_health_bar(self, s):
        p, size = (self.position.x - 20, self.position.y - self.size - 15), (40, 6)
        pygame.draw.rect(s, RED, (*p, *size))
        pygame.draw.rect(
            s, GREEN, (*p, size[0] * (self.health / self.max_health), size[1])
        )


class Player(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, h=200)
        self.speed = 250
        (
            self.shots_fired,
            self.shots_hit,
            self.enemies_killed,
            self.damage_taken_session,
        ) = (0, 0, 0, 0)
        self.start_time, self.score, self.waves_survived = pygame.time.get_ticks(), 0, 0

    def get_accuracy(self):
        return self.shots_hit / max(self.shots_fired, 1)

    def get_survival_time(self):
        return (pygame.time.get_ticks() - self.start_time) / 1000

    def calculate_score(self):
        self.score = (
            self.enemies_killed * 100
            + self.waves_survived * 500
            + int(self.get_accuracy() * 1000)
            + int(self.get_survival_time() * 10)
        )
        return self.score

    def handle_input(self, keys):
        self.velocity = Vector2(0, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.velocity.y = -1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.velocity.y = 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.velocity.x = -1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.velocity.x = 1
        if self.velocity.magnitude() > 0:
            self.velocity = self.velocity.normalize() * self.speed

    def shoot_at(self, tp: Vector2, g: "Game"):
        if self.weapon.shoot(pygame.time.get_ticks()):
            self.shots_fired += 1
            g.audio.play_sound("gunshot")
            d = (tp - self.position).normalize()
            g.bullets.append(
                Bullet(
                    self.position + d * (self.size + 5),
                    d * 700,
                    self.weapon.damage,
                    self,
                )
            )

    def draw(self, s):
        Graphics.draw_player(s, self.position.x, self.position.y, self.size)
        self.draw_health_bar(s)


class EnemyAI(Entity):
    def __init__(self, x, y, diff=1.0):
        super().__init__(x, y, h=50)
        self.diff = diff
        self.state, self.speed = EntityState.PATROLLING, 80 * diff
        self.det_range, self.atk_range = 350, 250
        self.patrol_pts = [
            Vector2(x, y),
            Vector2(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)),
        ]
        self.curr_patrol_idx = 0
        self.target = None
        self.path = []
        self.dec_timer, self.react_time = random.uniform(0, 1), max(
            0.2, 1.5 - (diff * 0.5)
        )

    def decide_action(self, p, grid):
        dist = self.position.distance_to(p.position)
        self.target, self.state = (
            (
                p,
                (
                    EntityState.ATTACKING
                    if dist <= self.atk_range
                    else EntityState.PURSUING
                ),
            )
            if dist < self.det_range
            else (None, EntityState.PATROLLING)
        )
        start = (
            int(self.position.x / GRID_CELL_SIZE),
            int(self.position.y / GRID_CELL_SIZE),
        )
        if self.target:
            self.path = AStar.find_path(
                grid,
                start,
                (
                    int(p.position.x / GRID_CELL_SIZE),
                    int(p.position.y / GRID_CELL_SIZE),
                ),
            )
        elif (
            not self.path
            or self.position.distance_to(self.patrol_pts[self.curr_patrol_idx]) < 50
        ):
            if self.position.distance_to(self.patrol_pts[self.curr_patrol_idx]) < 50:
                self.curr_patrol_idx = (self.curr_patrol_idx + 1) % len(self.patrol_pts)
                self.patrol_pts[self.curr_patrol_idx] = Vector2(
                    random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)
                )
            self.path = AStar.find_path(
                grid,
                start,
                (
                    int(self.patrol_pts[self.curr_patrol_idx].x / GRID_CELL_SIZE),
                    int(self.patrol_pts[self.curr_patrol_idx].y / GRID_CELL_SIZE),
                ),
            )

    def update(self, dt, p, obs, grid):
        self.dec_timer += dt
        if self.dec_timer > self.react_time:
            self.decide_action(p, grid)
            self.dec_timer = 0
        self.velocity = Vector2(0, 0)
        if self.path:
            t_pos = Vector2(
                self.path[0][0] * GRID_CELL_SIZE + GRID_CELL_SIZE / 2,
                self.path[0][1] * GRID_CELL_SIZE + GRID_CELL_SIZE / 2,
            )
            self.velocity = (t_pos - self.position).normalize() * self.speed
            if self.position.distance_to(t_pos) < GRID_CELL_SIZE:
                self.path.pop(0)
        super().update(dt, obs)

    def shoot_at(self, tp: Vector2, g: "Game"):
        if self.weapon.shoot(pygame.time.get_ticks()):
            g.audio.play_sound("gunshot")
            d = (tp - self.position).normalize()
            angle = math.atan2(d.y, d.x) + random.uniform(-0.1, 0.1) * (2.0 - self.diff)
            id = Vector2(math.cos(angle), math.sin(angle))
            g.bullets.append(
                Bullet(
                    self.position + id * (self.size + 5),
                    id * 500,
                    self.weapon.damage,
                    self,
                )
            )

    def draw(self, s):
        Graphics.draw_enemy(s, self.position.x, self.position.y, self.size, self.state)
        self.draw_health_bar(s)


@dataclass
class Bullet:
    position: Vector2
    velocity: Vector2
    damage: int
    owner: "Entity"
    lifetime: float = 2.0


@dataclass
class Effect:
    position: Vector2
    type: str
    duration: float
    max_duration: float


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tactical Survivor")
        self.clock, self.running, self.game_state = (
            pygame.time.Clock(),
            True,
            GameState.START_MENU,
        )
        self.audio, self.leaderboard, self.difficulty = (
            AudioManager(),
            Leaderboard(),
            AdaptiveDifficulty(),
        )
        self.font_lg, self.font_md, self.font_sm = (
            pygame.font.SysFont("Consolas", 64, True),
            pygame.font.SysFont("Consolas", 32),
            pygame.font.SysFont("Consolas", 18),
        )
        self.bg = Graphics.create_background(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.obstacles, self.grid, self.debug_path = [], [], False
        self.player_name_input = "Player"
        self.input_box = pygame.Rect(SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2, 300, 50)
        self.input_active = False

    def setup_new_game(self):
        self.player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        self.enemies, self.bullets, self.ammo_packs, self.effects = [], [], [], []
        self.wave, self.wave_cooldown, self.wave_timer = 0, 5, 5
        self.create_obstacles()
        self.update_grid()
        self.spawn_wave()

    def create_obstacles(self):
        self.obstacles = []
        safe_zone = pygame.Rect(
            SCREEN_WIDTH / 2 - 100, SCREEN_HEIGHT / 2 - 100, 200, 200
        )
        for _ in range(15):
            w, h = random.randint(50, 150), random.randint(50, 150)
            x, y = random.randint(0, SCREEN_WIDTH - w), random.randint(
                0, SCREEN_HEIGHT - h
            )
            if not pygame.Rect(x, y, w, h).colliderect(safe_zone):
                self.obstacles.append(
                    {
                        "rect": pygame.Rect(x, y, w, h),
                        "type": random.choice(["wall", "crate", "sandbag"]),
                    }
                )

    def update_grid(self):
        w, h = SCREEN_WIDTH // GRID_CELL_SIZE, SCREEN_HEIGHT // GRID_CELL_SIZE
        self.grid = [[0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if any(
                    o["rect"].colliderect(
                        pygame.Rect(
                            x * GRID_CELL_SIZE,
                            y * GRID_CELL_SIZE,
                            GRID_CELL_SIZE,
                            GRID_CELL_SIZE,
                        )
                    )
                    for o in self.obstacles
                ):
                    self.grid[y][x] = 1

    def spawn_wave(self):
        self.wave += 1
        self.player.waves_survived = self.wave
        num = 2 + self.wave * 2
        diff = self.difficulty.calculate_difficulty()
        for _ in range(num):
            x, y = random.choice(
                [
                    (random.randint(0, SCREEN_WIDTH), -50),
                    (random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT + 50),
                    (-50, random.randint(0, SCREEN_HEIGHT)),
                    (SCREEN_WIDTH + 50, random.randint(0, SCREEN_HEIGHT)),
                ]
            )
            self.enemies.append(EnemyAI(x, y, diff))
        if self.wave > 1 and self.wave % 3 == 0:
            self.ammo_packs.append(
                AmmoPack(
                    random.randint(50, SCREEN_WIDTH - 50),
                    random.randint(50, SCREEN_HEIGHT - 50),
                )
            )

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.debug_path = not self.debug_path

            if self.game_state == GameState.GET_PLAYER_NAME:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.input_active = self.input_box.collidepoint(event.pos)
                if event.type == pygame.KEYDOWN and self.input_active:
                    if event.key == pygame.K_RETURN:
                        name = (
                            self.player_name_input if self.player_name_input else "Anon"
                        )
                        self.leaderboard.add_score(
                            name,
                            self.player.score,
                            self.player.waves_survived,
                            self.player.enemies_killed,
                            self.player.get_accuracy(),
                            self.player.get_survival_time(),
                        )
                        self.game_state = GameState.LEADERBOARD
                    elif event.key == pygame.K_BACKSPACE:
                        self.player_name_input = self.player_name_input[:-1]
                    elif len(self.player_name_input) < 10:
                        self.player_name_input += event.unicode
                continue

            if self.game_state == GameState.PLAYING:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.player.shoot_at(Vector2(*pygame.mouse.get_pos()), self)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.player.weapon.reload()
                    self.audio.play_sound("reload")
            elif self.game_state == GameState.START_MENU:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.setup_new_game()
                        self.game_state = GameState.PLAYING
                    elif event.key == pygame.K_l:
                        self.game_state = GameState.LEADERBOARD
            elif (
                self.game_state in [GameState.GAME_OVER, GameState.LEADERBOARD]
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
            ):
                self.game_state = GameState.START_MENU

    def update(self, dt):
        if not self.player.alive:
            if self.game_state == GameState.PLAYING:
                score = self.player.calculate_score()
                if self.leaderboard.is_high_score(score):
                    self.player_name_input = ""
                    self.input_active = True
                    self.game_state = GameState.GET_PLAYER_NAME
                else:
                    self.game_state = GameState.GAME_OVER
            return

        self.player.handle_input(pygame.key.get_pressed())
        self.player.update(dt, self.obstacles)
        for e in self.enemies:
            e.update(dt, self.player, self.obstacles, self.grid)
            if e.state == EntityState.ATTACKING:
                e.shoot_at(self.player.position, self)
        for b in self.bullets[:]:
            b.position += b.velocity * dt
            b.lifetime -= dt
            if b.lifetime <= 0:
                self.bullets.remove(b)
                continue
            if (
                isinstance(b.owner, EnemyAI)
                and self.player.alive
                and b.position.distance_to(self.player.position) < self.player.size
            ):
                self.player.take_damage(b.damage, b.owner)
                self.player.damage_taken_session += b.damage
                self.bullets.remove(b)
            elif isinstance(b.owner, Player):
                for e in self.enemies:
                    if b.position.distance_to(e.position) < e.size:
                        e.take_damage(b.damage, b.owner)
                        self.player.shots_hit += 1
                        if b in self.bullets:
                            self.bullets.remove(b)
                        break
        for e in self.enemies[:]:
            if not e.alive:
                self.effects.append(Effect(e.position, "explosion", 0, 0.5))
                self.audio.play_sound("explosion")
                if isinstance(e.last_damager, Player):
                    self.player.enemies_killed += 1
                self.enemies.remove(e)
        pr = pygame.Rect(
            self.player.position.x - self.player.size / 2,
            self.player.position.y - self.player.size / 2,
            self.player.size,
            self.player.size,
        )
        for p in self.ammo_packs[:]:
            if pr.colliderect(p.rect):
                self.player.weapon.reload()
                self.audio.play_sound("pickup")
                self.ammo_packs.remove(p)
        for e in self.effects[:]:
            e.duration += dt
            if e.duration > e.max_duration:
                self.effects.remove(e)
        if not self.enemies:
            self.wave_timer -= dt
            if self.wave_timer <= 0:
                self.difficulty.update_stats(
                    accuracy=self.player.get_accuracy(),
                    survival_time=self.player.get_survival_time(),
                    damage_taken=self.player.damage_taken_session,
                    enemies_killed=self.player.enemies_killed,
                )
                self.spawn_wave()
                self.wave_timer = self.wave_cooldown

    def draw_hud(self):
        self.screen.blit(
            self.font_md.render(
                f"Health: {int(self.player.health)}", True, LIGHT_GREEN
            ),
            (10, 10),
        )
        self.screen.blit(
            self.font_md.render(
                f"Ammo: {self.player.weapon.current_ammo}/{self.player.weapon.ammo_capacity}",
                True,
                YELLOW,
            ),
            (10, 50),
        )
        st = self.font_md.render(f"Score: {self.player.calculate_score()}", True, WHITE)
        self.screen.blit(st, (SCREEN_WIDTH - st.get_width() - 10, 10))
        wt = self.font_md.render(f"Wave: {self.wave}", True, ORANGE)
        self.screen.blit(wt, (SCREEN_WIDTH - wt.get_width() - 10, 50))
        if not self.enemies:
            it = self.font_lg.render(
                f"Wave {self.wave + 1} in {math.ceil(self.wave_timer)}...", True, RED
            )
            self.screen.blit(
                it, it.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            )

    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        for o in self.obstacles:
            Graphics.draw_obstacle(self.screen, o)
        if self.debug_path:
            for y, r in enumerate(self.grid):
                for x, v in enumerate(r):
                    if v == 1:
                        pygame.draw.rect(
                            self.screen,
                            (100, 0, 0, 100),
                            (
                                x * GRID_CELL_SIZE,
                                y * GRID_CELL_SIZE,
                                GRID_CELL_SIZE,
                                GRID_CELL_SIZE,
                            ),
                        )
            for e in self.enemies:
                if e.path:
                    pygame.draw.lines(
                        self.screen,
                        CYAN,
                        False,
                        [
                            (
                                p[0] * GRID_CELL_SIZE + GRID_CELL_SIZE / 2,
                                p[1] * GRID_CELL_SIZE + GRID_CELL_SIZE / 2,
                            )
                            for p in e.path
                        ],
                        2,
                    )
        for p in self.ammo_packs:
            p.draw(self.screen)
        if self.player.alive:
            self.player.draw(self.screen)
        for e in self.enemies:
            e.draw(self.screen)
        for b in self.bullets:
            pygame.draw.circle(
                self.screen, YELLOW, (int(b.position.x), int(b.position.y)), 4
            )
        for e in self.effects:
            if e.type == "explosion":
                Graphics.draw_explosion(
                    self.screen,
                    e.position.x,
                    e.position.y,
                    f=int((e.duration / e.max_duration) * 10),
                )
        self.draw_hud()
        pygame.display.flip()

    def show_start_screen(self):
        self.screen.fill(DARK_GRAY)
        t = self.font_lg.render("Tactical Survivor", True, ORANGE)
        self.screen.blit(t, t.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3)))
        for i, (txt, k) in enumerate([("Start", "SPACE"), ("Leaderboard", "L")]):
            prompt = self.font_md.render(f"Press [{k}] for {txt}", True, WHITE)
            self.screen.blit(
                prompt,
                prompt.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + i * 50)),
            )
        pygame.display.flip()

    def show_name_input_screen(self):
        self.screen.fill(BLACK)
        title = self.font_lg.render("High Score!", True, GOLD)
        self.screen.blit(
            title, title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3))
        )
        prompt = self.font_md.render("Enter your name and press ENTER", True, WHITE)
        self.screen.blit(
            prompt, prompt.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50))
        )
        box_color = YELLOW if self.input_active else WHITE
        pygame.draw.rect(self.screen, box_color, self.input_box, 2)
        text_surface = self.font_md.render(self.player_name_input, True, WHITE)
        self.screen.blit(text_surface, (self.input_box.x + 10, self.input_box.y + 10))
        pygame.display.flip()

    def show_game_over_screen(self):
        self.screen.fill(BLACK)
        t = self.font_lg.render("GAME OVER", True, RED)
        self.screen.blit(t, t.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3)))
        st = self.font_md.render(f"Final Score: {self.player.score}", True, WHITE)
        self.screen.blit(st, st.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)))
        p = self.font_md.render("Press [SPACE] to return to menu", True, GRAY)
        self.screen.blit(
            p, p.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 100))
        )
        pygame.display.flip()

    def show_leaderboard_screen(self):
        self.screen.fill(DARK_GRAY)
        t = self.font_lg.render("Leaderboard", True, GOLD)
        self.screen.blit(t, t.get_rect(center=(SCREEN_WIDTH / 2, 80)))
        y = 180
        for txt, x in [("NAME", 150), ("SCORE", 400), ("WAVES", 600)]:
            self.screen.blit(self.font_sm.render(txt, True, WHITE), (x, y))
        y += 40
        for i, e in enumerate(self.leaderboard.get_top_scores()):
            c = [GOLD, SILVER, BRONZE][i] if i < 3 else WHITE
            self.screen.blit(self.font_md.render(f"{i + 1}.", True, c), (100, y))
            self.screen.blit(self.font_md.render(e["name"], True, c), (150, y))
            self.screen.blit(self.font_md.render(str(e["score"]), True, c), (400, y))
            self.screen.blit(self.font_md.render(str(e["waves"]), True, c), (600, y))
            y += 40
        p = self.font_md.render("Press [SPACE] to return", True, GRAY)
        self.screen.blit(p, p.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50)))
        pygame.display.flip()

    def run(self):
        self.audio.play_background_music()
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            if self.game_state == GameState.PLAYING:
                self.update(dt)
                self.draw()
            elif self.game_state == GameState.START_MENU:
                self.show_start_screen()
            elif self.game_state == GameState.GET_PLAYER_NAME:
                self.show_name_input_screen()
            elif self.game_state == GameState.GAME_OVER:
                self.show_game_over_screen()
            elif self.game_state == GameState.LEADERBOARD:
                self.show_leaderboard_screen()
        self.audio.cleanup()
        self.leaderboard.close()
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
