"""
dino_dqn_v9_infinite.py — Ночная полировка на высоких скоростях + бесконечный JS-режим без задержек (0 мс).
"""

import os
import json
import time
import math
import random
import asyncio
import copy
import threading
import re
import http.server
import socketserver
from functools import partial
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from playwright.async_api import async_playwright

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
GAME_DIR = r"C:\dino_coch\game"
GAME_URL = None

NUM_WORKERS = 1
TOTAL_EPISODES = 15000
GAMMA = 0.95
LR = 1e-4
BATCH_SIZE = 128
REPLAY_SIZE = 500000
TAU = 0.005
TRAIN_START = 300000
TRAIN_RATE_SLEEP = 0.025
INFERENCE_SYNC_EVERY = 100

EPS_START = 0.5
EPS_END = 0.03
EPS_DECAY_K = 3.0
EPS_BOOST_VALUE = 0.08
EPS_BOOST_DURATION = 150

HIGH_START_SPEED = 6.0
STEP_DELAY = 0.010
MAX_STEPS = 25000
N_STEP = 3

PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.5
PRIORITY_BETA_STEPS = 50000

BASE_V8_MODEL = "best_dqn_model_v8.pt"
MODEL_FILE = "dqn_model_v9.pt"
BEST_MODEL_FILE = "best_dqn_model_v9.pt"
STATE_FILE = "dqn_state_v9.json"
STATS_DIR = "stats_v9"

# --- РЕЖИМЫ РАБОТЫ ---
# TRAIN_MODE = True  -> ночное обучение
# TRAIN_MODE = False -> бесконечный JS-режим (0 мс задержка)
TRAIN_MODE = False
LOAD_MODEL = True
SAVE_MODEL = False
HEADLESS = False
SEED = 42

WIN_W = 650
WIN_H = 350

AGENT_NAMES = ["Эль патрон", "Эль дрифто", "Пантера", "Серцеед", "Коч братан", "Никита"]
NOOP, JUMP, DUCK = 0, 1, 2
ACTIONS = [NOOP, JUMP, DUCK]

# ==============================================================================
# 2. ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# ==============================================================================
episodes_done = 0
training_steps = 0
start_episode = 0
target_episodes = TOTAL_EPISODES
epsilon_boost_until = 0
best_avg_score = 0.0
agent_stats = {}
save_lock = asyncio.Lock()
model_lock = threading.Lock()

os.makedirs(STATS_DIR, exist_ok=True)

for _name in AGENT_NAMES:
    agent_stats[_name] = {
        "max_score": 0, "min_score": 99999, "total_reward": 0.0,
        "episodes_played": 0, "last_100_scores": [], "avg_last_100": 0.0,
    }

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def start_local_server(game_dir, port=0):
    handler = partial(QuietHTTPHandler, directory=game_dir)
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


# ==============================================================================
# 2.5. ПАТЧ index.js — ПТИЦЫ С ПЕРВОГО ШАГА
# ==============================================================================
def ensure_pterodactyl_from_start():
    path = os.path.join(GAME_DIR, "index.js")
    if not os.path.exists(path):
        print(f"⚠️  {path} не найден, патч птиц пропущен")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    ptero_pattern = re.compile(
        r"(type:\s*['\"]PTERODACTYL['\"][^}]*?minSpeed:\s*)([\d.]+)",
        re.DOTALL
    )

    def replacer(match):
        old_val = match.group(2)
        if old_val == "0":
            return match.group(0)
        return match.group(1) + "0"

    new_content, count = ptero_pattern.subn(replacer, content)

    if count == 0:
        print(f"⚠️  PTERODACTYL не найден в {path}. Проверь формат.")
        return

    if new_content == content:
        print("✅ index.js уже пропатчен: птицы с minSpeed=0")
        return

    backup_path = path + ".original"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"💾 Бэкап оригинала: {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("🔧 index.js пропатчен: птицы доступны с первого шага (minSpeed=0)")


# ==============================================================================
# 3. НЕЙРОСЕТЬ (11 ВХОДОВ -> 256 -> 256 -> 3 ДЕЙСТВИЯ)
# ==============================================================================
class DQN(nn.Module):
    def __init__(self, in_dim=11, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


model = DQN().to(DEVICE)
target_model = DQN().to(DEVICE)
target_model.load_state_dict(model.state_dict())
target_model.eval()
inference_model = DQN().to(DEVICE)
inference_model.load_state_dict(model.state_dict())
inference_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)


def export_weights_to_json():
    with model_lock:
        sd = inference_model.state_dict()
        weights = {
            "w1": sd["net.0.weight"].detach().cpu().numpy().tolist(),
            "b1": sd["net.0.bias"].detach().cpu().numpy().tolist(),
            "w2": sd["net.2.weight"].detach().cpu().numpy().tolist(),
            "b2": sd["net.2.bias"].detach().cpu().numpy().tolist(),
            "w3": sd["net.4.weight"].detach().cpu().numpy().tolist(),
            "b3": sd["net.4.bias"].detach().cpu().numpy().tolist(),
        }
    return json.dumps(weights)


# ==============================================================================
# 4. PRIORITIZED REPLAY BUFFER
# ==============================================================================
class PrioritizedReplay:
    def __init__(self, capacity=REPLAY_SIZE, alpha=PRIORITY_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, s2, d):
        max_prio = float(self.priorities[:self.size].max()) if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((s, a, r, s2, d))
            self.size += 1
        else:
            self.buffer[self.pos] = (s, a, r, s2, d)
        self.priorities[self.pos] = max(max_prio, 1e-5) ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n, beta):
        prios = self.priorities[:self.size]
        probs = prios / prios.sum()
        indices = np.random.choice(self.size, n, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]
        s, a, r, s2, d = zip(*batch)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = (abs(float(err)) + 1e-5) ** self.alpha

    def __len__(self):
        return self.size


replay = PrioritizedReplay()


# ==============================================================================
# 5. N-STEP RETURNS
# ==============================================================================
class NStepBuffer:
    def __init__(self, n=N_STEP, gamma=GAMMA):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def get_transition(self):
        if len(self.buffer) < self.n:
            return None
        first_item = self.buffer[0]
        s0, a0 = first_item[0], first_item[1]
        R, gamma_pow = 0.0, 1.0
        for i in range(self.n):
            item = self.buffer[i]
            r, done = item[2], item[4]
            R += gamma_pow * r
            gamma_pow *= self.gamma
            if done:
                return (s0, a0, R, item[3], True)
        last_item = self.buffer[self.n - 1]
        return (s0, a0, R, last_item[3], last_item[4])

    def flush_partial(self):
        result = []
        while len(self.buffer) > 0:
            first_item = self.buffer[0]
            s0, a0 = first_item[0], first_item[1]
            R, gamma_pow = 0.0, 1.0
            for i in range(len(self.buffer)):
                item = self.buffer[i]
                r, done = item[2], item[4]
                R += gamma_pow * r
                gamma_pow *= self.gamma
                if done:
                    break
            last_item = self.buffer[-1]
            result.append((s0, a0, R, last_item[3], last_item[4]))
            self.buffer.popleft()
        return result

    def clear(self):
        self.buffer.clear()


# ==============================================================================
# 6. ВЫБОР ДЕЙСТВИЯ И ОБУЧЕНИЕ
# ==============================================================================
def nn_choose_action(state_vec, epsilon):
    if TRAIN_MODE and random.random() < epsilon:
        return random.choice(ACTIONS)
    with torch.no_grad():
        x = torch.FloatTensor(state_vec).unsqueeze(0).to(DEVICE)
        return int(inference_model(x).argmax().item())


def soft_update():
    with model_lock:
        for tp, p in zip(target_model.parameters(), model.parameters()):
            tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)


def sync_inference():
    with model_lock:
        inference_model.load_state_dict(model.state_dict())


def train_step(batch):
    s, a, r, s2, d, indices, weights = batch
    s_t = torch.FloatTensor(s).to(DEVICE)
    a_t = torch.LongTensor(a).to(DEVICE)
    r_t = torch.FloatTensor(r).to(DEVICE)
    s2_t = torch.FloatTensor(s2).to(DEVICE)
    d_t = torch.FloatTensor(d).to(DEVICE)
    w_t = torch.FloatTensor(weights).to(DEVICE)

    with model_lock:
        q = model(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_actions = model(s2_t).argmax(1, keepdim=True)
            q_next = target_model(s2_t).gather(1, best_actions).squeeze(1)
            target = r_t + (GAMMA ** N_STEP) * q_next * (1.0 - d_t)

    td_errors = (q - target).detach().cpu().numpy()
    losses = torch.nn.functional.huber_loss(q, target, reduction='none', delta=2.0)
    loss = (w_t * losses).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    replay.update_priorities(indices, td_errors)
    return float(loss.item())


def features(st):
    return np.array([
        min(st["distance"], 500.0) / 500.0,
        min(st["width"], 100.0) / 100.0,
        min(st["yPos"], 150.0) / 150.0,
        min(st["speed"], 15.0) / 15.0,
        1.0 if st["isBird"] else 0.0,
        1.0 if st["jumping"] else 0.0,
        min(st["distance"] + st["width"], 600.0) / 600.0,
        1.0 if st["isLarge"] else 0.0,
        min(st["gap"], 500.0) / 500.0,
        min(st["yPos2"], 150.0) / 150.0,
        1.0 if st["isBird2"] else 0.0,
    ], dtype=np.float32)


def compute_reward(action, st, cleared_obstacle, step, agent_record, died):
    if died:
        return -15.0

    reward = 0.02
    if cleared_obstacle:
        reward += 1.0

    distance = st["distance"]
    speed = st["speed"]
    is_bird = st["isBird"]
    y_pos = st["yPos"]

    speed_factor = max(1.0, speed / 6.0)
    danger_zone = 240.0 * speed_factor

    if distance <= danger_zone:
        if is_bird:
            if y_pos < 60:
                # Высокая птица — не прыгать
                if action == NOOP:
                    reward += 0.15
                elif action == JUMP:
                    reward -= 0.15
            elif y_pos < 90:
                # Средняя птица — подкат
                if action == DUCK:
                    reward += 0.25
                elif action == JUMP:
                    reward -= 0.30
                elif action == NOOP:
                    reward -= 0.20
            else:
                # Низкая птица (yPos ~100) — нужно ПРЫГАТЬ
                if action == JUMP:
                    reward += 0.20
                elif action == DUCK:
                    reward -= 0.25
                elif action == NOOP:
                    reward -= 0.20
        else:
            # Кактус — прыгать
            if action == JUMP:
                reward += 0.20
            elif action == DUCK:
                reward -= 0.25
    else:
        if action == JUMP and distance > 450:
            reward -= 0.25

    if step == agent_record + 1:
        reward += 0.5

    return reward


def compute_epsilon():
    if not TRAIN_MODE:
        return 0.0
    if training_steps < epsilon_boost_until:
        return EPS_BOOST_VALUE
    span = max(1, target_episodes - start_episode)
    progress = (episodes_done - start_episode) / span
    progress = max(0.0, min(1.0, progress))
    return EPS_END + (EPS_START - EPS_END) * math.exp(-progress * EPS_DECAY_K)


def compute_beta():
    frac = min(1.0, training_steps / PRIORITY_BETA_STEPS)
    return PRIORITY_BETA_START + frac * (1.0 - PRIORITY_BETA_START)


# ==============================================================================
# 7. СОХРАНЕНИЕ / ЗАГРУЗКА
# ==============================================================================
def _write_json(path, data):
    tmp = f"{path}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        if os.path.exists(tmp):
            os.replace(tmp, path)
    except Exception as e:
        print(f"⚠️  JSON {path}: {e}")


def safe_filename(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def load_all():
    global episodes_done, training_steps, start_episode, target_episodes, best_avg_score

    target_path = None
    if not TRAIN_MODE:
        for candidate in [BEST_MODEL_FILE, MODEL_FILE, BASE_V8_MODEL]:
            if os.path.exists(candidate):
                target_path = candidate
                break
    else:
        for candidate in [MODEL_FILE, BEST_MODEL_FILE, BASE_V8_MODEL]:
            if os.path.exists(candidate):
                target_path = candidate
                break

    if LOAD_MODEL and target_path:
        checkpoint = torch.load(target_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        target_model.load_state_dict(checkpoint.get("target", checkpoint["model"]))
        inference_model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and TRAIN_MODE and target_path == MODEL_FILE:
            optimizer.load_state_dict(checkpoint["optimizer"])
            training_steps = int(checkpoint.get("training_steps", 0))
        print(f"✅ Загружены веса нейросети из {target_path} (steps={training_steps})")
    else:
        print("🆕 Создана новая нейросеть DQN (11 входов -> 256 -> 256 -> 3).")

    if LOAD_MODEL and os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        episodes_done = int(data.get("episodes_done", 0))
        best_avg_score = float(data.get("best_avg_score", 0.0))

    start_episode = episodes_done
    target_episodes = episodes_done + TOTAL_EPISODES
    print(f"Старт с эпизода {start_episode}, цель: {target_episodes}, лучший средний счёт: {best_avg_score:.1f}")


async def save_all():
    global best_avg_score
    async with save_lock:
        if not SAVE_MODEL:
            return

        current_max_avg = max(
            (s["avg_last_100"] for s in agent_stats.values() if s["episodes_played"] >= 30),
            default=0.0
        )
        is_new_best = current_max_avg > best_avg_score
        if is_new_best:
            best_avg_score = current_max_avg

        def _save():
            with model_lock:
                payload = {
                    "model": model.state_dict(),
                    "target": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "training_steps": training_steps,
                }
                torch.save(payload, MODEL_FILE)
                if is_new_best:
                    torch.save(payload, BEST_MODEL_FILE)

        try:
            await asyncio.to_thread(_save)
            if is_new_best:
                print(f"🌟 НОВЫЙ НОЧНОЙ РЕКОРД СРЕДНЕГО СЧЁТА ({best_avg_score:.1f})! Сохранено в {BEST_MODEL_FILE}")
        except Exception as e:
            print(f"⚠️  Ошибка сохранения весов: {e}")

        await asyncio.to_thread(_write_json, STATE_FILE, {
            "episodes_done": episodes_done,
            "best_avg_score": best_avg_score
        })

        stats_copy = copy.deepcopy(agent_stats)
        for name, stats in stats_copy.items():
            await asyncio.to_thread(
                _write_json,
                f"{STATS_DIR}/stats_{safe_filename(name)}.json",
                stats
            )
        print(f"[Сохранение] эп={episodes_done}/{target_episodes} "
              f"steps={training_steps} ε={compute_epsilon():.3f} "
              f"buffer={len(replay)}")


# ==============================================================================
# 8. ФОНОВЫЕ ЗАДАЧИ ОБУЧЕНИЯ
# ==============================================================================
async def trainer_task(stop_event):
    global training_steps, epsilon_boost_until
    last_log = time.time()
    collapse_counter = 0

    while not stop_event.is_set():
        if not TRAIN_MODE or len(replay) < TRAIN_START:
            await asyncio.sleep(0.2)
            continue

        beta = compute_beta()
        batch = replay.sample(BATCH_SIZE, beta)
        try:
            loss = await asyncio.to_thread(train_step, batch)
        except Exception as e:
            print(f"[Trainer] {e}")
            await asyncio.sleep(0.5)
            continue

        training_steps += 1
        soft_update()
        if training_steps % INFERENCE_SYNC_EVERY == 0:
            sync_inference()

        if loss < 0.0005:
            collapse_counter += 1
        else:
            collapse_counter = 0

        if collapse_counter >= 2000:
            epsilon_boost_until = training_steps + EPS_BOOST_DURATION
            print(f"[Trainer] ⚠️  Встряска исследования: ε={EPS_BOOST_VALUE} на {EPS_BOOST_DURATION} шагов")
            collapse_counter = 0

        if time.time() - last_log > 20:
            print(f"[Trainer] steps={training_steps} loss={loss:.4f} "
                  f"buffer={len(replay)} ε={compute_epsilon():.3f} β={beta:.2f}")
            last_log = time.time()

        await asyncio.sleep(TRAIN_RATE_SLEEP)


async def saver_task(stop_event):
    while not stop_event.is_set():
        await asyncio.sleep(60)
        if stop_event.is_set():
            break
        try:
            await save_all()
        except Exception as e:
            print(f"[Saver] {e}")


# ==============================================================================
# 9. СВЯЗЬ С ИГРОЙ И ВСТРОЕННЫЙ JS-ИНФЕРЕНС (0 МС ЗАДЕРЖКИ)
# ==============================================================================
GET_STATE_JS = """
() => {
    const runner = window.Runner?.instance_;
    if (!runner) return { status: "NO_RUNNER" };
    if (runner.crashed) return { status: "DEAD" };
    if (!runner.playing) return { status: "NOT_PLAYING" };

    const dino = runner.tRex;
    const dinoWidth = dino.config ? dino.config.WIDTH : 44;
    const dinoRight = dino.xPos + dinoWidth;
    const obstacles = runner.horizon?.obstacles || [];

    let upcoming = [];
    for (const o of obstacles) {
        const d = o.xPos - dinoRight;
        if (d > -o.width) {
            upcoming.push({ obs: o, dist: d });
            if (upcoming.length === 2) break;
        }
    }

    if (upcoming.length === 0) {
        return {
            status: "OK", distance: 1000, speed: runner.currentSpeed,
            yPos: 105, jumping: !!dino.jumping,
            isBird: false, isLarge: false, width: 20,
            gap: 1000, yPos2: 105, isBird2: false
        };
    }

    const first = upcoming[0];
    const best = first.obs;
    const typeName = String(
        (best.typeConfig && best.typeConfig.type) || best.type || "UNKNOWN"
    ).toUpperCase();

    let gap = 1000, yPos2 = 105, isBird2 = false;
    if (upcoming.length > 1) {
        const second = upcoming[1].obs;
        gap = second.xPos - (best.xPos + best.width);
        yPos2 = second.yPos;
        const type2 = String(
            (second.typeConfig && second.typeConfig.type) || second.type || "UNKNOWN"
        ).toUpperCase();
        isBird2 = (type2 === "PTERODACTYL");
    }

    return {
        status: "OK",
        distance: first.dist,
        speed: runner.currentSpeed,
        yPos: best.yPos,
        jumping: !!dino.jumping,
        isBird: (typeName === "PTERODACTYL"),
        isLarge: (typeName === "CACTUS_LARGE"),
        width: best.width,
        gap: gap,
        yPos2: yPos2,
        isBird2: isBird2
    };
}
"""

INJECT_JS_BRAIN_TEMPLATE = """
({ weightsJson }) => {
    const W = JSON.parse(weightsJson);

    function denseRelu(input, weight, bias) {
        const out = new Array(bias.length);
        for (let i = 0; i < bias.length; i++) {
            let sum = bias.at(i);
            const row = weight.at(i);
            for (let j = 0; j < input.length; j++) {
                sum += row.at(j) * input.at(j);
            }
            out[i] = sum > 0 ? sum : 0;
        }
        return out;
    }

    function denseLinear(input, weight, bias) {
        const out = new Array(bias.length);
        for (let i = 0; i < bias.length; i++) {
            let sum = bias.at(i);
            const row = weight.at(i);
            for (let j = 0; j < input.length; j++) {
                sum += row.at(j) * input.at(j);
            }
            out[i] = sum;
        }
        return out;
    }

    function predictAction(features) {
        const h1 = denseRelu(features, W.w1, W.b1);
        const h2 = denseRelu(h1, W.w2, W.b2);
        const q = denseLinear(h2, W.w3, W.b3);
        const q0 = q.at(0), q1 = q.at(1), q2 = q.at(2);
        if (q1 > q0 && q1 >= q2) return 1;
        if (q2 > q0 && q2 > q1) return 2;
        return 0;
    }

    if (window.__aiLoopRunning) return true;
    window.__aiLoopRunning = true;
    let isDucking = false;

    function aiLoop() {
        const runner = window.Runner?.instance_;
        
        if (runner) {
            if (runner.distanceMeter) {
                runner.distanceMeter.maxScore = Number.MAX_SAFE_INTEGER;
            }
            // Удерживаем скорость в идеальном для модели темпе
            if (runner.currentSpeed > 10.6) {
                runner.currentSpeed = 10.6;
            }
        }

        if (runner && runner.playing && !runner.crashed && !runner.playingIntro) {
            const dino = runner.tRex;
            if (dino && !dino.jumping) {
                const dinoWidth = dino.config ? dino.config.WIDTH : 44;
                const dinoRight = dino.xPos + dinoWidth;
                const obstacles = runner.horizon?.obstacles || [];

                let upcoming = [];
                for (const o of obstacles) {
                    const d = o.xPos - dinoRight;
                    if (d > -o.width) {
                        upcoming.push({ obs: o, dist: d });
                        if (upcoming.length === 2) break;
                    }
                }

                let dist = 1000, width = 20, yPos = 105, isBird = 0, isLarge = 0;
                let gap = 1000, yPos2 = 105, isBird2 = 0;

                if (upcoming.length > 0) {
                    const firstItem = upcoming.at(0);
                    const best = firstItem.obs;
                    dist = firstItem.dist;
                    width = best.width;
                    yPos = best.yPos;
                    const t1 = String((best.typeConfig && best.typeConfig.type) || best.type || "").toUpperCase();
                    isBird = (t1 === "PTERODACTYL") ? 1.0 : 0.0;
                    isLarge = (t1 === "CACTUS_LARGE") ? 1.0 : 0.0;

                    if (upcoming.length > 1) {
                        const second = upcoming.at(1).obs;
                        gap = second.xPos - (best.xPos + best.width);
                        yPos2 = second.yPos;
                        const t2 = String((second.typeConfig && second.typeConfig.type) || second.type || "").toUpperCase();
                        isBird2 = (t2 === "PTERODACTYL") ? 1.0 : 0.0;
                    }
                }

                const feat = [
                    Math.min(dist, 500.0) / 500.0,
                    Math.min(width, 100.0) / 100.0,
                    Math.min(yPos, 150.0) / 150.0,
                    Math.min(runner.currentSpeed, 15.0) / 15.0,
                    isBird,
                    dino.jumping ? 1.0 : 0.0,
                    Math.min(dist + width, 600.0) / 600.0,
                    isLarge,
                    Math.min(gap, 500.0) / 500.0,
                    Math.min(yPos2, 150.0) / 150.0,
                    isBird2
                ];

                let action = predictAction(feat);
                if (dist > 550 && action === 1) action = 0;

                // Страховочный щит вплотную к препятствию
                if (dist > -10 && dist < 75) {
                    if (isBird === 1.0 && yPos < 90) {
                        action = 2;
                    } else {
                        action = 1;
                    }
                }

                if (action === 1) {
                    if (isDucking) {
                        dino.setDuck(false);
                        isDucking = false;
                    }
                    if (!dino.jumping && !dino.ducking) {
                        dino.startJump(runner.currentSpeed);
                    }
                } else if (action === 2) {
                    if (!isDucking && !dino.jumping) {
                        dino.setDuck(true);
                        isDucking = true;
                    }
                } else {
                    if (isDucking) {
                        dino.setDuck(false);
                        isDucking = false;
                    }
                }
            }
        } else if (isDucking && runner?.tRex) {
            runner.tRex.setDuck(false);
            isDucking = false;
        }

        requestAnimationFrame(aiLoop);
    }

    requestAnimationFrame(aiLoop);
    return true;
}
"""



async def force_restart_js(page):
    try:
        result = await asyncio.wait_for(page.evaluate("""
() => {
    const r = window.Runner?.instance_;
    if (!r) return false;
    try { r.restart(); return true; } catch (e) { return false; }
}
"""), timeout=2.0)
        return bool(result)
    except Exception:
        return False


async def boost_speed(page, target_speed):
    if target_speed <= 6.0:
        return
    try:
        await asyncio.wait_for(page.evaluate(f"""
() => {{
    const r = window.Runner?.instance_;
    if (r && r.currentSpeed < {target_speed}) {{
        r.currentSpeed = {target_speed};
    }}
}}
"""), timeout=1.0)
    except Exception:
        pass


async def get_state(page):
    try:
        result = await asyncio.wait_for(page.evaluate(GET_STATE_JS), timeout=2.0)
        if result["status"] != "OK":
            return None, result["status"]
        return {
            "distance": float(result["distance"]),
            "width": float(result["width"]),
            "yPos": float(result["yPos"]),
            "speed": float(result["speed"]),
            "isBird": bool(result["isBird"]),
            "isLarge": bool(result.get("isLarge", False)),
            "jumping": bool(result["jumping"]),
            "gap": float(result["gap"]),
            "yPos2": float(result["yPos2"]),
            "isBird2": bool(result["isBird2"]),
        }, "OK"
    except asyncio.TimeoutError:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"ERROR: {e}"


async def start_game(page, target_speed=6.0, weights_json=None, agent_name=None):
    await page.keyboard.up("ArrowDown")
    await asyncio.sleep(0.1)

    for round_num in (1, 2, 3):
        if round_num == 2:
            await force_restart_js(page)
            await asyncio.sleep(0.15)
        elif round_num == 3:
            # КРАЙНЯЯ АВАРИЙНАЯ СТРАХОВКА: reload только если игра полностью зависла
            try:
                await page.reload(wait_until="load")
                await asyncio.sleep(0.2)
                await page.mouse.click(WIN_W // 2, WIN_H // 2)
                await asyncio.sleep(0.1)
                # Ждём инициализации движка игры перед реинъекцией
                await page.wait_for_function(
                    "() => !!window.Runner?.instance_",
                    timeout=5000
                )
                # ПОВТОРНАЯ ИНЪЕКЦИЯ МОЗГА ПОСЛЕ RELOAD
                if not TRAIN_MODE and weights_json and agent_name:
                    await page.evaluate(
                        f"({INJECT_JS_BRAIN_TEMPLATE})",
                        {"weightsJson": weights_json, "agentName": agent_name}
                    )
            except Exception:
                return None, "RELOAD_FAILED"

        for _ in range(4):
            try:
                await page.mouse.click(WIN_W // 2, WIN_H // 2)
                await page.keyboard.down("Space")
                await asyncio.sleep(0.04)
                await page.keyboard.up("Space")
            except Exception:
                pass

        for _ in range(5):
            await asyncio.sleep(0.04)

        st, _ = await get_state(page)
        if st is not None:
            await boost_speed(page, target_speed)
            return st, "OK"

    return None, "START_FAILED"


# ==============================================================================
# 10. ВОРКЕР АГЕНТА (ОБУЧЕНИЕ ИЛИ БЕСКОНЕЧНЫЙ JS-ЗАБЕГ)
# ==============================================================================
async def worker(worker_id, agent_name, context, stop_event, weights_json=None):
    global episodes_done
    try:
        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda: stop_event.set())

        await page.add_init_script(f"document.title = '{agent_name}';")
        await page.goto(GAME_URL, wait_until="load")
        await asyncio.sleep(0.3)
        await page.bring_to_front()
        await page.mouse.click(WIN_W // 2, WIN_H // 2)
        await asyncio.sleep(0.2)

        if not TRAIN_MODE:
            await page.evaluate(
                f"({INJECT_JS_BRAIN_TEMPLATE})",
                {"weightsJson": weights_json, "agentName": agent_name}
            )

        # ---------- БЕСКОНЕЧНЫЙ JS-РЕЖИМ ----------
        if not TRAIN_MODE:
            run_num = 0
            # Скорость старта ближе к тренировочной (10.2 для 4 агентов, 9.0 для остальных)
            demo_speed = HIGH_START_SPEED if worker_id <= 4 else 9.0
            while not stop_event.is_set():
                run_num += 1
                st, _ = await start_game(
                    page,
                    target_speed=demo_speed,
                    weights_json=weights_json,
                    agent_name=agent_name
                )
                if st is None:
                    print(f"[{agent_name}] Не удалось запустить игру, retry через 1с")
                    await asyncio.sleep(1.0)
                    continue

                print(f"🚀 [{agent_name}] Бесконечный забег #{run_num} запущен "
                      f"(скорость {demo_speed}, 0 мс задержка)!")

                while not stop_event.is_set():
                    await asyncio.sleep(1.0)
                    try:
                        status_info = await page.evaluate("""
() => {
    const r = window.Runner?.instance_;
    if (!r) return { crashed: true, score: 0, speed: 0 };
    const score = Math.floor(r.distanceRan * 0.025);
    return { crashed: !!r.crashed, score: score, speed: r.currentSpeed };
}
""")
                    except Exception:
                        status_info = {"crashed": True, "score": 0, "speed": 0}

                    if status_info["crashed"]:
                        print(f"💥 [{agent_name}] Забег #{run_num} окончен на счёте: "
                              f"{status_info['score']} очков (скорость {status_info['speed']:.1f}). Перезапуск...")
                        break
                    elif status_info["score"] > 0 and status_info["score"] % 5000 < 50:
                        print(f"🔥 [{agent_name}] В пути: {status_info['score']} очков "
                              f"(скорость {status_info['speed']:.2f})")

                # после краша — цикл продолжается, новый забег
                await asyncio.sleep(0.3)

            return  # выходим только если stop_event

        # ---------- РЕЖИМ ОБУЧЕНИЯ ----------
        worker_start_speed = HIGH_START_SPEED if worker_id <= 4 else 6.0
        is_ducking = False
        nstep = NStepBuffer()

        while episodes_done < target_episodes and not stop_event.is_set():
            episodes_done += 1
            ep_num = episodes_done
            epsilon = compute_epsilon()

            if is_ducking:
                await page.keyboard.up("ArrowDown")
                is_ducking = False

            st, status = await start_game(
                page,
                target_speed=worker_start_speed,
                weights_json=weights_json,
                agent_name=agent_name
            )
            nstep.clear()

            if st is None:
                episodes_done -= 1
                await asyncio.sleep(0.3)
                continue

            total_reward = 0.0
            jump_decision = None
            step = 0
            agent_record = agent_stats[agent_name]["max_score"]
            prev_dist = st["distance"]

            for step in range(1, MAX_STEPS + 1):
                if stop_event.is_set():
                    break

                if st["jumping"]:
                    await asyncio.sleep(STEP_DELAY)
                    nst, status = await get_state(page)
                    if nst is None:
                        if jump_decision is not None:
                            s_jump, a_jump = jump_decision
                            s2_dummy = np.zeros(11, dtype=np.float32)
                            nstep.push(s_jump, a_jump, -15.0, s2_dummy, True)
                            for trans in nstep.flush_partial():
                                replay.push(*trans)
                            total_reward -= 15.0
                            print(f"[{agent_name} | Эп.{ep_num}] Смерть в прыжке, шагов: {step}")
                        break
                    st = nst
                    continue

                s_vec = features(st)
                action = nn_choose_action(s_vec, epsilon)

                if st["distance"] > 550 and action == JUMP:
                    action = NOOP

                if action == JUMP:
                    jump_decision = (s_vec, action)
                    if is_ducking:
                        await page.keyboard.up("ArrowDown")
                        is_ducking = False
                    await page.keyboard.press("Space")
                elif action == DUCK:
                    jump_decision = None
                    if not is_ducking:
                        await page.keyboard.down("ArrowDown")
                        is_ducking = True
                else:
                    jump_decision = None
                    if is_ducking:
                        await page.keyboard.up("ArrowDown")
                        is_ducking = False

                await asyncio.sleep(STEP_DELAY)
                nst, status = await get_state(page)

                if nst is None:
                    died = (status == "DEAD")
                    reward = compute_reward(action, st, False, step, agent_record, died)
                    s2_dummy = np.zeros(11, dtype=np.float32)
                    nstep.push(s_vec, action, reward, s2_dummy, died)
                    for trans in nstep.flush_partial():
                        replay.push(*trans)
                    total_reward += reward
                    print(f"[{agent_name} | Эп.{ep_num}] Конец ({status}). "
                          f"Шагов: {step}, R: {total_reward:.2f}, ε={epsilon:.3f}")
                    break

                cleared = (prev_dist < 40 and nst["distance"] > prev_dist + 100)
                prev_dist = nst["distance"]
                s2 = features(nst)
                reward = compute_reward(action, st, cleared, step, agent_record, False)
                nstep.push(s_vec, action, reward, s2, False)

                trans = nstep.get_transition()
                if trans is not None:
                    replay.push(*trans)
                total_reward += reward
                st = nst

            if step == MAX_STEPS and not stop_event.is_set():
                print(f"[{agent_name} | Эп.{ep_num}] Лимит {MAX_STEPS}. Перезапуск.")
                await force_restart_js(page)

            stats = agent_stats[agent_name]
            stats["episodes_played"] += 1
            stats["total_reward"] += total_reward

            if step > stats["max_score"]:
                stats["max_score"] = step
                print(f"🏆 [{agent_name}] Ночной рекорд: {step} шагов! (ε={epsilon:.3f})")

            if step < stats["min_score"]:
                stats["min_score"] = step

            stats["last_100_scores"].append(step)
            if len(stats["last_100_scores"]) > 100:
                stats["last_100_scores"].pop(0)
            stats["avg_last_100"] = round(
                sum(stats["last_100_scores"]) / len(stats["last_100_scores"]), 1
            )

            if not stop_event.is_set() and is_ducking:
                try:
                    await page.keyboard.up("ArrowDown")
                    is_ducking = False
                except Exception:
                    pass

    except Exception as e:
        if not stop_event.is_set():
            print(f"[{agent_name}] Ошибка: {e}")
            stop_event.set()
# ==============================================================================
# 11. ТОЧКА ВХОДА
# ==============================================================================
async def main():
    global GAME_URL
    mode_str = "НОЧНАЯ ПОЛИРОВКА (скорость 10.2+)" if TRAIN_MODE else "БЕСКОНЕЧНЫЙ JS-РЕЖИМ (0 мс задержка)"
    print(f"Запускаем {NUM_WORKERS} агентов | Режим: {mode_str}...")

    load_all()

    if not os.path.isdir(GAME_DIR):
        print(f"❌ Папка {GAME_DIR} не найдена.")
        return

    if not os.path.exists(os.path.join(GAME_DIR, "index.html")):
        print(f"❌ В {GAME_DIR} нет index.html.")
        return

    ensure_pterodactyl_from_start()

    httpd, port = start_local_server(GAME_DIR)
    GAME_URL = f"http://127.0.0.1:{port}/index.html"
    print(f"🌐 Локальный сервер: {GAME_URL}")

    weights_json = export_weights_to_json() if not TRAIN_MODE else None

    async with async_playwright() as p:
        browser = await p.firefox.launch(
            headless=HEADLESS,
            args=[f"--width={WIN_W}", f"--height={WIN_H}"],
        )

        contexts, tasks = [], []
        stop_event = asyncio.Event()

        try:
            for i in range(NUM_WORKERS):
                agent_name = AGENT_NAMES[i % len(AGENT_NAMES)]
                ctx = await browser.new_context(
                    viewport={"width": WIN_W, "height": WIN_H}
                )
                contexts.append(ctx)
                tasks.append(asyncio.create_task(
                    worker(i + 1, agent_name, ctx, stop_event, weights_json)
                ))

            if TRAIN_MODE:
                tasks.append(asyncio.create_task(trainer_task(stop_event)))
                tasks.append(asyncio.create_task(saver_task(stop_event)))

            gather = asyncio.gather(*tasks, return_exceptions=True)
            stop_wait = asyncio.create_task(stop_event.wait())
            await asyncio.wait([gather, stop_wait], return_when=asyncio.FIRST_COMPLETED)

            for t in tasks:
                if not t.done():
                    t.cancel()
        finally:
            for ctx in contexts:
                try:
                    await ctx.close()
                except Exception:
                    pass
            try:
                await browser.close()
            except Exception:
                pass

        httpd.shutdown()

    if TRAIN_MODE:
        await save_all()

    print("\nРАБОТА ЗАВЕРШЕНА.")


if __name__ == "__main__":
    asyncio.run(main())