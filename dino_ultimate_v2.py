"""
dino_dqn_v7.py — птицы с первого шага + чёткие награды за птиц.

Правки после 20 минут деградации:
1. Патч index.js: minSpeed=0 для PTERODACTYL — птицы с первого шага.
2. compute_reward: чёткие награды за три высоты птиц:
   - yPos=50 (высоко): NOOP +0.25, JUMP -0.10
   - yPos=75 (средне): DUCK +0.50, JUMP -0.50, NOOP -0.30
   - yPos=100 (низко): JUMP +0.25, DUCK -0.25
3. MIN_START_SPEED = 6.0 (было 8.5) — не нужно, птицы уже с нуля.
4. Остальное — из v6 (REPLAY_SIZE=500k, LR=5e-5, GAMMA=0.93, TAU=0.005).

При первом запуске патчит index.js, при повторном — проверяет, что уже пропатчено.
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
import tkinter as tk
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

NUM_WORKERS = 6
TOTAL_EPISODES = 5000
GAMMA = 0.93
LR = 5e-5
BATCH_SIZE = 128
REPLAY_SIZE = 500000
TAU = 0.005
TRAIN_START = 10000
TRAIN_RATE_SLEEP = 0.03
INFERENCE_SYNC_EVERY = 100

EPS_START = 0.5
EPS_END = 0.10
EPS_DECAY_K = 5
EPS_BOOST_VALUE = 0.30
EPS_BOOST_DURATION = 200

MIN_START_SPEED = 6.0              # было 8.5 — птицы уже с нуля, скорость не нужна
STEP_DELAY = 0.010
MAX_STEPS = 5000
N_STEP = 3

PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_STEPS = 50000

MODEL_FILE = "dqn_model_v7.pt"
STATE_FILE = "dqn_state_v7.json"
STATS_DIR = "stats_v7"

TRAIN_MODE = True
LOAD_MODEL = False
SAVE_MODEL = True
HEADLESS = True
SEED = 42

WIN_W = 650
WIN_H = 350

AGENT_NAMES = ["Эль патрон", "Эль дрифто", "Пантера", "Серцеед", "Коч братан", "Никита"]
ACTIONS = [0, 1, 2]
NOOP, JUMP, DUCK = 0, 1, 2

# ==============================================================================
# 2. ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# ==============================================================================
episodes_done = 0
training_steps = 0
start_episode = 0
target_episodes = 0
epsilon_boost_until = 0
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
    """Патчит index.js: minSpeed=0 для PTERODACTYL — птицы с первого шага."""
    path = os.path.join(GAME_DIR, "index.js")
    if not os.path.exists(path):
        print(f"⚠️  {path} не найден, патч птиц пропущен")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Регекс: находим блок PTERODACTYL и меняем minSpeed
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
        print(f"✅ index.js уже пропатчен: птицы с minSpeed=0")
        return

    backup_path = path + ".original"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"💾 Бэкап оригинала: {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"🔧 index.js пропатчен: птицы доступны с первого шага (minSpeed=0)")


# ==============================================================================
# 3. НЕЙРОСЕТЬ
# ==============================================================================
class DQN(nn.Module):
    def __init__(self, in_dim=8, hidden=256, out_dim=3):
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
loss_fn = nn.HuberLoss(delta=2.0)


# ==============================================================================
# 4. PRIORITIZED REPLAY
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
        s0, a0, _, _, _ = self.buffer[0]
        R, gamma_pow = 0.0, 1.0
        for i in range(self.n):
            _, _, r, _, done = self.buffer[i]
            R += gamma_pow * r
            gamma_pow *= self.gamma
            if done:
                return (s0, a0, R, self.buffer[i][3], True)
        s_last = self.buffer[self.n - 1][3]
        done = self.buffer[self.n - 1][4]
        return (s0, a0, R, s_last, done)

    def flush_partial(self):
        result = []
        while len(self.buffer) > 0:
            s0, a0, _, _, _ = self.buffer[0]
            R, gamma_pow = 0.0, 1.0
            for i in range(len(self.buffer)):
                _, _, r, _, _ = self.buffer[i]
                R += gamma_pow * r
                gamma_pow *= self.gamma
            s_last = self.buffer[-1][3]
            result.append((s0, a0, R, s_last, True))
            self.buffer.popleft()
        return result

    def clear(self):
        self.buffer.clear()


# ==============================================================================
# 6. ВЫБОР ДЕЙСТВИЯ И ОБУЧЕНИЕ
# ==============================================================================
def nn_choose_action(state_vec, epsilon):
    if random.random() < epsilon:
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
            target = r_t + GAMMA * q_next * (1.0 - d_t)
        td_errors = (q - target).detach().cpu().numpy()
        losses = torch.nn.functional.huber_loss(q, target, reduction='none', delta=2.0)
        loss = (w_t * losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

    replay.update_priorities(indices, td_errors)
    return float(loss.item())


def features(distance, width, y_pos, speed, is_bird, is_large, jumping):
    return np.array([
        min(distance, 500.0) / 500.0,
        min(width, 100.0) / 100.0,
        min(y_pos, 150.0) / 150.0,
        min(speed, 15.0) / 15.0,
        1.0 if is_bird else 0.0,
        1.0 if jumping else 0.0,
        min(distance + width, 600.0) / 600.0,
        1.0 if is_large else 0.0,
    ], dtype=np.float32)


def compute_reward(action, distance, speed, is_bird, y_pos,
                   step, agent_record, died):
    """
    Награда с чёткими правилами для трёх высот птиц.

    Птицы:
      - yPos=50 (высоко): NOOP +0.25, JUMP -0.10
      - yPos=75 (средне): DUCK +0.50, JUMP -0.50, NOOP -0.30
      - yPos=100 (низко): JUMP +0.25, DUCK -0.25
    Кактусы: JUMP +0.25, DUCK -0.25
    """
    if died:
        return -5.0

    reward = 0.01
    speed_factor = max(1.0, speed / 6.0)
    danger_zone = 250.0 * speed_factor

    if distance <= danger_zone:
        if is_bird:
            if y_pos < 60:
                # Высокая птица (50) — безопасна
                if action == NOOP:
                    reward += 0.25
                elif action == JUMP:
                    reward -= 0.10
            elif y_pos < 90:
                # Средняя птица (75) — критично присесть
                if action == DUCK:
                    reward += 0.50
                elif action == JUMP:
                    reward -= 0.50
                elif action == NOOP:
                    reward -= 0.30
            else:
                # Низкая птица (100) — прыгать
                if action == JUMP:
                    reward += 0.25
                elif action == DUCK:
                    reward -= 0.25
        else:
            # Кактус
            if action == JUMP:
                reward += 0.25
            elif action == DUCK:
                reward -= 0.25

    if step == agent_record + 1:
        reward += 0.5

    return reward


def compute_epsilon():
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
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e2:
            print(f"⚠️  {path}: {e2}")


def safe_filename(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def load_all():
    global episodes_done, training_steps, start_episode, target_episodes
    start_episode = 0
    target_episodes = TOTAL_EPISODES
    print(f"Новая модель (v7). TOTAL_EPISODES={TOTAL_EPISODES}")


async def save_all():
    async with save_lock:
        if not SAVE_MODEL:
            return
        def _save():
            with model_lock:
                torch.save({
                    "model": model.state_dict(),
                    "target": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "training_steps": training_steps,
                }, MODEL_FILE)
        try:
            await asyncio.to_thread(_save)
        except Exception as e:
            print(f"⚠️  Сохранение: {e}")

        await asyncio.to_thread(_write_json, STATE_FILE,
                                {"episodes_done": episodes_done})
        stats_copy = copy.deepcopy(agent_stats)
        for name, stats in stats_copy.items():
            await asyncio.to_thread(_write_json,
                                    f"{STATS_DIR}/stats_{safe_filename(name)}.json",
                                    stats)
        print(f"[Сохранение] эп={episodes_done}/{target_episodes} "
              f"steps={training_steps} ε={compute_epsilon():.3f} "
              f"buffer={len(replay)}")


# ==============================================================================
# 8. ФОНОВЫЕ ЗАДАЧИ
# ==============================================================================
async def trainer_task(stop_event):
    global training_steps, epsilon_boost_until
    last_log = time.time()
    collapse_counter = 0

    while not stop_event.is_set():
        if len(replay) < TRAIN_START:
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

        if training_steps % 5000 == 0:
            with torch.no_grad():
                sample = torch.FloatTensor(batch[0][:16]).to(DEVICE)
                q = model(sample)
                print(f"[Q-diag] min={q.min():.3f} max={q.max():.3f} "
                      f"spread={(q.max() - q.min()):.3f}")

        if loss < 0.0005:
            collapse_counter += 1
        else:
            collapse_counter = 0

        if collapse_counter >= 2000:
            epsilon_boost_until = training_steps + EPS_BOOST_DURATION
            print(f"[Trainer] ⚠️  КОЛЛАПС: loss<0.0005 2000 шагов. "
                  f"ε={EPS_BOOST_VALUE} на {EPS_BOOST_DURATION} шагов")
            collapse_counter = 0

        if time.time() - last_log > 15:
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
# 9. СВЯЗЬ С ИГРОЙ
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
    let best = null, bestDist = Infinity;
    for (const o of obstacles) {
        const d = o.xPos - dinoRight;
        if (d > -o.width && d < bestDist) { bestDist = d; best = o; }
    }
    if (!best) return {
        status: "OK", distance: 1000, speed: runner.currentSpeed,
        yPos: 105, jumping: !!dino.jumping,
        isBird: false, isLarge: false, width: 20
    };
    const typeName = String(
        (best.typeConfig && best.typeConfig.type) || best.type || "UNKNOWN"
    ).toUpperCase();
    const isBird = typeName === "PTERODACTYL";
    const isLarge = typeName === "CACTUS_LARGE";
    return {
        status: "OK", distance: bestDist, speed: runner.currentSpeed,
        yPos: best.yPos, jumping: !!dino.jumping,
        isBird: isBird, isLarge: isLarge, width: best.width
    };
}
"""


async def force_restart_js(page):
    try:
        alive = await asyncio.wait_for(
            page.evaluate("() => !!window.Runner?.instance_"), timeout=2.0)
        if not alive:
            return False
        result = await asyncio.wait_for(page.evaluate("""
            () => {
                const r = window.Runner?.instance_;
                if (!r) return { ok: false };
                try { r.restart(); return { ok: true }; }
                catch (e) { return { ok: false, reason: String(e) }; }
            }
        """), timeout=2.0)
        return bool(result.get("ok"))
    except Exception:
        return False


async def boost_speed(page):
    try:
        await asyncio.wait_for(page.evaluate(f"""
            () => {{
                const r = window.Runner?.instance_;
                if (r && r.currentSpeed < {MIN_START_SPEED}) {{
                    r.currentSpeed = {MIN_START_SPEED};
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
        }, "OK"
    except asyncio.TimeoutError:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"ERROR: {e}"


async def start_game(page):
    await page.keyboard.up("ArrowDown")
    await asyncio.sleep(0.15)
    for round_num in (1, 2, 3):
        if round_num == 2:
            await force_restart_js(page)
            await asyncio.sleep(0.15)
        elif round_num == 3:
            try:
                await page.reload(wait_until="load")
                await asyncio.sleep(0.2)
                await page.mouse.click(WIN_W // 2, WIN_H // 2)
                await asyncio.sleep(0.1)
            except Exception:
                return None, "RELOAD_FAILED"
        for attempt in range(5):
            try:
                await page.mouse.click(WIN_W // 2, WIN_H // 2)
                await page.keyboard.down("Space")
                await asyncio.sleep(0.05)
                await page.keyboard.up("Space")
            except Exception:
                pass
            for _ in range(6):
                await asyncio.sleep(0.04)
                st, _ = await get_state(page)
                if st is not None:
                    await boost_speed(page)
                    return st, "OK"
    return None, "START_FAILED"


# ==============================================================================
# 10. ВОРКЕР
# ==============================================================================
async def worker(worker_id, agent_name, context, stop_event):
    global episodes_done
    try:
        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda: stop_event.set())
        await page.add_init_script(f"document.title = '{agent_name}';")
        await page.goto(GAME_URL, wait_until="load")
        await asyncio.sleep(0.2)
        await page.bring_to_front()
        await page.mouse.click(WIN_W // 2, WIN_H // 2)
        await asyncio.sleep(0.2)

        is_ducking = False
        nstep = NStepBuffer()

        while episodes_done < target_episodes and not stop_event.is_set():
            episodes_done += 1
            ep_num = episodes_done
            epsilon = compute_epsilon() if TRAIN_MODE else 0.0

            st, status = await start_game(page)
            is_ducking = False
            nstep.clear()
            if st is None:
                episodes_done -= 1
                await asyncio.sleep(0.3)
                continue

            total_reward = 0.0
            last = None
            step = 0
            agent_record = agent_stats[agent_name]["max_score"]

            for step in range(1, MAX_STEPS + 1):
                if stop_event.is_set():
                    break

                if st["jumping"]:
                    await asyncio.sleep(STEP_DELAY)
                    nst, status = await get_state(page)
                    if nst is None:
                        if TRAIN_MODE and last is not None:
                            s_prev, a_prev = last
                            s2_dummy = np.zeros(8, dtype=np.float32)
                            nstep.push(s_prev, a_prev, -5.0, s2_dummy, True)
                            for trans in nstep.flush_partial():
                                replay.push(*trans)
                        total_reward -= 5.0
                        print(f"[{agent_name} | Эп.{ep_num}] Смерть в прыжке, шагов: {step}")
                        break
                    if TRAIN_MODE and last is not None:
                        s_prev, a_prev = last
                        s_air = features(nst["distance"], nst["width"], nst["yPos"],
                                         nst["speed"], nst["isBird"], nst["isLarge"],
                                         nst["jumping"])
                        nstep.push(s_prev, a_prev, 0.01, s_air, False)
                        trans = nstep.get_transition()
                        if trans is not None:
                            replay.push(*trans)
                        last = (s_air, a_prev)
                    st = nst
                    continue

                s_vec = features(st["distance"], st["width"], st["yPos"],
                                 st["speed"], st["isBird"], st["isLarge"],
                                 st["jumping"])
                action = nn_choose_action(s_vec, epsilon)
                last = (s_vec, action)

                if st["distance"] > 600:
                    action = NOOP
                    last = (s_vec, action)

                if action == JUMP:
                    if is_ducking:
                        await page.keyboard.up("ArrowDown")
                        is_ducking = False
                    await page.keyboard.press("Space")
                elif action == DUCK:
                    if not is_ducking:
                        await page.keyboard.down("ArrowDown")
                        is_ducking = True
                else:
                    if is_ducking:
                        await page.keyboard.up("ArrowDown")
                        is_ducking = False

                await asyncio.sleep(STEP_DELAY)
                nst, status = await get_state(page)

                if nst is None:
                    died = status == "DEAD"
                    reward = compute_reward(
                        action, st["distance"], st["speed"],
                        st["isBird"], st["yPos"],
                        step, agent_record, died
                    )
                    if TRAIN_MODE:
                        s2_dummy = np.zeros(8, dtype=np.float32)
                        nstep.push(s_vec, action, reward, s2_dummy, died)
                        for trans in nstep.flush_partial():
                            replay.push(*trans)
                    total_reward += reward
                    print(f"[{agent_name} | Эп.{ep_num}] Конец ({status}). "
                          f"Шагов: {step}, R: {total_reward:.2f}, ε={epsilon:.3f}")
                    break

                s2 = features(nst["distance"], nst["width"], nst["yPos"],
                              nst["speed"], nst["isBird"], nst["isLarge"],
                              nst["jumping"])
                reward = compute_reward(
                    action, st["distance"], st["speed"],
                    st["isBird"], st["yPos"],
                    step, agent_record, False
                )
                if TRAIN_MODE:
                    nstep.push(s_vec, action, reward, s2, False)
                    trans = nstep.get_transition()
                    if trans is not None:
                        replay.push(*trans)
                total_reward += reward
                st = nst

            if step == MAX_STEPS and not stop_event.is_set():
                print(f"[{agent_name} | Эп.{ep_num}] Лимит {MAX_STEPS}. restart.")
                await force_restart_js(page)

            stats = agent_stats[agent_name]
            stats["episodes_played"] += 1
            stats["total_reward"] += total_reward
            if step > stats["max_score"]:
                stats["max_score"] = step
                print(f"🏆 [{agent_name}] Рекорд: {step} шагов! (ε={epsilon:.3f})")
            if step < stats["min_score"]:
                stats["min_score"] = step
            stats["last_100_scores"].append(step)
            if len(stats["last_100_scores"]) > 100:
                stats["last_100_scores"].pop(0)
            stats["avg_last_100"] = round(
                sum(stats["last_100_scores"]) / len(stats["last_100_scores"]), 1)

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

    print(f"Запускаем {NUM_WORKERS} агентов...")
    load_all()

    if not os.path.isdir(GAME_DIR):
        print(f"❌ Папка {GAME_DIR} не найдена.")
        return
    if not os.path.exists(os.path.join(GAME_DIR, "index.html")):
        print(f"❌ В {GAME_DIR} нет index.html.")
        return

    # Патчим index.js — птицы с первого шага
    ensure_pterodactyl_from_start()

    httpd, port = start_local_server(GAME_DIR)
    GAME_URL = f"http://127.0.0.1:{port}/index.html"
    print(f"🌐 Локальный сервер: {GAME_URL}")

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
                    viewport={"width": WIN_W, "height": WIN_H})
                contexts.append(ctx)
                tasks.append(asyncio.create_task(
                    worker(i + 1, agent_name, ctx, stop_event)))
            tasks.append(asyncio.create_task(trainer_task(stop_event)))
            tasks.append(asyncio.create_task(saver_task(stop_event)))

            gather = asyncio.gather(*tasks, return_exceptions=True)
            stop_wait = asyncio.create_task(stop_event.wait())
            await asyncio.wait([gather, stop_wait],
                               return_when=asyncio.FIRST_COMPLETED)

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
    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО (или прервано).")


if __name__ == "__main__":
    asyncio.run(main())