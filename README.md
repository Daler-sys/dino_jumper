# Dino Jumper 🦖

A **Deep Q-Network (DQN)** agent that learns to play the Chrome T-Rex runner game. The project includes a local copy of the game, training/inference scripts, and saved model weights.

> Repository: `Daler-sys/dino_jumper`

## What it does

- Runs the Chrome Dino game locally via **Playwright**.
- Uses a **PyTorch DQN** (11 inputs → 256 → 256 → 3 actions) to decide when to jump, duck, or do nothing.
- Supports **training mode** and **inference/JS-speed mode** (`TRAIN_MODE=False` for fast play).
- Includes saved v9 weights (`best_dqn_model_v9.pt`, `dqn_model_v9.pt`) and training state/stats.

## Agent / model overview

From `dino_dqn_v9.py` (v9):

- **Network**: `DQN(11, 256, 3)` with ReLU hidden layers.
- **Actions**: `NOOP`, `JUMP`, `DUCK`.
- **RL setup**: DQN with target network, prioritized replay buffer, n-step returns, epsilon schedule/boost.
- **Runtime**: Playwright-driven browser game; local HTTP server for the `game/` directory.
- **Modes**:
  - `TRAIN_MODE=True` — training loop.
  - `TRAIN_MODE=False` — inference mode (loads model, fast play).
- **Extra**: patches `game/index.js` to allow pterodactyls from the start (`minSpeed=0`) for harder/more complete training.

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── dino_dqn_v9.py           # main DQN v9 agent (train/inference)
├── dino_ultimate_v2.py      # earlier/alternate agent version
├── dino.py                  # simple Q-table baseline
├── best_dqn_model_v9.pt     # best saved v9 model
├── dqn_model_v9.pt          # latest v9 model
├── dqn_state_v9.json        # training state
├── stats_v9/                # per-agent/per-run stats
├── docs/                    # README screenshots
└── game/                    # local t-rex runner (submodule/source)
```

## Installation

Requires Python 3.12+ (see `pyproject.toml`).

```bash
git clone https://github.com/Daler-sys/dino_jumper.git
cd dino_jumper

# install dependencies
uv sync
# or: pip install numpy torch playwright

# install playwright browser
playwright install chromium
```

## Run

### Run the DQN v9 agent

```bash
python dino_dqn_v9.py
```

Key flags near the top of `dino_dqn_v9.py`:

```python
TRAIN_MODE = False   # False = inference, True = training
LOAD_MODEL = True    # load saved weights
SAVE_MODEL = False   # save during training
HEADLESS = False     # show browser window
```

The script serves the local game from `GAME_DIR` and controls it through Playwright.

### Run the simple baseline

```bash
python dino.py
```

`dino.py` is a small Q-learning baseline using distance buckets and a Q-table.

## Files to know

- `dino_dqn_v9.py` — main, most advanced agent.
- `best_dqn_model_v9.pt` — best checkpoint for inference.
- `game/index.js` — local game logic; v9 can patch pterodactyl spawn behavior.
- `stats_v9/*.json` — recorded run statistics.

## Notes

- The project is Windows-oriented in default paths (`GAME_DIR = r"C:\dino_coch\game"`); adjust `GAME_DIR` if running elsewhere.
- Training can be long; use inference mode with `best_dqn_model_v9.pt` for quick demo.
- Playwright opens a real Chromium window unless `HEADLESS=True`.

## Author

**Daler Ismail-zade** ([@Daler-sys](https://github.com/Daler-sys))
