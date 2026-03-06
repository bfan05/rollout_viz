# rollout-viz

A small Python package that visualizes **rollout accuracy over time** during GRPO/PPO-style training. It can integrate with [Weights & Biases (wandb)](https://wandb.ai) to log accuracy to your run, or run a **standalone local web UI** that updates live as rollouts are generated.

## Features

- **Accuracy over steps** — Track per-step and per-question accuracy as training progresses.
- **Standalone dashboard** — Table of questions with overall/initial/latest accuracy, difference, search, and expandable details (accuracy plot, step selector, rollout generations, length distribution).
- **Wandb integration** — Log `rollout/accuracy` and optional embedded HTML panels.
- **Flexible data sources** — Log from your training loop in-process, or point the CLI at a directory of JSONL files.

## Install

From the project root:

```bash
pip install -e .
```

With dev dependencies (tests):

```bash
pip install -e ".[dev]"
```

## Usage

### 1. With wandb

Use when you're already using wandb for your run. Rollout accuracy is logged as `rollout/accuracy` (and optionally as an interactive HTML panel).

```python
import rollout_viz

# Option A: You already called wandb.init() elsewhere
rollout_viz.init(use_wandb=True)

# Option B: Let rollout_viz start a wandb run
rollout_viz.init(use_wandb=True, project="my-project", name="run-1")

# Log each rollout in your training loop
for step in range(1, num_steps + 1):
    for rollout in rollouts_at_step:
        rollout_viz.log_rollout({
            "step": step,
            "input": rollout["prompt"],
            "output": rollout["completion"],
            "gts": rollout["ground_truth"],
            "score": 1.0 if rollout["correct"] else 0.0,
        })

# At the end, flush full series and final HTML to wandb
rollout_viz.flush_wandb()
```

**Supported field names:** `input`/`prompt`, `output`/`completion`, `gts`/`ground_truth`, `score`/`correct` (bool)/`reward`. Each entry must include `step`.

---

### 2. Standalone website (no wandb)

A local Flask server serves a dashboard that updates live as you call `log_rollout()` or as new JSONL files appear in `data_dir`.

**Option A: In-process server**

```python
import rollout_viz

rollout_viz.init(
    use_wandb=False,
    data_dir="./rollouts",
    run_standalone=True,
    standalone_port=5000,
)
# Open http://localhost:5000

rollout_viz.log_rollout({"step": 1, "input": "...", "output": "...", "gts": "4", "score": 1.0})
```

**Option B: CLI (watch a directory)**

If your trainer writes rollout files (e.g. `1.jsonl`, `2.jsonl`, …) to a directory:

```bash
rollout-viz --data-dir ./rollouts --port 5000
# Open http://localhost:5000
```

Files can be step-named (`1.jsonl`, `2.jsonl`, …) or a single `rollouts.jsonl` with a `step` field on each line.

---

### 3. Both wandb and standalone

You can run the local UI and log to wandb at the same time:

```python
rollout_viz.init(
    use_wandb=True,
    project="my-project",
    data_dir="./rollouts",
    run_standalone=True,
    standalone_port=5000,
)
rollout_viz.log_rollout(...)
```

---

## Data format

| Purpose        | Accepted keys                          |
|----------------|----------------------------------------|
| Prompt / input | `input`, `prompt`                      |
| Model output   | `output`, `completion`                 |
| Ground truth   | `gts`, `ground_truth`                 |
| Correct/score  | `score` (0/1), `correct` (bool), `reward` |
| Step           | `step` (required)                      |

`question_id` is optional; if missing, it is derived from the input (e.g. hash).

---

## API

| Function | Description |
|----------|-------------|
| `rollout_viz.init(use_wandb=..., project=..., data_dir=..., run_standalone=..., standalone_port=..., write_to_dir=...)` | Initialize; optionally start wandb and/or the standalone server. |
| `rollout_viz.log_rollout(entry)` | Log one rollout; updates wandb (if enabled), in-memory state, and optionally appends to `data_dir`. |
| `rollout_viz.flush_wandb()` | Log the full accuracy series and final HTML to wandb (call at end of run). |
| `rollout_viz.set_data_dir(path)` | Set or change the rollout data directory. |
| `rollout_viz.get_data_dir()` | Return the current data directory, if any. |
| `rollout_viz.is_standalone_running()` | Return whether the in-process standalone server is running. |

---

## CLI

```bash
rollout-viz [--data-dir <dir>] [--port <port>] [--host <host>]
```

- **`--data-dir`** — Directory containing rollout JSONL files (default: `.` or `ROLLOUT_VIZ_DATA_DIR`).
- **`--port`** — Port for the web server (default: 5000).
- **`--host`** — Host to bind (default: 0.0.0.0).

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `ROLLOUT_VIZ_DATA_DIR` | Default data directory for the CLI. |
| `ROLLOUT_VIZ_STANDALONE` | Set to `0` to disable starting the standalone server when using the Python API. |
| `ROLLOUT_VIZ_PORT` | Default port for the standalone server (Python API). |
| `ROLLOUT_VIZ_HTML_EVERY_N_STEPS` | When using wandb, log the interactive HTML panel every N steps (default: 5). |
| `ROLLOUT_VIZ_WANDB_HTML` | Set to `0` to disable logging the interactive HTML panel to wandb (accuracy scalars can still log). |

---

## Development

- Install with dev deps: `pip install -e ".[dev]"`
- Run tests: `pytest`
