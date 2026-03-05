# rollout-viz

A small Python package that visualizes **rollout accuracy over time** during training. It can integrate with [Weights & Biases (wandb)](https://wandb.ai) to log accuracy to your run, or run a **standalone local website** that updates live as rollouts are generated.

## Install

From the project root:

```bash
pip install -e .
```

Or install from a built wheel / from PyPI once published:

```bash
pip install rollout-viz
```

## Usage

### 1. With wandb (default)

Use this when you're already using wandb for your training run. Rollout accuracy is logged as `rollout/accuracy` and appears in your wandb dashboard over training steps.

```python
import rollout_viz

# Option A: You already called wandb.init() elsewhere
rollout_viz.init(use_wandb=True)

# Option B: Let rollout_viz start a wandb run
rollout_viz.init(use_wandb=True, project="my-project", name="run-1")

# Log each rollout as you generate it (e.g. in your GRPO/training loop)
for step in range(1, num_steps + 1):
    for rollout in rollouts_at_step:
        rollout_viz.log_rollout({
            "step": step,
            "input": rollout["prompt"],           # or "input"
            "output": rollout["completion"],      # or "output"
            "gts": rollout["ground_truth"],       # or "ground_truth"
            "score": 1.0 if rollout["correct"] else 0.0,  # or "correct", "reward"
        })

# At the end, flush so the full accuracy series is logged
rollout_viz.flush_wandb()
```

**Supported field names:** `input`/`prompt`, `output`/`completion`, `gts`/`ground_truth`, `score`/`correct` (bool)/`reward`. Each entry must include `step` (training step number).

The UI **updates as rollouts are generated**: each `log_rollout()` call updates in-memory state and, when using wandb, logs the current step’s accuracy to wandb.

---

### 2. Standalone website (no wandb)

Use this when you don’t want wandb. A local web server serves a dashboard that shows accuracy over time and per-question details; it **updates live** as you call `log_rollout()` or as new JSONL files appear in `data_dir`.

**Option A: In-process server (updates as you log)**

```python
import rollout_viz

# Start the website in a background thread; it will show data from memory + data_dir
rollout_viz.init(
    use_wandb=False,
    data_dir="./rollouts",
    run_standalone=True,
    standalone_port=5000,
)
# Open http://localhost:5000

# Log rollouts as usual; the page updates as you log
rollout_viz.log_rollout({"step": 1, "input": "...", "output": "...", "gts": "4", "score": 1.0})
```

**Option B: CLI server (watch a directory of JSONL files)**

If your training process writes rollout files (e.g. `1.jsonl`, `2.jsonl`, …) to a directory, run the standalone server in a separate terminal. The site will pick up new/changed files and refresh.

```bash
rollout-viz --data-dir ./rollouts --port 5000
# Open http://localhost:5000
```

Files should be named by step: `1.jsonl`, `2.jsonl`, … with one JSON object per line (same fields as above: `step`, `input`/`prompt`, `output`/`completion`, `gts`/`ground_truth`, `score`/`correct`/`reward`).

---

### 3. Both wandb and standalone

You can log to wandb and run the local website at the same time (e.g. for local debugging while also sending metrics to wandb).

```python
import rollout_viz

rollout_viz.init(
    use_wandb=True,
    project="my-project",
    data_dir="./rollouts",
    run_standalone=True,
    standalone_port=5000,
)
# Log rollouts as usual; both wandb and http://localhost:5000 update
rollout_viz.log_rollout(...)
```

---

## Data format

Each rollout entry can use any of these field names:

| Purpose        | Accepted keys                          |
|----------------|----------------------------------------|
| Prompt / input | `input`, `prompt`                      |
| Model output   | `output`, `completion`                 |
| Ground truth   | `gts`, `ground_truth`                  |
| Correct / score| `score` (0/1), `correct` (bool), `reward` |
| Step           | `step` (required)                      |

`question_id` is optional; if missing, it is derived from the input (e.g. hash).

---

## API summary

| Function | Description |
|----------|-------------|
| `rollout_viz.init(use_wandb=..., project=..., data_dir=..., run_standalone=..., standalone_port=...)` | Initialize; optionally start wandb and/or the standalone server. |
| `rollout_viz.log_rollout(entry)` | Log one rollout; updates wandb (if enabled) and in-memory / `data_dir`. |
| `rollout_viz.flush_wandb()` | Log the full accuracy-over-step series to wandb (call at end of run). |
| `rollout_viz.set_data_dir(path)` | Set or change the rollout data directory. |
| `rollout_viz.get_data_dir()` | Return the current data directory, if any. |
| `rollout_viz.is_standalone_running()` | Return whether the in-process standalone server is running. |

---

## CLI

- **`rollout-viz --data-dir <dir> [--port <port>] [--host <host>]`**  
  Run the standalone visualization server reading JSONL files from `<dir>`. Default port: 5000; default host: 0.0.0.0.  
  Environment: `ROLLOUT_VIZ_DATA_DIR` can set the default data directory.

---

## Development

- Install with dev deps: `pip install -e ".[dev]"`
- Run tests: `pytest`
