"""
rollout_viz: Wandb-integrated rollout accuracy visualization with optional standalone web UI.

Use with wandb (default):
    import rollout_viz
    rollout_viz.init(use_wandb=True, project="my-training")
    rollout_viz.log_rollout({"step": 1, "input": "...", "output": "...", "gts": "4", "score": 1.0})

Use standalone website (no wandb):
    rollout_viz.init(use_wandb=False, data_dir="./rollouts", run_standalone=True)
    rollout_viz.log_rollout(...)  # UI at http://localhost:5000 updates as you log

Or run the CLI to watch a directory: rollout-viz --data-dir ./rollouts
"""

from ._state import (
    init,
    log_rollout,
    set_data_dir,
    flush_wandb,
    get_data_dir,
    is_standalone_running,
)

__all__ = [
    "init",
    "log_rollout",
    "set_data_dir",
    "flush_wandb",
    "get_data_dir",
    "is_standalone_running",
]
