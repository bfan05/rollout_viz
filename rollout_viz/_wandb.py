"""Wandb logging for rollout accuracy over time."""

from typing import List, Dict, Any, Optional

from ._data import normalize_entry, calculate_accuracy_over_time


_wandb_run = None


def set_wandb_run(run):
    global _wandb_run
    _wandb_run = run


def get_wandb_run():
    return _wandb_run


def log_rollout_accuracy(data: List[Dict[str, Any]], step: int) -> None:
    """
    Log accuracy for the given step (and overall) to wandb.
    Call this after rollouts for a step are available (or after each rollout for live updates).
    """
    run = get_wandb_run()
    if run is None:
        return
    acc_by_step = calculate_accuracy_over_time(data, question_id=None)
    if step not in acc_by_step:
        return
    acc = acc_by_step[step]
    # Log so it appears as "rollout/accuracy" over training step in wandb
    run.log(
        {
            "rollout/accuracy": acc,
            "rollout/step_accuracy": acc,  # alias for clarity in custom charts
        },
        step=step,
    )
    # Optionally log per-question accuracy (sample a few to avoid clutter)
    # We skip that by default; user can add custom logging if needed.


def log_full_accuracy_series(data: List[Dict[str, Any]], up_to_step: Optional[int] = None) -> None:
    """Log the full accuracy-by-step series to wandb (e.g. at end of run)."""
    run = get_wandb_run()
    if run is None:
        return
    acc_by_step = calculate_accuracy_over_time(data, question_id=None)
    for s, acc in sorted(acc_by_step.items()):
        if up_to_step is not None and s > up_to_step:
            break
        run.log({"rollout/accuracy": acc}, step=s)
