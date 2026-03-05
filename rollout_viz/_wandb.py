"""Wandb logging for rollout accuracy over time."""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional

import wandb

from ._data import (
    normalize_entry,
    calculate_accuracy_over_time,
    get_question_ids,
    get_question_info,
    get_unique_steps,
    get_responses_at_step,
)
from ._server import _generate_plot, _classify_difficulty


_wandb_run = None
_last_html_step: Optional[int] = None
_HTML_LOG_EVERY_N_STEPS = int(os.environ.get("ROLLOUT_VIZ_HTML_EVERY_N_STEPS", "5"))


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
    global _last_html_step

    run = get_wandb_run()
    if run is None:
        return
    acc_by_step = calculate_accuracy_over_time(data, question_id=None)
    if step not in acc_by_step:
        return
    acc = acc_by_step[step]
    # Log so it appears as "rollout/accuracy" over training step in wandb
    # Let wandb/Trainer control the step value to avoid out-of-order step warnings.
    run.log(
        {
            "rollout/accuracy": acc,
            "rollout/step_accuracy": acc,  # alias for clarity in custom charts
        }
    )
    # Periodically log an updated interactive HTML visualization during training.
    if (
        _last_html_step is None
        or step >= _last_html_step + _HTML_LOG_EVERY_N_STEPS
    ):
        log_interactive_visualization(data)
        _last_html_step = step
    # Optionally log per-question accuracy (sample a few to avoid clutter)
    # We skip that by default; user can add custom logging if needed.


def log_full_accuracy_series(
    data: List[Dict[str, Any]], up_to_step: Optional[int] = None
) -> None:
    """Log the full accuracy-by-step series to wandb (e.g. at end of run)."""
    run = get_wandb_run()
    if run is None:
        return
    acc_by_step = calculate_accuracy_over_time(data, question_id=None)
    for s, acc in sorted(acc_by_step.items()):
        if up_to_step is not None and s > up_to_step:
            break
        # Again, let wandb/Trainer manage the step to keep it monotonic.
        run.log({"rollout/accuracy": acc})


def log_interactive_visualization(data: List[Dict[str, Any]]) -> None:
    """
    Log the same GRPO Training Visualization page used by the standalone server,
    with data embedded so it renders in WandB without API calls.
    """
    run = get_wandb_run()
    if run is None:
        return

    normalized: List[Dict[str, Any]] = []
    for entry in data:
        try:
            normalized.append(normalize_entry(entry))
        except Exception:
            continue
    if not normalized:
        return

    # Build questions list (same shape as /api/questions)
    questions_dict: Dict[str, Dict[str, Any]] = {}
    for entry in normalized:
        qid = entry.get("question_id")
        if not qid:
            continue
        if qid not in questions_dict:
            questions_dict[qid] = {
                "id": qid,
                "user_question": entry.get("user_question", ""),
                "ground_truth": entry.get("gts", ""),
                "steps": set(),
                "total_correct": 0,
                "total_rollouts": 0,
            }
        questions_dict[qid]["steps"].add(entry.get("step", 0))
        if entry.get("score", 0.0) == 1.0:
            questions_dict[qid]["total_correct"] += 1
        questions_dict[qid]["total_rollouts"] += 1

    question_list = list(questions_dict.values())
    for q in question_list:
        q["steps"] = sorted(list(q["steps"]))
        q["average_accuracy"] = (
            q["total_correct"] / q["total_rollouts"] if q["total_rollouts"] > 0 else 0.0
        )
        q["difficulty"] = _classify_difficulty(q["average_accuracy"])
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    question_list.sort(
        key=lambda x: (
            difficulty_order.get(x.get("difficulty", "medium"), 1),
            -x.get("average_accuracy", 0.5),
        )
    )
    for idx, q in enumerate(question_list):
        q["question_number"] = idx + 1

    question_ids = get_question_ids(normalized)
    steps = get_unique_steps(normalized)
    status = {
        "question_count": len(question_ids),
        "total_entries": len(normalized),
        "steps": steps,
    }
    question_info: Dict[str, Dict[str, Any]] = {}
    for qid in question_ids:
        question_info[qid] = get_question_info(normalized, qid)

    # Per-question plot images (base64 PNG)
    plot_images: Dict[str, str] = {}
    for qid in question_ids:
        try:
            plot_images[qid] = _generate_plot(
                normalized, question_id=qid, title="Question Accuracy Over Training Steps"
            )
        except Exception:
            pass

    # Responses per (question_id, step) - keys must be strings for JSON
    responses: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for qid in question_ids:
        info = question_info.get(qid, {})
        for step in info.get("steps", []):
            resp_list = get_responses_at_step(normalized, qid, step)
            if qid not in responses:
                responses[qid] = {}
            responses[qid][str(step)] = resp_list

    payload = {
        "status": status,
        "questions": question_list,
        "questionInfo": question_info,
        "plotImages": plot_images,
        "responses": responses,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    data_json = data_json.replace("</script>", "<\\/script>")

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(pkg_dir, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    injected = '<script type="application/json" id="rollout-viz-data">' + data_json + "</script>"
    html_content = template_content.replace("<!-- ROLLOUT_VIZ_EMBEDDED_PLACEHOLDER -->", injected)

    run.log({"question_accuracy_interactive": wandb.Html(html_content)})
