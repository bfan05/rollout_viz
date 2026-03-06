"""Shared data loading, normalization, and accuracy calculation for rollout data."""

import json
import os
import re
import hashlib
import glob
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple


def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an entry so it has input, output, gts, score, step, question_id.
    Supports both formats:
    - input/output/gts/score (from app) and prompt_id from hash of input
    - prompt/completion/ground_truth/correct (from DATA_FORMAT.md)
    """
    e = dict(entry)
    # Map prompt -> input, completion -> output, ground_truth -> gts
    if "input" not in e and "prompt" in e:
        e["input"] = e["prompt"]
    if "output" not in e and "completion" in e:
        e["output"] = e.get("completion", "")
    if "gts" not in e and "ground_truth" in e:
        e["gts"] = e["ground_truth"]
    if "score" not in e and "correct" in e:
        e["score"] = 1.0 if e["correct"] else 0.0
    if "score" not in e and "reward" in e:
        e["score"] = float(e["reward"]) if e["reward"] is not None else 0.0
    # question_id / prompt_id
    input_text = e.get("input", "")
    if input_text:
        e["question_id"] = get_question_id_from_input(input_text)
        e["user_question"] = extract_user_question(input_text)
    elif e.get("prompt_id"):
        e["question_id"] = str(e["prompt_id"])[:12]
    return e


def get_question_id_from_input(input_text: str) -> Optional[str]:
    if not input_text:
        return None
    return hashlib.md5(input_text.encode("utf-8")).hexdigest()[:12]


def extract_user_question(input_text: str) -> str:
    if not input_text:
        return ""
    user_match = re.search(r"user\n(.*?)\nassistant", input_text, re.DOTALL)
    if user_match:
        return user_match.group(1).strip()
    user_match = re.search(r"user\n(.*)", input_text, re.DOTALL)
    if user_match:
        return user_match.group(1).strip()
    return input_text[:200]


def load_rollout_data_from_dir(data_dir: str, max_steps: int = 500) -> List[Dict[str, Any]]:
    """
    Load and parse rollout data from a directory.

    Supports two formats:
    - Multi-file: 1.jsonl, 2.jsonl, ... where the filename encodes the step.
    - Single-file: rollouts.jsonl where each line has a `step` field (see DATA_FORMAT.md).
    """
    data = []
    if not data_dir or not os.path.isdir(data_dir):
        return data

    # Prefer the single-file rollouts.jsonl format if present.
    rollouts_path = os.path.join(data_dir, "rollouts.jsonl")
    if os.path.isfile(rollouts_path):
        jsonl_files = [rollouts_path]
        use_filename_steps = False
    else:
        jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        use_filename_steps = True

        def get_step_number(filepath: str) -> int:
            filename = os.path.basename(filepath)
            try:
                return int(filename.replace(".jsonl", ""))
            except ValueError:
                return 999999

        jsonl_files = sorted(jsonl_files, key=get_step_number)

    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if use_filename_steps:
                    # Multi-file format: derive step from filename (1.jsonl, 2.jsonl, ...).
                    try:
                        step = int(os.path.basename(file_path).replace(".jsonl", ""))
                    except ValueError:
                        continue
                    entry["step"] = step
                else:
                    # Single-file rollouts.jsonl: trust `step` field on each entry.
                    step = entry.get("step", 0)
                    if not isinstance(step, int):
                        # Skip entries with invalid or missing step.
                        continue

                if step > max_steps:
                    continue

                entry = normalize_entry(entry)
                if entry.get("question_id") or entry.get("input"):
                    data.append(entry)
    return data


def calculate_accuracy_over_time(
    data: List[Dict[str, Any]], question_id: Optional[str] = None, max_step: int = 10000
) -> Dict[int, float]:
    """
    Returns dict step -> accuracy (0-1) for each step that has data.
    """
    if question_id:
        data = [e for e in data if e.get("question_id") == question_id]
    step_data = defaultdict(list)
    for entry in data:
        step = entry.get("step", 0)
        if step <= max_step:
            step_data[step].append(entry)
    accuracy_by_step = {}
    for step in sorted(step_data.keys()):
        rollouts = step_data[step]
        if rollouts:
            correct = sum(1 for r in rollouts if r.get("score", 0.0) == 1.0)
            accuracy_by_step[step] = correct / len(rollouts)
    return accuracy_by_step


def get_unique_steps(data: List[Dict[str, Any]], max_step: int = 10000) -> List[int]:
    steps = sorted(set(e.get("step", 0) for e in data if 0 < e.get("step", 0) <= max_step))
    return steps


def get_question_ids(data: List[Dict[str, Any]]) -> List[str]:
    ids = sorted(set(e.get("question_id") for e in data if e.get("question_id")))
    return ids


def get_question_info(data: List[Dict[str, Any]], question_id: str) -> Dict[str, Any]:
    """Return system_prompt, user_question, ground_truth, total_rollouts, steps for a question."""
    question_data = [e for e in data if e.get("question_id") == question_id]
    if not question_data:
        return {
            "system_prompt": "",
            "user_question": "",
            "ground_truth": "",
            "total_rollouts": 0,
            "steps": [],
        }
    first = question_data[0]
    input_text = first.get("input", "")
    system_match = re.search(r"system\n(.*?)\nuser", input_text, re.DOTALL)
    system_prompt = system_match.group(1).strip() if system_match else "You are a helpful assistant."
    return {
        "system_prompt": system_prompt,
        "user_question": first.get("user_question", "") or extract_user_question(input_text),
        "ground_truth": first.get("gts", ""),
        "total_rollouts": len(question_data),
        "steps": sorted(set(e.get("step", 0) for e in question_data)),
    }


def get_responses_at_step(
    data: List[Dict[str, Any]], question_id: str, step: int
) -> List[Dict[str, Any]]:
    out = []
    for e in data:
        if e.get("question_id") == question_id and e.get("step") == step:
            output_text = e.get("output", "")
            out.append({
                "output": output_text,
                "score": e.get("score", 0.0),
                "ground_truth": e.get("gts", ""),
                "correct": e.get("score", 0.0) == 1.0,
                "length": len(output_text),  # character length; token count if present
                "length_tokens": e.get("response_length") or e.get("output_length"),
            })
    return out


def get_initial_and_latest_accuracy(
    data: List[Dict[str, Any]], question_id: str
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    For a question, return (initial_accuracy, latest_accuracy, first_step, last_step).
    Initial = accuracy on the first training step this question was used; latest = on the last.
    """
    question_data = [e for e in data if e.get("question_id") == question_id]
    if not question_data:
        return None, None, None, None
    steps = sorted(set(e.get("step", 0) for e in question_data if e.get("step", 0) > 0))
    if not steps:
        return None, None, None, None
    first_step = steps[0]
    last_step = steps[-1]
    acc_by_step = calculate_accuracy_over_time(data, question_id=question_id)
    initial_acc = acc_by_step.get(first_step)
    latest_acc = acc_by_step.get(last_step)
    return initial_acc, latest_acc, first_step, last_step


def get_table_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One row per question: question_id, user_question, initial_difficulty, latest_difficulty,
    difference (latest - initial), overall_accuracy, first_step, last_step.
    """
    qids = get_question_ids(data)
    rows = []
    for qid in qids:
        info = get_question_info(data, qid)
        initial_acc, latest_acc, first_step, last_step = get_initial_and_latest_accuracy(data, qid)
        user_question = info.get("user_question", "")
        # difference = latest - initial (None if either missing)
        diff = None
        if initial_acc is not None and latest_acc is not None:
            diff = round(latest_acc - initial_acc, 4)
        question_entries = [e for e in data if e.get("question_id") == qid]
        total = len(question_entries)
        correct = sum(1 for e in question_entries if e.get("score", 0.0) == 1.0)
        overall_accuracy = correct / total if total else None
        rows.append({
            "question_id": qid,
            "user_question": user_question,
            "initial_difficulty": initial_acc,
            "latest_difficulty": latest_acc,
            "difference": diff,
            "overall_accuracy": round(overall_accuracy, 4) if overall_accuracy is not None else None,
            "first_step": first_step,
            "last_step": last_step,
            "steps": info.get("steps", []),
        })
    # Sort by overall accuracy descending (highest first)
    def sort_key(r):
        oa = r.get("overall_accuracy")
        if oa is None:
            return -1.0
        return -oa
    rows.sort(key=sort_key)
    return rows


def get_rollout_length_distribution(
    data: List[Dict[str, Any]], question_id: str, step: int
) -> Dict[str, Any]:
    """
    Return rollout length stats at (question_id, step). Uses character length;
    if entries have response_length/output_length (token count), include that too.
    """
    responses = get_responses_at_step(data, question_id, step)
    lengths_chars = [r["length"] for r in responses]
    lengths_tokens = [r.get("length_tokens") for r in responses if r.get("length_tokens") is not None]
    result = {
        "lengths": lengths_chars,
        "count": len(lengths_chars),
        "min": min(lengths_chars) if lengths_chars else None,
        "max": max(lengths_chars) if lengths_chars else None,
        "mean": round(sum(lengths_chars) / len(lengths_chars), 2) if lengths_chars else None,
    }
    if lengths_tokens:
        result["lengths_tokens"] = lengths_tokens
        result["mean_tokens"] = round(sum(lengths_tokens) / len(lengths_tokens), 2)
    return result
