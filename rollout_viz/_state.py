"""Global state and public API for rollout_viz."""

import json
import os
import threading
from typing import List, Dict, Any, Optional

from . import _data
from . import _wandb


# Global state
_data_dir: Optional[str] = None
_use_wandb: bool = True
_write_to_dir: bool = True
_in_memory_data: List[Dict[str, Any]] = []
_in_memory_lock = threading.Lock()
_standalone_server_thread: Optional[threading.Thread] = None
_standalone_port: int = 5000
_initialized: bool = False


def init(
    use_wandb: bool = True,
    project: Optional[str] = None,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    standalone_port: int = 5000,
    run_standalone: bool = False,
    run: Any = None,
    write_to_dir: Optional[bool] = None,
    **wandb_kwargs: Any,
) -> None:
    """
    Initialize rollout_viz.

    Args:
        use_wandb: If True, log rollout accuracy to the current (or new) wandb run.
        project: Wandb project name (only used if we start a new run).
        name: Wandb run name (optional).
        data_dir: Directory to read/write rollout JSONL files (e.g. 1.jsonl, 2.jsonl).
                  Used by the standalone server and for persisting log_rollout() data.
        standalone_port: Port for the standalone web server when run_standalone=True.
        run_standalone: If True, start the standalone web server in a background thread so
                        the UI updates as rollouts are generated (can be used with or without wandb).
        **wandb_kwargs: Passed to wandb.init() if we start a new run (e.g. config=...).
    """
    global _data_dir, _use_wandb, _write_to_dir, _standalone_port, _initialized, _standalone_server_thread

    _data_dir = data_dir
    _use_wandb = use_wandb
    if write_to_dir is not None:
        _write_to_dir = write_to_dir
    _standalone_port = standalone_port
    _initialized = True

    if use_wandb:
        try:
            import wandb
            # If a specific wandb.Run instance was provided, attach directly to it.
            active_run = run
            if active_run is None:
                active_run = wandb.run
            if active_run is None:
                kwargs = {"project": project, "name": name, **wandb_kwargs}
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                wandb.init(**kwargs)
                active_run = wandb.run
            if active_run is not None:
                _wandb.set_wandb_run(active_run)
        except Exception:
            _wandb.set_wandb_run(None)

    if run_standalone:
        _start_standalone_server()


def _start_standalone_server() -> None:
    global _standalone_server_thread
    if _standalone_server_thread is not None:
        return
    from ._server import create_app, run_server
    app = create_app(get_data_callback=_get_data)
    _standalone_server_thread = threading.Thread(
        target=lambda: run_server(app, port=_standalone_port),
        daemon=True,
    )
    _standalone_server_thread.start()


def _get_data() -> List[Dict[str, Any]]:
    """Return combined data from data_dir (if set) and in-memory buffer."""
    with _in_memory_lock:
        in_mem = list(_in_memory_data)
    if _data_dir and os.path.isdir(_data_dir):
        from_dir = _data.load_rollout_data_from_dir(_data_dir)
        # Merge: from_dir takes precedence for steps that exist there; in_mem fills gaps
        steps_in_dir = set(e.get("step") for e in from_dir)
        extra = [e for e in in_mem if e.get("step") not in steps_in_dir]
        return from_dir + extra
    return in_mem


def set_data_dir(path: Optional[str]) -> None:
    """Set the data directory for reading/writing rollout JSONL files."""
    global _data_dir
    _data_dir = path
    if path:
        os.makedirs(path, exist_ok=True)


def get_data_dir() -> Optional[str]:
    return _data_dir


def log_rollout(entry: Dict[str, Any]) -> None:
    """
    Log a single rollout. Data is kept in memory and optionally appended to data_dir.
    If use_wandb was True in init(), accuracy is logged to wandb (updated per step).
    """
    global _in_memory_data
    normalized = _data.normalize_entry(entry)
    step = normalized.get("step", 0)

    with _in_memory_lock:
        _in_memory_data.append(normalized)

    if _data_dir and _write_to_dir:
        step_file = os.path.join(_data_dir, f"{step}.jsonl")
        with open(step_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if _use_wandb and _wandb.get_wandb_run() is not None:
        data = _get_data()
        _wandb.log_rollout_accuracy(data, step)


def flush_wandb() -> None:
    """Flush any pending wandb logs and optionally log full accuracy series."""
    if _use_wandb and _wandb.get_wandb_run() is not None:
        data = _get_data()
        _wandb.log_full_accuracy_series(data)
        _wandb.log_interactive_visualization(data)


def is_standalone_running() -> bool:
    return _standalone_server_thread is not None and _standalone_server_thread.is_alive()
