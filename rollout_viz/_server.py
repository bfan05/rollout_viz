"""Standalone Flask server for rollout visualization (used when use_wandb=False)."""

import os
import base64
from io import BytesIO
from typing import Callable, List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify

from . import _data


def _generate_plot(
    data: List[Dict[str, Any]], question_id: str = None, title: str = None
) -> str:
    """Generate accuracy-over-time plot as base64 PNG."""
    accuracy_data = _data.calculate_accuracy_over_time(data, question_id=question_id)
    plt.style.use("default")

    if not accuracy_data:
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.text(
            0.5, 0.5, "No data available",
            ha="center", va="center", transform=ax.transAxes, fontsize=14, color="#666",
        )
        ax.set_xlabel("Training Step", fontsize=12, color="#333")
        ax.set_ylabel("Accuracy", fontsize=12, color="#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        steps = sorted(accuracy_data.keys())
        accuracies = [accuracy_data[s] for s in steps]
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        color = "#2563eb"
        ax.plot(
            steps, accuracies, marker="o", linewidth=2.5, markersize=10,
            color=color, markerfacecolor=color, markeredgecolor="white", markeredgewidth=2, zorder=3,
        )
        ax.fill_between(steps, accuracies, alpha=0.15, color=color, zorder=1)
        if title:
            ax.set_title(title, fontsize=14, color="#333", fontweight="600", pad=15)
        ax.set_xlabel("Training Step", fontsize=13, color="#333", fontweight="500", labelpad=10)
        ax.set_ylabel("Accuracy", fontsize=13, color="#333", fontweight="500", labelpad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")
        ax.grid(True, alpha=0.2, color="#e5e7eb", linestyle="-", linewidth=1)
        ax.set_axisbelow(True)
        y_max = max(accuracies) if accuracies else 1.0
        y_padding = max(0.15, y_max * 0.2)
        ax.set_ylim([0, min(1.05, y_max + y_padding)])
        if len(steps) > 1:
            ax.set_xlim([min(steps) - 0.3, max(steps) + 0.3])
        else:
            ax.set_xlim([steps[0] - 0.5, steps[0] + 0.5])
        ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks()], fontsize=11, color="#666")
        ax.set_xticklabels(ax.get_xticks(), fontsize=11, color="#666")

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=120, facecolor="white", edgecolor="none")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64


def _classify_difficulty(accuracy: float) -> str:
    if accuracy >= 0.7:
        return "easy"
    if accuracy >= 0.3:
        return "medium"
    return "hard"


def create_app(get_data_callback: Callable[[], List[Dict[str, Any]]]) -> Flask:
    """Create Flask app that uses get_data_callback() for all data."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(pkg_dir, "templates")
    app = Flask(__name__, template_folder=template_dir)

    def data():
        return get_data_callback()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/question_ids")
    def api_question_ids():
        try:
            question_ids = _data.get_question_ids(data())
            return jsonify(question_ids)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/question_info/<question_id>")
    def api_question_info(question_id):
        info = _data.get_question_info(data(), question_id)
        return jsonify(info)

    @app.route("/api/plot/<question_id>")
    def api_plot(question_id):
        title = "Question Accuracy Over Training Steps"
        img_base64 = _generate_plot(data(), question_id=question_id, title=title)
        return jsonify({"image": img_base64})

    @app.route("/api/responses/<question_id>/<int:step>")
    def api_responses(question_id, step):
        responses = _data.get_responses_at_step(data(), question_id, step)
        question_info = _data.get_question_info(data(), question_id)
        return jsonify({
            "responses": responses,
            "step": step,
            "question_id": question_id,
            "user_question": question_info.get("user_question", ""),
            "ground_truth": question_info.get("ground_truth", ""),
        })

    @app.route("/api/steps")
    def api_steps():
        return jsonify(_data.get_unique_steps(data()))

    @app.route("/api/questions")
    def api_questions():
        try:
            d = data()
            questions = {}
            for entry in d:
                qid = entry.get("question_id")
                if not qid:
                    continue
                if qid not in questions:
                    questions[qid] = {
                        "id": qid,
                        "user_question": entry.get("user_question", ""),
                        "ground_truth": entry.get("gts", ""),
                        "steps": set(),
                        "total_correct": 0,
                        "total_rollouts": 0,
                    }
                questions[qid]["steps"].add(entry.get("step", 0))
                if entry.get("score", 0.0) == 1.0:
                    questions[qid]["total_correct"] += 1
                questions[qid]["total_rollouts"] += 1

            question_list = list(questions.values())
            for idx, q in enumerate(question_list):
                q["question_number"] = idx + 1
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
            return jsonify(question_list)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/status")
    def get_status():
        try:
            d = data()
            question_ids = _data.get_question_ids(d)
            steps = _data.get_unique_steps(d)
            return jsonify({
                "has_data": len(d) > 0,
                "question_count": len(question_ids),
                "total_entries": len(d),
                "steps": steps if steps else [],
                "data_source": "rollout_viz",
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "has_data": False,
                "error": str(e),
                "question_count": 0,
                "total_entries": 0,
                "steps": [],
                "data_source": "rollout_viz",
            }), 500

    return app


def run_server(app: Flask, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the Flask server (blocking). Use in a thread for background serving."""
    app.run(debug=False, host=host, port=port, use_reloader=False)
