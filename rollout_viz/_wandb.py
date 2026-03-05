"""Wandb logging for rollout accuracy over time."""

import json
from collections import defaultdict
from typing import List, Dict, Any, Optional

import wandb

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
        run.log({"rollout/accuracy": acc}, step=s)


def log_interactive_visualization(data: List[Dict[str, Any]]) -> None:
    """
    Create and log an interactive HTML visualization of per-question accuracy and rollouts.

    This replicates the rich WandB visualization previously implemented in simple_grpo,
    but uses normalized rollout_viz entries so it works for any project.
    """
    run = get_wandb_run()
    if run is None:
        return

    # Normalize entries to ensure question_id, user_question, gts, score, step are present.
    normalized: List[Dict[str, Any]] = []
    for entry in data:
        try:
            normalized.append(normalize_entry(entry))
        except Exception:
            continue

    if not normalized:
        return

    # Aggregate per-question, per-step stats and rollout details.
    question_metadata: Dict[str, Dict[str, Any]] = {}
    per_step_question_stats: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    rollout_data: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in normalized:
        qid = record.get("question_id") or record.get("prompt_id")
        step = record.get("step")
        if not qid or step is None:
            continue

        user_question = record.get("user_question") or record.get("input") or ""
        ground_truth = record.get("gts") or record.get("ground_truth", "")

        # Determine correctness and reward from score/correct/reward fields.
        score = record.get("score", 0.0)
        if isinstance(score, bool):
            is_correct = bool(score)
        else:
            try:
                is_correct = float(score) == 1.0
            except Exception:
                is_correct = bool(record.get("correct", False))
        reward_val = record.get("reward", score if isinstance(score, (int, float)) else 0.0)
        try:
            reward_val = float(reward_val)
        except Exception:
            reward_val = 0.0

        # Question metadata (question text, ground truth, first seen step).
        meta = question_metadata.get(qid)
        if meta is None:
            question_metadata[qid] = {
                "question": user_question,
                "ground_truth": ground_truth,
                "first_seen_step": step,
            }
        else:
            if not meta.get("question") and user_question:
                meta["question"] = user_question
            if not meta.get("ground_truth") and ground_truth:
                meta["ground_truth"] = ground_truth
            if step < meta.get("first_seen_step", step):
                meta["first_seen_step"] = step

        # Per-step stats for accuracy.
        stats = per_step_question_stats[qid][step]
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1

        # Rollout details for viewer.
        rollout_data[qid][step].append(
            {
                "completion": record.get("output") or record.get("completion", ""),
                "prediction": record.get("prediction", ""),
                "ground_truth": ground_truth,
                "correct": is_correct,
                "reward": reward_val,
            }
        )

    if not question_metadata:
        return

    # Build per-question accuracy history: list of (step, accuracy) pairs.
    accuracy_history: Dict[str, List[Any]] = {}
    for qid, step_dict in per_step_question_stats.items():
        history = []
        for step, stats in sorted(step_dict.items()):
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                history.append((step, acc))
        if history:
            accuracy_history[qid] = history

    if not accuracy_history:
        return

    # Build HTML with three tabs: overall accuracy, question browser, rollout viewer.
    html_parts: List[str] = []
    html_parts.append(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Question Accuracy Tracker</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }
                .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; }
                .tab { padding: 12px 24px; cursor: pointer; background: #f5f5f5; border: none; border-bottom: 2px solid transparent; margin-right: 4px; font-size: 14px; font-weight: 500; color: #000; }
                .tab:hover { background: #e8e8e8; }
                .tab.active { background: white; border-bottom: 2px solid #4CAF50; color: #4CAF50; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                .chart-container { margin: 20px 0; }
                .question-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }
                .question-box { padding: 15px; background: #f9f9f9; border: 2px solid #ddd; border-radius: 6px; cursor: pointer; transition: all 0.2s; }
                .question-box:hover { border-color: #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2); }
                .question-box.selected { border-color: #4CAF50; background: #e8f5e9; }
                .question-preview { font-size: 13px; color: #555; margin-top: 8px; line-height: 1.4; }
                .question-preview strong { color: #333; }
                .question-detail { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 6px; border-left: 4px solid #4CAF50; }
                .question-detail h3 { margin-top: 0; color: #333; }
                .question-detail p { margin: 8px 0; line-height: 1.6; }
                .question-detail-chart { margin: 20px 0; }
                .controls { margin: 20px 0; }
                .controls label { display: block; margin-bottom: 8px; font-weight: 500; }
                .controls select { padding: 10px; font-size: 14px; width: 100%; max-width: 400px; border: 2px solid #ddd; border-radius: 4px; }
                .controls select:focus { border-color: #4CAF50; outline: none; }
                .rollout-list { margin: 20px 0; }
                .rollout-item { padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #ddd; background: #f9f9f9; }
                .rollout-item.correct { border-left-color: #4CAF50; background: #e8f5e9; }
                .rollout-item.incorrect { border-left-color: #f44336; background: #ffebee; }
                .rollout-item strong { display: block; margin-bottom: 8px; }
                .rollout-completion { margin-top: 8px; padding: 10px; background: white; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
                .rollout-meta { margin-top: 8px; font-size: 12px; color: #666; }
                .rollout-meta span { margin-right: 20px; }
                .rollout-meta span:last-child { margin-right: 0; }
                h2 { color: #333; margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>📊 Per-Question Accuracy Over Time</h2>
                <div class="tabs">
                    <button class="tab active" onclick="switchTab(0)">📈 Overall Accuracy</button>
                    <button class="tab" onclick="switchTab(1)">📋 Question Browser</button>
                    <button class="tab" onclick="switchTab(2)">🔍 Rollout Viewer</button>
                </div>
                
                <!-- Tab 1: Overall Accuracy -->
                <div id="tab0" class="tab-content active">
                    <div class="chart-container">
                        <div id="overallChart"></div>
                    </div>
                </div>
                
                <!-- Tab 2: Question Browser -->
                <div id="tab1" class="tab-content">
                    <div id="questionDetail" class="question-detail" style="display: none;"></div>
                    <div class="question-grid" id="questionGrid"></div>
                </div>
                
                <!-- Tab 3: Rollout Viewer -->
                <div id="tab2" class="tab-content">
                    <div class="controls">
                        <label for="rolloutQuestionSelect">Question ID:</label>
                        <select id="rolloutQuestionSelect" onchange="updateRolloutStepDropdown()">
                            <option value="">-- Select Question --</option>
                        </select>
                    </div>
                    <div class="controls">
                        <label for="rolloutStepSelect">Training Step:</label>
                        <select id="rolloutStepSelect" onchange="displayRollouts()">
                            <option value="">-- Select Step --</option>
                        </select>
                    </div>
                    <div id="rolloutChartContainer" class="chart-container" style="display: none;">
                        <div id="rolloutChart"></div>
                    </div>
                    <div id="rolloutList" class="rollout-list"></div>
                </div>
            </div>
            
            <script>
                const accuracyData = {
        """
    )

    # Add data for each question
    for question_id, history in accuracy_history.items():
        sorted_history = sorted(history, key=lambda x: x[0])
        steps = [h[0] for h in sorted_history]
        accuracies = [h[1] for h in sorted_history]
        html_parts.append(f'"{question_id}": {{')
        html_parts.append(f"  steps: {json.dumps(steps)},")
        html_parts.append(f"  accuracies: {json.dumps(accuracies)}")
        html_parts.append("},")

    # Calculate average across all questions
    if accuracy_history:
        all_steps = set()
        for history in accuracy_history.values():
            all_steps.update([h[0] for h in history])

        avg_data: Dict[int, float] = {}
        for step in sorted(all_steps):
            accs = []
            for history in accuracy_history.values():
                for s, acc in history:
                    if s == step:
                        accs.append(acc)
            if accs:
                avg_data[step] = sum(accs) / len(accs)

        html_parts.append('"all": {')
        html_parts.append(f"  steps: {json.dumps(sorted(avg_data.keys()))},")
        html_parts.append(
            f"  accuracies: {json.dumps([avg_data[s] for s in sorted(avg_data.keys())])}"
        )
        html_parts.append("}")

    html_parts.append(
        """
                };
                
                const questionMetadata = {
        """
    )

    # Add question metadata
    for question_id, meta in question_metadata.items():
        html_parts.append(f'"{question_id}": {{')
        html_parts.append(f"  question: {json.dumps(meta.get('question', ''))},")
        html_parts.append(
            f"  ground_truth: {json.dumps(meta.get('ground_truth', ''))},"
        )
        html_parts.append(f"  first_seen_step: {meta.get('first_seen_step', 0)}")
        html_parts.append("},")

    html_parts.append(
        """
                };
                
                const rolloutData = {
        """
    )

    # Add rollout data
    for question_id, step_data in rollout_data.items():
        html_parts.append(f'"{question_id}": {{')
        for step, rollouts in step_data.items():
            html_parts.append(f'  "{step}": {json.dumps(rollouts)},')
        html_parts.append("},")

    html_parts.append(
        """
                };
                
                function switchTab(index) {
                    // Hide all tabs
                    for (let i = 0; i < 3; i++) {
                        document.getElementById(`tab${i}`).classList.remove('active');
                        document.querySelectorAll('.tab')[i].classList.remove('active');
                    }
                    // Show selected tab
                    document.getElementById(`tab${index}`).classList.add('active');
                    document.querySelectorAll('.tab')[index].classList.add('active');
                    
                    // Initialize tab content if needed
                    if (index === 0) {
                        updateOverallChart();
                    } else if (index === 1) {
                        renderQuestionGrid();
                    } else if (index === 2) {
                        populateRolloutQuestionDropdown();
                    }
                }
                
                // Helper function to calculate dynamic x-axis tick spacing (max 32 ticks)
                function getXAxisConfig(steps) {
                    if (!steps || steps.length === 0) {
                        return { tickmode: 'linear', dtick: 1 };
                    }
                    
                    const minStep = Math.min(...steps);
                    const maxStep = Math.max(...steps);
                    const stepRange = maxStep - minStep;
                    
                    if (stepRange === 0) {
                        return { tickmode: 'linear', dtick: 1 };
                    }
                    
                    // Calculate tick spacing to have at most 32 ticks
                    const maxTicks = 32;
                    const tickSpacing = Math.ceil(stepRange / maxTicks);
                    
                    return {
                        tickmode: 'linear',
                        dtick: tickSpacing,
                        title: 'Training Step (1, 2, 3, ...)',
                        gridcolor: '#e0e0e0'
                    };
                }
                
                // Helper function to calculate dynamic y-axis range
                function getYAxisConfig(accuracies) {
                    if (!accuracies || accuracies.length === 0) {
                        return { title: 'Accuracy', range: [0, 1], gridcolor: '#e0e0e0' };
                    }
                    
                    const minAcc = Math.min(...accuracies);
                    const maxAcc = Math.max(...accuracies);
                    const accRange = maxAcc - minAcc;
                    
                    // Add padding: 10% on each side, but ensure we don't go below 0 or above 1
                    const padding = Math.max(accRange * 0.1, 0.05);
                    let yMin = Math.max(0, minAcc - padding);
                    let yMax = Math.min(1, maxAcc + padding);
                    
                    // If the range is very small, ensure we have at least some visible range
                    if (yMax - yMin < 0.1) {
                        const center = (yMin + yMax) / 2;
                        yMin = Math.max(0, center - 0.1);
                        yMax = Math.min(1, center + 0.1);
                    }
                    
                    // If all values are very close to 0 or 1, use a reasonable default
                    if (yMax - yMin < 0.01) {
                        if (maxAcc < 0.1) {
                            yMin = 0;
                            yMax = 0.2;
                        } else if (minAcc > 0.9) {
                            yMin = 0.8;
                            yMax = 1;
                        } else {
                            yMin = 0;
                            yMax = 1;
                        }
                    }
                    
                    return {
                        title: 'Accuracy',
                        range: [yMin, yMax],
                        gridcolor: '#e0e0e0'
                    };
                }
                
                function updateOverallChart() {
                    const allData = accuracyData["all"];
                    const trace = {
                        x: allData.steps,
                        y: allData.accuracies,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Average Accuracy',
                        line: { width: 3, color: '#4CAF50' },
                        marker: { size: 6 }
                    };
                    
                    const layout = {
                        title: 'Overall Accuracy Over Time (All Questions)',
                        xaxis: getXAxisConfig(allData.steps),
                        yaxis: getYAxisConfig(allData.accuracies),
                        hovermode: 'closest',
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white'
                    };
                    
                    Plotly.newPlot('overallChart', [trace], layout);
                }
                
                function renderQuestionGrid() {
                    const grid = document.getElementById('questionGrid');
                    grid.innerHTML = '';
                    
                    const questions = Object.keys(questionMetadata).sort((a, b) => {
                        return questionMetadata[a].first_seen_step - questionMetadata[b].first_seen_step;
                    });
                    
                    questions.forEach(qid => {
                        const meta = questionMetadata[qid];
                        const questionText = meta.question || 'No question text';
                        const preview = questionText.length > 100 ? questionText.substring(0, 100) + '...' : questionText;
                        
                        const box = document.createElement('div');
                        box.className = 'question-box';
                        box.onclick = function() { showQuestionDetail(qid, this); };
                        box.innerHTML = `
                            <strong>ID: ${qid.substring(0, 12)}...</strong>
                            <div class="question-preview">${preview}</div>
                        `;
                        grid.appendChild(box);
                    });
                }
                
                function showQuestionDetail(questionId, clickedBox) {
                    const meta = questionMetadata[questionId];
                    const detailDiv = document.getElementById('questionDetail');
                    
                    // Update selected box
                    document.querySelectorAll('.question-box').forEach(box => {
                        box.classList.remove('selected');
                    });
                    if (clickedBox) {
                        clickedBox.classList.add('selected');
                    }
                    
                    // Get accuracy data for this question
                    const questionData = accuracyData[questionId];
                    const hasData = questionData && questionData.steps && questionData.steps.length > 0;
                    
                    detailDiv.style.display = 'block';
                    detailDiv.innerHTML = `
                        <h3>Question Details</h3>
                        <p><strong>Question ID:</strong> ${questionId}</p>
                        <p><strong>Question:</strong> ${meta.question || 'N/A'}</p>
                        <p><strong>Correct Answer:</strong> ${meta.ground_truth || 'N/A'}</p>
                        <p><strong>First Seen:</strong> Training Step ${meta.first_seen_step}</p>
                        ${hasData ? '<div class="question-detail-chart"><div id="questionDetailChart"></div></div>' : '<p><em>No accuracy data available yet for this question.</em></p>'}
                    `;
                    
                    // Render chart if data is available
                    if (hasData) {
                        setTimeout(() => {
                            const trace = {
                                x: questionData.steps,
                                y: questionData.accuracies,
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: 'Accuracy',
                                line: { width: 3, color: '#2196F3' },
                                marker: { size: 6 }
                            };
                            
                            const layout = {
                                title: 'Accuracy Over Time',
                                xaxis: getXAxisConfig(questionData.steps),
                                yaxis: getYAxisConfig(questionData.accuracies),
                                hovermode: 'closest',
                                plot_bgcolor: '#fafafa',
                                paper_bgcolor: 'white',
                                height: 400
                            };
                            
                            Plotly.newPlot('questionDetailChart', [trace], layout);
                        }, 100);
                    }
                }
                
                function populateRolloutQuestionDropdown() {
                    const select = document.getElementById('rolloutQuestionSelect');
                    select.innerHTML = '<option value="">-- Select Question --</option>';
                    
                    Object.keys(rolloutData).forEach(qid => {
                        const meta = questionMetadata[qid];
                        const questionText = meta.question || 'No question';
                        const display = questionText.length > 60 ? questionText.substring(0, 60) + '...' : questionText;
                        const option = document.createElement('option');
                        option.value = qid;
                        option.textContent = `${qid.substring(0, 12)}... - ${display}`;
                        select.appendChild(option);
                    });
                }
                
                function updateRolloutStepDropdown() {
                    const questionSelect = document.getElementById('rolloutQuestionSelect');
                    const stepSelect = document.getElementById('rolloutStepSelect');
                    const questionId = questionSelect.value;
                    const chartContainer = document.getElementById('rolloutChartContainer');
                    
                    stepSelect.innerHTML = '<option value="">-- Select Step --</option>';
                    
                    if (questionId && rolloutData[questionId]) {
                        const steps = Object.keys(rolloutData[questionId]).map(s => parseInt(s)).sort((a, b) => a - b);
                        steps.forEach(step => {
                            const option = document.createElement('option');
                            option.value = step;
                            option.textContent = `Step ${step}`;
                            stepSelect.appendChild(option);
                        });
                        
                        // Show and update accuracy chart
                        updateRolloutChart(questionId);
                    } else {
                        // Hide chart if no question selected
                        chartContainer.style.display = 'none';
                    }
                    
                    displayRollouts();
                }
                
                function updateRolloutChart(questionId) {
                    const chartContainer = document.getElementById('rolloutChartContainer');
                    const questionData = accuracyData[questionId];
                    
                    if (!questionData || !questionData.steps || questionData.steps.length === 0) {
                        chartContainer.style.display = 'none';
                        return;
                    }
                    
                    chartContainer.style.display = 'block';
                    
                    setTimeout(() => {
                        const trace = {
                            x: questionData.steps,
                            y: questionData.accuracies,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Accuracy',
                            line: { width: 3, color: '#2196F3' },
                            marker: { size: 6 }
                        };
                        
                        const layout = {
                            title: 'Accuracy Over Time',
                            xaxis: getXAxisConfig(questionData.steps),
                            yaxis: getYAxisConfig(questionData.accuracies),
                            hovermode: 'closest',
                            plot_bgcolor: '#fafafa',
                            paper_bgcolor: 'white',
                            height: 400
                        };
                        
                        Plotly.newPlot('rolloutChart', [trace], layout);
                    }, 100);
                }
                
                function displayRollouts() {
                    const questionSelect = document.getElementById('rolloutQuestionSelect');
                    const stepSelect = document.getElementById('rolloutStepSelect');
                    const questionId = questionSelect.value;
                    const step = stepSelect.value;
                    const listDiv = document.getElementById('rolloutList');
                    
                    listDiv.innerHTML = '';
                    
                    if (!questionId) {
                        return;
                    }
                    
                    // Update chart if question is selected (even if step isn't)
                    if (questionId) {
                        updateRolloutChart(questionId);
                    }
                    
                    if (!step) {
                        return;
                    }
                    
                    const rollouts = rolloutData[questionId] && rolloutData[questionId][step];
                    if (!rollouts || rollouts.length === 0) {
                        listDiv.innerHTML = '<p>No rollouts found for this question at this step.</p>';
                        return;
                    }
                    
                    rollouts.forEach((rollout, idx) => {
                        const item = document.createElement('div');
                        item.className = `rollout-item ${rollout.correct ? 'correct' : 'incorrect'}`;
                        item.innerHTML = `
                            <strong>Rollout ${idx + 1} - ${rollout.correct ? '✓ Correct' : '✗ Incorrect'}</strong>
                            <div class="rollout-meta">
                                <span><strong>Prediction:</strong> ${rollout.prediction || 'N/A'}</span>
                                <span><strong>Ground Truth:</strong> ${rollout.ground_truth || 'N/A'}</span>
                                <span><strong>Reward:</strong> ${rollout.reward.toFixed(2)}</span>
                            </div>
                            <div class="rollout-completion">${rollout.completion || 'N/A'}</div>
                        `;
                        listDiv.appendChild(item);
                    });
                }
                
                // Initialize
                updateOverallChart();
            </script>
        </body>
        </html>
        """
    )

    html_content = "".join(html_parts)

    # Log as HTML; this appears in Files > media > html in WandB.
    run.log({"question_accuracy_interactive": wandb.Html(html_content)})
