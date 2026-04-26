"""
Plotting helpers for the README + 90-second demo video.

Three plots judges actually look at:
  1. dataset_distribution.png   — class × source (heuristic / synthetic / curriculum)
  2. reward_curve.png            — mean reward over curriculum rounds, per profile
  3. confusion_matrix.png        — final eval, per-class

All plots are saved as PNG with axis labels and titles. Don't be cute about
formatting — readable beats pretty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "audible_env" / "data"
PLOTS_DIR = REPO / "training" / "plots"


def plot_dataset_distribution(combined_path: Path = DATA_DIR / "combined.parquet") -> Path:
    df = pd.read_parquet(combined_path)
    pivot = df.pivot_table(
        index="class_label", columns="source", values="text", aggfunc="count", fill_value=0,
    )
    pivot = pivot.reindex(
        ["IGNORE", "UPDATE_CONTEXT", "ACT_set_timer", "ACT_add_calendar_event",
         "ACT_play_music", "ACT_web_search", "ACT_smart_home_control"]
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_xlabel("class")
    ax.set_ylabel("count (rows = utterances × profiles)")
    ax.set_title("Training set composition by source")
    ax.legend(title="source", loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = PLOTS_DIR / "dataset_distribution.png"
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def plot_reward_curve(curriculum_log: Path) -> Path:
    log: List[Dict[str, Any]] = json.loads(curriculum_log.read_text())
    rounds = [r["round"] for r in log]

    fig, ax = plt.subplots(figsize=(9, 5))
    profiles = ["minimalist", "proactive", "work_focused"]
    for prof in profiles:
        ys = [r["metrics"]["per_profile"].get(prof, {}).get("mean_reward", 0.0) for r in log]
        ax.plot(rounds, ys, marker="o", label=prof)

    overall = [r["metrics"]["overall"]["mean_reward"] for r in log]
    ax.plot(rounds, overall, marker="s", linewidth=2.5, color="black", label="overall")

    ax.set_xlabel("curriculum round")
    ax.set_ylabel("mean reward (env rollouts)")
    ax.set_title("Self-improvement: reward over adversarial curriculum rounds")
    ax.set_xticks(rounds)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "reward_curve.png"
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def plot_eval_all(eval_all_path: Path) -> Path:
    """Plot baseline → round 1 → round 2 → round 3 with same-seed evals.

    The curriculum_log.json plot is misleading because each round in that log
    was evaluated against a different random seed (`seed=round_idx`).
    eval_all.json runs all four checkpoints with the same seed for a fair
    apples-to-apples comparison.
    """
    data = json.loads(eval_all_path.read_text())
    labels = list(data.keys())  # baseline, round_1, round_2, round_3
    xs = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean reward per profile + overall
    ax = axes[0]
    for prof, color in zip(("minimalist", "proactive", "work_focused"),
                           ("tab:blue", "tab:orange", "tab:green")):
        ys = [data[lbl]["summary"]["per_profile"][prof]["mean_reward"] for lbl in labels]
        ax.plot(xs, ys, marker="o", label=prof, color=color)
    overall = [data[lbl]["summary"]["overall"]["mean_reward"] for lbl in labels]
    ax.plot(xs, overall, marker="s", linewidth=2.5, color="black", label="overall")
    ax.set_xlabel("curriculum stage")
    ax.set_ylabel("mean reward  (env rollouts, max=2.0)")
    ax.set_title("Reward over curriculum (same seed, 400 rollouts)")
    ax.set_xticks(xs); ax.set_xticklabels([l.replace("_", " ") for l in labels])
    ax.legend(); ax.grid(alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.5)

    # Right: false-wake rate per profile (lower is better)
    ax = axes[1]
    for prof, color in zip(("minimalist", "proactive", "work_focused"),
                           ("tab:blue", "tab:orange", "tab:green")):
        ys = [data[lbl]["summary"]["per_profile"][prof]["false_wake_rate"] for lbl in labels]
        ax.plot(xs, ys, marker="o", label=prof, color=color)
    fw_overall = [data[lbl]["summary"]["overall"]["false_wake_rate"] for lbl in labels]
    ax.plot(xs, fw_overall, marker="s", linewidth=2.5, color="black", label="overall")
    ax.set_xlabel("curriculum stage")
    ax.set_ylabel("false-wake rate  (lower is better)")
    ax.set_title("False-wake rate over curriculum")
    ax.set_xticks(xs); ax.set_xticklabels([l.replace("_", " ") for l in labels])
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = PLOTS_DIR / "reward_curve.png"
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def plot_confusion(metrics_json: Path, class_names: List[str]) -> Path:
    metrics = json.loads(metrics_json.read_text())
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalized)")
    plt.colorbar(im, ax=ax)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    out = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def main() -> None:
    out = plot_dataset_distribution()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
