"""
Generate `training/notebook.ipynb` programmatically.

Writing the notebook by hand is error-prone (JSON escaping, cell IDs, etc.).
This script defines each cell as a Python list of source lines and uses
nbformat to assemble the .ipynb. Re-run any time the notebook needs updating.

Run:
    python -m training.build_notebook
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = Path(__file__).resolve().parent / "notebook.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(*lines: str):
    return new_code_cell("\n".join(lines))


def main() -> None:
    cells = [
        md(
            "# Audible — Self-Improving Ambient-Listening Gate\n"
            "\n"
            "Hackathon submission notebook for the **OpenEnv Hackathon (India 2026), Theme #4 — Self-Improvement**.\n"
            "\n"
            "This notebook trains a tiny **mobileBERT** classifier (~25M params, edge-deployable) to act as the\n"
            "**gate** of an always-on voice assistant: given an ambient utterance and the active user's preference\n"
            "profile, decide whether to ACT (and which of 5 tools to call), UPDATE_CONTEXT, or IGNORE. It then\n"
            "evaluates the trained model against our OpenEnv environment and shows reward + per-component metrics.\n"
            "\n"
            "**Run all → ~10–15 min on a free Colab T4**.\n"
            "\n"
            "Repo: https://github.com/me-tusharchandra/audible  •  Env Space: https://huggingface.co/spaces/me-tusharchandra/audible-env"
        ),
        md("## 1. Install dependencies"),
        code(
            "%%capture",
            "# OpenEnv core (env framework) + ML stack",
            "!pip install --quiet \"openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git\"",
            "!pip install --quiet transformers torch accelerate datasets scikit-learn matplotlib pandas pyarrow python-dotenv openai",
        ),
        md("## 2. Pull the project repo\n\n"
           "The notebook trains against the `combined.parquet` dataset that lives in this repo, then evaluates against the audible env (started locally inside the Colab VM)."),
        code(
            "import os, subprocess, sys, time, pathlib",
            "REPO_DIR = pathlib.Path('/content/audible')",
            "if not REPO_DIR.exists():",
            "    !git clone --depth 1 https://github.com/me-tusharchandra/audible.git {REPO_DIR}",
            "%cd {REPO_DIR}",
            "sys.path.insert(0, str(REPO_DIR))",
            "print('repo at:', REPO_DIR)",
        ),
        md("## 3. Configure secrets (optional — only needed for synthetic data + curriculum)"),
        code(
            "from google.colab import userdata  # comment out if running locally",
            "try:",
            "    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')",
            "    print('OPENAI_API_KEY set — synthetic generation + curriculum available')",
            "except Exception as e:",
            "    print('No OPENAI_API_KEY in Colab Secrets — skipping synth/curriculum (eval still works)')",
        ),
        md(
            "## 4. Inspect the dataset\n"
            "\n"
            "Combined dataset = friend's binary CSV (heuristic-mapped to 7 classes) + LLM-generated synthetic\n"
            "scenarios that target ambient/profile-divergent cases the binary data lacks. ~7.5k unique utterances\n"
            "× 3 user profiles = ~22k labeled rows."
        ),
        code(
            "import pandas as pd",
            "df = pd.read_parquet('audible_env/data/combined.parquet')",
            "print(f'rows: {len(df):,}   utterances: {df[\"text\"].nunique():,}')",
            "print('\\nclass distribution per profile:')",
            "print(df.pivot_table(index='class_label', columns='profile', values='text', aggfunc='count', fill_value=0).to_string())",
        ),
        md(
            "## 5. Train mobileBERT (gating classifier)\n"
            "\n"
            "1 epoch on the combined dataset takes ~3 min on a Colab T4. The training script is a custom\n"
            "`WeightedTrainer` (HF `Trainer` subclass) with class-weighted CE loss to compensate for the\n"
            "natural class imbalance. Outputs go to `training/runs/<timestamp>/`.\n"
            "\n"
            "> *Note on TRL*: TRL's `PPOTrainer`/`GRPOTrainer` are built for generative LMs and don't fit\n"
            "> encoder-only classifiers. Our `WeightedTrainer` is a `transformers.Trainer` subclass — the same\n"
            "> base class TRL extends, with class-weighted CE in `compute_loss`."
        ),
        code(
            "!python -m training.train --epochs 1 --batch 32 2>&1 | tail -30",
        ),
        md("## 6. Start the OpenEnv server (locally inside this VM)"),
        code(
            "import subprocess, time, urllib.request",
            "ENV_PORT = 8765",
            "env_proc = subprocess.Popen(",
            "    ['python', '-m', 'uvicorn', 'server.app:app',",
            "     '--host', '127.0.0.1', '--port', str(ENV_PORT)],",
            "    cwd='audible_env', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,",
            ")",
            "for _ in range(30):",
            "    try:",
            "        urllib.request.urlopen(f'http://127.0.0.1:{ENV_PORT}/health', timeout=1).read()",
            "        print('env server up'); break",
            "    except Exception:",
            "        time.sleep(0.5)",
            "else:",
            "    raise RuntimeError('env server did not start')",
        ),
        md(
            "## 7. Evaluate the trained gate against the env\n"
            "\n"
            "Runs ~400 rollouts: env serves an utterance + profile, the gate classifies, env returns reward\n"
            "via the composite rubric (gate correctness + tool match + profile alignment + false-wake penalty).\n"
            "Aggregates per-profile so we can see how each preference profile is performing separately."
        ),
        code(
            "import json, pathlib",
            "from training.eval_env import GatePolicy, run_rollouts, summarize",
            "",
            "runs = sorted(pathlib.Path('training/runs').glob('*'))",
            "model_dir = runs[-1]",
            "print('using model:', model_dir.name)",
            "",
            "policy = GatePolicy(model_dir)",
            "rollouts = run_rollouts(policy, base_url=f'http://127.0.0.1:{ENV_PORT}', n=400, seed=0)",
            "summary = summarize(rollouts)",
            "print(json.dumps(summary, indent=2))",
        ),
        md("## 8. Visualise per-profile metrics"),
        code(
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "profiles = ['minimalist', 'proactive', 'work_focused']",
            "metrics_to_plot = ['mean_reward', 'decision_accuracy', 'tool_accuracy_when_act', 'false_wake_rate']",
            "vals = np.array([[summary['per_profile'][p][m] for m in metrics_to_plot] for p in profiles])",
            "",
            "fig, ax = plt.subplots(figsize=(10, 4))",
            "x = np.arange(len(metrics_to_plot)); w = 0.25",
            "for i, p in enumerate(profiles):",
            "    ax.bar(x + i * w, vals[i], w, label=p)",
            "ax.set_xticks(x + w); ax.set_xticklabels(metrics_to_plot, rotation=15)",
            "ax.legend(); ax.set_title('Per-profile env-rollout metrics')",
            "ax.grid(alpha=0.3, axis='y')",
            "plt.tight_layout(); plt.show()",
        ),
        md(
            "## 9. Theme #4: adaptive curriculum (optional, requires OPENAI_API_KEY)\n"
            "\n"
            "After each training round we:\n"
            "1. Eval the current gate against the env\n"
            "2. Mine the worst-reward rollouts as adversarial seeds\n"
            "3. Generate new scenarios via OpenAI: *\"here are utterances the gate gets wrong — make harder variations\"*\n"
            "4. Append to the dataset and retrain\n"
            "\n"
            "Across rounds the reward curve climbs because the gate gets stronger; the generator keeps escalating\n"
            "because it sees the gate's current weak spots. **That co-evolution is what makes this self-improvement.**\n"
            "\n"
            "Skipped here (each round adds ~10 min of compute). To run locally:\n"
            "```bash\n"
            "python -m training.curriculum --rounds 3 --gen-per-round 200\n"
            "```\n"
            "Pre-computed curriculum results are in `training/plots/reward_curve.png`."
        ),
        md(
            "## 10. Cleanup"
        ),
        code(
            "env_proc.terminate(); env_proc.wait(timeout=5); print('env server stopped')",
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.10"}

    NOTEBOOK_PATH.write_text(nbformat.writes(nb))
    print(f"wrote {NOTEBOOK_PATH} ({NOTEBOOK_PATH.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
