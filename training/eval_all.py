"""
Re-evaluate all four checkpoints (baseline + 3 curriculum rounds) with the
SAME seed and the same number of rollouts, so the comparison is apples-to-apples.

The first pipeline run used `seed=round_idx` which means each round was
evaluated on a *different* random scenario distribution — that buried the
real signal under noise. This script fixes that.

Output:
    training/runs/eval_all.json   — clean baseline → round 1 → round 2 → round 3 metrics
    training/plots/reward_curve.png — re-rendered with apples-to-apples numbers
"""

from __future__ import annotations

import json
from pathlib import Path

from training.eval_env import GatePolicy, run_rollouts, start_server, summarize

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "training" / "runs"
OUT_PATH = RUNS_DIR / "eval_all.json"

# Identify the four checkpoints in chronological order.
CHECKPOINTS = {
    "baseline":   "20260425-183254",  # Phase 2 baseline (3 epochs on combined.parquet)
    "round_1":    "20260425-191113",  # post-round-1 retrain
    "round_2":    "20260425-192537",  # post-round-2 retrain
    "round_3":    "20260425-193905",  # post-round-3 retrain (final)
}

SEED = 42
N_ROLLOUTS = 400
PORT = 8770


def main() -> None:
    results: dict = {}

    for label, run_id in CHECKPOINTS.items():
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            print(f"[{label}] SKIP — {run_dir} missing")
            continue

        print(f"\n=== {label} ({run_id}) ===")
        policy = GatePolicy(run_dir)
        server = start_server(PORT)
        try:
            rollouts = run_rollouts(
                policy, base_url=f"http://127.0.0.1:{PORT}",
                n=N_ROLLOUTS, seed=SEED,
            )
        finally:
            server.terminate()
            server.wait(timeout=5)

        summary = summarize(rollouts)
        o = summary["overall"]
        print(f"  mean_reward:           {o['mean_reward']:+.3f}")
        print(f"  decision_accuracy:     {o['decision_accuracy']:.3f}")
        print(f"  tool_acc_when_act:     {o['tool_accuracy_when_act']:.3f}")
        print(f"  false_wake_rate:       {o['false_wake_rate']:.3f}")
        for prof in ("minimalist", "proactive", "work_focused"):
            pp = summary["per_profile"][prof]
            print(f"  [{prof:13s}] reward={pp['mean_reward']:+.3f}  acc={pp['decision_accuracy']:.3f}  fw={pp['false_wake_rate']:.3f}")

        results[label] = {"run_id": run_id, "summary": summary}

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
