"""
Evaluate a trained mobileBERT gate against the OpenEnv environment.

Spins up the local server, then runs N rollouts via the WS client. For each
rollout: receive observation, run model inference, send action, collect
reward + per-component scores. Aggregates by profile so we can see how each
preference profile is performing separately.

Run:
    python -m training.eval_env --model training/runs/<RUN_ID> --rollouts 600
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from audible_env.client import AudibleEnv
from audible_env.models import PROFILE_DESCRIPTIONS, GateAction, GateObservation
from training.dataset import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def class_to_action(class_name: str) -> GateAction:
    if class_name == "IGNORE":
        return GateAction(decision="IGNORE")
    if class_name == "UPDATE_CONTEXT":
        return GateAction(decision="UPDATE_CONTEXT")
    assert class_name.startswith("ACT_"), class_name
    return GateAction(decision="ACT", tool=class_name[len("ACT_") :])


def _find_model_dir(run_dir: Path) -> Path:
    """If the run dir doesn't contain model weights at root, find the latest
    checkpoint-* subdir. Lets us load the best checkpoint even when
    train.py didn't explicitly call trainer.save_model() at the root."""
    if (run_dir / "config.json").exists() and (
        (run_dir / "model.safetensors").exists() or (run_dir / "pytorch_model.bin").exists()
    ):
        return run_dir
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model weights found at {run_dir} or in checkpoint-*/")
    return checkpoints[-1]


class GatePolicy:
    """Wraps a fine-tuned mobileBERT for inference at env-evaluation time."""

    def __init__(self, model_dir: Path, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights_dir = _find_model_dir(model_dir)
        # Tokenizer is saved at the run-dir root by train.py; model weights may
        # be in a checkpoint subdir.
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(weights_dir))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def act(self, obs: GateObservation) -> GateAction:
        description = PROFILE_DESCRIPTIONS[obs.user_profile]
        inputs = self.tokenizer(
            description,
            obs.utterance,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(self.device)
        logits = self.model(**inputs).logits
        class_id = int(logits.argmax(dim=-1).item())
        return class_to_action(CLASS_NAMES[class_id])


def start_server(port: int) -> subprocess.Popen:
    """Run uvicorn for the env in a child process; return the handle."""
    env_dir = REPO_ROOT / "audible_env"
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    proc = subprocess.Popen(
        [str(venv_python), "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=env_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    import urllib.request
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            return proc
        except Exception:
            time.sleep(0.3)
    proc.terminate()
    raise RuntimeError(f"server did not become ready on port {port}")


def run_rollouts(
    policy: GatePolicy, base_url: str, n: int, seed: int = 0
) -> List[Dict[str, Any]]:
    rollouts = []
    with AudibleEnv(base_url=base_url).sync() as client:
        for i in range(n):
            r = client.reset()
            obs = r.observation
            action = policy.act(obs)
            r = client.step(action)
            post = r.observation
            rollouts.append(
                {
                    "scenario_id": post.scenario_id,
                    "utterance": post.utterance,
                    "profile": post.user_profile,
                    "action_decision": action.decision,
                    "action_tool": action.tool,
                    "reward": r.reward,
                    "ground_truth": post.ground_truth,
                    "components": post.component_scores,
                }
            )
    return rollouts


def summarize(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_profile: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        by_profile[r["profile"]].append(r)

    overall = _profile_metrics(rollouts)
    per_profile = {p: _profile_metrics(rs) for p, rs in by_profile.items()}
    return {"overall": overall, "per_profile": per_profile, "n": len(rollouts)}


def _profile_metrics(rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rollouts:
        return {}
    rewards = [r["reward"] for r in rollouts]
    correct_decisions = sum(
        1 for r in rollouts if r["action_decision"] == (r["ground_truth"] or {}).get("decision")
    )
    correct_tools = sum(
        1 for r in rollouts
        if r["action_decision"] == "ACT" and (r["ground_truth"] or {}).get("decision") == "ACT"
        and r["action_tool"] == (r["ground_truth"] or {}).get("tool")
    )
    actionable = sum(1 for r in rollouts if (r["ground_truth"] or {}).get("decision") == "ACT")
    false_wakes = sum(
        1 for r in rollouts
        if r["action_decision"] == "ACT"
        and (r["ground_truth"] or {}).get("decision") in ("IGNORE", "UPDATE_CONTEXT")
    )
    non_act_total = sum(
        1 for r in rollouts
        if (r["ground_truth"] or {}).get("decision") in ("IGNORE", "UPDATE_CONTEXT")
    )
    return {
        "n": len(rollouts),
        "mean_reward": statistics.mean(rewards),
        "median_reward": statistics.median(rewards),
        "decision_accuracy": correct_decisions / len(rollouts),
        "tool_accuracy_when_act": correct_tools / actionable if actionable else 0.0,
        "false_wake_rate": false_wakes / non_act_total if non_act_total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path, help="Path to a training/runs/<id> dir")
    p.add_argument("--rollouts", type=int, default=600)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="Where to write metrics JSON (default: <model>/env_eval.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or (args.model / "env_eval.json")

    print(f"loading policy from {args.model}")
    policy = GatePolicy(args.model)

    print(f"starting env server on port {args.port}")
    server = start_server(args.port)
    try:
        rollouts = run_rollouts(
            policy, base_url=f"http://127.0.0.1:{args.port}",
            n=args.rollouts, seed=args.seed,
        )
    finally:
        server.terminate()
        server.wait(timeout=5)

    summary = summarize(rollouts)
    print("\n=== overall ===")
    for k, v in summary["overall"].items():
        print(f"  {k:25s} {v:.3f}" if isinstance(v, float) else f"  {k:25s} {v}")
    print("\n=== per profile ===")
    for prof, m in summary["per_profile"].items():
        print(f"  -- {prof} --")
        for k, v in m.items():
            print(f"    {k:25s} {v:.3f}" if isinstance(v, float) else f"    {k:25s} {v}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rollouts": rollouts}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
