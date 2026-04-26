"""
Phase 3: adaptive curriculum loop — the self-improvement engine.

Each round:
  1. Run env-rollout eval against the current trained model.
  2. Mine the worst failures (lowest reward, esp. false wakes).
  3. Use those failures as seeds for a new targeted OpenAI generation:
     "here are utterances the gate already gets wrong — generate adversarial
     variations that probe the same weakness more deeply."
  4. Append the new scenarios to the training set, retrain.
  5. Re-eval and log per-round metrics.

Across rounds the reward curve climbs because the gate gets stronger; the
*generator* keeps escalating because it sees the gate's current weak spots.
That co-evolution is what makes this Theme #4 self-improvement, not just
supervised SFT with extra steps.

Run:
    python -m training.curriculum --rounds 3 --gen-per-round 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from training.dataset import CLASS_NAMES
from training.eval_env import GatePolicy, run_rollouts, start_server, summarize
from training.synthetic_data import (
    GeneratedBatch,
    PROFILES,
    Scenario,
    SYSTEM_PROMPT,
    get_client,
    to_dataframe,
)

load_dotenv()

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "audible_env" / "data"
COMBINED = DATA_DIR / "combined.parquet"
CURRICULUM_DIR = DATA_DIR / "curriculum"
RUNS_DIR = REPO / "training" / "runs"


@dataclass
class Failure:
    utterance: str
    profile: str
    predicted_decision: str
    predicted_tool: str | None
    gt_decision: str
    gt_tool: str | None
    reward: float


def mine_failures(rollouts: List[Dict[str, Any]], top_k: int = 30) -> List[Failure]:
    """Pick the worst-reward rollouts as adversarial seeds.

    False wakes (predicted ACT, GT was IGNORE) score -1.0 and end up at the
    top of this list — exactly the failure mode we most want to address.
    """
    failures = [
        Failure(
            utterance=r["utterance"],
            profile=r["profile"],
            predicted_decision=r["action_decision"],
            predicted_tool=r["action_tool"],
            gt_decision=(r["ground_truth"] or {}).get("decision", "?"),
            gt_tool=(r["ground_truth"] or {}).get("tool"),
            reward=r["reward"],
        )
        for r in rollouts
    ]
    failures.sort(key=lambda f: f.reward)  # lowest first
    return failures[:top_k]


def adversarial_prompt(failures: List[Failure], n_to_generate: int) -> str:
    """Prompt the generator with the gate's current failure modes."""
    lines = []
    for f in failures[:15]:
        lines.append(
            f"- {f.utterance!r}  [profile={f.profile}, "
            f"gate said {f.predicted_decision}"
            + (f"/{f.predicted_tool}" if f.predicted_tool else "")
            + f", correct was {f.gt_decision}"
            + (f"/{f.gt_tool}" if f.gt_tool else "")
            + f", reward={f.reward:+.2f}]"
        )
    failure_block = "\n".join(lines)
    return f"""\
Here are {len(failures[:15])} cases the current gate gets wrong:

{failure_block}

Generate {n_to_generate} NEW utterances that probe the SAME failure modes more
adversarially — utterances structurally similar to the failures above but
with subtle variations the gate is even less likely to handle correctly.

For each new utterance, supply per-profile labels (minimalist, proactive,
work_focused). Cover the same mix of profiles as the failure list. Your goal
is to make the next training round teach the gate something it can't already
do, not to cover ground it already handles well.

Avoid duplicating the exact wording of the failures — vary the phrasing while
preserving the underlying ambiguity that's tripping up the model.
"""


def generate_adversarial(
    client: OpenAI, failures: List[Failure], n: int = 200, batch_size: int = 10
) -> pd.DataFrame:
    """Run K parallel adversarial-generation calls seeded with the failures."""
    n_calls = (n + batch_size - 1) // batch_size
    print(f"  generating {n_calls * batch_size} adversarial scenarios...")

    all_rows: list[pd.DataFrame] = []
    for i in range(n_calls):
        prompt = adversarial_prompt(failures, batch_size)
        try:
            resp = client.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=GeneratedBatch,
                temperature=0.95,  # higher = more divergent variations
            )
            batch = resp.choices[0].message.parsed.scenarios
            all_rows.append(to_dataframe(batch, category="adversarial_curriculum"))
            print(f"    [{i + 1}/{n_calls}] +{len(batch)}")
        except Exception as e:
            print(f"    [{i + 1}/{n_calls}] FAILED: {e}")

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def run_round(
    round_idx: int,
    rounds_log: list[dict],
    n_gen: int,
    n_rollouts: int,
    epochs: int,
    port: int,
) -> None:
    """One curriculum round: eval → mine → generate → retrain."""
    print(f"\n{'=' * 60}\nROUND {round_idx}\n{'=' * 60}")

    # 1. Find the latest model checkpoint
    runs = sorted(RUNS_DIR.glob("*"), key=lambda p: p.name)
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        raise RuntimeError(f"No models in {RUNS_DIR} — train one before curriculum")
    latest = runs[-1]
    print(f"using model: {latest.name}")

    # 2. Eval against env
    print("evaluating against env...")
    policy = GatePolicy(latest)
    server = start_server(port)
    try:
        rollouts = run_rollouts(
            policy, base_url=f"http://127.0.0.1:{port}", n=n_rollouts, seed=round_idx,
        )
    finally:
        server.terminate()
        server.wait(timeout=5)
    summary = summarize(rollouts)
    print(f"  mean reward: {summary['overall']['mean_reward']:+.3f}")
    print(f"  decision acc: {summary['overall']['decision_accuracy']:.3f}")
    print(f"  false wake:   {summary['overall']['false_wake_rate']:.3f}")

    # 3. Mine failures
    failures = mine_failures(rollouts, top_k=30)
    print(f"\nworst failures (top 5):")
    for f in failures[:5]:
        print(f"  reward={f.reward:+.2f}  '{f.utterance}'  ({f.profile})")
        print(f"    gate: {f.predicted_decision}/{f.predicted_tool}, gt: {f.gt_decision}/{f.gt_tool}")

    # 4. Generate adversarial data
    client = get_client()
    new_data = generate_adversarial(client, failures, n=n_gen)
    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)
    new_path = CURRICULUM_DIR / f"round_{round_idx:02d}.parquet"
    new_data.to_parquet(new_path, index=False)
    print(f"\nwrote {new_path} ({len(new_data)} rows)")

    # 5. Append to combined dataset
    base = pd.read_parquet(COMBINED)
    new_data["class_id"] = new_data["class_label"].map({c: i for i, c in enumerate(CLASS_NAMES)})
    cols = base.columns.intersection(new_data.columns).tolist()
    augmented = pd.concat([base, new_data[cols]], ignore_index=True)
    augmented = augmented.drop_duplicates(subset=["text", "profile"], keep="last").reset_index(drop=True)
    augmented.to_parquet(COMBINED, index=False)
    print(f"updated {COMBINED} ({len(augmented)} rows total)")

    # 6. Retrain (use the venv's python explicitly so we don't pick up system python)
    print(f"\nretraining ({epochs} epochs)...")
    t0 = time.time()
    venv_python = REPO / ".venv" / "bin" / "python"
    subprocess.run(
        [str(venv_python), "-m", "training.train", "--epochs", str(epochs), "--batch", "32"],
        check=True,
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    print(f"  trained in {time.time() - t0:.0f}s")

    # 7. Log metrics
    rounds_log.append(
        {
            "round": round_idx,
            "n_train": len(augmented),
            "metrics": summary,
            "n_failures_used": len(failures),
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--gen-per-round", type=int, default=200)
    p.add_argument("--rollouts-per-eval", type=int, default=400)
    p.add_argument("--epochs-per-round", type=int, default=2)
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rounds_log: list[dict] = []
    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CURRICULUM_DIR / "curriculum_log.json"

    for r in range(1, args.rounds + 1):
        try:
            run_round(
                round_idx=r,
                rounds_log=rounds_log,
                n_gen=args.gen_per_round,
                n_rollouts=args.rollouts_per_eval,
                epochs=args.epochs_per_round,
                port=args.port,
            )
        except Exception as e:
            print(f"\n!!! round {r} FAILED: {type(e).__name__}: {e}")
            print("    skipping to next round; partial log preserved.")
            rounds_log.append({"round": r, "error": f"{type(e).__name__}: {e}"})
        # checkpoint after every round so a later failure preserves earlier results
        log_path.write_text(json.dumps(rounds_log, indent=2, default=str))
        print(f"  log checkpoint -> {log_path} ({len(rounds_log)} rounds recorded)")

    print(f"\ndone. log: {log_path}")


if __name__ == "__main__":
    main()
