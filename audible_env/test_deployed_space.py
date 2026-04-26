"""
Smoke test against the deployed HF Space — fires once the Space is up.
Verifies the wire contract end-to-end: WebSocket connection, reset(), step(),
ground-truth + component scores propagate through serialization.

Run AFTER `bash training/deploy.sh` and the Space has finished BUILDING.
"""

from __future__ import annotations

from audible_env.client import AudibleEnv
from audible_env.models import GateAction

DEPLOYED_URL = "https://me-tusharchandra-audible-env.hf.space"


def main() -> None:
    print(f"connecting to {DEPLOYED_URL}...")
    with AudibleEnv(base_url=DEPLOYED_URL).sync() as env:
        result = env.reset()
        obs = result.observation
        print(f"reset OK   -> utterance={obs.utterance!r}  profile={obs.user_profile}")
        print(f"            available_tools={[t['name'] for t in obs.available_tools]}")

        # Make a wrong-on-purpose action so we can confirm the rubric punishes false-wakes
        # in the deployed env exactly the way it does locally.
        action = GateAction(decision="ACT", tool="set_timer")
        result = env.step(action)
        post = result.observation
        print(f"\nstep OK    -> reward={result.reward:+.2f}  done={result.done}")
        print(f"            ground_truth={post.ground_truth}")
        print(f"            component_scores={post.component_scores}")

        st = env.state()
        print(f"\nstate OK   -> episode_id={st.episode_id}  step_count={st.step_count}")

    print("\nDeployed-space smoke test passed.")


if __name__ == "__main__":
    main()
