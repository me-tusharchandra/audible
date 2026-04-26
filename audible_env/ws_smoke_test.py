"""WebSocket smoke test against a running local server.

Uses the typed AudibleEnv client (the same one TRL training will use) to
do a real reset/step round-trip over the WS endpoint. Validates the wire
contract end-to-end including metadata propagation.
"""

import json
from typing import Any, Dict

from audible_env.client import AudibleEnv
from audible_env.models import GateAction


class _DebugAudibleEnv(AudibleEnv):
    """Subclass that prints the raw step payload for one-shot debugging."""

    def _parse_result(self, payload: Dict[str, Any]):
        print("=== raw step payload ===")
        print(json.dumps(payload, indent=2))
        print("========================")
        return super()._parse_result(payload)


def main() -> None:
    with _DebugAudibleEnv(base_url="http://127.0.0.1:8765").sync() as client:
        result = client.reset()
        obs = result.observation
        print(f"reset OK  -> utterance={obs.utterance!r} profile={obs.user_profile}")

        action = GateAction(decision="ACT", tool="set_timer")
        result = client.step(action)
        obs = result.observation

        print(f"step  OK  -> reward={result.reward:+.2f} done={result.done}")
        print(f"            ground_truth={obs.ground_truth}")
        print(f"            components  ={obs.component_scores}")

    print("\nWS smoke test passed.")


if __name__ == "__main__":
    main()
