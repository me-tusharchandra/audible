"""
Smoke test: instantiate the environment directly (no HTTP), verify reset/step
contract works against the default echo logic. If this passes, the OpenEnv
scaffold is wired up correctly and we can move on to designing the real
GateAction / GateObservation / rubric.
"""

from models import AudibleAction
from server.audible_env_environment import AudibleEnvironment


def main() -> None:
    env = AudibleEnvironment()

    obs = env.reset()
    assert obs.echoed_message == "Audible Env environment ready!", obs
    assert obs.done is False
    print(f"reset OK     -> echoed={obs.echoed_message!r} reward={obs.reward}")

    obs = env.step(AudibleAction(message="hello world"))
    assert obs.echoed_message == "hello world"
    assert obs.message_length == 11
    assert obs.done is False
    print(f"step  OK     -> echoed={obs.echoed_message!r} reward={obs.reward}")

    state = env.state
    assert state.step_count == 1
    print(f"state OK     -> episode_id={state.episode_id} step_count={state.step_count}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
