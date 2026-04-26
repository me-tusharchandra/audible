"""
In-process smoke test for the Audible environment.

Verifies the full reset/step contract against the new GateAction /
GateObservation schema and the composite rubric, without any HTTP layer.
Hand-checks reward signs for the four canonical cases:

  1. correct ACT + correct tool        -> +2.0
  2. correct IGNORE                    -> +1.5
  3. wrong tool when ACT is correct    -> +1.5  (decision right, tool wrong)
  4. false wake (ACT when GT=IGNORE)   -> -1.0  (decision wrong + false-wake hit)
"""

from models import PROFILE_NAMES, GateAction
from server.audible_env_environment import AudibleEnvironment
from server.scenarios import SCENARIOS


def _scenario_by_id(sid: int) -> dict:
    for s in SCENARIOS:
        if s["id"] == sid:
            return s
    raise KeyError(sid)


def _force_scenario(env: AudibleEnvironment, sid: int, profile: str) -> None:
    """Pin the env to a specific (scenario, profile) — bypasses random sampling."""
    env._current = (_scenario_by_id(sid), profile)


def case(label: str, env: AudibleEnvironment, sid: int, profile: str,
         action: GateAction, expected: float) -> None:
    env.reset()
    _force_scenario(env, sid, profile)
    obs = env.step(action)
    assert obs.reward == expected, (
        f"{label}: expected reward={expected}, got {obs.reward} "
        f"(components={obs.component_scores})"
    )
    print(f"  {label:55s} reward={obs.reward:+.2f}  components={obs.component_scores}")


def main() -> None:
    env = AudibleEnvironment()

    # ---- general sanity: reset returns a valid observation ----
    obs = env.reset(seed=0)
    assert obs.user_profile in PROFILE_NAMES
    assert len(obs.available_tools) == 5
    assert obs.done is False
    print(f"reset OK -> utterance={obs.utterance!r} profile={obs.user_profile}")

    # ---- four canonical reward cases ----
    print("\nrubric check:")

    # 1. Direct command (id=1: 'Set a timer for 10 minutes'), correct ACT+tool
    case(
        "correct ACT + correct tool",
        env, sid=1, profile="minimalist",
        action=GateAction(decision="ACT", tool="set_timer"),
        expected=2.0,
    )

    # 2. False-wake bait (id=9: 'Hold on a sec'), correctly ignored
    case(
        "correct IGNORE on false-wake bait",
        env, sid=9, profile="minimalist",
        action=GateAction(decision="IGNORE"),
        expected=1.5,
    )

    # 3. Right decision but wrong tool (id=1, ACT but pick play_music instead)
    case(
        "correct ACT, wrong tool",
        env, sid=1, profile="minimalist",
        action=GateAction(decision="ACT", tool="play_music"),
        expected=1.5,
    )

    # 4. False wake (id=9 ground truth IGNORE, agent predicts ACT)
    case(
        "false wake (ACT when GT=IGNORE)",
        env, sid=9, profile="minimalist",
        action=GateAction(decision="ACT", tool="set_timer"),
        expected=-1.0,
    )

    # 5. Profile divergence: id=6 ('Play some lo-fi') — work_focused IGNOREs
    case(
        "profile divergence: work_focused ignores music",
        env, sid=6, profile="work_focused",
        action=GateAction(decision="IGNORE"),
        expected=1.5,
    )
    case(
        "profile divergence: proactive plays music",
        env, sid=6, profile="proactive",
        action=GateAction(decision="ACT", tool="play_music"),
        expected=2.0,
    )

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
