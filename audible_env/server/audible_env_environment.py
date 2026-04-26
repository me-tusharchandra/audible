"""
Ambient-listening gating environment.

Episode = single step. reset() samples a (scenario, profile) pair and returns
the utterance + context + tool palette. step() takes the agent's GateAction,
scores it via the composite GateRubric, and returns done=True with the reward
attached and ground truth + per-component scores in metadata for eval.
"""

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        PROFILE_NAMES,
        TOOL_PALETTE,
        GateAction,
        GateObservation,
    )
    from .rubric import GateRubric
    from .scenarios import SCENARIOS
except ImportError:  # local script execution
    from models import PROFILE_NAMES, TOOL_PALETTE, GateAction, GateObservation
    from server.rubric import GateRubric
    from server.scenarios import SCENARIOS


class AudibleEnvironment(Environment):
    """Single-step gating environment.

    On reset, samples a scenario + profile and exposes the utterance to the
    agent. On step, scores the agent's classification against the per-profile
    ground truth via the composite rubric and ends the episode.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__(rubric=GateRubric())
        self._state: State = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()
        self._current: Optional[tuple[dict, str]] = None  # (scenario, profile)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GateObservation:
        if seed is not None:
            self._rng.seed(seed)

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        scenario = self._rng.choice(SCENARIOS)
        profile = self._rng.choice(PROFILE_NAMES)
        self._current = (scenario, profile)

        return GateObservation(
            utterance=scenario["utterance"],
            context_history=list(scenario.get("context_history", [])),
            user_profile=profile,
            available_tools=list(TOOL_PALETTE),
            done=False,
            reward=None,
            scenario_id=scenario["id"],
        )

    def step(
        self,
        action: GateAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GateObservation:
        if self._current is None:
            raise RuntimeError("step() called before reset()")

        self._state.step_count += 1
        scenario, profile = self._current
        gt = scenario["labels"][profile]

        post_obs = GateObservation(
            utterance=scenario["utterance"],
            context_history=list(scenario.get("context_history", [])),
            user_profile=profile,
            available_tools=list(TOOL_PALETTE),
            done=True,
            reward=None,
            scenario_id=scenario["id"],
            ground_truth=gt,
        )

        assert self.rubric is not None  # set in __init__
        reward = float(self.rubric(action, post_obs))
        post_obs.reward = reward
        post_obs.component_scores = {
            name: child.last_score for name, child in self.rubric.named_children()
        }

        return post_obs

    @property
    def state(self) -> State:
        return self._state
