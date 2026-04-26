"""Audible environment client — typed wrapper over the OpenEnv HTTP/WS server."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import GateAction, GateObservation


class AudibleEnv(EnvClient[GateAction, GateObservation, State]):
    """
    Client for the Audible ambient-listening gating environment.

    Example:
        >>> with AudibleEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     print(obs.utterance, obs.user_profile)
        ...
        ...     action = GateAction(decision="ACT", tool="set_timer")
        ...     result = client.step(action)
        ...     print(result.reward, result.observation.metadata["ground_truth"])
    """

    def _step_payload(self, action: GateAction) -> Dict[str, Any]:
        return {
            "decision": action.decision,
            "tool": action.tool,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GateObservation]:
        obs_data = payload.get("observation", {})
        observation = GateObservation(
            utterance=obs_data.get("utterance", ""),
            context_history=list(obs_data.get("context_history", [])),
            user_profile=obs_data.get("user_profile", "minimalist"),
            available_tools=list(obs_data.get("available_tools", [])),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            scenario_id=obs_data.get("scenario_id"),
            ground_truth=obs_data.get("ground_truth"),
            component_scores=obs_data.get("component_scores"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
