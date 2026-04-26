"""
Audible: data models for the ambient-listening gating environment.

Shared by client and server. Defines the typed Action/Observation contract
plus the fixed tool palette and profile names so the agent always sees the
same option set in observations.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

ToolName = Literal[
    "set_timer",
    "add_calendar_event",
    "play_music",
    "web_search",
    "smart_home_control",
]

Decision = Literal["ACT", "UPDATE_CONTEXT", "IGNORE"]

ProfileName = Literal["minimalist", "proactive", "work_focused"]

TOOL_PALETTE: List[Dict[str, str]] = [
    {"name": "set_timer", "description": "Start a countdown timer for the user."},
    {"name": "add_calendar_event", "description": "Create a calendar event."},
    {"name": "play_music", "description": "Play music — playlist, song, or genre."},
    {"name": "web_search", "description": "Look up a fact on the web."},
    {"name": "smart_home_control", "description": "Control smart-home devices (lights, thermostat, etc.)."},
]

PROFILE_NAMES: List[ProfileName] = ["minimalist", "proactive", "work_focused"]

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "minimalist": (
        "Acts only on explicit, first-person imperative commands directed at the assistant. "
        "Anything ambiguous or indirect should be ignored."
    ),
    "proactive": (
        "Acts on indirect cues too — 'I wonder…', 'it's a bit bright in here'. "
        "Errs on the side of being helpful."
    ),
    "work_focused": (
        "Acts on timers, calendar, and web search. Never plays music or controls "
        "smart-home devices, even when explicitly asked."
    ),
}


class GateAction(Action):
    """Agent's classification of an ambient utterance.

    decision: ACT (call a tool), UPDATE_CONTEXT (note this for later), or IGNORE.
    tool:     which tool to invoke — only meaningful when decision == "ACT".
    """

    decision: Decision = Field(..., description="ACT | UPDATE_CONTEXT | IGNORE")
    tool: Optional[ToolName] = Field(default=None, description="Tool name when decision==ACT")


class GateObservation(Observation):
    """One ambient utterance to classify, with the active user's profile.

    The agent's *input* features are: utterance, context_history, user_profile,
    available_tools. The remaining fields are post-step diagnostics — `None`
    on observations returned by reset(), populated by step() so the trainer
    can compute reward, log per-component scores, and evaluate held-out sets.

    These cannot live in `metadata` because OpenEnv's serializer strips the
    metadata field from the wire payload (see core.serialize_observation).
    """

    utterance: str = Field(..., description="The utterance to classify")
    context_history: List[str] = Field(
        default_factory=list,
        description="Up to N prior conversational turns that frame this utterance",
    )
    user_profile: ProfileName = Field(..., description="Active user preference profile")
    available_tools: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Tools the agent may select from when decision==ACT",
    )

    # ---- post-step diagnostics (None on reset, populated on step) ----
    scenario_id: Optional[int] = Field(
        default=None, description="Identifier of the underlying scenario"
    )
    ground_truth: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-profile ground-truth label: {'decision': ..., 'tool': ...}",
    )
    component_scores: Optional[Dict[str, float]] = Field(
        default=None, description="Per-rubric-component scores for introspection"
    )
