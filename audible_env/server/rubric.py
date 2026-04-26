"""
Composite rubric for the gating classifier.

Each component is a separate Rubric so per-component scores are introspectable
via `env.rubric.named_children()` — useful for plotting precision/recall on each
axis separately and for spotting which component the model is gaming.

Final reward range: roughly [-1.0, +2.0]. False wake is the most painful error
because spurious activations are the worst UX failure of always-on listening.
"""

from typing import Any

from openenv.core.rubrics import Rubric


def _ground_truth(observation: Any) -> dict:
    """Extract the per-profile ground-truth label set by the env on post-step
    observations. Returns an empty dict if absent (e.g., on reset observations)."""
    return getattr(observation, "ground_truth", None) or {}


class GateCorrectnessRubric(Rubric):
    """1.0 if action.decision matches ground-truth decision, else 0.0."""

    def forward(self, action: Any, observation: Any) -> float:
        gt = _ground_truth(observation)
        return 1.0 if action.decision == gt.get("decision") else 0.0


class ToolCorrectnessRubric(Rubric):
    """1.0 if the chosen tool matches ground truth (only when both decisions are ACT).

    Returns 0 when either side is non-ACT — tool selection is irrelevant unless
    the agent decided to actually act, AND the ground truth says it should.
    """

    def forward(self, action: Any, observation: Any) -> float:
        gt = _ground_truth(observation)
        if action.decision != "ACT" or gt.get("decision") != "ACT":
            return 0.0
        return 1.0 if action.tool == gt.get("tool") else 0.0


class FalseWakePenaltyRubric(Rubric):
    """-1.0 if the agent predicted ACT but the ground truth was IGNORE/UPDATE_CONTEXT.

    Worst UX failure of an always-on listener: speaking up when nobody asked.
    Multiplied by a large weight in the composite to dominate other signals.
    """

    def forward(self, action: Any, observation: Any) -> float:
        gt = _ground_truth(observation)
        if action.decision == "ACT" and gt.get("decision") in ("IGNORE", "UPDATE_CONTEXT"):
            return -1.0
        return 0.0


class ProfileAlignmentRubric(Rubric):
    """1.0 if the action honors the active profile's preference.

    For Phase-1 seed data this is equivalent to gate_correctness because we
    encode per-profile labels directly. Once Phase-3 brings in LLM-generated
    profile-agnostic data, this rubric will use the profile description to
    decide whether the chosen action matches the profile's intent — at which
    point its score will diverge from gate_correctness.
    """

    def forward(self, action: Any, observation: Any) -> float:
        gt = _ground_truth(observation)
        return 1.0 if action.decision == gt.get("decision") else 0.0


class GateRubric(Rubric):
    """Weighted composite. Children auto-registered via attribute assignment.

    Component weights:
      +1.0 * gate_correctness   — base reward for the right decision
      +0.5 * tool_correctness   — secondary reward for the right tool
      +0.5 * profile_alignment  — secondary reward for honoring the profile
      +1.0 * false_wake         — multiplier on a -1.0 penalty (so net -1.0)

    Reward range: [-1.0, +2.0]. WeightedSum is not used because its weights
    must sum to 1.0, which would force normalization and dilute the false-wake
    penalty's bite.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate_correctness = GateCorrectnessRubric()
        self.tool_correctness = ToolCorrectnessRubric()
        self.profile_alignment = ProfileAlignmentRubric()
        self.false_wake = FalseWakePenaltyRubric()

    def forward(self, action: Any, observation: Any) -> float:
        return (
            1.0 * self.gate_correctness(action, observation)
            + 0.5 * self.tool_correctness(action, observation)
            + 0.5 * self.profile_alignment(action, observation)
            + 1.0 * self.false_wake(action, observation)
        )
