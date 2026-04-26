---
title: Audible — Ambient-Listening Gating Environment
emoji: 🎛️
colorFrom: green
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - gating-classifier
  - mobilebert
  - self-improvement
  - hackathon
---

# Audible — Ambient-Listening Gating Environment

OpenEnv environment for training an always-on voice-assistant **gating classifier**.
Given a single ambient utterance plus the active user's preference profile, decide:

- `ACT(<tool>)` — fire one of 5 tools (`set_timer`, `add_calendar_event`, `play_music`, `web_search`, `smart_home_control`)
- `UPDATE_CONTEXT` — silently note this for later
- `IGNORE` — do nothing

Built for the **Meta OpenEnv Hackathon 2026, Theme #4 — Self-Improvement**.

## Why this is interesting

Wake-word ("Hey Siri") gating is a solved problem. **Always-on listening isn't.** The hard cases are ambient utterances where a tool keyword appears but the speaker isn't actually addressing the assistant:

- *"Hold on a sec, grabbing my keys"* — `set_timer`-shaped, but it's not a command
- *"Did you set the timer for the cookies?"* — addressing another person
- *"I wonder what the weather's like in Paris"* — proactive should ACT, others shouldn't
- *"It's a bit bright in here"* — proactive should turn down lights, work-focused should ignore

A naive keyword classifier wakes spuriously on all of these. A binary "actionable / non-actionable" classifier (the standard baseline) misses the **per-user personalization** layer entirely. This environment trains a small, edge-deployable model (mobileBERT, ~25M params) to handle both, with three preference profiles personalizing the gate's behavior.

## Action / observation contract

```python
GateObservation       # what the agent sees
    utterance: str
    context_history: list[str]
    user_profile: "minimalist" | "proactive" | "work_focused"
    available_tools: list[{name, description}]   # 5 tools

GateAction            # what the agent emits
    decision: "ACT" | "UPDATE_CONTEXT" | "IGNORE"
    tool: optional ToolName, only when decision == "ACT"
```

### Profiles

| Profile | Behavior |
|---|---|
| `minimalist` | Acts only on direct first-person imperative commands clearly addressed to the assistant. Anything ambient → IGNORE. |
| `proactive` | Acts on direct commands AND indirect cues ("I wonder…", "it's freezing") — but not omniscient; vague cues still IGNORE. |
| `work_focused` | Acts on `set_timer` / `add_calendar_event` / `web_search`. Never plays music or controls smart home, even when explicitly asked. |

## Reward (composite rubric)

Four components, combined non-uniformly so **false wakes are the most painful error** (because spurious activation is the worst UX failure of always-on listening):

| Component | Weight | Range | Fires when |
|---|---|---|---|
| `gate_correctness` | +1.0 | {0, 1} | decision matches ground truth |
| `tool_correctness` | +0.5 | {0, 1} | both decisions ACT and tool matches |
| `profile_alignment` | +0.5 | {0, 1} | action honors profile preference |
| `false_wake_penalty` | +1.0 | {-1, 0} | predicted ACT but ground truth = IGNORE/UPDATE |

Reward range: `[-1.0, +2.0]`. Per-component scores propagate in `observation.component_scores` so training can plot accuracy on each axis separately.

## Episode structure

Single step. `reset()` samples one (scenario, profile) pair from the held-out seed scenario set; `step(action)` scores the agent's classification with the composite rubric and returns `done=True` with the reward attached.

## Quick start

```python
from audible_env import AudibleEnv, GateAction

with AudibleEnv(base_url="https://me-tusharchandra-audible-env.hf.space").sync() as env:
    obs = env.reset().observation
    print(obs.utterance, obs.user_profile)        # one ambient utterance
    action = GateAction(decision="ACT", tool="set_timer")
    result = env.step(action)
    print(result.reward, result.observation.ground_truth)
```

Or pull the Docker image and run locally:

```bash
docker pull registry.hf.space/me-tusharchandra-audible-env:latest
docker run -p 8000:8000 registry.hf.space/me-tusharchandra-audible-env:latest
```

## Repo layout

| File | Purpose |
|---|---|
| `models.py` | Typed `GateAction` / `GateObservation` + `TOOL_PALETTE` + `PROFILE_DESCRIPTIONS` |
| `client.py` | Typed WS/HTTP client subclassing `EnvClient` |
| `server/audible_env_environment.py` | `Environment` subclass — single-step episodes |
| `server/rubric.py` | Four-component composite `Rubric` |
| `server/scenarios.py` | Seed scenarios with per-profile labels |
| `server/app.py` | FastAPI app via `openenv.core.create_app` |

## Submission

Training pipeline, curriculum loop (Theme #4 self-improvement), results, plots, and demo all live in the parent repository. See the top-level `README.md` for the full submission with metrics and demo links.
