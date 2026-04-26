"""
Seed scenarios for Phase 1 — hand-curated to cover the full spectrum of
gate decisions across the three profiles.

Each scenario carries per-profile ground-truth labels so the same utterance
can teach different gate behavior for different users — this is the whole
point of the personalization layer.

Phase 3 augments this list with LLM-generated adversarial scenarios.
"""

from typing import Any, Dict, List

SCENARIOS: List[Dict[str, Any]] = [
    # ---- Direct commands, all profiles ACT --------------------------------
    {
        "id": 1,
        "utterance": "Set a timer for 10 minutes",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "set_timer"},
            "proactive": {"decision": "ACT", "tool": "set_timer"},
            "work_focused": {"decision": "ACT", "tool": "set_timer"},
        },
    },
    {
        "id": 2,
        "utterance": "Add a meeting with Sarah tomorrow at 3pm",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "add_calendar_event"},
            "proactive": {"decision": "ACT", "tool": "add_calendar_event"},
            "work_focused": {"decision": "ACT", "tool": "add_calendar_event"},
        },
    },
    {
        "id": 3,
        "utterance": "What's the weather in Tokyo?",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "web_search"},
            "proactive": {"decision": "ACT", "tool": "web_search"},
            "work_focused": {"decision": "ACT", "tool": "web_search"},
        },
    },
    {
        "id": 4,
        "utterance": "How tall is Mount Everest?",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "web_search"},
            "proactive": {"decision": "ACT", "tool": "web_search"},
            "work_focused": {"decision": "ACT", "tool": "web_search"},
        },
    },
    {
        "id": 5,
        "utterance": "Block off Thursday afternoon for deep work",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "add_calendar_event"},
            "proactive": {"decision": "ACT", "tool": "add_calendar_event"},
            "work_focused": {"decision": "ACT", "tool": "add_calendar_event"},
        },
    },
    # ---- Direct commands, profile-divergent -------------------------------
    {
        "id": 6,
        "utterance": "Play some lo-fi",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "play_music"},
            "proactive": {"decision": "ACT", "tool": "play_music"},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    {
        "id": 7,
        "utterance": "Turn off the kitchen lights",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "smart_home_control"},
            "proactive": {"decision": "ACT", "tool": "smart_home_control"},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    {
        "id": 8,
        "utterance": "Can you put on some classical?",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "play_music"},
            "proactive": {"decision": "ACT", "tool": "play_music"},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    # ---- Ambient — confusable, all profiles IGNORE (false-wake bait) -----
    {
        "id": 9,
        "utterance": "Hold on a sec, I need to grab something",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "IGNORE", "tool": None},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    {
        "id": 10,
        "utterance": "I love this song that's playing",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "IGNORE", "tool": None},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    {
        "id": 11,
        "utterance": "Did you set a timer for the cookies?",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "IGNORE", "tool": None},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    {
        "id": 12,
        "utterance": "Don't bother setting any timer for this",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "IGNORE", "tool": None},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    # ---- Ambient — divergent (proactive ACTs on indirect cues) ----------
    {
        "id": 13,
        "utterance": "I wonder what the weather's like in Paris",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "ACT", "tool": "web_search"},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    {
        "id": 14,
        "utterance": "It's a bit bright in here",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "ACT", "tool": "smart_home_control"},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    # ---- Multi-speaker chatter (all IGNORE) -----------------------------
    {
        "id": 15,
        "utterance": "Did you finish that report? Yeah, sent it last night",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "IGNORE", "tool": None},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    {
        "id": 16,
        "utterance": "Why does the AC always break in summer?",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "IGNORE", "tool": None},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    # ---- UPDATE_CONTEXT — notable info worth remembering, no action -----
    {
        "id": 17,
        "utterance": "Sarah and I might catch up tomorrow",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "UPDATE_CONTEXT", "tool": None},
            "proactive": {"decision": "UPDATE_CONTEXT", "tool": None},
            "work_focused": {"decision": "UPDATE_CONTEXT", "tool": None},
        },
    },
    {
        "id": 18,
        "utterance": "My flight on Friday got delayed by two hours",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "UPDATE_CONTEXT", "tool": None},
            "proactive": {"decision": "UPDATE_CONTEXT", "tool": None},
            "work_focused": {"decision": "UPDATE_CONTEXT", "tool": None},
        },
    },
    # ---- Context-dependent: implicit reference ---------------------------
    {
        "id": 19,
        "utterance": "Yeah, that one",
        "context_history": ["What music should I put on?"],
        "labels": {
            "minimalist": {"decision": "IGNORE", "tool": None},
            "proactive": {"decision": "ACT", "tool": "play_music"},
            "work_focused": {"decision": "IGNORE", "tool": None},
        },
    },
    # ---- Explicit imperative with reminder semantics --------------------
    {
        "id": 20,
        "utterance": "Remind me to call mom at 6",
        "context_history": [],
        "labels": {
            "minimalist": {"decision": "ACT", "tool": "set_timer"},
            "proactive": {"decision": "ACT", "tool": "set_timer"},
            "work_focused": {"decision": "ACT", "tool": "set_timer"},
        },
    },
]
