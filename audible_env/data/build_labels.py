"""
Map friend's binary actionable/non-actionable dataset into our 7-class action
space, cross-producted with the three user profiles.

Rules are intentionally simple keyword heuristics — noisy at scale beats clean
at small scale for hackathon iteration. Phase 3 will refine with LLM-judge
relabeling on the cases the trained gate keeps getting wrong.

Output classes (7):
    IGNORE
    UPDATE_CONTEXT
    ACT_set_timer
    ACT_add_calendar_event
    ACT_play_music
    ACT_web_search
    ACT_smart_home_control

Profiles:
    minimalist     — uses heuristic labels as-is.
    proactive      — same as minimalist on this dataset; ambient-cue divergence
                     is captured in scenarios.py, not in friend's binary data.
    work_focused   — collapses ACT_play_music and ACT_smart_home_control
                     to IGNORE (this profile silences entertainment + home).

Run: python audible_env/data/build_labels.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent
EXTERNAL = DATA_DIR / "external"
OUT_PATH = DATA_DIR / "labeled.parquet"

CLASS_NAMES = [
    "IGNORE",
    "UPDATE_CONTEXT",
    "ACT_set_timer",
    "ACT_add_calendar_event",
    "ACT_play_music",
    "ACT_web_search",
    "ACT_smart_home_control",
]

# Keyword patterns — order matters (more specific first).
TIMER_RE = re.compile(
    r"\b(set\s+(?:a\s+)?(?:timer|alarm|reminder)|remind\s+me|"
    r"wake\s+me|countdown|alarm\s+for)\b",
    re.I,
)
CALENDAR_RE = re.compile(
    r"\b(schedule|meeting|calendar|appointment|book\s+(?:a|the)\s+\w+|"
    r"add\s+(?:a|an)?\s*(?:event|meeting)|block\s+off|reserve|cancel\s+(?:my|the)\s+meeting)\b",
    re.I,
)
MUSIC_RE = re.compile(
    r"\b(play|put\s+on|queue|shuffle)\b.*\b(music|song|playlist|album|track|"
    r"lo[- ]fi|jazz|classical|pop|rock|spotify)\b|"
    r"\bplay\s+(?:some\s+)?\w+",
    re.I,
)
SMART_HOME_RE = re.compile(
    r"\b(turn\s+(?:on|off|up|down)|dim|brighten|set\s+(?:the\s+)?(?:lights?|"
    r"thermostat|temperature|ac|heat))\b|"
    r"\b(lights?|thermostat|temperature|fan|blinds|curtains|door|lock)\b.*"
    r"\b(on|off|up|down|warm|cool|open|close|lock|unlock)\b",
    re.I,
)
SEARCH_RE = re.compile(
    r"^\s*(what|who|when|where|why|how|which|is|are|do|does|can|could|"
    r"will|would|should|tell\s+me|find|search|look\s+up|google)\b",
    re.I,
)
# Update-context candidates: declarative facts about future plans, schedule changes,
# or notable personal information that an assistant should remember but not act on.
UPDATE_CTX_RE = re.compile(
    r"\b(my|i)\s+(?:flight|train|plane|appointment|interview|trip|meeting|call)\b.*"
    r"\b(delayed|cancell?ed|moved|reschedul?ed|tomorrow|next\s+week|on\s+\w+day)\b|"
    r"\b(might|may|maybe|probably|thinking\s+of|planning\s+to)\b",
    re.I,
)


def heuristic_class(text: str, binary_label: int) -> str:
    """Map (text, binary 0/1) to one of the 7 classes via keyword rules."""
    if binary_label == 0:
        if UPDATE_CTX_RE.search(text):
            return "UPDATE_CONTEXT"
        return "IGNORE"

    # binary_label == 1 — actionable
    if TIMER_RE.search(text):
        return "ACT_set_timer"
    if CALENDAR_RE.search(text):
        return "ACT_add_calendar_event"
    if MUSIC_RE.search(text):
        return "ACT_play_music"
    if SMART_HOME_RE.search(text):
        return "ACT_smart_home_control"
    if SEARCH_RE.search(text):
        return "ACT_web_search"
    # Fallback — most queries that don't match a tool become web_search.
    return "ACT_web_search"


def apply_profile(klass: str, profile: str) -> str:
    """Adjust the per-utterance class according to user profile preferences."""
    if profile == "work_focused" and klass in ("ACT_play_music", "ACT_smart_home_control"):
        return "IGNORE"
    return klass


def class_to_action(klass: str) -> tuple[str, Optional[str]]:
    """Decompose a class label into (decision, tool) for the env's GateAction."""
    if klass == "IGNORE":
        return "IGNORE", None
    if klass == "UPDATE_CONTEXT":
        return "UPDATE_CONTEXT", None
    assert klass.startswith("ACT_"), klass
    return "ACT", klass[len("ACT_") :]


def main() -> None:
    frames = []
    for csv in ("final_dataset.csv", "second_finetune_data.csv"):
        df = pd.read_csv(EXTERNAL / csv)
        df["source"] = csv
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"raw: {len(combined)} rows")

    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"after dedup: {len(combined)} rows")

    combined["heuristic_class"] = [
        heuristic_class(t, l) for t, l in zip(combined["text"], combined["label"])
    ]
    print("\nheuristic class distribution (profile-agnostic):")
    for k, v in Counter(combined["heuristic_class"]).most_common():
        print(f"  {k:30s} {v:5d}  ({v / len(combined):.1%})")

    rows = []
    for _, r in combined.iterrows():
        for profile in ("minimalist", "proactive", "work_focused"):
            klass = apply_profile(r["heuristic_class"], profile)
            decision, tool = class_to_action(klass)
            rows.append(
                {
                    "text": r["text"],
                    "profile": profile,
                    "binary_label": int(r["label"]),
                    "class_label": klass,
                    "class_id": CLASS_NAMES.index(klass),
                    "decision": decision,
                    "tool": tool,
                    "source": r["source"],
                }
            )
    out = pd.DataFrame(rows)
    print(f"\nfinal: {len(out)} rows ({len(combined)} utterances × 3 profiles)")
    print("\nclass distribution per profile:")
    pivot = out.pivot_table(
        index="class_label", columns="profile", values="text", aggfunc="count", fill_value=0
    )
    pivot = pivot.reindex(CLASS_NAMES)
    print(pivot.to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
