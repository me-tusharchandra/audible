"""
Generate synthetic ambient-listening scenarios with OpenAI structured outputs.

The friend's binary dataset has ~6.6k utterances but is missing the very thing
that makes our problem interesting: ambient/ambiguous cases where a tool
keyword appears but the user isn't actually addressing the assistant. We
generate those cases here, balanced across categories and profiles.

Per-category generation lets us hit the rare-but-important edge cases at
controlled volume instead of hoping the LLM rolls them organically.

Run iteratively from a notebook or:
    python -m training.synthetic_data --category ambient_confusable --n 10  # peek
    python -m training.synthetic_data --full                                # full run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = REPO_ROOT / "audible_env" / "data" / "synthetic"

MODEL = "gpt-4o-mini"
PROFILES = ["minimalist", "proactive", "work_focused"]

# Categories — quotas chosen to overweight the things friend's data lacks.
CATEGORIES_AND_QUOTA = {
    "ambient_confusable": 200,        # contains a tool keyword but should NOT trigger
    "indirect_address": 150,          # proactive should ACT, others IGNORE
    "multi_speaker_chatter": 120,     # two people talking, not the assistant
    "rhetorical_question": 80,        # question shape but no answer wanted
    "update_worthy": 80,              # notable info, no action right now
    "direct_command_balanced": 200,   # tool-balanced: 40 per tool to fix imbalance
}

ToolName = Literal[
    "set_timer", "add_calendar_event", "play_music", "web_search", "smart_home_control"
]
Decision = Literal["ACT", "UPDATE_CONTEXT", "IGNORE"]


class ProfileLabel(BaseModel):
    decision: Decision
    tool: Optional[ToolName] = None


class Scenario(BaseModel):
    utterance: str = Field(
        ...,
        description=(
            "A single natural-sounding utterance (10-25 words) as if transcribed "
            "from ambient audio. Conversational, lowercase OK, contractions OK."
        ),
    )
    why_interesting: str = Field(
        ...,
        description="One sentence on what makes this case ambiguous or instructive.",
    )
    minimalist: ProfileLabel = Field(..., description="Label for the minimalist profile")
    proactive: ProfileLabel = Field(..., description="Label for the proactive profile")
    work_focused: ProfileLabel = Field(..., description="Label for the work_focused profile")


class GeneratedBatch(BaseModel):
    scenarios: List[Scenario]


SYSTEM_PROMPT = """\
You generate training data for an always-on voice assistant gating classifier.
The assistant is listening to ambient audio. Each "utterance" you produce is
something the microphone might transcribe — not a perfectly composed sentence,
but the kind of casual, mid-conversation speech a real person produces.

For each utterance, the gate decides one of:
  ACT(<tool>)         — fire one of 5 tools: set_timer, add_calendar_event,
                        play_music, web_search, smart_home_control
  UPDATE_CONTEXT      — silently note this for later (notable but no action now)
  IGNORE              — do nothing

## Tool semantics — be precise

  set_timer           — start a countdown / reminder / alarm at a future time
  add_calendar_event  — CREATE a new calendar event. Reading existing calendar
                        is NOT a tool we have, so calendar LOOKUPS go to
                        web_search ("when is my next meeting" → web_search).
  play_music          — actively start audio playback
  web_search          — look up a fact, person, place, definition, or any
                        answerable question
  smart_home_control  — change the state of a physical device

## User profiles

The SAME utterance can warrant different labels under different profiles —
this divergence is the most valuable signal in your output:

  minimalist:    Acts ONLY on direct first-person imperatives clearly addressed
                 to the assistant ("set a timer", "what's the weather"). Anything
                 indirect, third-person, or ambient -> IGNORE.

  proactive:     Acts on direct commands AND on indirect cues that imply a
                 useful action ("I wonder what time it is" -> web_search,
                 "it's freezing in here" -> smart_home_control). BUT proactive
                 is not omniscient — if the cue is too vague (e.g., "this is
                 frustrating"), proactive should still IGNORE. Aim for ~50%
                 of indirect-cue cases to ACT, ~50% to IGNORE — calibration
                 matters.

  work_focused:  Acts on set_timer / add_calendar_event / web_search. NEVER acts
                 on play_music or smart_home_control even when explicitly asked.

## Style — natural ambient speech

GOOD examples:
- "set a 10 minute timer"
- "what's the weather like in tokyo"
- "ugh, this traffic is the worst"
- "did sam send the deck?"
- "we're moving standup to eleven"
- "hey can you put on some lo-fi"

BAD examples (do NOT generate these):
- "I would like to inquire about the weather forecast"        (too formal)
- "I should probably consider adjusting the thermostat"        (formulaic, every line starts "I should")
- "Hello assistant, please initiate a timer of ten minutes"   (no one talks like this)

Diversity rules:
- DO NOT start every utterance with "I". Vary openers — questions, exclamations,
  direct address to other people, sentence fragments, etc.
- Use contractions ("it's", "we're", "don't"). People don't speak in full forms.
- Mix lengths — sometimes 4 words, sometimes 15.
- Tense should vary. Not everything is "I should have / I forgot / I was thinking".

## What makes a great training example

- The decision is genuinely arguable. The interesting cases require you to
  pause and think.
- Profile divergence is organic — you can FEEL why proactive labels it
  differently from minimalist. Don't force divergence; report what's true.
- The "why_interesting" line explains the ambiguity in one sentence.
"""


CATEGORY_PROMPTS = {
    "ambient_confusable": """\
Generate {n} utterances where a TOOL KEYWORD appears but the speaker is NOT
asking the assistant to fire that tool. The whole point: a naive classifier
that just looks for keywords would wake spuriously.

Examples of the shape (don't copy, generate fresh):
- "Did you set the timer for the cookies?" (asking another person)
- "I love this song that's playing"        (commenting, not requesting music)
- "Hold on a sec, grabbing my keys"        ("sec" / "set"-like, not a real command)
- "Sarah's flight got cancelled"           (mentions calendar-ish thing, not a task)
- "It's so dark in this room"              (could be smart_home, but no command)

Most should be IGNORE under all 3 profiles (these are false-wake bait).
A FEW (~20%) should be UPDATE_CONTEXT (when the info is genuinely worth remembering).
A few may divergence (proactive ACTs on indirect cues like "it's so dark") —
that's GOOD; tag them with the appropriate tool for proactive.
""",
    "indirect_address": """\
Generate {n} utterances where the speaker is talking aloud (to themselves or
the room) about something the assistant could help with — but they're NOT
issuing a direct command. These are the canonical "proactive should act,
others should ignore" cases.

Examples of the shape:
- "I wonder what the high is in Boston tomorrow"  (web_search for proactive)
- "It's getting kind of stuffy in here"           (smart_home for proactive)
- "Now I really want to hear that one acoustic version"  (play_music for proactive)
- "Forgot when my dentist appointment was"        (calendar lookup for proactive)
- "Need to remember to call mom at 6"             (set_timer for proactive)

Pattern: proactive ACTs with the right tool, minimalist + work_focused IGNORE
(unless work_focused would still fire because the tool is in its allowed set —
e.g., set_timer / add_calendar_event / web_search remain allowed for work_focused).

Distribute across all 5 tool categories so we get balanced coverage.
""",
    "multi_speaker_chatter": """\
Generate {n} utterances representing two-person conversation snippets the
microphone overhears — neither speaker is addressing the assistant. The
utterance should sound like one or both sides of an interpersonal exchange.

Examples of the shape:
- "Did you finish the report?  Yeah, sent it last night"
- "Ugh, this traffic. We're gonna be late again."
- "He said maybe Thursday, but I'm not holding my breath"
- "What time did she say? Six?"

Almost all should be IGNORE under all 3 profiles. A few (~15%) might be
UPDATE_CONTEXT when the snippet contains schedule-relevant info worth
remembering ("dinner moved to seven").
""",
    "rhetorical_question": """\
Generate {n} utterances shaped like questions but where the speaker isn't
expecting an answer. These look like web_search bait but should not fire.

Examples of the shape:
- "Why is parking always so impossible downtown"
- "How am I supposed to get all this done by Friday"
- "Why does the AC always break in summer"

Mostly IGNORE under all 3 profiles. The proactive profile MIGHT act on a few
ambiguous ones where the rhetorical question is genuinely answerable — use
your judgment, but lean toward IGNORE since rhetorical=no-answer-wanted.
""",
    "update_worthy": """\
Generate {n} utterances containing notable personal information the assistant
should silently remember, but no action is needed right now. These train the
UPDATE_CONTEXT class which is currently underrepresented.

Examples of the shape:
- "My flight on Friday got delayed by two hours"
- "We're moving the team standup to 11 starting next week"
- "Mom's birthday is on the 23rd this year"
- "I'm starting a new job in October"

All 3 profiles should label these UPDATE_CONTEXT. Make the info specific and
varied (dates, names, schedule changes, status updates).
""",
    "direct_command_balanced": """\
Generate {n} clear, natural direct commands. CRITICAL: distribute roughly
EVENLY across all 5 tool types — the friend's dataset is heavily skewed to
web_search and we need balanced coverage of timer / calendar / music / home.

Approximately {per_tool} per tool:
  set_timer            ("set a 10 minute timer", "wake me up at 7")
  add_calendar_event   ("schedule a meeting with sam tomorrow at 2", "block off thursday afternoon")
  play_music           ("play some lo-fi", "put on the classical playlist")
  web_search           ("what's the weather in tokyo", "how tall is everest")
  smart_home_control   ("turn off the kitchen lights", "set the thermostat to 70")

Vary the phrasing — different verbs, different framings (questions, imperatives,
"can you" softeners). Most should be ACT under minimalist+proactive.
work_focused IGNOREs play_music and smart_home_control even when explicit.
""",
}


def get_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing — set it in .env")
    return OpenAI()


def generate_batch(
    client: OpenAI,
    category: str,
    n: int,
    temperature: float = 0.9,
) -> List[Scenario]:
    """One API call → up to `n` scenarios in `category`."""
    if category not in CATEGORY_PROMPTS:
        raise ValueError(f"unknown category: {category}")

    prompt = CATEGORY_PROMPTS[category].format(
        n=n, per_tool=max(n // 5, 2)
    )

    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=GeneratedBatch,
        temperature=temperature,
    )
    return response.choices[0].message.parsed.scenarios


def show_batch(scenarios: List[Scenario], category: str) -> None:
    """Pretty-print a batch for prompt iteration / inspection."""
    print(f"\n=== {category} ({len(scenarios)} scenarios) ===")
    for i, s in enumerate(scenarios, 1):
        print(f"\n[{i}] {s.utterance!r}")
        print(f"    why: {s.why_interesting}")
        for prof in PROFILES:
            label: ProfileLabel = getattr(s, prof)
            print(f"    {prof:13s} → {label.decision}" + (f" / {label.tool}" if label.tool else ""))


def to_dataframe(scenarios: List[Scenario], category: str) -> pd.DataFrame:
    """Flatten to (utterance × profile) rows, matching labeled.parquet's schema."""
    rows = []
    for s in scenarios:
        for profile in PROFILES:
            label: ProfileLabel = getattr(s, profile)
            klass = "IGNORE" if label.decision == "IGNORE" else (
                "UPDATE_CONTEXT" if label.decision == "UPDATE_CONTEXT"
                else f"ACT_{label.tool}"
            )
            rows.append(
                {
                    "text": s.utterance,
                    "why_interesting": s.why_interesting,
                    "category": category,
                    "profile": profile,
                    "class_label": klass,
                    "decision": label.decision,
                    "tool": label.tool,
                    "source": "synthetic_v1",
                }
            )
    return pd.DataFrame(rows)


def generate_full_dataset(
    quotas: dict[str, int] = CATEGORIES_AND_QUOTA,
    batch_size: int = 10,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Generate the full quota across all categories with parallel API calls.

    Each task generates `batch_size` scenarios; we issue ceil(quota / batch_size)
    tasks per category and run up to `max_workers` in parallel.
    """
    client = get_client()
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, int]] = []
    for cat, quota in quotas.items():
        n_calls = (quota + batch_size - 1) // batch_size
        for _ in range(n_calls):
            tasks.append((cat, batch_size))
    print(f"Issuing {len(tasks)} parallel batches across {len(quotas)} categories...")

    all_dfs: list[pd.DataFrame] = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(generate_batch, client, cat, n): cat for cat, n in tasks}
        done = 0
        for fut in as_completed(futures):
            cat = futures[fut]
            try:
                scenarios = fut.result()
                all_dfs.append(to_dataframe(scenarios, cat))
                done += 1
                print(f"  [{done:3d}/{len(tasks)}] {cat:25s} +{len(scenarios):2d} ({time.time() - start:.0f}s)")
            except Exception as e:
                print(f"  FAILED {cat}: {e}")

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default=None,
                   help="Generate only this category (peek mode)")
    p.add_argument("--n", type=int, default=10, help="Scenarios per batch (peek mode)")
    p.add_argument("--full", action="store_true", help="Run full quotas")
    p.add_argument("--out", default=None, help="Output parquet path (full mode)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = get_client()

    if args.category:
        scenarios = generate_batch(client, args.category, args.n)
        show_batch(scenarios, args.category)
        return

    if args.full:
        df = generate_full_dataset()
        out = Path(args.out) if args.out else SYNTH_DIR / "synthetic_v1.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"\n=== summary ===")
        print(f"rows: {len(df)} ({len(df) // 3} unique utterances × 3 profiles)")
        print(f"\nclass distribution:")
        print(df["class_label"].value_counts().to_string())
        print(f"\nper-category counts (utterances):")
        print((df.groupby("category")["text"].nunique()).to_string())
        print(f"\nwrote {out}")
        return

    print("nothing to do — pass --category <name> for a peek or --full for a full run")


if __name__ == "__main__":
    main()
