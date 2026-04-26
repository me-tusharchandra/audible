import type { ClassifyResponse, Profile } from "./types";

// Tiny client-side fallback so the demo never breaks. Not the main path —
// just enough to make the example chips behave sensibly when the backend
// is cold or unreachable.
export function heuristicClassify(
  utteranceRaw: string,
  profile: Profile,
): ClassifyResponse {
  const u = utteranceRaw.toLowerCase().trim();
  const has = (...needles: string[]) => needles.some((n) => u.includes(n));

  // Filler / ambient cues — speaker isn't addressing the assistant.
  if (
    has("hold on", "give me a sec", "give me a second", "grabbing", "wait,") ||
    /\bwait\b/.test(u)
  ) {
    return { decision: "IGNORE", tool: null, confidence: 0.82 };
  }

  // Someone else in the room is being addressed. Past-tense "did you" and
  // declaratives like "you should" are the strong signals — present-tense
  // "can you" / "could you" are overwhelmingly polite requests TO the
  // assistant, so don't catch them here.
  if (/\bdid you\b/.test(u) || u.startsWith("you should")) {
    if (has("timer", "calendar", "play", "music", "lights", "alarm")) {
      return { decision: "IGNORE", tool: null, confidence: 0.78 };
    }
  }

  // Timer.
  if (has("timer", "alarm", "countdown")) {
    if (profile === "minimalist" && !has("set", "start", "please", "begin")) {
      return { decision: "IGNORE", tool: null, confidence: 0.7 };
    }
    return { decision: "ACT", tool: "set_timer", confidence: 0.91 };
  }

  // Calendar.
  if (has("calendar", "schedule", "remind me", "meeting", "standup", "appointment")) {
    return { decision: "ACT", tool: "add_calendar_event", confidence: 0.88 };
  }

  // Music.
  if (has("play", "song", "music", "spotify", "playlist")) {
    if (profile === "work_focused") {
      return { decision: "IGNORE", tool: null, confidence: 0.84 };
    }
    return { decision: "ACT", tool: "play_music", confidence: 0.86 };
  }

  // Smart-home cues.
  if (
    has(
      "lights",
      "lamp",
      "thermostat",
      "temperature",
      "bright in here",
      "freezing",
      "too cold",
      "too warm",
      "ac ",
      "heater",
    )
  ) {
    if (profile === "work_focused") {
      return { decision: "IGNORE", tool: null, confidence: 0.83 };
    }
    if (profile === "minimalist") {
      const explicit =
        /\b(turn|set|dim|brighten|raise|lower|switch)\b/.test(u) ||
        u.startsWith("please");
      if (!explicit) {
        return { decision: "IGNORE", tool: null, confidence: 0.72 };
      }
    }
    return { decision: "ACT", tool: "smart_home_control", confidence: 0.85 };
  }

  // Indirect / curious wonderings.
  if (has("wonder", "what's the", "what is the", "how's the weather", "how is the weather")) {
    if (profile === "proactive") {
      return { decision: "ACT", tool: "web_search", confidence: 0.79 };
    }
    return { decision: "UPDATE_CONTEXT", tool: null, confidence: 0.66 };
  }

  // Generic fall-through.
  return { decision: "IGNORE", tool: null, confidence: 0.6 };
}
