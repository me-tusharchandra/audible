import type { Profile, Tool } from "./types";

function extractMusicQuery(utterance: string): string {
  const cleaned = utterance
    .trim()
    .replace(/^(hey|ok(?:ay)?|alexa|siri|audible)[,!. ]+/i, "")
    .replace(/^(can you|could you|please|i'd like you to|i want you to)\s+/i, "")
    .replace(
      /^(play me|play|put on|put|spin|queue up|start|i wanna hear|i want to hear|i'd like to hear)\s+/i,
      "",
    )
    .replace(/^(some|a little|a bit of)\s+/i, "")
    .trim();
  return cleaned || utterance.trim();
}

export function buildToolUrl(
  tool: Tool | null,
  utterance: string,
): string | null {
  if (!tool) return null;
  const u = utterance.trim();
  if (!u) return null;
  if (tool === "web_search") {
    return `https://www.google.com/search?q=${encodeURIComponent(u)}`;
  }
  if (tool === "play_music") {
    return `https://music.youtube.com/search?q=${encodeURIComponent(extractMusicQuery(u))}`;
  }
  return null;
}

export const LAUNCHABLE_TOOLS: ReadonlySet<Tool> = new Set([
  "web_search",
  "play_music",
]);

// Lightweight, *speculation-only* predictor for which URL we should open in
// the synchronous click handler. Deliberately more lenient than the full
// gating heuristic — its only job is to keep the popup-blocker happy. If the
// real model later disagrees, DemoBlock closes / redirects the tab.
export function predictLaunchUrl(
  utterance: string,
  profile: Profile,
): string | null {
  const u = utterance.toLowerCase();

  // Music: any explicit music verb / noun. work_focused profile never plays
  // music, so don't speculate for that one.
  const musicCues =
    /\b(play|playing|put on|spin|queue|listen to|listening to|song|songs|music|tune|tunes|track|tracks|album|playlist|spotify|hum|sing)\b/;
  if (profile !== "work_focused" && musicCues.test(u)) {
    return buildToolUrl("play_music", utterance);
  }

  // Web search: question forms, curious wonderings, "look up X" style.
  // minimalist profile is conservative and rarely launches search.
  const searchCues =
    /\b(wonder|wondering|curious|what'?s|what is|what are|where'?s|where is|when'?s|when is|how'?s|how is|how do|how can|how much|how many|tell me about|look up|search for|find out|google|weather|capital of|population of|news about|recipe for|meaning of)\b/;
  if (profile !== "minimalist" && searchCues.test(u)) {
    return buildToolUrl("web_search", utterance);
  }

  return null;
}
