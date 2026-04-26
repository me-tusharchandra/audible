export type Profile = "minimalist" | "proactive" | "work_focused";

export type Decision = "ACT" | "UPDATE_CONTEXT" | "IGNORE";

export type Tool =
  | "set_timer"
  | "add_calendar_event"
  | "play_music"
  | "web_search"
  | "smart_home_control";

export interface ClassifyResponse {
  decision: Decision;
  tool: Tool | null;
  confidence: number;
}

export type Feedback = "up" | "down" | null;

export interface ResultRow {
  id: string;
  utterance: string;
  profile: Profile;
  decision: Decision;
  tool: Tool | null;
  confidence: number;
  fromHeuristic: boolean;
  timestamp: number;
  launchUrl: string | null;
  autoLaunched: boolean;
  feedback: Feedback;
}

export const PROFILES: { id: Profile; label: string; description: string }[] = [
  {
    id: "minimalist",
    label: "Minimalist",
    description:
      "Acts only on explicit, first-person imperative commands directed at the assistant. Anything ambiguous or indirect should be ignored.",
  },
  {
    id: "proactive",
    label: "Proactive",
    description:
      "Acts on indirect cues too — 'I wonder…', 'it's a bit bright in here'. Errs on the side of being helpful.",
  },
  {
    id: "work_focused",
    label: "Work focused",
    description:
      "Acts on timers, calendar, and web search. Never plays music or controls smart-home devices, even when explicitly asked.",
  },
];

export const EXAMPLE_UTTERANCES: string[] = [
  "Hey, set a timer for 5 minutes",
  "Hold on a sec, grabbing my keys",
  "I wonder what the weather's like in Paris",
  "Play me some Pink Floyd",
  "Did you set the timer for the cookies?",
  "It's a bit bright in here",
  "Add tomorrow's standup to my calendar",
  "What's the capital of Mongolia?",
];
