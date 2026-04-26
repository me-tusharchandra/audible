"use client";

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { classifyUtterance, warmupClassifier } from "@/lib/classify";
import { buildToolUrl, LAUNCHABLE_TOOLS, predictLaunchUrl } from "@/lib/launch";
import {
  EXAMPLE_UTTERANCES,
  PROFILES,
  type Feedback,
  type Profile,
  type ResultRow,
} from "@/lib/types";
import { DecisionChip } from "./DecisionChip";

type Mode = "mic" | "text";

const MAX_RESULTS = 10;
// Only auto-open the launchable tools (web_search / play_music) when the
// model is genuinely confident. Below this, we still show a manual Launch ↗
// link so the user can fire it themselves if they actually want it.
const AUTO_LAUNCH_CONFIDENCE = 0.85;

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export function DemoBlock() {
  const textareaId = useId();
  const [profile, setProfile] = useState<Profile>("proactive");
  const [mode, setMode] = useState<Mode>("mic");
  const [textInput, setTextInput] = useState("");
  const [results, setResults] = useState<ResultRow[]>([]);
  const [isClassifying, setIsClassifying] = useState(false);

  // Mic state.
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [micSupported, setMicSupported] = useState<boolean | null>(null);
  const profileRef = useRef<Profile>(profile);

  // Keep latest profile available to event handlers without re-binding.
  useEffect(() => {
    profileRef.current = profile;
  }, [profile]);

  // Warm the classifier on mount so the first real classification doesn't pay
  // the lazy-load penalty on the Space.
  useEffect(() => {
    void warmupClassifier();
  }, []);

  const activeProfile = useMemo(
    () => PROFILES.find((p) => p.id === profile)!,
    [profile],
  );

  // Detect browser support on mount only — avoids hydration mismatch.
  useEffect(() => {
    if (typeof window === "undefined") {
      setMicSupported(false);
      return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    setMicSupported(Boolean(SR));
  }, []);

  // Stop recognition when the component unmounts.
  useEffect(() => {
    return () => {
      try {
        recognitionRef.current?.stop();
      } catch {
        // already stopped
      }
      recognitionRef.current = null;
    };
  }, []);

  const pushResult = useCallback(
    async (
      utterance: string,
      speculativeTab: Window | null,
      speculativeUrl: string | null,
    ) => {
      const cleaned = utterance.trim();
      if (!cleaned) {
        speculativeTab?.close();
        return;
      }
      setIsClassifying(true);
      try {
        const { result, fromHeuristic } = await classifyUtterance(
          cleaned,
          profileRef.current,
        );
        const realUrl =
          result.decision === "ACT"
            ? buildToolUrl(result.tool, cleaned)
            : null;

        let autoLaunched = false;
        const launchTrustsModel =
          !!realUrl && result.confidence >= AUTO_LAUNCH_CONFIDENCE;
        if (speculativeTab && !speculativeTab.closed) {
          if (launchTrustsModel) {
            // Only redirect if the model picked a different URL than the
            // heuristic guess — avoids an unnecessary reload when they agree.
            if (realUrl !== speculativeUrl) {
              try {
                speculativeTab.location.href = realUrl;
              } catch {
                // some browsers throw on cross-origin assignment after navigation
              }
            }
            autoLaunched = true;
          } else {
            // Either model says don't act, or its confidence is below the
            // auto-launch threshold — close the speculative tab. Manual
            // Launch ↗ still appears in the row if there's a tool URL.
            try {
              speculativeTab.close();
            } catch {
              // ignore
            }
          }
        }

        const row: ResultRow = {
          id: newId(),
          utterance: cleaned,
          profile: profileRef.current,
          decision: result.decision,
          tool: result.tool,
          confidence: result.confidence,
          fromHeuristic,
          timestamp: Date.now(),
          launchUrl: realUrl,
          autoLaunched,
          feedback: null,
        };
        setResults((prev) => [row, ...prev].slice(0, MAX_RESULTS));
      } catch {
        speculativeTab?.close();
      } finally {
        setIsClassifying(false);
      }
    },
    [],
  );

  const startListening = useCallback(() => {
    if (typeof window === "undefined") return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      setMicSupported(false);
      return;
    }
    try {
      const recognition = new SR();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      recognition.maxAlternatives = 1;

      recognition.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const r = event.results[i];
          const transcript = r[0]?.transcript ?? "";
          if (r.isFinal) {
            // No user-gesture context here — popup blocker would eat any tab
            // we tried to open. Render a Launch ↗ link in the row instead.
            void pushResult(transcript, null, null);
          } else {
            interim += transcript;
          }
        }
        setInterimTranscript(interim);
      };
      recognition.onerror = () => {
        setIsListening(false);
        setInterimTranscript("");
      };
      recognition.onend = () => {
        setIsListening(false);
        setInterimTranscript("");
      };

      recognition.start();
      recognitionRef.current = recognition;
      setIsListening(true);
    } catch {
      setIsListening(false);
    }
  }, [pushResult]);

  const stopListening = useCallback(() => {
    try {
      recognitionRef.current?.stop();
    } catch {
      // ignore
    }
    setIsListening(false);
    setInterimTranscript("");
  }, []);

  const handleTextSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      const value = textInput;
      const cleaned = value.trim();
      setTextInput("");
      if (!cleaned) return;

      // Predict the tool URL synchronously and open it inside the click
      // handler. Opening a real https tab is far more reliable than the
      // about:blank-then-redirect approach — Chrome aggressively kills empty
      // tabs that aren't navigated before the cold-start classifier returns.
      // If the real model later disagrees, pushResult redirects or closes.
      let speculativeUrl: string | null = null;
      let speculativeTab: Window | null = null;
      if (typeof window !== "undefined") {
        speculativeUrl = predictLaunchUrl(cleaned, profileRef.current);
        if (speculativeUrl) {
          speculativeTab = window.open(speculativeUrl, "_blank");
        }
      }
      void pushResult(cleaned, speculativeTab, speculativeUrl);
    },
    [textInput, pushResult],
  );

  const setFeedback = useCallback((id: string, value: Feedback) => {
    setResults((prev) =>
      prev.map((r) =>
        r.id === id
          ? { ...r, feedback: r.feedback === value ? null : value }
          : r,
      ),
    );
  }, []);

  const handleExampleClick = useCallback((utterance: string) => {
    setTextInput(utterance);
    setMode("text");
  }, []);

  return (
    <section
      id="demo"
      className="border-b border-zinc-800/80 scroll-mt-12"
    >
      <div className="mx-auto w-full max-w-5xl px-6 py-20 sm:py-24">
        <div className="max-w-2xl">
          <p className="text-xs uppercase tracking-widest text-zinc-500">
            Live demo
          </p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-50 sm:text-4xl">
            Try it live
          </h2>
          <p className="mt-3 text-zinc-400">
            Pick a user profile, then speak or type. Watch how the same
            utterance gets a different gating decision per profile.
          </p>
        </div>

        {/* Profile selector */}
        <div className="mt-10">
          <div
            role="radiogroup"
            aria-label="User profile"
            className="inline-flex flex-wrap gap-1 rounded-lg border border-zinc-800 bg-zinc-900/40 p-1"
          >
            {PROFILES.map((p) => {
              const active = p.id === profile;
              return (
                <button
                  key={p.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  onClick={() => setProfile(p.id)}
                  className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition-colors ${
                    active
                      ? "bg-zinc-50 text-zinc-950"
                      : "text-zinc-400 hover:text-zinc-100"
                  }`}
                >
                  {p.label}
                </button>
              );
            })}
          </div>
          <p className="mt-3 max-w-2xl text-sm text-zinc-400">
            <span className="font-mono text-xs text-zinc-500">
              {activeProfile.id}:
            </span>{" "}
            {activeProfile.description}
          </p>
        </div>

        {/* Mode tabs + input */}
        <div className="mt-10 rounded-xl border border-zinc-800 bg-zinc-900/30">
          <div className="flex border-b border-zinc-800">
            <button
              type="button"
              onClick={() => setMode("mic")}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                mode === "mic"
                  ? "bg-zinc-900/70 text-zinc-50"
                  : "text-zinc-400 hover:text-zinc-100"
              }`}
            >
              <span aria-hidden className="mr-1.5">🎤</span> Mic
            </button>
            <button
              type="button"
              onClick={() => setMode("text")}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors border-l border-zinc-800 ${
                mode === "text"
                  ? "bg-zinc-900/70 text-zinc-50"
                  : "text-zinc-400 hover:text-zinc-100"
              }`}
            >
              <span aria-hidden className="mr-1.5">⌨️</span> Text
            </button>
          </div>

          {/* Mic mode */}
          {mode === "mic" ? (
            <div className="flex flex-col items-center gap-3 px-6 py-10">
              {micSupported === false ? (
                <p className="text-center text-sm text-zinc-400">
                  Your browser doesn&apos;t expose the Web Speech API. Switch to
                  the Text tab to keep going.
                </p>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={isListening ? stopListening : startListening}
                    aria-pressed={isListening}
                    className={`relative inline-flex items-center gap-3 rounded-full px-6 py-3 text-sm font-medium transition-colors ${
                      isListening
                        ? "bg-red-500/10 text-red-300 ring-1 ring-inset ring-red-500/40"
                        : "bg-zinc-50 text-zinc-950 hover:bg-zinc-200"
                    }`}
                  >
                    {isListening ? (
                      <>
                        <span className="pulse-dot inline-block h-2.5 w-2.5 rounded-full bg-red-500" />
                        Stop listening
                      </>
                    ) : (
                      <>
                        <span aria-hidden>🎤</span>
                        Start listening
                      </>
                    )}
                  </button>
                  <div className="min-h-[1.5rem] text-center">
                    {isListening ? (
                      <p className="text-xs text-zinc-500">
                        Listening…{" "}
                        {interimTranscript ? (
                          <span className="italic text-zinc-300">
                            “{interimTranscript}”
                          </span>
                        ) : (
                          <span className="italic text-zinc-600">
                            (say something — finalised utterances get
                            classified)
                          </span>
                        )}
                      </p>
                    ) : null}
                  </div>
                </>
              )}
            </div>
          ) : (
            <form onSubmit={handleTextSubmit} className="flex flex-col gap-3 px-6 py-6">
              <label
                htmlFor={textareaId}
                className="text-xs uppercase tracking-widest text-zinc-500"
              >
                Utterance
              </label>
              <textarea
                id={textareaId}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                rows={3}
                placeholder="Type something the assistant might overhear…"
                className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-zinc-600 focus:outline-none"
              />
              <div className="flex items-center justify-end">
                <button
                  type="submit"
                  disabled={!textInput.trim() || isClassifying}
                  className="inline-flex items-center justify-center rounded-md bg-zinc-50 px-4 py-2 text-sm font-medium text-zinc-950 transition-colors hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isClassifying ? "Classifying…" : "Classify"}
                </button>
              </div>
            </form>
          )}

          <p className="border-t border-zinc-800 px-6 py-3 text-xs text-zinc-500">
            Mic input uses the Web Speech API and works best in Chrome/Edge.
            Firefox &amp; Safari users — text input still works.
          </p>
        </div>

        {/* Result stack */}
        <div className="mt-8">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-xs uppercase tracking-widest text-zinc-500">
              Results
            </p>
            {(() => {
              const up = results.filter((r) => r.feedback === "up").length;
              const down = results.filter((r) => r.feedback === "down").length;
              if (up + down === 0) return null;
              return (
                <p className="text-[11px] text-zinc-500">
                  <span className="font-mono text-emerald-400">{up} 👍</span>
                  <span className="mx-2 text-zinc-700">·</span>
                  <span className="font-mono text-rose-400">{down} 👎</span>
                  <span className="ml-2 hidden text-zinc-500 sm:inline">
                    → routed into the next curriculum round
                  </span>
                </p>
              );
            })()}
          </div>
          <div className="max-h-[28rem] divide-y divide-zinc-800 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900/30">
            {results.length === 0 ? (
              <div className="px-5 py-8 text-center text-sm text-zinc-500">
                No utterances yet. Try one of the examples below or use the mic.
              </div>
            ) : (
              results.map((r) => (
                <div
                  key={r.id}
                  className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-start sm:justify-between"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-zinc-100">{r.utterance}</p>
                    <p className="mt-1 font-mono text-[10px] uppercase tracking-wider text-zinc-500">
                      {r.profile}
                    </p>
                    <div className="mt-2 flex items-center gap-1.5">
                      <button
                        type="button"
                        aria-label="Correct classification"
                        aria-pressed={r.feedback === "up"}
                        onClick={() => setFeedback(r.id, "up")}
                        className={`inline-flex h-7 w-7 items-center justify-center rounded-md border text-sm transition-colors ${
                          r.feedback === "up"
                            ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-300"
                            : "border-zinc-800 bg-zinc-900/40 text-zinc-500 hover:border-zinc-700 hover:text-zinc-200"
                        }`}
                      >
                        👍
                      </button>
                      <button
                        type="button"
                        aria-label="Wrong classification"
                        aria-pressed={r.feedback === "down"}
                        onClick={() => setFeedback(r.id, "down")}
                        className={`inline-flex h-7 w-7 items-center justify-center rounded-md border text-sm transition-colors ${
                          r.feedback === "down"
                            ? "border-rose-500/50 bg-rose-500/10 text-rose-300"
                            : "border-zinc-800 bg-zinc-900/40 text-zinc-500 hover:border-zinc-700 hover:text-zinc-200"
                        }`}
                      >
                        👎
                      </button>
                      {r.feedback ? (
                        <span className="ml-1 text-[10px] text-zinc-500">
                          fed back into curriculum
                        </span>
                      ) : null}
                    </div>
                  </div>
                  <div className="flex flex-col items-start gap-1 sm:items-end">
                    <DecisionChip decision={r.decision} tool={r.tool} />
                    <span className="font-mono text-[10px] text-zinc-500">
                      {(r.confidence * 100).toFixed(1)}% conf
                    </span>
                    {r.launchUrl && r.tool && LAUNCHABLE_TOOLS.has(r.tool) ? (
                      <a
                        href={r.launchUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[11px] text-emerald-400 underline-offset-2 hover:underline"
                      >
                        {r.autoLaunched ? "Launched ↗" : "Launch ↗"}
                      </a>
                    ) : null}
                    {r.fromHeuristic ? (
                      <span className="text-[10px] text-amber-400/80">
                        (heuristic — backend warming up)
                      </span>
                    ) : null}
                  </div>
                </div>
              ))
            )}
          </div>
          <p className="mt-2 text-[11px] text-zinc-500">
            👍 / 👎 mark each label as correct or wrong — in the real loop, these signals would route into the next adversarial curriculum round to retrain the gate.
          </p>
        </div>

        {/* Example chips */}
        <div className="mt-8">
          <p className="mb-3 text-xs uppercase tracking-widest text-zinc-500">
            Try these
          </p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_UTTERANCES.map((u) => (
              <button
                key={u}
                type="button"
                onClick={() => handleExampleClick(u)}
                className="rounded-full border border-zinc-800 bg-zinc-900/40 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:border-zinc-700 hover:bg-zinc-800/60 hover:text-zinc-100"
              >
                {u}
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
