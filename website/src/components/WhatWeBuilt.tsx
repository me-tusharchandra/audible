interface Card {
  title: string;
  body: string;
}

const CARDS: Card[] = [
  {
    title: "OpenEnv environment",
    body: "Composite 4-component rubric — gate correctness, tool correctness, profile alignment, false-wake penalty. Single-step episodes, 3 personalised user profiles. Reward range [-1.0, +2.0].",
  },
  {
    title: "mobileBERT classifier",
    body: "24.6M params, sentence-pair input (profile description + utterance), 7-class head. Edge-deployable (~95MB on-device).",
  },
  {
    title: "Adaptive curriculum",
    body: "Eval → mine failures → adversarial generation (gpt-4o-mini Structured Outputs) → retrain. The Self-Improvement angle. Round 1 dropped proactive false-wakes 4×.",
  },
];

export function WhatWeBuilt() {
  return (
    <section className="border-b border-zinc-800/80">
      <div className="mx-auto w-full max-w-5xl px-6 py-20 sm:py-24">
        <p className="text-xs uppercase tracking-widest text-zinc-500">
          What we built
        </p>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-50 sm:text-4xl">
          Three pieces, one feedback loop.
        </h2>
        <div className="mt-10 grid gap-4 md:grid-cols-3">
          {CARDS.map((c) => (
            <div
              key={c.title}
              className="flex flex-col rounded-xl border border-zinc-800 bg-zinc-900/30 p-6"
            >
              <h3 className="text-base font-semibold text-zinc-100">
                {c.title}
              </h3>
              <p className="mt-3 text-sm leading-relaxed text-zinc-400">
                {c.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
