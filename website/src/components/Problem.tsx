export function Problem() {
  const failures: { title: string; example: string }[] = [
    {
      title: "Filler that mimics commands",
      example: '"Hold on a sec, grabbing my keys" — sounds like a timer cue but isn\'t.',
    },
    {
      title: "Speech aimed at someone else",
      example: '"Did you set the timer for the cookies?" — directed at a person, not the assistant.',
    },
    {
      title: "Indirect cues with no addressee",
      example: '"It\'s a bit bright in here" — proactive users want lights dimmed; minimalists want silence.',
    },
  ];

  return (
    <section className="border-b border-zinc-800/80">
      <div className="mx-auto w-full max-w-5xl px-6 py-20 sm:py-24">
        <p className="text-xs uppercase tracking-widest text-zinc-500">
          The problem
        </p>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-50 sm:text-4xl">
          Wake-words are solved. Always-on isn&apos;t.
        </h2>
        <div className="mt-6 grid max-w-3xl gap-5 text-zinc-400">
          <p>
            Wake-word gating — &quot;Hey Siri&quot;, &quot;Alexa&quot; — is a
            mature problem. The hard cases live in always-on listening, where
            the assistant has to decide which utterances are <em>actually</em>{" "}
            addressed to it.
          </p>
          <p>
            Most failures aren&apos;t about acoustics. They&apos;re ambient
            speech that contains a tool keyword but isn&apos;t a command — and
            two reasonable users with the same input would want completely
            different responses.
          </p>
        </div>
        <div className="mt-10 grid gap-3 md:grid-cols-3">
          {failures.map((f) => (
            <div
              key={f.title}
              className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-5"
            >
              <p className="text-sm font-medium text-zinc-200">{f.title}</p>
              <p className="mt-2 text-sm text-zinc-400">{f.example}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
