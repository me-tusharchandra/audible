export function Hero() {
  return (
    <section className="border-b border-zinc-800/80">
      <div className="mx-auto w-full max-w-5xl px-6 pt-20 pb-24 sm:pt-28 sm:pb-32">
        <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/40 px-3 py-1 text-xs text-zinc-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          Meta OpenEnv Hackathon 2026 — Self-Improvement
        </div>
        <h1 className="mt-6 text-5xl font-semibold tracking-tight text-zinc-50 sm:text-7xl">
          Audible
        </h1>
        <p className="mt-5 max-w-3xl text-2xl font-medium tracking-tight text-zinc-200 sm:text-3xl">
          An always-on voice assistant that learns when not to listen.
        </p>
        <p className="mt-6 max-w-2xl text-base leading-relaxed text-zinc-400 sm:text-lg">
          A self-improving OpenEnv environment that teaches a 24.6M-parameter
          mobileBERT to gate ambient utterances per user.{" "}
          <span className="text-zinc-100 font-medium">
            4× false-wake reduction
          </span>{" "}
          in one round of adaptive curriculum.
        </p>
        <div className="mt-10 flex flex-col items-start gap-3 sm:flex-row sm:items-center">
          <a
            href="#demo"
            className="inline-flex items-center justify-center rounded-md bg-zinc-50 px-5 py-2.5 text-sm font-medium text-zinc-950 transition-colors hover:bg-zinc-200"
          >
            Try the live demo <span aria-hidden className="ml-1.5">↓</span>
          </a>
          <a
            href="https://huggingface.co/spaces/me-tusharchandra/audible-env"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center rounded-md border border-zinc-800 bg-transparent px-5 py-2.5 text-sm font-medium text-zinc-200 transition-colors hover:border-zinc-700 hover:bg-zinc-900/60"
          >
            View on Hugging Face <span aria-hidden className="ml-1.5">→</span>
          </a>
        </div>
      </div>
    </section>
  );
}
