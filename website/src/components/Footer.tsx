export function Footer() {
  return (
    <footer>
      <div className="mx-auto w-full max-w-5xl px-6 py-12">
        <p className="text-sm text-zinc-300">
          Built for Meta OpenEnv Hackathon 2026 — Self-Improvement theme.
        </p>
        <p className="mt-2 max-w-2xl text-xs text-zinc-500">
          mobileBERT (Google) · OpenEnv (Meta) · gpt-4o-mini (OpenAI, for
          synthetic data).
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-4 text-xs text-zinc-500">
          <a
            href="https://github.com/me-tusharchandra/audible"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-400 transition-colors hover:text-zinc-100"
          >
            github.com/me-tusharchandra/audible
          </a>
          <span aria-hidden>·</span>
          <span className="font-mono">v1.0</span>
        </div>
      </div>
    </footer>
  );
}
