interface LinkCard {
  title: string;
  description: string;
  href: string;
  hint: string;
}

const LINKS: LinkCard[] = [
  {
    title: "HF Space",
    description: "The OpenEnv environment.",
    href: "https://huggingface.co/spaces/me-tusharchandra/audible-env",
    hint: "huggingface.co/spaces/me-tusharchandra/audible-env",
  },
  {
    title: "Live Web UI",
    description: "Single-step env interaction.",
    href: "https://me-tusharchandra-audible-env.hf.space/web",
    hint: "me-tusharchandra-audible-env.hf.space/web",
  },
  {
    title: "Blog post / writeup",
    description: "How we built it & what we learned.",
    href: "https://huggingface.co/spaces/me-tusharchandra/audible-env/blob/main/BLOG.md",
    hint: "BLOG.md",
  },
  {
    title: "GitHub repo",
    description: "Source for the env, classifier, curriculum loop.",
    href: "https://github.com/me-tusharchandra/audible",
    hint: "github.com/me-tusharchandra/audible",
  },
  {
    title: "Friend's prior work",
    description: "The binary baseline that started this.",
    href: "https://github.com/pranjal-pravesh/actionable-gating-classifier",
    hint: "github.com/pranjal-pravesh/actionable-gating-classifier",
  },
];

export function Explore() {
  return (
    <section className="border-b border-zinc-800/80">
      <div className="mx-auto w-full max-w-5xl px-6 py-20 sm:py-24">
        <p className="text-xs uppercase tracking-widest text-zinc-500">
          Try it / explore
        </p>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-50 sm:text-4xl">
          Dig deeper.
        </h2>
        <div className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex flex-col rounded-xl border border-zinc-800 bg-zinc-900/30 p-5 transition-colors hover:border-zinc-700 hover:bg-zinc-900/60"
            >
              <div className="flex items-start justify-between">
                <h3 className="text-base font-semibold text-zinc-100">
                  {l.title}
                </h3>
                <span
                  aria-hidden
                  className="text-zinc-500 transition-colors group-hover:text-zinc-200"
                >
                  ↗
                </span>
              </div>
              <p className="mt-1 text-sm text-zinc-400">{l.description}</p>
              <p className="mt-3 break-all font-mono text-[11px] text-zinc-500">
                {l.hint}
              </p>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
