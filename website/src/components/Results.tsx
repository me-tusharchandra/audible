import Image from "next/image";

interface Stat {
  value: string;
  label: string;
}

const STATS: Stat[] = [
  { value: "96.9%", label: "Accuracy on held-out eval" },
  { value: "0.875", label: "F1-macro" },
  { value: "24.6M", label: "Params (mobileBERT)" },
  { value: "4×", label: "False-wake reduction" },
];

interface Plot {
  src: string;
  alt: string;
  caption: string;
}

const PLOTS: Plot[] = [
  {
    src: "/plots/training_loss.png",
    alt: "Training loss curve",
    caption:
      "Loss converges cleanly from init spike to ~0.08 over 1677 steps. LR follows configured warmup-then-decay.",
  },
  {
    src: "/plots/reward_curve.png",
    alt: "Reward curve and false-wake rate",
    caption:
      "Reward + false-wake rate over 4 curriculum stages. Round 1 is the win.",
  },
  {
    src: "/plots/confusion_matrix.png",
    alt: "Confusion matrix",
    caption:
      "Row-normalized confusion matrix on held-out eval (baseline).",
  },
  {
    src: "/plots/dataset_distribution.png",
    alt: "Dataset distribution",
    caption:
      "22K training rows by class & source (heuristic / synthetic / curriculum).",
  },
];

export function Results() {
  return (
    <section className="border-b border-zinc-800/80">
      <div className="mx-auto w-full max-w-5xl px-6 py-20 sm:py-24">
        <p className="text-xs uppercase tracking-widest text-zinc-500">
          Results
        </p>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-50 sm:text-4xl">
          One round of curriculum.
        </h2>

        <dl className="mt-10 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-zinc-800 bg-zinc-800 md:grid-cols-4">
          {STATS.map((s) => (
            <div
              key={s.label}
              className="flex flex-col gap-1 bg-zinc-950 px-5 py-6"
            >
              <dt className="font-mono text-3xl font-semibold text-zinc-50 sm:text-4xl">
                {s.value}
              </dt>
              <dd className="text-xs uppercase tracking-wider text-zinc-500">
                {s.label}
              </dd>
            </div>
          ))}
        </dl>

        <div className="mt-12 grid gap-10">
          {PLOTS.map((p) => (
            <figure
              key={p.src}
              className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40"
            >
              <div className="relative w-full bg-zinc-950">
                <Image
                  src={p.src}
                  alt={p.alt}
                  width={1400}
                  height={650}
                  sizes="(max-width: 768px) 100vw, 1024px"
                  className="h-auto w-full object-contain"
                />
              </div>
              <figcaption className="border-t border-zinc-800 px-5 py-3 text-xs text-zinc-500">
                {p.caption}
              </figcaption>
            </figure>
          ))}
        </div>
      </div>
    </section>
  );
}
