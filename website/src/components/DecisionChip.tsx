import type { Decision, Tool } from "@/lib/types";

interface DecisionChipProps {
  decision: Decision;
  tool: Tool | null;
}

const STYLES: Record<Decision, string> = {
  ACT: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30",
  UPDATE_CONTEXT: "bg-blue-500/10 text-blue-400 ring-blue-500/30",
  IGNORE: "bg-zinc-700/30 text-zinc-400 ring-zinc-700/40",
};

export function DecisionChip({ decision, tool }: DecisionChipProps) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${STYLES[decision]}`}
    >
      <span>{decision}</span>
      {decision === "ACT" && tool ? (
        <span className="font-mono text-[11px] text-zinc-300">
          {tool}
        </span>
      ) : null}
    </span>
  );
}
