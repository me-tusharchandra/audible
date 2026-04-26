import { heuristicClassify } from "./heuristic";
import type { ClassifyResponse, Profile } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://me-tusharchandra-audible-env.hf.space";

const CLASSIFY_TIMEOUT_MS = 12_000;

// Fire-and-forget request to warm the model on the Space — first call after
// idle takes ~5–10s while the lazy loader pulls the safetensors into memory.
// Subsequent calls are <300ms.
export async function warmupClassifier(): Promise<void> {
  try {
    await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ utterance: "warmup", profile: "proactive" }),
      keepalive: true,
    });
  } catch {
    // ignore — warmup is best-effort
  }
}

export interface ClassifyOutcome {
  result: ClassifyResponse;
  fromHeuristic: boolean;
}

export async function classifyUtterance(
  utterance: string,
  profile: Profile,
): Promise<ClassifyOutcome> {
  const trimmed = utterance.trim();
  if (!trimmed) {
    return {
      result: { decision: "IGNORE", tool: null, confidence: 1 },
      fromHeuristic: true,
    };
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CLASSIFY_TIMEOUT_MS);
    const res = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ utterance: trimmed, profile }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = (await res.json()) as Partial<ClassifyResponse>;
    if (
      !data ||
      typeof data.decision !== "string" ||
      typeof data.confidence !== "number"
    ) {
      throw new Error("Bad payload");
    }
    return {
      result: {
        decision: data.decision as ClassifyResponse["decision"],
        tool:
          (data.tool as ClassifyResponse["tool"]) === undefined
            ? null
            : (data.tool as ClassifyResponse["tool"]),
        confidence: data.confidence,
      },
      fromHeuristic: false,
    };
  } catch {
    return {
      result: heuristicClassify(trimmed, profile),
      fromHeuristic: true,
    };
  }
}
