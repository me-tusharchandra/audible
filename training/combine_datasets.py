"""
Merge friend's heuristic-labeled data with synthetic LLM-generated scenarios
into a single training set.

Strategy:
  - Friend's data: high volume, lexical diversity, but tool labels are noisy
    (regex fallback collapses many ACTs to web_search).
  - Synthetic data: lower volume, but precise per-profile labels and the
    ambient/divergent cases the friend's binary data misses entirely.

Combined gives both: friend's data anchors the model on natural commands and
non-actionable distractors, synthetic data teaches the hard ambient cases and
the rare tools.

Run:
    python -m training.combine_datasets
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.dataset import CLASS_NAMES

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "audible_env" / "data"
HEURISTIC = DATA_DIR / "labeled.parquet"
SYNTHETIC = DATA_DIR / "synthetic" / "synthetic_v1.parquet"
OUT = DATA_DIR / "combined.parquet"


def load_heuristic() -> pd.DataFrame:
    df = pd.read_parquet(HEURISTIC)
    return df[["text", "profile", "class_label", "decision", "tool", "source"]].assign(
        source="heuristic_friend"
    )


def load_synthetic() -> pd.DataFrame:
    if not SYNTHETIC.exists():
        return pd.DataFrame(
            columns=["text", "profile", "class_label", "decision", "tool", "source"]
        )
    df = pd.read_parquet(SYNTHETIC)
    return df[["text", "profile", "class_label", "decision", "tool", "source"]]


def main() -> None:
    h = load_heuristic()
    s = load_synthetic()
    print(f"heuristic_friend: {len(h):>6d} rows ({h['text'].nunique():>5d} utterances)")
    print(f"synthetic_v1:     {len(s):>6d} rows ({s['text'].nunique():>5d} utterances)")

    combined = pd.concat([h, s], ignore_index=True)
    # Dedupe at the (text, profile) level — synthetic wins on collision because
    # its labels are higher-quality.
    combined = combined.sort_values(by="source", ascending=False)  # 'synthetic_v1' > 'heuristic_friend'
    combined = combined.drop_duplicates(subset=["text", "profile"], keep="first").reset_index(drop=True)
    print(f"\ncombined (after dedup): {len(combined)} rows ({combined['text'].nunique()} utterances)")

    combined["class_id"] = combined["class_label"].map({c: i for i, c in enumerate(CLASS_NAMES)})

    print("\nclass distribution per profile (combined):")
    pivot = combined.pivot_table(
        index="class_label", columns="profile", values="text",
        aggfunc="count", fill_value=0,
    ).reindex(CLASS_NAMES)
    print(pivot.to_string())

    print("\nsource × class:")
    print(combined.groupby(["source", "class_label"]).size().unstack(fill_value=0).reindex(columns=CLASS_NAMES, fill_value=0).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
