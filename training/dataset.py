"""
Build a HuggingFace Dataset from labeled.parquet for mobileBERT training.

Input format is a sentence pair (profile_description, utterance) — mobileBERT
was pretrained with NSP, so the [SEP]-separated two-sentence layout is a
natural fit and lets the model condition its prediction on the profile.

The friend's binary data is heavily skewed — ~50% IGNORE, ~44% ACT_web_search,
and the four other tool classes are <2% each. We train with class-weighted
loss to keep the rare tools learnable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, DatasetDict
from transformers import AutoTokenizer

from audible_env.models import PROFILE_DESCRIPTIONS

_DATA_DIR = Path(__file__).resolve().parents[1] / "audible_env" / "data"
COMBINED_PARQUET = _DATA_DIR / "combined.parquet"
LABELED_PARQUET = _DATA_DIR / "labeled.parquet"


def _resolve_dataset_path() -> Path:
    """Prefer combined.parquet if it exists (heuristic + synthetic),
    otherwise fall back to the heuristic-only labeled.parquet."""
    return COMBINED_PARQUET if COMBINED_PARQUET.exists() else LABELED_PARQUET

CLASS_NAMES = [
    "IGNORE",
    "UPDATE_CONTEXT",
    "ACT_set_timer",
    "ACT_add_calendar_event",
    "ACT_play_music",
    "ACT_web_search",
    "ACT_smart_home_control",
]
NUM_CLASSES = len(CLASS_NAMES)


def load_labeled() -> pd.DataFrame:
    path = _resolve_dataset_path()
    df = pd.read_parquet(path)
    # combined.parquet has the same schema; ensure class_id present
    if "class_id" not in df.columns:
        df["class_id"] = df["class_label"].map({c: i for i, c in enumerate(CLASS_NAMES)})
    return df


def to_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """Convert the labeled dataframe into a stratified train/eval DatasetDict."""
    df = df[["text", "profile", "class_id"]].rename(columns={"class_id": "labels"})
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.cast_column("labels", ClassLabel(names=CLASS_NAMES))
    return ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="labels")


def build_tokenize_fn(tokenizer: AutoTokenizer, max_length: int = 128):
    """Returns a function that builds (profile_description, utterance) pairs."""

    def _tokenize(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        descriptions = [PROFILE_DESCRIPTIONS[p] for p in batch["profile"]]
        return tokenizer(
            descriptions,
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return _tokenize


def class_weights(labels: List[int], device: str = "cpu", smoothing: float = 0.1) -> torch.Tensor:
    """Inverse-frequency class weights, smoothed to keep rare classes from dominating.

    `smoothing` sets a floor so a class with one example doesn't get an
    unbounded weight that explodes the gradient.
    """
    counts = Counter(labels)
    n = len(labels)
    raw = np.array(
        [n / (counts[c] + smoothing * n) for c in range(NUM_CLASSES)],
        dtype=np.float32,
    )
    raw = raw / raw.mean()  # normalize so mean weight ≈ 1.0
    return torch.tensor(raw, device=device)


def prepare(
    tokenizer_name: str = "google/mobilebert-uncased",
    test_size: float = 0.2,
    max_length: int = 128,
    seed: int = 42,
) -> Tuple[DatasetDict, AutoTokenizer, torch.Tensor]:
    """One-shot helper: load + split + tokenize + compute class weights."""
    df = load_labeled()
    splits = to_dataset(df, test_size=test_size, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenize = build_tokenize_fn(tokenizer, max_length=max_length)
    splits = splits.map(tokenize, batched=True, remove_columns=["text", "profile"])

    weights = class_weights(splits["train"]["labels"])
    return splits, tokenizer, weights


if __name__ == "__main__":
    df = load_labeled()
    print(f"loaded {len(df)} rows from {_resolve_dataset_path()}")
    print("\nclass distribution:")
    counts = Counter(df["class_label"])
    for c in CLASS_NAMES:
        print(f"  {c:30s} {counts[c]:5d}")

    splits = to_dataset(df)
    print(f"\nsplits: train={len(splits['train'])} eval={len(splits['test'])}")
    print(f"\nfirst train example: {splits['train'][0]}")

    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    tokenize = build_tokenize_fn(tokenizer)
    one_batch = splits["train"].select(range(2)).map(tokenize, batched=True)
    print(f"\ntokenized first row keys: {list(one_batch[0].keys())}")
    print(f"input_ids length: {len(one_batch[0]['input_ids'])}")

    weights = class_weights(splits["train"]["labels"])
    print(f"\nclass weights (mean={weights.mean():.2f}):")
    for c, w in zip(CLASS_NAMES, weights.tolist()):
        print(f"  {c:30s} {w:.3f}")
