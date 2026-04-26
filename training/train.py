"""
Train mobileBERT as the gating classifier.

Path A — encoder classifier with HuggingFace Trainer + class-weighted CE loss.
TRL's generative-LM trainers don't fit encoder-only models; HF Trainer is the
class TRL extends, so the training loop, loggers, and checkpointing are all
the same machinery.

Run from repo root:
    python -m training.train                          # default 3 epochs
    python -m training.train --epochs 1 --batch 32    # smoke run

Outputs:
    training/runs/<timestamp>/  — checkpoints + tensorboard logs
    training/runs/<timestamp>/eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from training.dataset import CLASS_NAMES, NUM_CLASSES, prepare

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "training" / "runs"


@dataclass
class TrainConfig:
    model_name: str = "google/mobilebert-uncased"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    test_size: float = 0.2


class WeightedTrainer(Trainer):
    """HF Trainer with a class-weighted cross-entropy loss to compensate for
    the heavy class imbalance in friend's binary-derived labels."""

    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self._class_weights.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float((preds == labels).mean()),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=TrainConfig.model_name)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = p.parse_args()
    return TrainConfig(
        model_name=args.model,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== run {run_id} ===")
    print(f"out: {out_dir}")
    print(f"cfg: {cfg}")

    splits, tokenizer, weights = prepare(
        tokenizer_name=cfg.model_name,
        test_size=cfg.test_size,
        max_length=cfg.max_length,
        seed=cfg.seed,
    )
    print(f"train={len(splits['train'])} eval={len(splits['test'])}")
    print(f"class weights: {dict(zip(CLASS_NAMES, [round(w, 2) for w in weights.tolist()]))}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=NUM_CLASSES,
        id2label={i: c for i, c in enumerate(CLASS_NAMES)},
        label2id={c: i for i, c in enumerate(CLASS_NAMES)},
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",  # tensorboard optional — install it and switch back if you want curves
        seed=cfg.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        compute_metrics=compute_metrics,
        class_weights=weights,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"\nfinal eval metrics: {eval_metrics}")

    # Per-class breakdown on the held-out split for the final checkpoint
    preds_out = trainer.predict(splits["test"])
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist()

    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "config": cfg.__dict__,
                "summary": eval_metrics,
                "per_class": report,
                "confusion_matrix": cm,
                "class_names": CLASS_NAMES,
            },
            indent=2,
        )
    )
    print(f"wrote {metrics_path}")

    # Save the best model + tokenizer at the run-dir root (eval_env loads from here).
    # load_best_model_at_end=True ensures `trainer.model` is the best epoch's weights.
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)
    print(f"\ndone. model + tokenizer at {out_dir}")


if __name__ == "__main__":
    main()
