"""
scripts/train_bert.py
======================
Stage 4 — Fine-tune DistilBERT for prompt-injection classification.

WHY THIS EXISTS
---------------
The rule-based baseline catches direct attacks but cannot generalize
(paraphrases, obfuscation, indirect injection). DistilBERT, fine-tuned
on our labeled dataset, learns semantic patterns and should generalize
better.

WHAT THIS SCRIPT DOES
---------------------
1. Loads train.csv and validation.csv (Stage 2 outputs).
2. Tokenizes texts using DistilBERT tokenizer.
3. Builds a PyTorch Dataset + DataLoader.
4. Initializes DistilBERTForSequenceClassification (2 labels).
5. Trains for N epochs with AdamW optimizer + linear LR schedule.
6. Validates each epoch, computes val loss + accuracy + F1.
7. Implements early stopping on validation loss.
8. Saves the model + tokenizer to models/distilbert_v1/.
9. Writes a training report (losses, metrics, timestamps) to results/.

EXPECTED RUNTIME
----------------
- 84 training examples, 3 epochs, CPU, batch_size=16:
  ~5-15 minutes on a modern Windows laptop.
- 84 training examples, 3 epochs, GPU:
  ~30-60 seconds.
- First run will download DistilBERT weights (~268 MB).

CRITICAL HONESTY
---------------
With only 84 training examples, the model may:
- Overfit (training accuracy >> validation accuracy)
- Have high variance (different seeds → different results)
- Not dramatically outperform the rule-based baseline

We will report whatever happens — no fabricated results.

USAGE
-----
    # Train with default config:
    python scripts/train_bert.py

    # Override hyperparameters:
    python scripts/train_bert.py --epochs 5 --batch-size 8

    # Train and save to a custom directory:
    python scripts/train_bert.py --output-dir models/distilbert_v1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Lazy torch imports — error message is clear if missing
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
except ImportError as e:
    print("ERROR: PyTorch / Transformers not installed.")
    print("Install with:")
    print("  pip install torch==2.5.1 transformers==4.46.3 tokenizers==0.20.3")
    print("For CPU-only torch on Windows:")
    print("  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu")
    raise

from src.utils.config import load_config
from src.evaluation.metrics import compute_binary_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptInjectionDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized prompts.

    Each item: (input_ids, attention_mask, label)
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def evaluate_model(model, dataloader, device) -> dict:
    """Run model on a dataloader, return metrics + average loss."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    n_batches = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_binary_metrics(all_labels, all_preds)
    return {
        "loss": total_loss / max(1, n_batches),
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "confusion_matrix": metrics.confusion_matrix,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(train_csv: Path,
          val_csv: Path,
          output_dir: Path,
          model_name: str,
          max_length: int,
          batch_size: int,
          learning_rate: float,
          epochs: int,
          weight_decay: float,
          early_stopping_patience: int,
          force_cpu: bool,
          seed: int) -> dict:
    """Full training pipeline. Returns a report dict."""

    # ----- Setup -----
    set_seed(seed)
    print(f"[train] Seed: {seed}")
    print(f"[train] Train CSV: {train_csv}  ({train_csv.exists()})")
    print(f"[train] Val CSV:   {val_csv}  ({val_csv.exists()})")

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")

    # ----- Device -----
    if force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"[train] Device: CPU (forced={force_cpu}, cuda_available={torch.cuda.is_available()})")
    else:
        device = torch.device("cuda")
        print(f"[train] Device: CUDA ({torch.cuda.get_device_name(0)})")

    # ----- Load data -----
    train_df = pd.read_csv(train_csv, encoding="utf-8")
    val_df = pd.read_csv(val_csv, encoding="utf-8")
    print(f"[train] Loaded train: {len(train_df)} rows")
    print(f"[train] Loaded val:   {len(val_df)} rows")

    # ----- Tokenizer + Model -----
    print(f"[train] Loading tokenizer + model: {model_name}")
    print(f"[train] (First run downloads ~268 MB of DistilBERT weights)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    model.to(device)

    # ----- Datasets + DataLoaders -----
    train_dataset = PromptInjectionDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = PromptInjectionDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ----- Optimizer + Scheduler -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # ----- Training loop -----
    print(f"\n[train] Starting training: {epochs} epochs, batch_size={batch_size}")
    print(f"[train]   learning_rate={learning_rate}, weight_decay={weight_decay}")
    print(f"[train]   train batches/epoch: {len(train_loader)}, val batches: {len(val_loader)}\n")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        val_metrics = evaluate_model(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "epoch_time_sec": round(epoch_time, 2),
        })

        print(f"  Epoch {epoch+1}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}  "
              f"({epoch_time:.1f}s)")

        # Early stopping on validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            # Save best checkpoint
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  ✓ New best val_loss — saved to {output_dir}")
        else:
            epochs_without_improvement += 1
            print(f"  · No improvement ({epochs_without_improvement}/{early_stopping_patience})")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[train] Early stopping at epoch {epoch+1} "
                      f"(no improvement for {early_stopping_patience} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\n[train] Training complete in {total_time:.1f}s")
    print(f"[train] Best validation loss: {best_val_loss:.4f}")
    print(f"[train] Model saved to: {output_dir}")

    # ----- Save training report -----
    report = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "hyperparameters": {
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
        },
        "dataset_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
        },
        "device": str(device),
        "history": history,
        "best_val_loss": best_val_loss,
        "total_training_time_sec": round(total_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for prompt-injection detection.")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config.")
    parser.add_argument("--train-csv", type=str, default="data/train/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/validation/validation.csv")
    parser.add_argument("--output-dir", type=str, default="models/distilbert_v1",
                        help="Where to save the trained model.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config batch_size.")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override config learning_rate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bert_cfg = cfg["detection"]["bert"]

    # Resolve paths
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    if not train_csv.is_absolute():
        train_csv = PROJECT_ROOT / train_csv
    if not val_csv.is_absolute():
        val_csv = PROJECT_ROOT / val_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Override hyperparameters
    epochs = args.epochs or bert_cfg["epochs"]
    batch_size = args.batch_size or bert_cfg["batch_size"]
    learning_rate = args.learning_rate or bert_cfg["learning_rate"]

    report = train(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=output_dir,
        model_name=bert_cfg["model_name"],
        max_length=bert_cfg["max_length"],
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        weight_decay=bert_cfg["weight_decay"],
        early_stopping_patience=bert_cfg["early_stopping_patience"],
        force_cpu=bert_cfg.get("train_on_cpu", True),
        seed=args.seed,
    )

    # Save report
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"bert_training_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[train] Training report saved to: {report_path}")


if __name__ == "__main__":
    main()
