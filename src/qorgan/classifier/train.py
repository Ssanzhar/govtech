"""Fine-tune the XLM-R multi-task scam classifier and export it for the `xlmr` backend
(D3-1). Class-weighted BCE on both heads (the positive-heavy corpus makes the balanced
`pos_weight` down-weight positives, which already nudges toward low FPR -- CLAUDE.md §3.5);
temperature is fitted on the val split for calibrated confidence.

Model + tokenizer are injectable so smoke tests run a tiny random-config model + fake
tokenizer fully offline; the CLI builds the real `xlm-roberta-base`. The export is
self-describing (`metadata.json` carries base-model name, label space, max length,
threshold, temperature) so `predict.py` can reconstruct and load it without guessing.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qorgan.classifier import calibrate, labels
from qorgan.config import get_config
from qorgan.data.schema import SCAM_RISK_THRESHOLD, Dialogue

_DEFAULT_BASE_MODEL = "xlm-roberta-base"
_DEFAULT_EPOCHS = 5
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_LR = 2e-5
_DEFAULT_MAX_LENGTH = 256
_DEFAULT_TACTIC_THRESHOLD = 0.5
_WEIGHT_DECAY = 0.01
# Gradient-norm clip: keeps the higher fine-tuning LR from destabilising the encoder.
_MAX_GRAD_NORM = 1.0
# Labeled risk >= this is a scam in the binary risk target (matches eval/run.py).
_TRUTH_THRESHOLD = SCAM_RISK_THRESHOLD
# Cap class weights: rare tactics otherwise yield huge pos_weights whose gradient spikes
# collapse the shared encoder to a constant output.
_MAX_POS_WEIGHT = 10.0
# Fraction of total steps spent linearly warming up the LR (stabilises early fine-tuning).
_WARMUP_RATIO = 0.1

_METADATA_FILE = "metadata.json"
_WEIGHTS_FILE = "model.pt"


def pick_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in CI/dev
        return torch.device("cuda")
    return torch.device("cpu")


def _scam_target(dialogue: Dialogue) -> int:
    return 1 if dialogue.label.risk >= _TRUTH_THRESHOLD else 0


def build_targets(
    dialogues: Sequence[Dialogue], label_space: Sequence[str]
) -> tuple[list[int], list[list[float]]]:
    """Binary risk targets + multi-hot tactic matrix aligned to `label_space`."""
    risk_targets = [_scam_target(d) for d in dialogues]
    tactic_matrix = [
        labels.encode_tactics([t.id for t in d.label.tactic_tags], label_space) for d in dialogues
    ]
    return risk_targets, tactic_matrix


def _batches(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _forward_batch(model, tokenizer, texts: Sequence[str], device, max_length: int):
    enc = tokenizer(
        list(texts), return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out["risk_logit"], out["tactic_logits"]


def train_model(
    train_dialogues: Sequence[Dialogue],
    *,
    model: nn.Module,
    tokenizer: Any,
    label_space: Sequence[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
) -> nn.Module:
    """Class-weighted multi-task training loop; returns the trained model."""
    if not train_dialogues:
        raise ValueError("train_dialogues must not be empty")

    from transformers import get_linear_schedule_with_warmup

    risk_targets, tactic_matrix = build_targets(train_dialogues, label_space)
    risk_pos_weight = torch.tensor(_clamp(labels.pos_weight(risk_targets)), device=device)
    tactic_pos_weight = torch.tensor(
        [_clamp(w) for w in labels.tactic_pos_weights(tactic_matrix)], device=device
    )
    risk_loss_fn = nn.BCEWithLogitsLoss(pos_weight=risk_pos_weight)
    tactic_loss_fn = nn.BCEWithLogitsLoss(pos_weight=tactic_pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=_WEIGHT_DECAY)

    num_batches = max(1, (len(train_dialogues) + batch_size - 1) // batch_size)
    total_steps = num_batches * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(_WARMUP_RATIO * total_steps), total_steps
    )

    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in _batches(list(train_dialogues), batch_size):
            texts = [d.transcript() for d in batch]
            risk_tgt = torch.tensor([_scam_target(d) for d in batch], dtype=torch.float, device=device)
            tactic_tgt = torch.tensor(
                [labels.encode_tactics([t.id for t in d.label.tactic_tags], label_space) for d in batch],
                dtype=torch.float,
                device=device,
            )
            optimizer.zero_grad()
            risk_logit, tactic_logit = _forward_batch(model, tokenizer, texts, device, max_length)
            loss = risk_loss_fn(risk_logit, risk_tgt) + tactic_loss_fn(tactic_logit, tactic_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs}  avg_loss={epoch_loss / num_batches:.4f}", flush=True)
    return model


def _clamp(weight: float) -> float:
    return min(weight, _MAX_POS_WEIGHT)


def fit_calibration(
    model: nn.Module,
    val_dialogues: Sequence[Dialogue],
    *,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> float:
    """Fit the risk-head temperature on the val split (defaults to 1.0 if val is empty)."""
    if not val_dialogues:
        return 1.0
    model.eval()
    logits: list[float] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch in _batches(list(val_dialogues), batch_size):
            risk_logit, _ = _forward_batch(model, tokenizer, [d.transcript() for d in batch], device, max_length)
            logits.extend(risk_logit.detach().cpu().reshape(-1).tolist())
            targets.extend(_scam_target(d) for d in batch)
    return calibrate.fit_temperature(logits, targets)


def export_bundle(
    model: nn.Module,
    *,
    base_model: str,
    label_space: Sequence[str],
    max_length: int,
    tactic_threshold: float,
    temperature: float,
    out_dir: Path,
) -> dict:
    """Write `model.pt` + `metadata.json` to `out_dir`; return the metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / _WEIGHTS_FILE)
    metadata = {
        "base_model": base_model,
        "label_space": list(label_space),
        "num_tactics": len(label_space),
        "max_length": max_length,
        "tactic_threshold": tactic_threshold,
        "temperature": temperature,
    }
    (out_dir / _METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_and_export(
    train_dialogues: Sequence[Dialogue],
    val_dialogues: Sequence[Dialogue],
    *,
    model: nn.Module,
    tokenizer: Any,
    base_model: str,
    label_space: Sequence[str],
    out_dir: Path,
    device: torch.device | None = None,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    lr: float = _DEFAULT_LR,
    max_length: int = _DEFAULT_MAX_LENGTH,
    tactic_threshold: float = _DEFAULT_TACTIC_THRESHOLD,
) -> dict:
    """End-to-end: train -> fit calibration on val -> export the self-describing bundle."""
    active_device = device or pick_device()
    train_model(
        train_dialogues,
        model=model,
        tokenizer=tokenizer,
        label_space=label_space,
        device=active_device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
    )
    temperature = fit_calibration(
        model, val_dialogues, tokenizer=tokenizer, device=active_device, batch_size=batch_size, max_length=max_length
    )
    return export_bundle(
        model,
        base_model=base_model,
        label_space=label_space,
        max_length=max_length,
        tactic_threshold=tactic_threshold,
        temperature=temperature,
        out_dir=out_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI (real training)
    """CLI: fine-tune on the processed splits and export to `config.xlmr_model_dir`."""
    from transformers import AutoTokenizer

    from qorgan.classifier.model import build_model
    from qorgan.eval.run import load_split

    cfg = get_config()
    parser = argparse.ArgumentParser(description="Fine-tune + export the XLM-R scam classifier.")
    parser.add_argument("--processed-dir", type=Path, default=cfg.data_dir / "processed")
    parser.add_argument("--out-dir", type=Path, default=cfg.xlmr_model_dir)
    parser.add_argument("--base-model", default=_DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--max-length", type=int, default=_DEFAULT_MAX_LENGTH)
    args = parser.parse_args(argv)

    label_space = labels.default_label_space()
    model = build_model(args.base_model, len(label_space))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    metadata = train_and_export(
        load_split(args.processed_dir, "train"),
        load_split(args.processed_dir, "val"),
        model=model,
        tokenizer=tokenizer,
        base_model=args.base_model,
        label_space=label_space,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
    print(json.dumps(metadata, indent=2))
    print(f"exported -> {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
