"""
Reusable train loop for BiLSTMBaseline. Every run gets its own timestamped
folder under runs/ with config snapshot + metrics.json + best checkpoint --
minimal experiment tracking without extra infra (MLflow/DVC can replace this
later if the project grows past notebook-scale).
"""
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, EARLY_STOP_PATIENCE, EPOCHS, LR, RUNS_DIR, SEED


def _json_default(o):
    """json.dump default= hook -- numpy scalars (e.g. participant_id as
    np.int64 from a pandas/numpy array) aren't natively JSON serializable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for feat, label in loader:
            feat, label = feat.to(device), label.to(device)
            logits, _ = model(feat)
            loss = criterion(logits, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * feat.size(0)
            correct += (logits.argmax(-1) == label).sum().item()
            n += feat.size(0)

    return total_loss / n, correct / n


def train_baseline(model, train_ds, val_ds, run_name: str, device: str = "cpu",
                    epochs: int = EPOCHS, lr: float = LR, batch_size: int = BATCH_SIZE,
                    patience: int = EARLY_STOP_PATIENCE, extra_config: dict = None):
    set_seed()
    run_dir = RUNS_DIR / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as fh:
        json.dump({
            "run_name": run_name, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "patience": patience,
            **(extra_config or {}),
        }, fh, indent=2, default=_json_default)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    history = []
    best_val_acc, best_epoch, bad_epochs = 0.0, -1, 0

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f} acc {train_acc:.3f}  "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, bad_epochs = val_acc, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch} (no val improvement in {patience} epochs)")
                break

    with open(run_dir / "metrics.json", "w") as fh:
        json.dump({"history": history, "best_val_acc": best_val_acc, "best_epoch": best_epoch},
                   fh, indent=2, default=_json_default)

    return run_dir, best_val_acc
