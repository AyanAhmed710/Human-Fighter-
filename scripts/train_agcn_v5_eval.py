"""
Standalone, fresh run of the CURRENT src/models/agcn.py (v5: joint+bone+
real-MediaPipe-Hands-curl+elbow, 578,352 params) -- mirrors
notebooks/agcn/01_2s_agcn.ipynb cells 1/3/5/7/9/10 exactly, but run directly
so the result is verifiably from the code as it exists right now, not a
possibly-stale notebook cell output (notebook's cell 7/9 outputs had
execution_count=None -- edited since last actually run, not trustworthy).

Usage: python scripts/train_agcn_v5_eval.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import ACTIONS, SEQ_LEN
from src.data.dataset import ClipSequenceDataset, load_labels, split_dataframe
from src.data.graph_dataset import build_graph_cache
from src.evaluate import evaluate_dataset
from src.models.agcn import TwoStreamAGCN
from src.train import train_baseline

DEVICE = "cpu"


def main():
    t0 = time.time()
    df = load_labels()
    train_df, val_df, test_df = split_dataframe(df)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_paths = build_graph_cache(train_df, split="train", augment=True)
    val_paths = build_graph_cache(val_df, split="val", augment=False)
    test_paths = build_graph_cache(test_df, split="test", augment=False)
    train_ds = ClipSequenceDataset(train_paths)
    val_ds = ClipSequenceDataset(val_paths)
    test_ds = ClipSequenceDataset(test_paths)

    sample_feat, _ = train_ds[0]
    print(f"train: {len(train_ds)} (incl. augmented) | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"feat shape: {tuple(sample_feat.shape)}")
    assert tuple(sample_feat.shape) == (SEQ_LEN, 22, 6), "shape mismatch -- stale cache"

    model = TwoStreamAGCN(base_channels=32, num_classes=3, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    run_dir, best_val_acc = train_baseline(
        model, train_ds, val_ds, run_name="agcn_v5_elbow", device=DEVICE,
        extra_config={"base_channels": 32, "total_packed_nodes": 22, "seq_len": SEQ_LEN,
                      "variant": "v5_elbow"},
    )
    print(f"\nbest val acc: {best_val_acc:.3f}  |  run saved to {run_dir}")
    print(f"learned stream weights (joint, bone, hand, elbow): "
          f"{model.joint_w.item():.3f}, {model.bone_w.item():.3f}, "
          f"{model.hand_w.item():.3f}, {model.elbow_w.item():.3f}")

    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))
    model.to(DEVICE)
    model.eval()
    report, cm = evaluate_dataset(model, test_ds, device=DEVICE, class_names=ACTIONS)
    print("\n=== 2s-AGCN v5 -- test (deterministic) ===")
    print(pd.DataFrame(report).T)
    print(cm)

    print(f"\ntotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
