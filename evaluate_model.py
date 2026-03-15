#!/usr/bin/env python3
"""
evaluate_model.py — Generate confusion matrix and per-class metrics
===================================================================
Runs the trained DINOv2 classifier on cached val features and produces
a detailed evaluation report with confusion matrix, precision, recall, F1.
"""

import json
import numpy as np
from pathlib import Path

FEATURES_DIR = Path(__file__).parent / "results" / "cached_features"
RESULTS_DIR = Path(__file__).parent / "results" / "dino_v2_classify"
CLASSES = ["healthy_vegetation", "stressed_soil"]


def load_and_evaluate():
    """Load cached val features and best model, compute metrics."""
    # Load cached features
    val_features = np.load(FEATURES_DIR / "val_features.npy")
    val_labels = np.load(FEATURES_DIR / "val_labels.npy")
    
    print(f"  Val features: {val_features.shape}")
    print(f"  Val labels:   {val_labels.shape}")
    print(f"  Classes: {CLASSES}")
    print(f"  Distribution: {dict(zip(CLASSES, [int((val_labels == i).sum()) for i in range(len(CLASSES))]))}")
    
    # Load model weights (just the linear layer)
    import torch
    checkpoint = torch.load(RESULTS_DIR / "best_model.pth", map_location="cpu")
    
    # Extract head weights from full model state dict
    state = checkpoint["model_state_dict"]
    W = state["head.1.weight"].numpy()  # [2, 384]
    b = state["head.1.bias"].numpy()    # [2]
    
    print(f"\n  Model epoch: {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%")
    print(f"  Weight shape: {W.shape}, bias shape: {b.shape}")
    
    # Compute predictions (simple linear: logits = X @ W.T + b)
    logits = val_features @ W.T + b  # [N, 2]
    predictions = np.argmax(logits, axis=1)
    confidences = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
    
    # Confusion matrix
    num_classes = len(CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(val_labels, predictions):
        cm[int(true)][int(pred)] += 1
    
    # Per-class metrics
    metrics = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(num_classes)) - tp
        fn = sum(cm[i][j] for j in range(num_classes)) - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(tp + fn),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }
    
    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    
    # Mean confidence for correct and incorrect predictions
    correct_mask = predictions == val_labels
    mean_conf_correct = float(confidences[correct_mask].max(axis=1).mean())
    mean_conf_incorrect = float(confidences[~correct_mask].max(axis=1).mean()) if (~correct_mask).any() else 0
    
    # Print results
    print("\n" + "═" * 62)
    print("  CONFUSION MATRIX")
    print("═" * 62)
    print(f"\n  {'':>25s}  Predicted")
    print(f"  {'':>25s}  {'Healthy':>10s}  {'Stressed':>10s}")
    print(f"  {'─'*50}")
    print(f"  Actual  Healthy      {cm[0][0]:>10d}  {cm[0][1]:>10d}")
    print(f"  Actual  Stressed     {cm[1][0]:>10d}  {cm[1][1]:>10d}")
    
    print(f"\n{'═' * 62}")
    print("  PER-CLASS METRICS")
    print("═" * 62)
    print(f"\n  {'Class':<25s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'─'*65}")
    for cls in CLASSES:
        m = metrics[cls]
        print(f"  {cls:<25s} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10d}")
    
    print(f"\n  Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Mean confidence (correct):   {mean_conf_correct:.4f}")
    print(f"  Mean confidence (incorrect): {mean_conf_incorrect:.4f}")
    print(f"  Total val samples: {cm.sum()}")
    
    # Save results
    report = {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "per_class": metrics,
        "mean_confidence_correct": round(mean_conf_correct, 4),
        "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
        "total_samples": int(cm.sum()),
    }
    
    report_path = RESULTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — Model Evaluation Report                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    load_and_evaluate()
