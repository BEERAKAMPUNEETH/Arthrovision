import os
import json
import pandas as pd
from training.xray.train_baselines import train_model

def run_full_baseline():
    print("=" * 60)
    print("RUNNING STANDARDIZED FULL-DATA RESNET50 BASELINE (PHASE 3)")
    print("=" * 60)
    
    # 1. Train the standard image-only baseline on the full dataset
    # We set eval_test=False to lock the test set completely!
    metrics = train_model(
        model_name="resnet_baseline_full",
        fusion_type="image_only",
        ordinal=False,
        use_weights=False,  # No class-weights yet to establish standard cross-entropy baseline
        use_sampler=False,  # No samplers yet
        eval_test=False     # Lock test set!
    )
    
    # Save validation metrics to full_baseline folder
    os.makedirs("results/full_baseline", exist_ok=True)
    results_path = "results/full_baseline/resnet_baseline_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nBaseline training complete. Metrics saved to {results_path}")
    print("\n" + "=" * 50)
    print("BASELINE VALIDATION PERFORMANCE")
    print("=" * 50)
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"  Quadratic Kappa: {metrics['cohen_kappa_quadratic']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

if __name__ == "__main__":
    run_full_baseline()
