import os
import json
import pandas as pd
from training.xray.train_baselines import train_model

def run_experiments():
    print("=" * 60)
    print("RUNNING PROPOSED TAFF & ORDINAL TAFF MODELS (PHASES 9 & 10)")
    print("=" * 60)
    
    results = {}
    
    # 1. Load prior validation metrics
    baseline_path = "results/full_baseline/resnet_baseline_metrics.json"
    concat_path = "results/resnet_tda_concat/concat_metrics.json"
    
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            results["ResNet50"] = json.load(f)
        print("Loaded ResNet50 baseline metrics.")
    else:
        print("Warning: ResNet50 baseline metrics not found.")
        results["ResNet50"] = None
        
    if os.path.exists(concat_path):
        with open(concat_path, "r") as f:
            results["ResNet50 + TDA"] = json.load(f)
        print("Loaded ResNet50 + TDA concatenation metrics.")
    else:
        print("Warning: ResNet50 + TDA metrics not found.")
        results["ResNet50 + TDA"] = None
        
    # 2. Train Phase 9: Standard TAFF Model (Fusion: taff, Ordinal: False)
    # (Takes around 3 hours on CPU)
    print("\n" + "=" * 50)
    print("TRAINING PHASE 9: PROPOSED TAFF MODEL")
    print("=" * 50)
    taff_metrics = train_model(
        model_name="taff_proposed_full",
        fusion_type="taff",
        ordinal=False,
        loss_type="cross_entropy",
        use_sampler=False,
        eval_test=False # Lock test set!
    )
    results["TAFF"] = taff_metrics
    
    # Save TAFF metrics
    os.makedirs("results/full_taff", exist_ok=True)
    taff_save_path = "results/full_taff/taff_metrics.json"
    with open(taff_save_path, "w") as f:
        json.dump(taff_metrics, f, indent=4)
    print(f"Saved TAFF metrics to {taff_save_path}")
    
    # 3. Train Phase 10: Ordinal TAFF Model (Fusion: taff, Ordinal: True)
    # (Takes around 3 hours on CPU)
    print("\n" + "=" * 50)
    print("TRAINING PHASE 10: ORDINAL TAFF MODEL")
    print("=" * 50)
    taff_ordinal_metrics = train_model(
        model_name="taff_ordinal_proposed_full",
        fusion_type="taff",
        ordinal=True,
        loss_type="cross_entropy", # Losses are overridden inside train_model when ordinal=True (coral_loss)
        use_sampler=False,
        eval_test=False # Lock test set!
    )
    results["Ordinal TAFF"] = taff_ordinal_metrics
    
    # Save Ordinal TAFF metrics
    os.makedirs("results/full_ordinal", exist_ok=True)
    ordinal_save_path = "results/full_ordinal/taff_ordinal_metrics.json"
    with open(ordinal_save_path, "w") as f:
        json.dump(taff_ordinal_metrics, f, indent=4)
    print(f"Saved Ordinal TAFF metrics to {ordinal_save_path}")
    
    # 4. Save comparison report
    print("\n" + "=" * 60)
    print("FINAL VALIDATION COMPARISON TABLE")
    print("=" * 60)
    
    comparison_rows = []
    for model_name, metrics in results.items():
        if metrics is not None:
            comparison_rows.append({
                "Model": model_name,
                "Accuracy": f"{metrics['accuracy']:.2%}",
                "Balanced Accuracy": f"{metrics['balanced_accuracy']:.2%}",
                "Macro F1": f"{metrics['f1_macro']:.4f}",
                "Kappa": f"{metrics['cohen_kappa_quadratic']:.4f}",
                "MAE": f"{metrics['mae']:.4f}"
            })
            
    df_comparison = pd.DataFrame(comparison_rows)
    print(df_comparison.to_string(index=False))
    
    # Save to JSON
    json_path = "results/taff_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison_rows, f, indent=4)
    print(f"\nSaved comparison JSON report to {json_path}")
    
    # Save to CSV
    csv_path = "results/taff_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV report to {csv_path}")

if __name__ == "__main__":
    run_experiments()
