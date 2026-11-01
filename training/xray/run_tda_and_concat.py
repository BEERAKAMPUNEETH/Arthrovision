import os
import json
import pandas as pd
from training.xray.train_baselines import train_model

def run_experiments():
    print("=" * 60)
    print("RUNNING TDA-ONLY & CONCATENATION MODELS (PHASES 7 & 8)")
    print("=" * 60)
    
    results = {}
    
    # 1. Load the ResNet50 baseline metrics from results/full_baseline/resnet_baseline_metrics.json
    baseline_path = "results/full_baseline/resnet_baseline_metrics.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_metrics = json.load(f)
        results["ResNet50 baseline"] = baseline_metrics
        print("Loaded ResNet50 baseline metrics successfully.")
    else:
        print("Warning: ResNet50 baseline metrics file not found. Ensure Phase 3 has run.")
        results["ResNet50 baseline"] = None
        
    # 2. Train Phase 7: TDA-only MLP model
    # (Since it uses no ResNet passes, training is extremely fast, taking seconds on CPU)
    print("\n" + "=" * 50)
    print("TRAINING PHASE 7: TDA-ONLY MLP MODEL")
    print("=" * 50)
    tda_metrics = train_model(
        model_name="tda_only_full",
        fusion_type="tda_only",
        ordinal=False,
        loss_type="cross_entropy",
        use_sampler=False,
        eval_test=False # Lock test set!
    )
    results["TDA-only"] = tda_metrics
    
    # Save TDA-only metrics
    os.makedirs("results/full_tda", exist_ok=True)
    tda_save_path = "results/full_tda/tda_only_metrics.json"
    with open(tda_save_path, "w") as f:
        json.dump(tda_metrics, f, indent=4)
    print(f"Saved TDA-only metrics to {tda_save_path}")
    
    # 3. Train Phase 8: ResNet50 + TDA Concatenation
    # (Uses visual ResNet passes, takes around 3 hours on CPU)
    print("\n" + "=" * 50)
    print("TRAINING PHASE 8: RESNET50 + TDA CONCATENATION MODEL")
    print("=" * 50)
    concat_metrics = train_model(
        model_name="resnet_tda_concat_full",
        fusion_type="concat",
        ordinal=False,
        loss_type="cross_entropy",
        use_sampler=False,
        eval_test=False # Lock test set!
    )
    results["ResNet50 + TDA concat"] = concat_metrics
    
    # Save Concatenation metrics
    os.makedirs("results/resnet_tda_concat", exist_ok=True)
    concat_save_path = "results/resnet_tda_concat/concat_metrics.json"
    with open(concat_save_path, "w") as f:
        json.dump(concat_metrics, f, indent=4)
    print(f"Saved Concatenation metrics to {concat_save_path}")
    
    # 4. Save comparison report
    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE")
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
    
    comparison_save_path = "results/tda_concat_comparison_report.json"
    with open(comparison_save_path, "w") as f:
        json.dump(comparison_rows, f, indent=4)
    print(f"\nSaved comparison report to {comparison_save_path}")

if __name__ == "__main__":
    run_experiments()
