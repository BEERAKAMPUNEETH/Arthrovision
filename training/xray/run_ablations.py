import os
import json
import pandas as pd
from training.xray.train_baselines import train_model

def run_ablations():
    print("=" * 60)
    print("RUNNING ABLATION EXPERIMENTS (PHASE 11)")
    print("=" * 60)
    
    results = {}
    
    # 1. Load prior results
    paths = {
        "Visual-only": "results/full_baseline/resnet_baseline_metrics.json",
        "TDA-only": "results/full_tda/tda_only_metrics.json",
        "Concatenation": "results/resnet_tda_concat/concat_metrics.json",
        "TAFF": "results/full_taff/taff_metrics.json",
        "Ordinal TAFF": "results/full_ordinal/taff_ordinal_metrics.json"
    }
    
    for name, path in paths.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                results[name] = json.load(f)
            print(f"Loaded {name} metrics successfully.")
        else:
            print(f"Warning: {name} metrics file {path} not found.")
            results[name] = None
            
    # 2. Train H0-only Concatenation Model
    # (Takes around 3 hours on CPU)
    print("\n" + "=" * 50)
    print("TRAINING H0-ONLY CONCATENATION MODEL")
    print("=" * 50)
    h0_metrics = train_model(
        model_name="concat_h0_only",
        fusion_type="concat",
        ordinal=False,
        loss_type="cross_entropy",
        use_sampler=False,
        eval_test=False, # Lock test set!
        ablation_type="h0_only"
    )
    results["H0-only Concat"] = h0_metrics
    
    # Save H0-only metrics
    os.makedirs("results/ablation", exist_ok=True)
    h0_path = "results/ablation/h0_only_metrics.json"
    with open(h0_path, "w") as f:
        json.dump(h0_metrics, f, indent=4)
    print(f"Saved H0-only metrics to {h0_path}")
    
    # 3. Train H1-only Concatenation Model
    # (Takes around 3 hours on CPU)
    print("\n" + "=" * 50)
    print("TRAINING H1-ONLY CONCATENATION MODEL")
    print("=" * 50)
    h1_metrics = train_model(
        model_name="concat_h1_only",
        fusion_type="concat",
        ordinal=False,
        loss_type="cross_entropy",
        use_sampler=False,
        eval_test=False, # Lock test set!
        ablation_type="h1_only"
    )
    results["H1-only Concat"] = h1_metrics
    
    # Save H1-only metrics
    h1_path = "results/ablation/h1_only_metrics.json"
    with open(h1_path, "w") as f:
        json.dump(h1_metrics, f, indent=4)
    print(f"Saved H1-only metrics to {h1_path}")
    
    # 4. Save compilation report
    print("\n" + "=" * 60)
    print("ABLATION STUDIES SUMMARY TABLE")
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
    
    report_save_path = "results/ablation/ablation_report.json"
    with open(report_save_path, "w") as f:
        json.dump(comparison_rows, f, indent=4)
    print(f"\nSaved ablation report to {report_save_path}")

if __name__ == "__main__":
    run_ablations()
