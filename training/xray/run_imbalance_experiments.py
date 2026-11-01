import os
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from training.xray.train_baselines import train_model

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def run_experiments():
    parser = argparse.ArgumentParser(description="Run class imbalance experiments using the strongest visual backbone (ResNet50)")
    parser.add_argument("--sanity", action="store_true", help="Run a quick 1-epoch subset sanity check")
    args = parser.parse_args()
    
    config_path = "configs/config_xray.yaml"
    original_config = load_yaml(config_path)
    
    # Target directory
    os.makedirs("results/class_imbalance", exist_ok=True)
    
    # 1. Setup configurations to compare
    configs_to_test = [
        {
            "name": "A: Standard Cross Entropy",
            "loss_type": "cross_entropy",
            "use_sampler": False
        },
        {
            "name": "B: Class-weighted Cross Entropy",
            "loss_type": "weighted_cross_entropy",
            "use_sampler": False
        },
        {
            "name": "C: Weighted Random Sampler",
            "loss_type": "cross_entropy",
            "use_sampler": True
        },
        {
            "name": "D: Focal Loss",
            "loss_type": "focal_loss",
            "use_sampler": False
        }
    ]
    
    try:
        if args.sanity:
            print("\n" + "=" * 50)
            print("RUNNING SANITY CHECK (SUBSET MODE, 1 EPOCH)")
            print("=" * 50)
            temp_config = load_yaml(config_path)
            temp_config["mode"] = "development"
            if "development" not in temp_config:
                temp_config["development"] = {}
            temp_config["development"]["subset_mode"] = True
            temp_config["development"]["epochs"] = 1
            temp_config["development"]["warmup_epochs"] = 0
            write_yaml(config_path, temp_config)
        else:
            print("\n" + "=" * 50)
            print("RUNNING FULL CLASS IMBALANCE EXPERIMENTS (10 EPOCHS)")
            print("=" * 50)
            temp_config = load_yaml(config_path)
            temp_config["mode"] = "full_training"
            if "full_training" not in temp_config:
                temp_config["full_training"] = {}
            temp_config["full_training"]["subset_mode"] = False
            temp_config["full_training"]["epochs"] = 10
            temp_config["full_training"]["warmup_epochs"] = 2
            write_yaml(config_path, temp_config)
            
        results = {}
        
        for c in configs_to_test:
            # We map visual baseline results if it was standard cross entropy and no sampler
            # Wait! Since we already trained resnet50_visual_baseline with standard cross entropy and no sampler on the full dataset,
            # we can check if its metrics file exists at results/visual_baselines/resnet50_visual_baseline_metrics.json and reuse it!
            # This is a huge optimization that saves 50 minutes of CPU training!
            # However, for a sanity check, we always train all configurations.
            
            model_name = f"resnet50_imbalance_{c['loss_type']}_{c['use_sampler']}"
            metrics_path = f"results/class_imbalance/{model_name}_metrics.json"
            
            # Reusing baseline if matching
            if c["loss_type"] == "cross_entropy" and not c["use_sampler"] and not args.sanity:
                baseline_path = "results/visual_baselines/resnet50_visual_baseline_metrics.json"
                if os.path.exists(baseline_path):
                    print(f"\n[REUSE] Reusing results from visual baseline for {c['name']}...")
                    with open(baseline_path, "r") as f:
                        metrics = json.load(f)
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=4)
                    results[c["name"]] = metrics
                    continue
            
            # Check for resumption
            if os.path.exists(metrics_path) and not args.sanity:
                print(f"\n[SKIP] Found existing metrics for {c['name']} at {metrics_path}. Loading cached results...")
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            else:
                print(f"\nEvaluating configuration: {c['name']}...")
                metrics = train_model(
                    model_name=model_name,
                    fusion_type="image_only",
                    ordinal=False,
                    loss_type=c["loss_type"],
                    use_sampler=c["use_sampler"],
                    eval_test=False,
                    backbone_type="resnet50"
                )
                
                # Save individual metrics
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                    
            results[c["name"]] = metrics
            
        # Compile comparison results
        comparison_path_json = "results/class_imbalance/class_imbalance_comparison.json"
        with open(comparison_path_json, "w") as f:
            json.dump(results, f, indent=4)
            
        # Convert to DataFrame and save CSV
        records = []
        for name, metrics in results.items():
            records.append({
                "Strategy": name,
                "Accuracy": metrics["accuracy"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Macro F1": metrics["f1_macro"],
                "Weighted F1": metrics["f1_weighted"],
                "Quadratic Kappa": metrics["cohen_kappa_quadratic"],
                "MAE": metrics["mae"]
            })
            
        df = pd.DataFrame(records)
        comparison_path_csv = "results/class_imbalance/class_imbalance_comparison.csv"
        df.to_csv(comparison_path_csv, index=False)
        
        print("\n" + "=" * 50)
        print("CLASS IMBALANCE COMPARISON RESULTS")
        print("=" * 50)
        print(df.to_string(index=False))
        print("=" * 50)
        print(f"Results saved to results/class_imbalance/")
        
    finally:
        # ALWAYS restore original config file
        write_yaml(config_path, original_config)
        print("\nRestored original configuration file.")

if __name__ == "__main__":
    run_experiments()
