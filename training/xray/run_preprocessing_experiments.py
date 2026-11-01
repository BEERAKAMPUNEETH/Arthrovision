import os
import yaml
import json
import pandas as pd
from training.xray.train_baselines import train_model

def load_config():
    with open("configs/config_xray.yaml", "r") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode", "full_training")
    mode_settings = config.get(mode, {})
    for k, v in mode_settings.items():
        config["training"][k] = v
    return config

def save_config(config):
    with open("configs/config_xray.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def run_experiments():
    print("=" * 60)
    print("RUNNING PREPROCESSING EXPERIMENTS (PHASE 6)")
    print("=" * 60)
    
    # Preprocessing configurations to compare
    configs_to_test = [
        {"name": "A: Resize + Normalization", "use_clahe": False, "use_roi": False},
        {"name": "B: Resize + Normalization + CLAHE", "use_clahe": True, "use_roi": False},
        {"name": "C: Resize + Normalization + ROI", "use_clahe": False, "use_roi": True},
        {"name": "D: Resize + Normalization + CLAHE + ROI", "use_clahe": True, "use_roi": True}
    ]
    
    results = []
    
    original_config = load_config()
    
    for c in configs_to_test:
        print(f"\nEvaluating configuration: {c['name']}")
        
        # Load and modify config
        config = load_config()
        config["preprocessing"]["use_clahe"] = c["use_clahe"]
        config["preprocessing"]["use_roi"] = c["use_roi"]
        
        # Force subset_mode: True and epochs: 2 (warmup: 1) for fast preprocessing selection
        config["full_training"]["subset_mode"] = True
        config["full_training"]["epochs"] = 2
        config["full_training"]["warmup_epochs"] = 1
        
        # Set mode to full_training
        config["mode"] = "full_training"
        save_config(config)
        
        # Train ResNet50 baseline with standard Cross Entropy (No weights/samplers yet)
        try:
            metrics = train_model(
                model_name=f"resnet_baseline_temp_{c['use_clahe']}_{c['use_roi']}",
                fusion_type="image_only",
                ordinal=False,
                use_weights=False,
                use_sampler=False,
                eval_test=False
            )
            
            results.append({
                "Configuration": c["name"],
                "use_clahe": c["use_clahe"],
                "use_roi": c["use_roi"],
                "Accuracy": metrics["accuracy"],
                "Balanced Acc": metrics["balanced_accuracy"],
                "Macro F1": metrics["f1_macro"],
                "Weighted F1": metrics["f1_weighted"],
                "Kappa": metrics["cohen_kappa_quadratic"],
                "MAE": metrics["mae"]
            })
            
        except Exception as e:
            print(f"Error during configuration training: {e}")
            import traceback
            traceback.print_exc()
            
    # Restore original config
    save_config(original_config)
    
    # Save preprocessing results to JSON
    os.makedirs("results/full_baseline", exist_ok=True)
    results_path = "results/full_baseline/preprocessing_experiments.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nPreprocessing experiments completed. Results saved to {results_path}")
    
    # Print comparison table
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Determine the best configuration based on Validation F1/Accuracy
    if results:
        best_cfg = max(results, key=lambda x: (x["Macro F1"], x["Accuracy"]))
        print(f"\n[WINNER] Best Preprocessing Configuration based on validation Macro F1/Accuracy: {best_cfg['Configuration']}")
        return best_cfg
    return None

if __name__ == "__main__":
    run_experiments()
