import os
import yaml
import json
import argparse
import pandas as pd
from training.xray.train_baselines import train_model

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def run_experiments():
    parser = argparse.ArgumentParser(description="Train visual baselines (ResNet50, EfficientNet, ConvNeXt)")
    parser.add_argument("--sanity", action="store_true", help="Run a quick 1-epoch subset sanity check")
    args = parser.parse_args()
    
    config_path = "configs/config_xray.yaml"
    original_config = load_yaml(config_path)
    
    try:
        if args.sanity:
            print("\n" + "=" * 50)
            print("RUNNING SANITY CHECK (SUBSET MODE, 1 EPOCH)")
            print("=" * 50)
            temp_config = load_yaml(config_path)
            # Override for sanity check
            temp_config["mode"] = "development"
            if "development" not in temp_config:
                temp_config["development"] = {}
            temp_config["development"]["subset_mode"] = True
            temp_config["development"]["epochs"] = 1
            temp_config["development"]["warmup_epochs"] = 0
            write_yaml(config_path, temp_config)
        else:
            print("\n" + "=" * 50)
            print("RUNNING FULL VISUAL BASELINES TRAINING (10 EPOCHS)")
            print("=" * 50)
            temp_config = load_yaml(config_path)
            temp_config["mode"] = "full_training"
            if "full_training" not in temp_config:
                temp_config["full_training"] = {}
            temp_config["full_training"]["subset_mode"] = False
            temp_config["full_training"]["epochs"] = 10
            temp_config["full_training"]["warmup_epochs"] = 2
            write_yaml(config_path, temp_config)
            
        # Target directory
        os.makedirs("results/visual_baselines", exist_ok=True)
        
        # Baselines configurations
        baselines = [
            ("resnet50_visual_baseline", "resnet50"),
            ("efficientnet_visual_baseline", "efficientnet"),
            ("convnext_visual_baseline", "convnext")
        ]
        
        results = {}
        
        for model_name, backbone in baselines:
            metrics_path = f"results/visual_baselines/{model_name}_metrics.json"
            
            # Check if metrics file already exists and load it to skip retraining (unless sanity run)
            if os.path.exists(metrics_path) and not args.sanity:
                print(f"\n[SKIP] Found existing metrics for {model_name} at {metrics_path}. Loading cached results...")
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            else:
                print(f"\nTraining visual baseline: {model_name} with backbone: {backbone}...")
                # We train visual-only (image_only) and keep the test set LOCKED (eval_test=False)
                metrics = train_model(
                    model_name=model_name,
                    fusion_type="image_only",
                    ordinal=False,
                    loss_type="cross_entropy",
                    use_sampler=False,
                    eval_test=False,
                    backbone_type=backbone
                )
                
                # Save visual baseline metrics
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                    
            results[backbone] = metrics
            
        # Save comparison results
        comparison_path_json = "results/visual_baselines/visual_baselines_comparison.json"
        with open(comparison_path_json, "w") as f:
            json.dump(results, f, indent=4)
            
        # Convert to DataFrame and save CSV
        records = []
        for backbone, metrics in results.items():
            records.append({
                "Backbone": backbone,
                "Accuracy": metrics["accuracy"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Macro F1": metrics["f1_macro"],
                "Weighted F1": metrics["f1_weighted"],
                "Quadratic Kappa": metrics["cohen_kappa_quadratic"],
                "MAE": metrics["mae"]
            })
            
        df = pd.DataFrame(records)
        comparison_path_csv = "results/visual_baselines/visual_baselines_comparison.csv"
        df.to_csv(comparison_path_csv, index=False)
        
        print("\n" + "=" * 50)
        print("VISUAL BASELINES SELECTION RESULTS")
        print("=" * 50)
        print(df.to_string(index=False))
        print("=" * 50)
        print(f"Results saved to results/visual_baselines/")
        
    finally:
        # ALWAYS restore original config file
        write_yaml(config_path, original_config)
        print("\nRestored original configuration file.")

if __name__ == "__main__":
    run_experiments()
