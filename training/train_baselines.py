import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from training.train import train_classifier, extract_and_cache_features
from evaluation.metrics import evaluate_predictions

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_experiments():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    train_npz = os.path.join(processed_dir, "train_features.npz")
    val_npz = os.path.join(processed_dir, "val_features.npz")
    test_npz = os.path.join(processed_dir, "test_features.npz")
    
    # Extract features if not cached
    if not (os.path.exists(train_npz) and os.path.exists(val_npz) and os.path.exists(test_npz)):
        print("Cached feature files not found. Initiating fine-tuning & extraction...")
        extract_and_cache_features(config)
        
    test_data = np.load(test_npz)
    device = torch.device(config["training"]["device"])
    
    X_test_vis = torch.tensor(test_data["vis_feats"]).to(device)
    X_test_topo = torch.tensor(test_data["topo_feats"]).to(device)
    X_test_clin = torch.tensor(test_data["clin_feats"]).to(device)
    y_test = test_data["labels"]
    
    # Experiment Matrix configurations
    experiments = {
        "1. Image-only ResNet50 (Baseline 1)": {
            "use_visual": True, "use_topological": False, "use_clinical": False,
            "fusion_type": "simple_concat", "ordinal": False, "model_path": "baseline_1_image_only.pth"
        },
        "2. Topology-only GUDHI (Baseline 2)": {
            "use_visual": False, "use_topological": True, "use_clinical": False,
            "fusion_type": "simple_concat", "ordinal": False, "model_path": "baseline_2_topo_only.pth"
        },
        "3. Clinical-only (Baseline 3)": {
            "use_visual": False, "use_topological": False, "use_clinical": True,
            "fusion_type": "simple_concat", "ordinal": False, "model_path": "baseline_3_clinical_only.pth"
        },
        "4. ResNet50 + Clinical (Baseline 4 / Ablation: No TDA)": {
            "use_visual": True, "use_topological": False, "use_clinical": True,
            "fusion_type": "simple_concat", "ordinal": False, "model_path": "baseline_4_image_clinical.pth"
        },
        "5. ResNet50 + GUDHI (Baseline 5 / Ablation: No Clinical)": {
            "use_visual": True, "use_topological": True, "use_clinical": False,
            "fusion_type": "simple_concat", "ordinal": False, "model_path": "baseline_5_image_topo.pth"
        },
        "6. ResNet50 + GUDHI + Clinical + Concat (Baseline 6 / Ablation: No TAFF)": {
            "use_visual": True, "use_topological": True, "use_clinical": True,
            "fusion_type": "simple_concat", "ordinal": False, "model_path": "baseline_6_concat.pth"
        },
        "7. Proposed Model: ResNet50 + GUDHI + Clinical + TAFF": {
            "use_visual": True, "use_topological": True, "use_clinical": True,
            "fusion_type": "topology_aware", "ordinal": False, "model_path": "proposed_multimodal.pth"
        },
        "8. Proposed Model (Ordinal): TAFF + CORAL": {
            "use_visual": True, "use_topological": True, "use_clinical": True,
            "fusion_type": "topology_aware", "ordinal": True, "model_path": "proposed_ordinal.pth"
        },
        "Proposed Model (Ablation: H0-only)": {
            "use_visual": True, "use_topological": True, "use_clinical": True,
            "fusion_type": "topology_aware", "ordinal": False, "model_path": "proposed_h0_only.pth",
            "topo_ablate": "h0"
        },
        "Proposed Model (Ablation: H0+H1-only)": {
            "use_visual": True, "use_topological": True, "use_clinical": True,
            "fusion_type": "topology_aware", "ordinal": False, "model_path": "proposed_h0_h1_only.pth",
            "topo_ablate": "h0_h1"
        }
    }
    
    results = {}
    test_preds_by_model = {}
    
    for name, exp_config in experiments.items():
        print(f"\nTraining Model: {name}")
        
        # Load and potentially ablate topological features for this run
        t_topo_test = X_test_topo.clone()
        if "topo_ablate" in exp_config:
            if exp_config["topo_ablate"] == "h0":
                # Zero out H1 (indices 6-11) and H2 (indices 12-15)
                t_topo_test[:, 6:] = 0.0
            elif exp_config["topo_ablate"] == "h0_h1":
                # Zero out H2 (indices 12-15)
                t_topo_test[:, 12:] = 0.0
                
        # Train model
        model = train_classifier(config, exp_config, train_npz, val_npz)
        torch.save(model.state_dict(), os.path.join(models_dir, exp_config["model_path"]))
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            t_vis = X_test_vis if exp_config["use_visual"] else None
            t_topo = t_topo_test if exp_config["use_topological"] else None
            t_clin = X_test_clin if exp_config["use_clinical"] else None
            
            test_logits = model(t_vis, t_topo, t_clin)
            test_probs = model.predict_probabilities(test_logits).cpu().numpy()
            test_preds = np.argmax(test_probs, axis=1)
            
            metrics = evaluate_predictions(y_test, test_preds, test_probs)
            
        results[name] = metrics
        test_preds_by_model[name] = test_preds
        
        # Quick console report
        print(f"  F1 Macro: {metrics['f1_macro']:.4f} | Cohen Kappa: {metrics['cohen_kappa_quadratic']:.4f} | MAE: {metrics['mae']:.4f}")

    # Paired Statistical Validation (Proposed Model vs baseline 1 and baseline 6)
    # We compare prediction errors (absolute distance to ground truth) via a paired t-test
    proposed_errors = np.abs(test_preds_by_model["7. Proposed Model: ResNet50 + GUDHI + Clinical + TAFF"] - y_test)
    resnet_errors = np.abs(test_preds_by_model["1. Image-only ResNet50 (Baseline 1)"] - y_test)
    concat_errors = np.abs(test_preds_by_model["6. ResNet50 + GUDHI + Clinical + Concat (Baseline 6 / Ablation: No TAFF)"] - y_test)
    
    # proposed vs resnet
    t_stat_res, p_val_res = stats.ttest_rel(proposed_errors, resnet_errors)
    # proposed vs concat
    t_stat_cat, p_val_cat = stats.ttest_rel(proposed_errors, concat_errors)
    
    statistical_metrics = {
        "resnet_vs_proposed": {"t_statistic": float(t_stat_res), "p_value": float(p_val_res)},
        "concat_vs_proposed": {"t_statistic": float(t_stat_cat), "p_value": float(p_val_cat)}
    }
    
    # 5-Fold Cross-Validation on Proposed Model
    print("\n--- Running 5-Fold Patient-Level Cross-Validation on Proposed Model ---")
    
    # Load all raw splits data to merge back into single dataset
    split_csv = os.path.join(config["data"]["split_dir"], "dataset_splits.csv")
    full_df = pd.read_csv(split_csv)
    
    # Load pre-extracted train, val, test and align them
    train_np = np.load(train_npz)
    val_np = np.load(val_npz)
    test_np = np.load(test_npz)
    
    all_vis = np.concatenate([train_np["vis_feats"], val_np["vis_feats"], test_np["vis_feats"]], axis=0)
    all_topo = np.concatenate([train_np["topo_feats"], val_np["topo_feats"], test_np["topo_feats"]], axis=0)
    all_clin = np.concatenate([train_np["clin_feats"], val_np["clin_feats"], test_np["clin_feats"]], axis=0)
    all_labels = np.concatenate([train_np["labels"], val_np["labels"], test_np["labels"]], axis=0)
    all_pats = full_df["patient_id"].values
    
    gkf = GroupKFold(n_splits=5)
    cv_metrics = []
    
    fold = 1
    for train_idx, val_idx in gkf.split(all_vis, all_labels, groups=all_pats):
        # Package temporary NPZ files for this fold
        fold_train_npz = os.path.join(processed_dir, f"fold_{fold}_train.npz")
        fold_val_npz = os.path.join(processed_dir, f"fold_{fold}_val.npz")
        
        np.savez(fold_train_npz, vis_feats=all_vis[train_idx], topo_feats=all_topo[train_idx], clin_feats=all_clin[train_idx], labels=all_labels[train_idx])
        np.savez(fold_val_npz, vis_feats=all_vis[val_idx], topo_feats=all_topo[val_idx], clin_feats=all_clin[val_idx], labels=all_labels[val_idx])
        
        model_fold = train_classifier(
            config, 
            experiments["7. Proposed Model: ResNet50 + GUDHI + Clinical + TAFF"], 
            fold_train_npz, 
            fold_val_npz
        )
        
        # Evaluate on fold validation split
        model_fold.eval()
        with torch.no_grad():
            f_vis = torch.tensor(all_vis[val_idx]).to(device)
            f_topo = torch.tensor(all_topo[val_idx]).to(device)
            f_clin = torch.tensor(all_clin[val_idx]).to(device)
            
            logits = model_fold(f_vis, f_topo, f_clin)
            probs = model_fold.predict_probabilities(logits).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            fold_metrics = evaluate_predictions(all_labels[val_idx], preds, probs)
            cv_metrics.append(fold_metrics)
            
        # Clean up files
        os.remove(fold_train_npz)
        os.remove(fold_val_npz)
        fold += 1
        
    # Aggregate CV results
    cv_accs = [m["accuracy"] for m in cv_metrics]
    cv_f1s = [m["f1_macro"] for m in cv_metrics]
    cv_kappas = [m["cohen_kappa_quadratic"] for m in cv_metrics]
    cv_maes = [m["mae"] for m in cv_metrics]
    
    cv_summary = {
        "accuracy_mean": float(np.mean(cv_accs)),
        "accuracy_std": float(np.std(cv_accs)),
        "f1_macro_mean": float(np.mean(cv_f1s)),
        "f1_macro_std": float(np.std(cv_f1s)),
        "kappa_mean": float(np.mean(cv_kappas)),
        "kappa_std": float(np.std(cv_kappas)),
        "mae_mean": float(np.mean(cv_maes)),
        "mae_std": float(np.std(cv_maes))
    }
    
    # Save combined outcomes to experimental_results.json
    final_output = {
        "experiments": results,
        "statistical_validation": statistical_metrics,
        "cross_validation": cv_summary
    }
    
    results_path = os.path.join(processed_dir, "experimental_results.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("\n\n================ EXPERIMENT REPORT COMPARISON ================")
    print("| Model Configuration | Accuracy | Bal. Acc. | F1 Macro | Cohen Kappa | MAE |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, metrics in results.items():
        print(f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['f1_macro']:.4f} | {metrics['cohen_kappa_quadratic']:.4f} | {metrics['mae']:.4f} |")
    
    print("\n5-Fold Patient-Level Cross-Validation (Proposed Model):")
    print(f"  Accuracy: {cv_summary['accuracy_mean']:.4%} ± {cv_summary['accuracy_std']:.3f}")
    print(f"  F1 Macro: {cv_summary['f1_macro_mean']:.4f} ± {cv_summary['f1_macro_std']:.3f}")
    print(f"  Cohen Kappa: {cv_summary['kappa_mean']:.4f} ± {cv_summary['kappa_std']:.3f}")
    print(f"  MAE: {cv_summary['mae_mean']:.4f} ± {cv_summary['mae_std']:.3f}")
    print("===============================================================\n")

if __name__ == "__main__":
    run_experiments()
