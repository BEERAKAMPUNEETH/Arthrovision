import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error, confusion_matrix
from scipy.stats import binom

from feature_extraction.xray.dataset import KneeXrayDataset
from feature_extraction.xray.scaler import XrayTopologicalScaler
from models.xray.classifier import HybridXrayClassifier

def load_config():
    with open("configs/config_xray.yaml", "r") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode", "full_training")
    mode_settings = config.get(mode, {})
    for k, v in mode_settings.items():
        config["training"][k] = v
    return config

def evaluate_on_test():
    print("=" * 60)
    print("FINAL EVALUATION AND STATISTICAL SIGNIFICANCE TESTING ON TEST SET")
    print("=" * 60)
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load test cache data
    test_cache = "data/processed_xray/test_cache.npz"
    train_cache = "data/processed_xray/train_cache.npz"
    if not os.path.exists(test_cache):
        raise FileNotFoundError(f"Test cache {test_cache} not found.")
        
    # Fit scaler on train cached data first
    train_raw = np.load(train_cache, allow_pickle=True)
    tda_scaler = XrayTopologicalScaler()
    tda_scaler.fit(train_raw["tda_features"])
    
    test_ds = KneeXrayDataset(test_cache, tda_scaler=tda_scaler, split="test", tda_only=False)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)
    
    test_lbls = test_ds.labels
    
    # 2. Evaluate ResNet50 Baseline
    baseline_path = "models/xray/resnet_baseline_full.pth"
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint {baseline_path} not found.")
        
    print(f"\nEvaluating ResNet50 baseline on test set using {baseline_path}...")
    baseline_model = HybridXrayClassifier(
        d_tda=16,
        d_tda_emb=config["model"]["topological_embedding_dim"],
        d_vis_emb=config["model"]["visual_embedding_dim"],
        d_fusion=config["model"]["fusion_embedding_dim"],
        num_classes=config["model"]["num_classes"],
        fusion_type="image_only",
        ordinal=False
    ).to(device)
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    baseline_model.eval()
    
    baseline_preds = []
    with torch.no_grad():
        for imgs, tda_feats, _ in test_loader:
            imgs = imgs.to(device)
            tda_feats = tda_feats.to(device)
            logits = baseline_model(imgs, tda_feats)
            probs = baseline_model.predict_probabilities(logits)
            preds = torch.argmax(probs, dim=1)
            baseline_preds.extend(preds.cpu().numpy())
            
    baseline_preds = np.array(baseline_preds)
    
    # 3. Evaluate ResNet50 + TDA Concatenation Model
    concat_path = "models/xray/resnet_tda_concat_full.pth"
    if not os.path.exists(concat_path):
        raise FileNotFoundError(f"Concatenation checkpoint {concat_path} not found.")
        
    print(f"\nEvaluating ResNet50 + TDA Concatenation on test set using {concat_path}...")
    concat_model = HybridXrayClassifier(
        d_tda=16,
        d_tda_emb=config["model"]["topological_embedding_dim"],
        d_vis_emb=config["model"]["visual_embedding_dim"],
        d_fusion=config["model"]["fusion_embedding_dim"],
        num_classes=config["model"]["num_classes"],
        fusion_type="concat",
        ordinal=False
    ).to(device)
    concat_model.load_state_dict(torch.load(concat_path, map_location=device))
    concat_model.eval()
    
    concat_preds = []
    with torch.no_grad():
        for imgs, tda_feats, _ in test_loader:
            imgs = imgs.to(device)
            tda_feats = tda_feats.to(device)
            logits = concat_model(imgs, tda_feats)
            probs = concat_model.predict_probabilities(logits)
            preds = torch.argmax(probs, dim=1)
            concat_preds.extend(preds.cpu().numpy())
            
    concat_preds = np.array(concat_preds)
    
    # 4. Compute Test Metrics
    def get_metrics(lbls, preds):
        return {
            "accuracy": float(accuracy_score(lbls, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(lbls, preds)),
            "f1_macro": float(f1_score(lbls, preds, average='macro')),
            "f1_weighted": float(f1_score(lbls, preds, average='weighted')),
            "cohen_kappa_quadratic": float(cohen_kappa_score(lbls, preds, weights='quadratic')),
            "mae": float(mean_absolute_error(lbls, preds)),
            "confusion_matrix": confusion_matrix(lbls, preds).tolist()
        }
        
    baseline_metrics = get_metrics(test_lbls, baseline_preds)
    concat_metrics = get_metrics(test_lbls, concat_preds)
    
    # 5. Perform McNemar's Significance Test
    # Contingency Table:
    #                 Concat Correct    Concat Incorrect
    # Base Correct         a                 b
    # Base Incorrect       c                 d
    correct_base = (baseline_preds == test_lbls)
    correct_concat = (concat_preds == test_lbls)
    
    a = np.sum(correct_base & correct_concat)
    b = np.sum(correct_base & ~correct_concat)
    c = np.sum(~correct_base & correct_concat)
    d = np.sum(~correct_base & ~correct_concat)
    
    contingency_table = [[int(a), int(b)], [int(c), int(d)]]
    print(f"\nContingency Table (McNemar):")
    print(f"  Both Correct (a):             {a}")
    print(f"  Base Correct / Concat Wrong:  {b}")
    print(f"  Base Wrong / Concat Correct:  {c}")
    print(f"  Both Incorrect (d):           {d}")
    
    # Run McNemar exact binomial test
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        p_value = 2.0 * binom.cdf(k, n_discordant, 0.5)
        p_value = min(1.0, float(p_value))
        
    print(f"  McNemar exact p-value: {p_value:.6f}")
    
    # 6. Save report
    report = {
        "baseline_test_metrics": baseline_metrics,
        "concat_test_metrics": concat_metrics,
        "mcnemar_contingency_table": contingency_table,
        "p_value": p_value,
        "is_statistically_significant": bool(p_value < 0.05)
    }
    
    os.makedirs("results", exist_ok=True)
    report_path = "results/test_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 50)
    print("Metric               | ResNet50 Baseline | ResNet50 + TDA Concat")
    print("-" * 65)
    print(f"Accuracy             | {baseline_metrics['accuracy']:.2%}           | {concat_metrics['accuracy']:.2%}")
    print(f"Balanced Accuracy    | {baseline_metrics['balanced_accuracy']:.2%}           | {concat_metrics['balanced_accuracy']:.2%}")
    print(f"Macro F1-Score       | {baseline_metrics['f1_macro']:.4f}             | {concat_metrics['f1_macro']:.4f}")
    print(f"Quadratic Kappa      | {baseline_metrics['cohen_kappa_quadratic']:.4f}             | {concat_metrics['cohen_kappa_quadratic']:.4f}")
    print(f"MAE                  | {baseline_metrics['mae']:.4f}             | {concat_metrics['mae']:.4f}")
    print("-" * 65)
    print(f"Statistical Significance (p < 0.05): {report['is_statistically_significant']} (p = {p_value:.6f})")
    print(f"\nFull report saved to {report_path}")

if __name__ == "__main__":
    evaluate_on_test()
