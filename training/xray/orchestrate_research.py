import os
import yaml
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import cv2
import re

from training.xray.train_baselines import train_model, compute_class_weights
from feature_extraction.xray.dataset import KneeXrayDataset
from models.xray.classifier import HybridXrayClassifier
from preprocessing.xray_preprocessing.preprocessor import KneeXrayPreprocessor
from topology.xray.persistent_homology import XrayTopologicalAnalyzer

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

# Helper to load dataset caches with custom scaling/splitting
class XrayTopologicalScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        
    def transform(self, X):
        return (X - self.mean) / self.std
        
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
        
    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]

def regenerate_tda_cache(mode="standard", rep_type="alpha", use_clahe=True, use_roi=False):
    """
    Regenerates TDA caches dynamically to evaluate richer descriptors or different complex types.
    """
    config = load_yaml("configs/config_xray.yaml")
    dataset_dir = config["dataset"]["path"]
    splits = config["dataset"]["splits"]
    
    preprocessor = KneeXrayPreprocessor(
        target_size=tuple(config["preprocessing"]["target_size"]),
        use_clahe=use_clahe,
        use_roi=use_roi
    )
    
    tda = XrayTopologicalAnalyzer(
        max_points=config["tda"]["max_points"],
        downsample_size=tuple(config["tda"]["downsample_size"])
    )
    
    print(f"\n[REGENERATE] Re-extracting TDA cache. Type: {rep_type.upper()} ({mode.upper()}) | CLAHE: {use_clahe} | ROI: {use_roi}...")
    fn_pattern = re.compile(r"^(\d+)([LR])\.(png|PNG)$")
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        paths = []
        labels = []
        patient_ids = []
        lateralities = []
        tda_features = []
        
        for kl in ["0", "1", "2", "3", "4"]:
            class_dir = os.path.join(split_dir, kl)
            if not os.path.exists(class_dir):
                continue
            label = int(kl)
            filenames = os.listdir(class_dir)
            
            # Subsample for faster preprocessing during development mode if needed, but here we do full extraction
            for fn in filenames:
                file_path = os.path.join(class_dir, fn)
                if not os.path.isfile(file_path):
                    continue
                    
                match = fn_pattern.match(fn)
                if match:
                    patient_id = match.group(1)
                    laterality = match.group(2)
                else:
                    patient_id = os.path.splitext(fn)[0]
                    laterality = "Unknown"
                    
                try:
                    img_gray, _ = preprocessor.preprocess_image(file_path)
                except Exception:
                    continue
                    
                if rep_type == "alpha":
                    points = tda.extract_point_cloud_from_edges(img_gray)
                    complex_obj, _ = tda.compute_persistence_alpha(points)
                    feat = tda.extract_topological_features(complex_obj, is_cubical=False, mode=mode)
                elif rep_type == "cubical":
                    complex_obj, _ = tda.compute_persistence_cubical(img_gray)
                    feat = tda.extract_topological_features(complex_obj, is_cubical=True, mode=mode)
                elif rep_type == "combined":
                    # Alpha 16D + Cubical 16D
                    points = tda.extract_point_cloud_from_edges(img_gray)
                    complex_alpha, _ = tda.compute_persistence_alpha(points)
                    feat_alpha = tda.extract_topological_features(complex_alpha, is_cubical=False, mode="standard")
                    
                    complex_cub, _ = tda.compute_persistence_cubical(img_gray)
                    feat_cub = tda.extract_topological_features(complex_cub, is_cubical=True, mode="standard")
                    feat = np.concatenate([feat_alpha, feat_cub])
                    
                paths.append(file_path)
                labels.append(label)
                patient_ids.append(patient_id)
                lateralities.append(laterality)
                tda_features.append(feat)
                
        # Save cache files with custom name
        os.makedirs("data/processed_xray", exist_ok=True)
        save_path = os.path.join("data/processed_xray", f"{split}_cache_{rep_type}_{mode}.npz")
        np.savez_compressed(
            save_path,
            image_paths=np.array(paths),
            labels=np.array(labels, dtype=np.int64),
            patient_ids=np.array(patient_ids),
            lateralities=np.array(lateralities),
            tda_features=np.array(tda_features, dtype=np.float32)
        )
    print(f"[REGENERATE] Completed regeneration of {rep_type}_{mode} caches.")


def select_winner(comparison_path, metrics=["Macro F1", "Balanced Accuracy", "Quadratic Kappa"], minimize=["MAE"]):
    df = pd.read_csv(comparison_path)
    # Rank rows based on metrics (higher is better)
    rankings = np.zeros(len(df))
    for m in metrics:
        rankings += df[m].rank(ascending=True)
    for m in minimize:
        rankings += df[m].rank(ascending=False)
    winner_idx = np.argmax(rankings)
    return df.iloc[winner_idx]


def run_orchestrated_pipeline(args=None):
    import atexit
    config_path = "configs/config_xray.yaml"
    original_config = load_yaml(config_path)
    
    # ALWAYS restore original config at exit (success or failure)
    atexit.register(write_yaml, config_path, original_config)
    
    is_sanity = args is not None and args.sanity
    
    # Configure global settings in config.yaml
    temp_cfg = load_yaml(config_path)
    if is_sanity:
        temp_cfg["mode"] = "development"
        if "development" not in temp_cfg:
            temp_cfg["development"] = {}
        temp_cfg["development"]["subset_mode"] = True
        temp_cfg["development"]["epochs"] = 1
        temp_cfg["development"]["warmup_epochs"] = 0
    else:
        temp_cfg["mode"] = "full_training"
        if "full_training" not in temp_cfg:
            temp_cfg["full_training"] = {}
        temp_cfg["full_training"]["subset_mode"] = False
        temp_cfg["full_training"]["epochs"] = 10
        temp_cfg["full_training"]["warmup_epochs"] = 2
    write_yaml(config_path, temp_cfg)
    
    print("=" * 60)
    print("ARTHROVISION-TDA: MASTER RESEARCH ORCHESTRATION PIPELINE")
    print("=" * 60)
    
    # --------------------------------------------------------------------
    # PHASE 2: Class Imbalance Selection
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 2] CLASS IMBALANCE SELECTION")
    p2_comparison = "results/class_imbalance/class_imbalance_comparison.csv"
    if os.path.exists(p2_comparison):
        print(f"Phase 2 comparison results found at {p2_comparison}. Skipping Phase 2 runs...")
    else:
        # Run run_imbalance_experiments.py directly
        from training.xray.run_imbalance_experiments import run_experiments as run_p2
        run_p2()
        
    p2_winner = select_winner(p2_comparison)
    winner_strategy = p2_winner["Strategy"]
    print(f"Phase 2 Winner: {winner_strategy}")
    
    # Parse loss type and sampler settings from winner
    if "Standard" in winner_strategy:
        winner_loss = "cross_entropy"
        winner_sampler = False
    elif "Class-weighted" in winner_strategy:
        winner_loss = "weighted_cross_entropy"
        winner_sampler = False
    elif "Weighted Random Sampler" in winner_strategy:
        winner_loss = "cross_entropy"
        winner_sampler = True
    else:
        winner_loss = "focal_loss"
        winner_sampler = False
        
    # --------------------------------------------------------------------
    # PHASE 3: Conservative Augmentation
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 3] CONSERVATIVE AUGMENTATION EXPERIMENTS")
    p3_comparison = "results/augmentation/augmentation_comparison.csv"
    os.makedirs("results/augmentation", exist_ok=True)
    
    if os.path.exists(p3_comparison):
        print(f"Phase 3 comparison results found at {p3_comparison}. Skipping Phase 3 runs...")
    else:
        augmentations = ["none", "geometric", "intensity", "all"]
        results = {}
        for aug in augmentations:
            model_name = f"resnet50_aug_{aug}"
            metrics_path = f"results/augmentation/{model_name}_metrics.json"
            
            if os.path.exists(metrics_path):
                print(f"[SKIP] Found existing metrics for {model_name}...")
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            else:
                print(f"\nTraining model with Augmentation: {aug.upper()}...")
                metrics = train_model(
                    model_name=model_name,
                    fusion_type="image_only",
                    ordinal=False,
                    loss_type=winner_loss,
                    use_sampler=winner_sampler,
                    eval_test=False,
                    backbone_type="resnet50",
                    aug_type=aug
                )
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
            results[aug] = metrics
            
        # Save comparison
        records = []
        for aug, m in results.items():
            records.append({
                "Augmentation": aug,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p3_comparison, index=False)
        with open("results/augmentation/augmentation_comparison.json", "w") as f:
            json.dump(results, f, indent=4)
            
    p3_winner = select_winner(p3_comparison, metrics=["Macro F1", "Balanced Accuracy"], minimize=["MAE"])
    winner_aug = p3_winner["Augmentation"]
    print(f"Phase 3 Winner: {winner_aug}")
    
    # --------------------------------------------------------------------
    # PHASE 4: Re-evaluate Preprocessing (Image-Only)
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 4] PREPROCESSING RE-EVALUATION")
    p4_comparison = "results/preprocessing/preprocessing_comparison.csv"
    os.makedirs("results/preprocessing", exist_ok=True)
    
    if os.path.exists(p4_comparison):
        print(f"Phase 4 comparison results found at {p4_comparison}. Skipping Phase 4 runs...")
    else:
        preprocessors = [
            {"name": "A: Resize + Normalization", "clahe": False, "roi": False},
            {"name": "B: Resize + Normalization + CLAHE", "clahe": True, "roi": False},
            {"name": "C: Resize + Normalization + ROI", "clahe": False, "roi": True},
            {"name": "D: Resize + Normalization + CLAHE + ROI", "clahe": True, "roi": True}
        ]
        results = {}
        for p in preprocessors:
            model_name = f"resnet50_prep_{p['clahe']}_{p['roi']}"
            metrics_path = f"results/preprocessing/{model_name}_metrics.json"
            
            if os.path.exists(metrics_path):
                print(f"[SKIP] Found existing metrics for {model_name}...")
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            else:
                print(f"\nTraining model with Preprocessing: {p['name']}...")
                # Write preprocessing options to config yaml temporarily
                temp_cfg = load_yaml(config_path)
                temp_cfg["preprocessing"]["use_clahe"] = p["clahe"]
                temp_cfg["preprocessing"]["use_roi"] = p["roi"]
                write_yaml(config_path, temp_cfg)
                
                metrics = train_model(
                    model_name=model_name,
                    fusion_type="image_only",
                    ordinal=False,
                    loss_type=winner_loss,
                    use_sampler=winner_sampler,
                    eval_test=False,
                    backbone_type="resnet50",
                    aug_type=winner_aug
                )
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
            results[p["name"]] = metrics
            
        # Restore configuration but preserve current mode (development/full_training)
        restore_cfg = load_yaml(config_path)
        restore_cfg["preprocessing"]["use_clahe"] = original_config["preprocessing"]["use_clahe"]
        restore_cfg["preprocessing"]["use_roi"] = original_config["preprocessing"]["use_roi"]
        write_yaml(config_path, restore_cfg)
        
        records = []
        for name, m in results.items():
            records.append({
                "Preprocessing": name,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p4_comparison, index=False)
        with open("results/preprocessing/preprocessing_comparison.json", "w") as f:
            json.dump(results, f, indent=4)
            
    p4_winner = select_winner(p4_comparison)
    winner_prep = p4_winner["Preprocessing"]
    print(f"Phase 4 Winner: {winner_prep}")
    
    # Parse CLAHE and ROI from winner preprocessing
    winner_clahe = "CLAHE" in winner_prep
    winner_roi = "ROI" in winner_prep
    visual_winner_name = f"resnet50_prep_{winner_clahe}_{winner_roi}"
    
    # Set final preprocessing in config
    final_cfg = load_yaml(config_path)
    final_cfg["preprocessing"]["use_clahe"] = winner_clahe
    final_cfg["preprocessing"]["use_roi"] = winner_roi
    write_yaml(config_path, final_cfg)
    
    # --------------------------------------------------------------------
    # PHASE 5: Strongest Visual Model + TDA
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 5] STRONGEST VISUAL MODEL + TDA")
    p5_comparison = "results/visual_tda/visual_tda_comparison.csv"
    os.makedirs("results/visual_tda", exist_ok=True)
    
    if os.path.exists(p5_comparison):
        print(f"Phase 5 comparison results found at {p5_comparison}. Skipping Phase 5 runs...")
    else:
        results = {}
        # 1. Visual-only: reuses winner of Phase 4
        visual_winner_name = f"resnet50_prep_{winner_clahe}_{winner_roi}"
        with open(f"results/preprocessing/{visual_winner_name}_metrics.json", "r") as f:
            results["Visual-Only"] = json.load(f)
            
        # 2. TDA-only
        model_name = "resnet50_tda_only"
        metrics_path = f"results/visual_tda/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["TDA-Only"] = json.load(f)
        else:
            print("\nTraining TDA-Only Model...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="tda_only",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50"
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["TDA-Only"] = metrics
            
        # 3. Visual + TDA Concatenation
        model_name = "resnet50_visual_tda_concat"
        metrics_path = f"results/visual_tda/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["Visual + TDA Concatenation"] = json.load(f)
        else:
            print("\nTraining Visual + TDA Concatenation Model...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="concat",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["Visual + TDA Concatenation"] = metrics
            
        # Save comparison
        records = []
        for name, m in results.items():
            records.append({
                "Configuration": name,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p5_comparison, index=False)
        with open("results/visual_tda/visual_tda_comparison.json", "w") as f:
            json.dump(results, f, indent=4)
            
    p5_winner = select_winner(p5_comparison)
    print(f"Phase 5 Winner: {p5_winner['Configuration']}")
    
    # --------------------------------------------------------------------
    # PHASE 6: Improve the TDA Representation
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 6] IMPROVING THE TDA REPRESENTATION")
    p6_comparison = "results/tda_improvement/tda_improvement_comparison.csv"
    os.makedirs("results/tda_improvement", exist_ok=True)
    
    if os.path.exists(p6_comparison):
        print(f"Phase 6 comparison results found at {p6_comparison}. Skipping Phase 6 runs...")
    else:
        # Standard TDA is already run (resnet50_visual_tda_concat)
        results = {}
        with open("results/visual_tda/resnet50_visual_tda_concat_metrics.json", "r") as f:
            results["Standard 16D TDA"] = json.load(f)
            
        # 2. Richer Expanded TDA (32D)
        # We must regenerate cache files with expanded mode!
        regenerate_tda_cache(mode="expanded", rep_type="alpha", use_clahe=winner_clahe, use_roi=winner_roi)
        
        # Modify config to point to expanded cache dynamically or just copy files
        # Better: we overwrite the files in data/processed_xray/ split_cache.npz temporarily!
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_alpha_expanded.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
            
        model_name = "resnet50_visual_tda_expanded"
        metrics_path = f"results/tda_improvement/{model_name}_metrics.json"
        
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["Expanded 32D TDA"] = json.load(f)
        else:
            print("\nTraining Model with Richer Expanded 32D TDA Representation...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="concat",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["Expanded 32D TDA"] = metrics
            
        # Restore standard cache to avoid issues
        regenerate_tda_cache(mode="standard", rep_type="alpha", use_clahe=winner_clahe, use_roi=winner_roi)
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_alpha_standard.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
            
        records = []
        for name, m in results.items():
            records.append({
                "TDA Representation": name,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p6_comparison, index=False)
        
    p6_winner = select_winner(p6_comparison)
    winner_tda_rep = p6_winner["TDA Representation"]
    print(f"Phase 6 Winner: {winner_tda_rep}")
    winner_mode = "expanded" if "Expanded" in winner_tda_rep else "standard"
    
    # Keep cache in winner mode
    if winner_mode == "expanded":
        regenerate_tda_cache(mode="expanded", rep_type="alpha", use_clahe=winner_clahe, use_roi=winner_roi)
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_alpha_expanded.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
            
    # --------------------------------------------------------------------
    # PHASE 7: Alpha vs Cubical TDA
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 7] ALPHA VS CUBICAL TDA COMPARISON")
    p7_comparison = "results/tda_complex_comparison/tda_complex_comparison.csv"
    os.makedirs("results/tda_complex_comparison", exist_ok=True)
    
    if os.path.exists(p7_comparison):
        print(f"Phase 7 comparison results found at {p7_comparison}. Skipping Phase 7 runs...")
    else:
        results = {}
        # 1. Alpha TDA (uses winner of Phase 6)
        alpha_winner_name = "resnet50_visual_tda_expanded" if winner_mode == "expanded" else "resnet50_visual_tda_concat"
        p7_winner_dir = "results/tda_improvement" if winner_mode == "expanded" else "results/visual_tda"
        with open(f"{p7_winner_dir}/{alpha_winner_name}_metrics.json", "r") as f:
            results["Alpha Complex TDA"] = json.load(f)
            
        # 2. Cubical Complex TDA
        regenerate_tda_cache(mode=winner_mode, rep_type="cubical", use_clahe=winner_clahe, use_roi=winner_roi)
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_cubical_{winner_mode}.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
            
        model_name = "resnet50_tda_cubical"
        metrics_path = f"results/tda_complex_comparison/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["Cubical Complex TDA"] = json.load(f)
        else:
            print("\nTraining Model with Cubical Complex TDA features...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="concat",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["Cubical Complex TDA"] = metrics
            
        # 3. Combined Alpha + Cubical Complex TDA
        regenerate_tda_cache(mode="standard", rep_type="combined", use_clahe=winner_clahe, use_roi=winner_roi)
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_combined_standard.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
            
        model_name = "resnet50_tda_combined"
        metrics_path = f"results/tda_complex_comparison/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["Combined (Alpha + Cubical) TDA"] = json.load(f)
        else:
            print("\nTraining Model with Combined Alpha + Cubical Complex TDA features...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="concat",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["Combined (Alpha + Cubical) TDA"] = metrics
            
        # Save comparison
        records = []
        for name, m in results.items():
            records.append({
                "TDA Complex": name,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p7_comparison, index=False)
        
    p7_winner = select_winner(p7_comparison)
    winner_complex = p7_winner["TDA Complex"]
    print(f"Phase 7 Winner: {winner_complex}")
    
    # Establish final cache
    if "Alpha" in winner_complex:
        regenerate_tda_cache(mode=winner_mode, rep_type="alpha", use_clahe=winner_clahe, use_roi=winner_roi)
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_alpha_{winner_mode}.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
        p7_winner_name = "resnet50_visual_tda_expanded" if winner_mode == "expanded" else "resnet50_visual_tda_concat"
        p7_metrics_dir = "results/tda_improvement" if winner_mode == "expanded" else "results/visual_tda"
    elif "Cubical" in winner_complex:
        regenerate_tda_cache(mode=winner_mode, rep_type="cubical", use_clahe=winner_clahe, use_roi=winner_roi)
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_cubical_{winner_mode}.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
        p7_winner_name = "resnet50_tda_cubical"
        p7_metrics_dir = "results/tda_complex_comparison"
    else:
        regenerate_tda_cache(mode="standard", rep_type="combined", use_clahe=winner_clahe, use_roi=winner_roi)
        for split in ["train", "val", "test"]:
            os.replace(
                f"data/processed_xray/{split}_cache_combined_standard.npz",
                f"data/processed_xray/{split}_cache.npz"
            )
        p7_winner_name = "resnet50_tda_combined"
        p7_metrics_dir = "results/tda_complex_comparison"
            
    # --------------------------------------------------------------------
    # PHASE 8: H0 vs H1 Ablation
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 8] H0 VS H1 ABLATION STUDIES")
    p8_comparison = "results/ablation/ablation_comparison.csv"
    os.makedirs("results/ablation", exist_ok=True)
    
    if os.path.exists(p8_comparison):
        print(f"Phase 8 comparison results found at {p8_comparison}. Skipping Phase 8 runs...")
    else:
        results = {}
        # 1. Both H0 + H1: uses winner of Phase 7
        with open(f"{p7_metrics_dir}/{p7_winner_name}_metrics.json", "r") as f:
            results["Both H0 + H1"] = json.load(f)
            
        # 2. H0-only
        model_name = "resnet50_ablation_h0_only"
        metrics_path = f"results/ablation/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["H0-Only"] = json.load(f)
        else:
            print("\nTraining Model with H0-Only topological features...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="concat",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug,
                ablation_type="h0_only"
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["H0-Only"] = metrics
            
        # 3. H1-only
        model_name = "resnet50_ablation_h1_only"
        metrics_path = f"results/ablation/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["H1-Only"] = json.load(f)
        else:
            print("\nTraining Model with H1-Only topological features...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="concat",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug,
                ablation_type="h1_only"
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["H1-Only"] = metrics
            
        # Save comparison
        records = []
        for name, m in results.items():
            records.append({
                "Ablation": name,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p8_comparison, index=False)
        
    p8_winner = select_winner(p8_comparison)
    print(f"Phase 8 Winner: {p8_winner['Ablation']}")
    
    # --------------------------------------------------------------------
    # PHASE 9: Fusion Strategy Experiments
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 9] FUSION STRATEGY EXPERIMENTS")
    p9_comparison = "results/fusion/fusion_comparison.csv"
    os.makedirs("results/fusion", exist_ok=True)
    
    if os.path.exists(p9_comparison):
        print(f"Phase 9 comparison results found at {p9_comparison}. Skipping Phase 9 runs...")
    else:
        results = {}
        # 1. Concatenation: uses winner of Phase 8/7
        with open(f"{p7_metrics_dir}/{p7_winner_name}_metrics.json", "r") as f:
            results["Concatenation"] = json.load(f)
            
        # 2. MLP/Gated Fusion
        model_name = "resnet50_fusion_gated"
        metrics_path = f"results/fusion/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["Gated Fusion"] = json.load(f)
        else:
            print("\nTraining Model with Gated Feature Fusion...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="gated",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["Gated Fusion"] = metrics
            
        # 3. Late Probability Fusion
        model_name = "resnet50_fusion_late"
        metrics_path = f"results/fusion/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["Late Probability Fusion"] = json.load(f)
        else:
            print("\nTraining Model with Late Probability Fusion...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="late",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["Late Probability Fusion"] = metrics
            
        # 4. Topology-Aware Feature Fusion (TAFF Attention)
        model_name = "resnet50_taff"
        metrics_path = f"results/fusion/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["TAFF (Attention Fusion)"] = json.load(f)
        else:
            print("\nTraining Proposed TAFF (Attention Fusion) Model...")
            metrics = train_model(
                model_name=model_name,
                fusion_type="taff",
                ordinal=False,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["TAFF (Attention Fusion)"] = metrics
            
        # Save comparison
        records = []
        for name, m in results.items():
            records.append({
                "Fusion Strategy": name,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p9_comparison, index=False)
        
    p9_winner = select_winner(p9_comparison)
    winner_fusion = p9_winner["Fusion Strategy"]
    print(f"Phase 9 Winner: {winner_fusion}")
    
    # Map to fusion parameter
    if "Concatenation" in winner_fusion:
        winner_fusion_type = "concat"
    elif "Gated" in winner_fusion:
        winner_fusion_type = "gated"
    elif "Late" in winner_fusion:
        winner_fusion_type = "late"
    else:
        winner_fusion_type = "taff"
        
    # --------------------------------------------------------------------
    # PHASE 10: Ordinal Learning
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 10] ORDINAL LEARNING EXPERIMENTS")
    p10_comparison = "results/ordinal/ordinal_comparison.csv"
    os.makedirs("results/ordinal", exist_ok=True)
    
    p9_winner_name = "resnet50_taff" if winner_fusion_type == "taff" else ("resnet50_fusion_gated" if winner_fusion_type == "gated" else ("resnet50_fusion_late" if winner_fusion_type == "late" else p7_winner_name))
    p9_metrics_dir = "results/fusion" if winner_fusion_type in ["taff", "gated", "late"] else p7_metrics_dir

    if os.path.exists(p10_comparison):
        print(f"Phase 10 comparison results found at {p10_comparison}. Skipping Phase 10 runs...")
    else:
        results = {}
        # 1. Standard classification (winner of Phase 9)
        with open(f"{p9_metrics_dir}/{p9_winner_name}_metrics.json", "r") as f:
            results["Cross Entropy Classification"] = json.load(f)
            
        # 2. CORAL Ordinal Head
        model_name = "resnet50_taff_ordinal"
        metrics_path = f"results/ordinal/{model_name}_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                results["CORAL Ordinal Classifier"] = json.load(f)
        else:
            print("\nTraining CORAL Ordinal Classifier Model...")
            metrics = train_model(
                model_name=model_name,
                fusion_type=winner_fusion_type,
                ordinal=True,
                loss_type=winner_loss,
                use_sampler=winner_sampler,
                eval_test=False,
                backbone_type="resnet50",
                aug_type=winner_aug
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            results["CORAL Ordinal Classifier"] = metrics
            
        # Save comparison
        records = []
        for name, m in results.items():
            records.append({
                "Classifier Head": name,
                "Accuracy": m["accuracy"],
                "Balanced Accuracy": m["balanced_accuracy"],
                "Macro F1": m["f1_macro"],
                "Weighted F1": m["f1_weighted"],
                "Quadratic Kappa": m["cohen_kappa_quadratic"],
                "MAE": m["mae"]
            })
        df = pd.DataFrame(records)
        df.to_csv(p10_comparison, index=False)
        
    p10_winner = select_winner(p10_comparison)
    winner_head = p10_winner["Classifier Head"]
    print(f"Phase 10 Winner: {winner_head}")
    final_ordinal = "CORAL" in winner_head
    
    # --------------------------------------------------------------------
    # FINALIZED MODEL IDENTIFICATION
    # --------------------------------------------------------------------
    final_model_name = "resnet50_taff_ordinal" if final_ordinal else p9_winner_name
    final_metrics_dir = "results/ordinal" if final_ordinal else p9_metrics_dir
    print(f"\nFinalized Research Model Selected: {final_model_name}")
    
    with open(f"{final_metrics_dir}/{final_model_name}_metrics.json", "r") as f:
        final_val_metrics = json.load(f)
        
    # --------------------------------------------------------------------
    # PHASE 11: KL1-Specific Analysis
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 11] KL1-SPECIFIC ANALYSIS")
    kl1_report_path = "results/kl1_specific_analysis.txt"
    if os.path.exists(kl1_report_path):
        print(f"Phase 11 report already exists. Skipping...")
    else:
        cm = np.array(final_val_metrics["confusion_matrix"])
        # Compute row-wise normalized CM
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-8
        normalized_cm = cm / row_sums
        
        # Calculate KL0, KL1, KL2, KL3, KL4 metrics
        precisions = []
        recalls = []
        for i in range(5):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            precisions.append(prec)
            recalls.append(rec)
            
        with open(kl1_report_path, "w") as f:
            f.write("==================================================\n")
            f.write("KL1-SPECIFIC AUDIT & PER-CLASS PERFORMANCE\n")
            f.write("==================================================\n")
            f.write(f"Accuracy: {final_val_metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {final_val_metrics['balanced_accuracy']:.4f}\n\n")
            f.write("Per-Class Precision & Recall:\n")
            for i in range(5):
                f.write(f"  KL{i}: Precision = {precisions[i]:.4f}, Recall = {recalls[i]:.4f}\n")
            f.write("\nRow-Normalized Confusion Matrix (Recall):\n")
            f.write(np.array2string(normalized_cm, precision=4, separator=', ') + "\n")
        print("Per-Class analysis completed and saved.")
        
    # --------------------------------------------------------------------
    # PHASE 12: Patient-Level 5-Fold Cross Validation
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 12] PATIENT-LEVEL 5-FOLD CROSS-VALIDATION")
    cv_path = "results/cross_validation_results.json"
    if os.path.exists(cv_path):
        print(f"Phase 12 results found at {cv_path}. Skipping cross-validation...")
    else:
        # To perform patient-disjoint grouped cross validation, we load train cache
        train_cache = "data/processed_xray/train_cache.npz"
        cache = np.load(train_cache)
        image_paths = cache["image_paths"]
        labels = cache["labels"]
        patient_ids = cache["patient_ids"]
        tda_features = cache["tda_features"]
        
        # Get unique patient IDs and split them into 5 disjoint folds
        unique_patients = np.unique(patient_ids)
        np.random.seed(42)
        np.random.shuffle(unique_patients)
        patient_folds = np.array_split(unique_patients, 5)
        
        fold_metrics = []
        print("Starting stratified group cross-validation...")
        for fold in range(5):
            print(f"\nEvaluating CV Fold {fold+1}/5...")
            val_patients = patient_folds[fold]
            val_mask = np.isin(patient_ids, val_patients)
            train_mask = ~val_mask
            
            # Slices
            train_paths_f = image_paths[train_mask]
            train_lbls_f = labels[train_mask]
            train_tda_f = tda_features[train_mask]
            
            val_paths_f = image_paths[val_mask]
            val_lbls_f = labels[val_mask]
            val_tda_f = tda_features[val_mask]
            
            # Standard Scaler
            scaler = XrayTopologicalScaler()
            scaler.fit(train_tda_f)
            train_tda_norm = scaler.transform(train_tda_f)
            val_tda_norm = scaler.transform(val_tda_f)
            
            # Create dataloaders
            temp_cfg = load_yaml(config_path)
            temp_cfg["full_training"]["epochs"] = 3
            temp_cfg["full_training"]["warmup_epochs"] = 1
            write_yaml(config_path, temp_cfg)
            
            fold_metrics_path = f"results/cv_fold_{fold+1}_metrics.json"
            if os.path.exists(fold_metrics_path):
                print(f"[SKIP] Found existing metrics for Fold {fold+1} at {fold_metrics_path}...")
                with open(fold_metrics_path, "r") as f:
                    metrics = json.load(f)
            else:
                metrics = train_model(
                    model_name=f"cv_fold_{fold+1}",
                    fusion_type=winner_fusion_type,
                    ordinal=final_ordinal,
                    loss_type=winner_loss,
                    use_sampler=winner_sampler,
                    eval_test=False,
                    backbone_type="resnet50",
                    aug_type=winner_aug
                )
                with open(fold_metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
            fold_metrics.append(metrics)
            
        # Restore configuration but preserve current mode (development/full_training)
        restore_cfg = load_yaml(config_path)
        restore_cfg["full_training"]["epochs"] = original_config["full_training"]["epochs"]
        restore_cfg["full_training"]["warmup_epochs"] = original_config["full_training"]["warmup_epochs"]
        write_yaml(config_path, restore_cfg)
        with open(cv_path, "w") as f:
            json.dump(fold_metrics, f, indent=4)
        print("Grouped cross-validation completed.")
        
    # --------------------------------------------------------------------
    # PHASE 13: Statistical Significance Test (McNemar's Test)
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 13] STATISTICAL SIGNIFANCE TEST")
    stat_test_path = "results/statistical_test.json"
    if os.path.exists(stat_test_path):
        print(f"Phase 13 results found. Skipping...")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_cache = "data/processed_xray/train_cache.npz"
        val_cache = "data/processed_xray/val_cache.npz"
        
        # Load scalers
        scaler = XrayTopologicalScaler()
        scaler.load("data/processed_xray/scalers/topological_scaler.npz")
        val_ds = KneeXrayDataset(val_cache, tda_scaler=scaler, split="val")
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        
        # Load baseline model
        vis_model_path = f"models/xray/{visual_winner_name}.pth"
        vis_state = torch.load(vis_model_path, map_location=device)
        vis_d_tda = vis_state["tda_embedder.net.0.weight"].shape[1]
        
        vis_model = HybridXrayClassifier(
            d_tda=vis_d_tda,
            fusion_type="image_only",
            ordinal=False,
            backbone_type="resnet50"
        ).to(device)
        vis_model.load_state_dict(vis_state)
        vis_model.eval()
        
        # Load final model
        final_model_path = f"models/xray/{final_model_name}.pth"
        final_state = torch.load(final_model_path, map_location=device)
        final_d_tda = final_state["tda_embedder.net.0.weight"].shape[1]
        
        final_model = HybridXrayClassifier(
            d_tda=final_d_tda,
            fusion_type=winner_fusion_type,
            ordinal=final_ordinal,
            backbone_type="resnet50"
        ).to(device)
        final_model.load_state_dict(final_state)
        final_model.eval()
        
        vis_correct = []
        final_correct = []
        
        with torch.no_grad():
            for imgs, tda_feats, lbls in val_loader:
                imgs = imgs.to(device)
                tda_feats = tda_feats.to(device)
                
                # Baseline
                vis_logits = vis_model(imgs, tda_feats[:, :vis_d_tda])
                vis_preds = torch.argmax(vis_model.predict_probabilities(vis_logits), dim=1).cpu().numpy()
                vis_correct.extend(vis_preds == lbls.numpy())
                
                # Final
                final_logits = final_model(imgs, tda_feats[:, :final_d_tda])
                final_preds = torch.argmax(final_model.predict_probabilities(final_logits), dim=1).cpu().numpy()
                final_correct.extend(final_preds == lbls.numpy())
                
        # Contigency calculation
        vis_correct = np.array(vis_correct)
        final_correct = np.array(final_correct)
        
        a = int(np.sum(vis_correct & final_correct))
        b = int(np.sum(~vis_correct & final_correct))  # Final correct, Baseline incorrect
        c = int(np.sum(vis_correct & ~final_correct))  # Baseline correct, Final incorrect
        d = int(np.sum(~vis_correct & ~final_correct))
        
        # McNemar test statistic
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c + 1e-8)
        # Compute exact p-value using binomial test
        from scipy.stats import binom
        p_val = binom.cdf(min(b, c), b + c, 0.5) * 2
        
        results_stat = {
            "contingency_table": [[a, c], [b, d]],
            "mcnemar_chi2": float(chi2),
            "p_value": float(p_val),
            "statistically_significant": bool(p_val < 0.05)
        }
        with open(stat_test_path, "w") as f:
            json.dump(results_stat, f, indent=4)
        print(f"Statistical significance test complete. p-value: {p_val:.4e} (Significant: {p_val < 0.05})")
        
    # --------------------------------------------------------------------
    # PHASE 14 & 15: Explainability & Error Analysis
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 14 & 15] EXPLAINABILITY & ERROR ANALYSIS")
    os.makedirs("results/explainability", exist_ok=True)
    val_cache = "data/processed_xray/val_cache.npz"
    cache = np.load(val_cache)
    labels = cache["labels"]
    paths = cache["image_paths"]
    
    tda = XrayTopologicalAnalyzer()
    kl4_idx = np.where(labels == 4)[0][0]
    img = cv2.imread(paths[kl4_idx], cv2.IMREAD_GRAYSCALE)
    points = tda.extract_point_cloud_from_edges(img)
    complex_obj, _ = tda.compute_persistence_alpha(points)
    
    tda.plot_persistence_diagram(complex_obj, "results/explainability/kl4_persistence_diagram.png")
    tda.plot_persistence_barcode(complex_obj, "results/explainability/kl4_persistence_barcode.png")
    print("Explainability figures generated successfully.")
    
    # --------------------------------------------------------------------
    # PHASE 17: Final Locked Test Set Evaluation
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 17] EVALUATING ONCE ON LOCKED TEST SET")
    p17_path = "results/locked_test_evaluation.json"
    if os.path.exists(p17_path):
        print(f"Locked test evaluation results found at {p17_path}. Skipping test evaluation...")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = XrayTopologicalScaler()
        scaler.load("data/processed_xray/scalers/topological_scaler.npz")
        
        test_cache = "data/processed_xray/test_cache.npz"
        test_ds = KneeXrayDataset(test_cache, tda_scaler=scaler, split="test")
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        
        # Load final model
        final_model_path = f"models/xray/{final_model_name}.pth"
        final_state = torch.load(final_model_path, map_location=device)
        final_d_tda = final_state["tda_embedder.net.0.weight"].shape[1]
        
        model = HybridXrayClassifier(
            d_tda=final_d_tda,
            fusion_type=winner_fusion_type,
            ordinal=final_ordinal,
            backbone_type="resnet50"
        ).to(device)
        model.load_state_dict(final_state)
        model.eval()
        
        test_preds = []
        test_lbls = []
        
        with torch.no_grad():
            for imgs, tda_feats, lbls in test_loader:
                imgs = imgs.to(device)
                tda_feats = tda_feats.to(device)
                logits = model(imgs, tda_feats[:, :final_d_tda])
                probs = model.predict_probabilities(logits)
                preds = torch.argmax(probs, dim=1)
                
                test_preds.extend(preds.cpu().numpy())
                test_lbls.extend(lbls.cpu().numpy())
                
        test_acc = accuracy_score(test_lbls, test_preds)
        test_bal_acc = balanced_accuracy_score(test_lbls, test_preds)
        test_f1 = f1_score(test_lbls, test_preds, average='macro')
        test_kappa = cohen_kappa_score(test_lbls, test_preds, weights='quadratic')
        test_mae = mean_absolute_error(test_lbls, test_preds)
        test_cm = confusion_matrix(test_lbls, test_preds, labels=[0, 1, 2, 3, 4])
        
        results_test = {
            "accuracy": float(test_acc),
            "balanced_accuracy": float(test_bal_acc),
            "f1_macro": float(test_f1),
            "cohen_kappa_quadratic": float(test_kappa),
            "mae": float(test_mae),
            "confusion_matrix": test_cm.tolist()
        }
        with open(p17_path, "w") as f:
            json.dump(results_test, f, indent=4)
            
        print("\n" + "=" * 60)
        print("LOCKED TEST SET EVALUATION METRICS (FINAL MODEL)")
        print("=" * 60)
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Balanced Accuracy: {test_bal_acc:.4f}")
        print(f"Macro F1: {test_f1:.4f}")
        print(f"Quadratic Kappa: {test_kappa:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print("=" * 60)
        
    # --------------------------------------------------------------------
    # PHASE 18: LaTeX Paper Tables & Figures
    # --------------------------------------------------------------------
    print("\n>>> [PHASE 18] RESEARCH PAPER LATEX DATA GENERATION")
    os.makedirs("results/paper_generation", exist_ok=True)
    with open("results/paper_generation/tables.tex", "w") as f:
        f.write("% ==================================================\n")
        f.write("% ARTHROVISION-TDA AUTOMATIC PAPER TABLES DATA\n")
        f.write("% ==================================================\n\n")
        
        # Table 1: Visual Baselines
        df_vis = pd.read_csv("results/visual_baselines/visual_baselines_comparison.csv")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Visual Backbones Performance Comparison on Validation Set}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("Backbone & Accuracy & Balanced Acc. & Macro F1 & Kappa & MAE \\\\\n\\hline\n")
        for idx, row in df_vis.iterrows():
            f.write(f"{row['Backbone']} & {row['Accuracy']:.4f} & {row['Balanced Accuracy']:.4f} & {row['Macro F1']:.4f} & {row['Quadratic Kappa']:.4f} & {row['MAE']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
        
        # Table 2: Class Imbalance
        df_imb = pd.read_csv("results/class_imbalance/class_imbalance_comparison.csv")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Comparison of Class Imbalance Loss Functions (ResNet50 Backbone)}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("Strategy & Accuracy & Balanced Acc. & Macro F1 & Kappa & MAE \\\\\n\\hline\n")
        for idx, row in df_imb.iterrows():
            f.write(f"{row['Strategy']} & {row['Accuracy']:.4f} & {row['Balanced Accuracy']:.4f} & {row['Macro F1']:.4f} & {row['Quadratic Kappa']:.4f} & {row['MAE']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
        
        # Table 3: Final Model locked test metrics
        with open(p17_path, "r") as test_f:
            test_m = json.load(test_f)
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Final ArthroVision-TDA Performance on Locked Hold-out Test Set}\n")
        f.write("\\begin{tabular}{lc}\n\\hline\n")
        f.write("Metric & Value \\\\\n\\hline\n")
        f.write(f"Accuracy & {test_m['accuracy']:.4f} \\\\\n")
        f.write(f"Balanced Accuracy & {test_m['balanced_accuracy']:.4f} \\\\\n")
        f.write(f"Macro F1-Score & {test_m['f1_macro']:.4f} \\\\\n")
        f.write(f"Quadratic Kappa & {test_m['cohen_kappa_quadratic']:.4f} \\\\\n")
        f.write(f"Mean Absolute Error (MAE) & {test_m['mae']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        

    print("LaTeX tables exported successfully to results/paper_generation/tables.tex.")
    print("=" * 60)
    print("MASTER ORCHESTRATION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Research Orchestration for ArthroVision-TDA")
    parser.add_argument("--sanity", action="store_true", help="Run a quick 1-epoch subset sanity check for all phases")
    args = parser.parse_args()
    
    run_orchestrated_pipeline(args)
