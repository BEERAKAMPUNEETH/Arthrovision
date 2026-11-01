import os
import re
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from preprocessing.xray_preprocessing.preprocessor import KneeXrayPreprocessor
from topology.xray.persistent_homology import XrayTopologicalAnalyzer

def load_config():
    with open("configs/config_xray.yaml", "r") as f:
        return yaml.safe_load(f)

def run_extraction():
    config = load_config()
    dataset_dir = config["dataset"]["path"]
    splits = config["dataset"]["splits"]
    
    preprocessor = KneeXrayPreprocessor(
        target_size=tuple(config["preprocessing"]["target_size"]),
        use_clahe=config["preprocessing"]["use_clahe"],
        use_roi=config["preprocessing"]["use_roi"]
    )
    
    tda = XrayTopologicalAnalyzer(
        max_points=config["tda"]["max_points"],
        downsample_size=tuple(config["tda"]["downsample_size"])
    )
    
    rep_type = config["tda"]["representation"]
    print(f"Starting TDA feature extraction. Representation: {rep_type.upper()}")
    
    os.makedirs("data/processed_xray", exist_ok=True)
    fn_pattern = re.compile(r"^(\d+)([LR])\.(png|PNG)$")
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory {split_dir} not found.")
            continue
            
        print(f"\nProcessing split: {split.upper()}")
        
        paths = []
        labels = []
        patient_ids = []
        lateralities = []
        tda_features = []
        
        # We walk through classes 0 to 4
        for kl in ["0", "1", "2", "3", "4"]:
            class_dir = os.path.join(split_dir, kl)
            if not os.path.exists(class_dir):
                continue
                
            label = int(kl)
            filenames = os.listdir(class_dir)
            
            for fn in tqdm(filenames, desc=f"  Class {kl}"):
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
                    
                # Preprocess image
                try:
                    img_gray, _ = preprocessor.preprocess_image(file_path)
                except Exception as e:
                    print(f"Skipping {file_path} due to load error: {e}")
                    continue
                    
                # Compute TDA features
                if rep_type == "alpha":
                    points = tda.extract_point_cloud_from_edges(img_gray)
                    complex_obj, _ = tda.compute_persistence_alpha(points)
                    feat = tda.extract_topological_features(complex_obj, is_cubical=False)
                else:
                    complex_obj, _ = tda.compute_persistence_cubical(img_gray)
                    feat = tda.extract_topological_features(complex_obj, is_cubical=True)
                    
                paths.append(file_path)
                labels.append(label)
                patient_ids.append(patient_id)
                lateralities.append(laterality)
                tda_features.append(feat)
                
        # Save as npz file
        save_path = os.path.join("data/processed_xray", f"{split}_cache.npz")
        np.savez_compressed(
            save_path,
            image_paths=np.array(paths),
            labels=np.array(labels, dtype=np.int64),
            patient_ids=np.array(patient_ids),
            lateralities=np.array(lateralities),
            tda_features=np.array(tda_features, dtype=np.float32)
        )
        print(f"Saved split {split.upper()} cache to {save_path}")

if __name__ == "__main__":
    run_extraction()
