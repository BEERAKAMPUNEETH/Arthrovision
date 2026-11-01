import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image

DATASET_DIR = "C:/Users/punee/Downloads/archive/KneeXrayMini"
SPLITS = ["train", "val", "test", "auto_test"]
CLASS_MAPPING = {
    "0": "KL 0",
    "1": "KL 1",
    "2": "KL 2",
    "3": "KL 3",
    "4": "KL 4"
}

def analyze_dataset():
    print("=" * 60)
    print("ARTHROVISION-TDA: 2D KNEE X-RAY DATASET AUDIT")
    print("=" * 60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} does not exist.")
        return
        
    records = []
    leakage_dict = {}
    corrupted_files = []
    
    # Regex to extract Patient ID and Laterality: e.g. 9003126L.png -> Patient: 9003126, Laterality: L
    fn_pattern = re.compile(r"^(\d+)([LR])\.(png|PNG|jpg|JPG|jpeg|JPEG)$")
    
    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        split_patients = set()
        split_records = 0
        class_counts = {c: 0 for c in CLASS_MAPPING}
        dimensions = []
        formats = []
        
        for kl_class in CLASS_MAPPING:
            class_dir = os.path.join(split_dir, kl_class)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                    
                split_records += 1
                class_counts[kl_class] += 1
                
                # Check for corrupted file and read dimensions/format
                try:
                    with Image.open(file_path) as img:
                        dimensions.append(img.size)
                        formats.append(img.format)
                except Exception as e:
                    corrupted_files.append((file_path, str(e)))
                    continue
                
                # Parse filename for Patient ID
                match = fn_pattern.match(filename)
                if match:
                    patient_id = match.group(1)
                    laterality = match.group(2)
                else:
                    # Fallback to filename without extension
                    patient_id = os.path.splitext(filename)[0]
                    laterality = "Unknown"
                    
                split_patients.add(patient_id)
                records.append({
                    "split": split,
                    "kl_class": kl_class,
                    "filename": filename,
                    "patient_id": patient_id,
                    "laterality": laterality,
                    "path": file_path
                })
                
        leakage_dict[split] = split_patients
        
        # Display split statistics
        print(f"\nSplit: {split.upper()}")
        print(f"  Total Images: {split_records}")
        print(f"  Unique Patients: {len(split_patients)}")
        print("  Class Distribution:")
        for c, count in class_counts.items():
            print(f"    Class {c} ({CLASS_MAPPING[c]}): {count}")
            
        if dimensions:
            dims_arr = np.array(dimensions)
            print(f"  Image Dimensions (Min/Max): {dims_arr.min(axis=0)} / {dims_arr.max(axis=0)}")
            print(f"  Image Formats: {set(formats)}")
            
    df_all = pd.DataFrame(records)
    df_all.to_csv("data/processed/xray_metadata.csv", index=False)
    print(f"\nMetadata saved to data/processed/xray_metadata.csv")
    
    print("\n" + "=" * 60)
    print("PATIENT-LEVEL LEAKAGE AUDIT")
    print("=" * 60)
    
    # Audit for overlaps between train, val, test
    overlaps_found = False
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        if s1 in leakage_dict and s2 in leakage_dict:
            overlap = leakage_dict[s1].intersection(leakage_dict[s2])
            print(f"Overlap {s1.upper()} AND {s2.upper()}: {len(overlap)} patients")
            if len(overlap) > 0:
                overlaps_found = True
                # Print some example overlapping patient IDs
                examples = list(overlap)[:5]
                print(f"  Overlapping patient examples: {examples}")
                
    if overlaps_found:
        print("\n[LEAKAGE DETECTED] Patient-level data leakage exists in the default KneeXrayMini folder structure!")
    else:
        print("\n[NO LEAKAGE] Patient partitions are strictly disjoint. Default splits are scientifically valid at patient-level.")
        
    if corrupted_files:
        print(f"\nCorrupted/Invalid Files Found: {len(corrupted_files)}")
        for f, err in corrupted_files[:10]:
            print(f"  {f}: {err}")
    else:
        print("\n[INTEGRITY CHECK] No corrupted files found in dataset.")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    analyze_dataset()
