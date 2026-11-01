import os
import re
import json
import pandas as pd
import numpy as np

DATASET_DIR = "C:/Users/punee/Downloads/archive/KneeXrayMini"
SPLITS = ["train", "val", "test"]

def run_splits_audit():
    print("Running splits audit...")
    fn_pattern = re.compile(r"^(\d+)([LR])\.(png|PNG|jpg|JPG|jpeg|JPEG)$")
    
    # Store records
    records = []
    patient_to_splits = {}
    patient_knees = {} # patient_id -> list of (split, laterality, kl)
    
    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist.")
            continue
            
        for kl_class in ["0", "1", "2", "3", "4"]:
            class_dir = os.path.join(split_dir, kl_class)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                    
                match = fn_pattern.match(filename)
                if match:
                    patient_id = match.group(1)
                    laterality = match.group(2)
                else:
                    patient_id = os.path.splitext(filename)[0]
                    laterality = "Unknown"
                    
                records.append({
                    "split": split,
                    "kl_class": int(kl_class),
                    "patient_id": patient_id,
                    "laterality": laterality,
                    "filename": filename
                })
                
                if patient_id not in patient_to_splits:
                    patient_to_splits[patient_id] = set()
                patient_to_splits[patient_id].add(split)
                
                if patient_id not in patient_knees:
                    patient_knees[patient_id] = []
                patient_knees[patient_id].append((split, laterality, int(kl_class)))

    df = pd.DataFrame(records)
    
    # Calculate stats
    stats = {}
    for split in SPLITS:
        split_df = df[df["split"] == split]
        unique_pats = split_df["patient_id"].nunique()
        total_images = len(split_df)
        
        class_dist = {}
        for c in range(5):
            cnt = len(split_df[split_df["kl_class"] == c])
            pct = (cnt / total_images) * 100 if total_images > 0 else 0.0
            class_dist[f"KL{c}"] = {"count": cnt, "percentage": round(pct, 2)}
            
        stats[split] = {
            "total_images": total_images,
            "unique_patients": unique_pats,
            "class_distribution": class_dist
        }
        
    # Intersections
    train_pats = set(df[df["split"] == "train"]["patient_id"])
    val_pats = set(df[df["split"] == "val"]["patient_id"])
    test_pats = set(df[df["split"] == "test"]["patient_id"])
    
    overlaps = {
        "train_val": list(train_pats.intersection(val_pats)),
        "train_test": list(train_pats.intersection(test_pats)),
        "val_test": list(val_pats.intersection(test_pats))
    }
    
    # Check for same patient with both knees
    double_knees = 0
    double_knees_split_crossed = 0
    cross_split_patients = []
    
    for pat, knees in patient_knees.items():
        if len(knees) > 1:
            double_knees += 1
            splits_seen = set([k[0] for k in knees])
            if len(splits_seen) > 1:
                double_knees_split_crossed += 1
                cross_split_patients.append({
                    "patient_id": pat,
                    "knees": [{"split": k[0], "laterality": k[1], "kl_class": k[2]} for k in knees]
                })
                
    audit_report = {
        "split_statistics": stats,
        "overlaps": {
            "train_val_count": len(overlaps["train_val"]),
            "train_test_count": len(overlaps["train_test"]),
            "val_test_count": len(overlaps["val_test"]),
            "train_val_examples": overlaps["train_val"][:5],
            "train_test_examples": overlaps["train_test"][:5],
            "val_test_examples": overlaps["val_test"][:5]
        },
        "patient_bilaterality": {
            "patients_with_both_knees": double_knees,
            "patients_crossing_splits": double_knees_split_crossed,
            "crossing_patients": cross_split_patients
        }
    }
    
    # Save statistics report
    os.makedirs("results/dataset_statistics", exist_ok=True)
    report_path = "results/dataset_statistics/splits_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=4)
        
    print(f"Splits audit report saved to {report_path}")
    print("\n==================================================")
    print("AUDIT RESULTS SUMMARY")
    print("==================================================")
    for split in SPLITS:
        print(f"{split.upper()} split:")
        print(f"  Images: {stats[split]['total_images']}")
        print(f"  Unique Patients: {stats[split]['unique_patients']}")
        print("  Class counts:")
        for c in range(5):
            print(f"    KL{c}: {stats[split]['class_distribution'][f'KL{c}']['count']} ({stats[split]['class_distribution'][f'KL{c}']['percentage']}%)")
            
    print("\nLeakage Check:")
    print(f"  Train / Val overlap: {len(overlaps['train_val'])} patients")
    print(f"  Train / Test overlap: {len(overlaps['train_test'])} patients")
    print(f"  Val / Test overlap: {len(overlaps['val_test'])} patients")
    print(f"\nBilaterality:")
    print(f"  Patients with both knees in dataset: {double_knees}")
    print(f"  Bilaterals crossing splits (leakage): {double_knees_split_crossed}")
    
    if len(overlaps['train_val']) > 0 or len(overlaps['train_test']) > 0 or len(overlaps['val_test']) > 0:
        print("\n[WARNING] Patient leakage detected!")
    else:
        print("\n[SUCCESS] No patient leakage detected across splits.")

if __name__ == "__main__":
    run_splits_audit()
