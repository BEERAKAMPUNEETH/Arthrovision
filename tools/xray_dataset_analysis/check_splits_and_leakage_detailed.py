import os
import re
import json
import pandas as pd
import numpy as np

DATASET_DIR = "C:/Users/punee/Downloads/archive/KneeXrayMini"
SPLITS = ["train", "val", "test"]

def run_detailed_audit():
    print("Running detailed splits and patient ID audit...")
    fn_pattern = re.compile(r"^(\d+)([LR])\.(png|PNG|jpg|JPG|jpeg|JPEG)$")
    
    records = []
    
    # Walk all directories
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
                    "patient_id": patient_id,
                    "laterality": laterality,
                    "KL": int(kl_class),
                    "split": split,
                    "filename": filename
                })

    df = pd.DataFrame(records)
    
    # Save the table patient_id | laterality | KL | split to CSV
    os.makedirs("results/dataset_statistics", exist_ok=True)
    table_path = "results/dataset_statistics/patient_splits_table.csv"
    df[["patient_id", "laterality", "KL", "split"]].to_csv(table_path, index=False)
    print(f"Generated patient splits table saved to {table_path}")
    
    # Verify split intersections
    train_pats = set(df[df["split"] == "train"]["patient_id"])
    val_pats = set(df[df["split"] == "val"]["patient_id"])
    test_pats = set(df[df["split"] == "test"]["patient_id"])
    
    train_val_overlap = train_pats.intersection(val_pats)
    train_test_overlap = train_pats.intersection(test_pats)
    val_test_overlap = val_pats.intersection(test_pats)
    
    print(f"  TRAIN patient IDs INTERSECT VALIDATION patient IDs = {'empty (0)' if not train_val_overlap else str(len(train_val_overlap)) + ' patients'}")
    print(f"  TRAIN patient IDs INTERSECT TEST patient IDs = {'empty (0)' if not train_test_overlap else str(len(train_test_overlap)) + ' patients'}")
    print(f"  VALIDATION patient IDs INTERSECT TEST patient IDs = {'empty (0)' if not val_test_overlap else str(len(val_test_overlap)) + ' patients'}")
    
    # Check bilaterality constraints: If a patient has both knees, do they remain within the same split?
    patient_splits = df.groupby("patient_id")["split"].nunique()
    crossed_patients = patient_splits[patient_splits > 1].index.tolist()
    
    print("\nBilaterality Verification:")
    patients_with_both_knees = df.groupby("patient_id").filter(lambda g: len(g) > 1)["patient_id"].nunique()
    print(f"  Total unique patients: {df['patient_id'].nunique()}")
    print(f"  Patients with both knees (bilaterals): {patients_with_both_knees}")
    print(f"  Patients crossing splits (leakage): {len(crossed_patients)}")
    
    if len(crossed_patients) > 0:
        print(f"  Crossed patient examples: {crossed_patients[:5]}")
        for p in crossed_patients[:5]:
            p_knees = df[df["patient_id"] == p]
            print(f"    Patient {p}:")
            for _, r in p_knees.iterrows():
                print(f"      {r['laterality']} knee: KL {r['KL']} in {r['split']} split ({r['filename']})")
    
    # Generate final report dict
    report = {
        "total_records": len(df),
        "total_unique_patients": df["patient_id"].nunique(),
        "splits_info": {
            "train": {
                "images": len(df[df["split"] == "train"]),
                "patients": len(train_pats)
            },
            "val": {
                "images": len(df[df["split"] == "val"]),
                "patients": len(val_pats)
            },
            "test": {
                "images": len(df[df["split"] == "test"]),
                "patients": len(test_pats)
            }
        },
        "intersections": {
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap)
        },
        "bilaterality": {
            "patients_with_both_knees": patients_with_both_knees,
            "patients_crossing_splits": len(crossed_patients),
            "crossing_patient_ids": crossed_patients
        }
    }
    
    report_json_path = "results/dataset_statistics/splits_detailed_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nDetailed JSON report saved to {report_json_path}")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0 and len(crossed_patients) == 0:
        print("\n[SUCCESS] Phase 2 validation passed: Splits are strictly patient-disjoint and bilateral knees stay within the same splits.")
    else:
        print("\n[WARNING] Phase 2 validation failed: Patient ID overlap or bilateral knee split-cross leakage detected!")

if __name__ == "__main__":
    run_detailed_audit()
