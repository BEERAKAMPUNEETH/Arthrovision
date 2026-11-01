import os
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_synthetic_3d_knee(kl_grade, size=(64, 64, 32)):
    """
    Generates a synthetic 3D MRI volume and its segmentation mask for a knee joint.
    Classes in mask:
      0: Background
      1: Femur
      2: Tibia
      3: Cartilage
    """
    dx, dy, dz = size
    x, y, z = np.meshgrid(np.arange(dx), np.arange(dy), np.arange(dz), indexing="ij")
    
    # Bone structures shift closer as KL grade increases (joint space narrowing)
    # Cartilage thins out as KL grade increases
    narrowing_shift = kl_grade * 1.5  # shift femur down
    cartilage_thickness = max(0, 3 - kl_grade)
    
    # Femur condyles (two ellipsoids at the top)
    femur_m = ((x - 20) / 10)**2 + ((y - 32) / 14)**2 + ((z - (28 - narrowing_shift)) / 6)**2 <= 1.0
    femur_l = ((x - 44) / 10)**2 + ((y - 32) / 14)**2 + ((z - (28 - narrowing_shift)) / 6)**2 <= 1.0
    femur_mask = femur_m | femur_l
    
    # Add osteophytes (bone spurs) for KL >= 1 on femur edges
    if kl_grade >= 1:
        spur_m = ((x - 10) / 3)**2 + ((y - 32) / 8)**2 + ((z - (22 - narrowing_shift)) / 2)**2 <= 1.0
        spur_l = ((x - 54) / 3)**2 + ((y - 32) / 8)**2 + ((z - (22 - narrowing_shift)) / 2)**2 <= 1.0
        femur_mask = femur_mask | spur_m | spur_l
        
    # Tibia plateau
    tibia_mask = (z <= (8 + 2 * np.exp(-((x - 32)/18)**2 - ((y - 32)/18)**2)))
    
    # Cartilage layers
    if cartilage_thickness > 0:
        femur_dilated = ((x - 20) / 10)**2 + ((y - 32) / 14)**2 + ((z - (28 - narrowing_shift)) / (6 + cartilage_thickness * 0.5))**2 <= 1.0
        femur_dilated_l = ((x - 44) / 10)**2 + ((y - 32) / 14)**2 + ((z - (28 - narrowing_shift)) / (6 + cartilage_thickness * 0.5))**2 <= 1.0
        femur_cartilage = (femur_dilated | femur_dilated_l) & ~femur_mask
        
        tibia_dilated = (z <= (8 + cartilage_thickness + 2 * np.exp(-((x - 32)/18)**2 - ((y - 32)/18)**2)))
        tibia_cartilage = tibia_dilated & ~tibia_mask
        
        cartilage_mask = (femur_cartilage | tibia_cartilage) & ~(femur_mask | tibia_mask)
    else:
        cartilage_mask = np.zeros_like(femur_mask)
        
    # Combine into segmentation mask
    mask = np.zeros(size, dtype=np.uint8)
    mask[femur_mask] = 1
    mask[tibia_mask] = 2
    mask[cartilage_mask] = 3
    
    # Construct intensity image
    img = np.zeros(size, dtype=np.float32)
    img[femur_mask] = 0.8
    img[tibia_mask] = 0.7
    img[cartilage_mask] = 0.5
    
    # Add noise
    noise = np.random.normal(0, 0.05, size)
    img += noise
    img = np.clip(img, 0, 1.0)
    
    return img, mask

def main():
    config = load_config()
    np.random.seed(config["seed"])
    
    raw_dir = config["data"]["raw_dir"]
    images_dir = config["data"]["images_dir"]
    masks_dir = config["data"]["masks_dir"]
    split_dir = config["data"]["split_dir"]
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    
    # Generate clinical meta-data for 120 patients (total ~210 scans) to resolve small-dataset instabilities
    num_patients = 120
    patient_ids = [f"OAI_9{i:06d}" for i in range(num_patients)]
    
    records = []
    scan_counter = 0
    
    for pat_id in patient_ids:
        # Patient clinical covariates
        age = int(np.random.randint(45, 80))
        gender = int(np.random.choice([0, 1])) # 0: Female, 1: Male
        bmi = float(np.random.normal(25 + 5 * (age - 45) / 35, 4))
        bmi = max(18.0, min(bmi, 45.0))
        
        # Balanced probabilities across KL grades to prevent class representation collapse
        kl_probs = np.array([0.25, 0.22, 0.20, 0.18, 0.15])
        kl_probs = kl_probs / np.sum(kl_probs)
        
        # Scans for this patient (1 or 2 knees)
        num_knees = np.random.choice([1, 2], p=[0.3, 0.7])
        for knee in ["L", "R"]:
            if num_knees == 1 and knee == "R":
                continue
                
            kl_grade = int(np.random.choice([0, 1, 2, 3, 4], p=kl_probs))
            
            # Pain score: correlates with KL grade
            pain_score = float(np.clip(np.random.normal(kl_grade * 2.0, 1.5), 0, 10))
            
            scan_id = f"{pat_id}_{knee}"
            img_path = f"images/{scan_id}.nii.gz"
            mask_path = f"masks/{scan_id}_mask.nii.gz"
            
            # Generate 3D volume
            img, mask = generate_synthetic_3d_knee(kl_grade)
            
            # Convert to SimpleITK Images and save
            sitk_img = sitk.GetImageFromArray(img.transpose(2, 1, 0)) # sitk expects (Z, Y, X)
            sitk_img.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(sitk_img, os.path.join(raw_dir, img_path))
            
            sitk_mask = sitk.GetImageFromArray(mask.transpose(2, 1, 0))
            sitk_mask.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(sitk_mask, os.path.join(raw_dir, mask_path))
            
            records.append({
                "patient_id": pat_id,
                "scan_id": scan_id,
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "pain_score": pain_score,
                "kl_grade": kl_grade,
                "image_path": img_path,
                "mask_path": mask_path
            })
            scan_counter += 1
            
    df = pd.DataFrame(records)
    csv_path = os.path.join(raw_dir, "oai_clinical_subset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {scan_counter} synthetic knee MRI scans for {num_patients} patients.")
    print(f"Metadata saved to {csv_path}")
    
    # Perform strict patient-level splitting to prevent data leakage (70% train, 15% val, 15% test)
    unique_patients = df["patient_id"].unique()
    np.random.shuffle(unique_patients)
    
    n_train = int(len(unique_patients) * 0.70)
    n_val = int(len(unique_patients) * 0.15)
    
    train_pats = unique_patients[:n_train]
    val_pats = unique_patients[n_train:n_train+n_val]
    test_pats = unique_patients[n_train+n_val:]
    
    df["split"] = "train"
    df.loc[df["patient_id"].isin(val_pats), "split"] = "val"
    df.loc[df["patient_id"].isin(test_pats), "split"] = "test"
    
    split_csv_path = os.path.join(split_dir, "dataset_splits.csv")
    df.to_csv(split_csv_path, index=False)
    print(f"Split dataset at patient level. Saved splits to {split_csv_path}")
    print(df["split"].value_counts())

if __name__ == "__main__":
    main()
