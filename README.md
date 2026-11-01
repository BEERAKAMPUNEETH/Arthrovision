# ArthroVision-TDA

**ArthroVision-TDA: A Topology-Aware Hybrid Deep Learning Framework for Knee Osteoarthritis Detection and Severity Prediction**

ArthroVision-TDA is a classical academic research system designed to investigate whether combining deep visual features, topological structure features (persistent homology), and clinical patient covariates yields superior performance in knee osteoarthritis severity prediction (Kellgren-Lawrence Grade 0-4) compared to traditional image-only deep learning models.

---

## Central Research Architecture

```text
  +-----------------------+
  |  3D Knee MRI Scan     |
  +-----------+-----------+
              |
              +----------------------------+----------------------------+
              |                            |                            |
              v                            v                            v
      [Visual Branch]             [Segmentation Model]          [Clinical Branch]
   2D Joint Slice Extraction        MONAI 3D U-Net           Patient Covariates (Age,
              |                            |                 BMI, Pain, Gender)
              v                            v                            |
      ResNet50 Backbone          Joint Space Masking                    v
              |                            |             Normalization & Standard Imputation
              v                            v                            |
     Deep Visual Feature             [3D Reconstruction]                v
        Vector (2048)                  Marching Cubes             Clinical Feature
              |                            |                       Vector (4)
              |                            v                            |
              |                   3D Joint Point Cloud                  |
              |                            |                            |
              |                            v                            |
              |                   GUDHI persistentPH                    |
              |                            |                            |
              |                            v                            |
              |                   Topological Feature                   |
              |                       Vector (16)                       |
              |                            |                            |
              +--------------------+-------+----------------------------+
                                   |       |
                                   v       v
                            +------+-------+------+
                            |    Topology-Aware   |
                            | Feature Fusion (TAFF)|
                            +----------+----------+
                                       |
                                       v
                            +----------+----------+
                            |    Classification   |
                            |         Head        |
                            +----------+----------+
                                       |
                                       v
                             Predicted KL Grade [0-4]
```

---

## Project Structure

```text
ArthroVision-TDA/
│
├── configs/
│   └── config.yaml             # Hyperparameters, paths, and seed configurations
│
├── preprocessing/
│   ├── data_generator.py       # Simulates realistic knee MRIs and clinical profiles
│   └── standardizer.py         # Standardizes image intensities and resizes inputs
│
├── segmentation/
│   └── monai_unet.py           # Trains the MONAI 3D U-Net model on synthetic masks
│
├── reconstruction/
│   └── reconstructor.py        # Extracts 3D surface geometries using Marching Cubes
│
├── topology/
│   └── persistent_homology.py  # Computes persistent homology diagrams using GUDHI
│
├── feature_extraction/
│   ├── clinical.py             # Normalizes and cleans clinical attributes
│   ├── image.py                # ResNet50 vision feature extractor
│   └── topology.py             # Computes lifetimes, Betti numbers, and persistence entropy
│
├── fusion/
│   └── multi_modal_fusion.py   # Topology-Aware Feature Fusion (TAFF) and Concat module
│
├── models/
│   └── classifier.py           # Multi-modal classifier head supporting all baselines
│
├── training/
│   ├── train.py                # Visual, clinical, and topological feature cache loop
│   └── train_baselines.py      # Main pipeline executing comparative experiments
│
├── evaluation/
│   └── metrics.py              # Reports Sensitivity, Specificity, F1, Accuracy, ROC-AUC
│
├── explainability/
│   └── grad_cam.py             # Generates Grad-CAM overlays at native image resolution
│
├── api/
│   └── main.py                 # FastAPI backend for clinical inference
│
├── app/
│   └── main.py                 # Streamlit academic prototype visualization interface
│
├── tests/
│   └── test_pipeline.py        # End-to-end verification test suite
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Headless rendering-configured Docker recipe
└── README.md                   # System documentation
```

---

## Getting Started

### 1. Installation

Set up a virtual environment and install the pinned dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
To ensure complete reproducibility without needing NDA credentials, run the synthetic dataset generator. This simulates 35 knee MRI scans across 20 patients:

```powershell
python -m preprocessing.data_generator
```

This creates:
- The raw clinical file: `data/raw/oai_clinical_subset.csv`
- MRI scans & masks: `data/raw/images/` and `data/raw/masks/`
- Strict patient-level split file: `data/splits/dataset_splits.csv`

### 3. Train Segmentation Model
Train the 3D U-Net segmentation model using MONAI:

```powershell
python -m segmentation.monai_unet
```

### 4. Execute Baseline Experiments & Model Ablation
Run the automated pipeline to extract all multi-modal features, cache them to disk, and train the five models (proposed + 4 baselines/ablations) to compare performance:

```powershell
python -m training.train_baselines
```

The script will output a comparative markdown table summarizing accuracy, F1-score, precision, and sensitivity.

### 5. Launch FastAPI Backend

```powershell
python -m uvicorn api.main:app --port 8000
```

### 6. Launch Streamlit Visualization Interface

```powershell
python -m streamlit run app/main.py
```

---

## Explainability Visualizations

- **Grad-CAM**: Generates class activation maps highlighting areas of narrow joint spaces or bone spur locations, overlaid directly onto original-resolution slices.
- **Topological Barcodes**: Displays the persistence of connected components ($H_0$), loops ($H_1$), and voids ($H_2$) to interpret structural changes mathematically.

## Testing

Verify the codebase components via `pytest`:

```powershell
python -m pytest tests/
```
