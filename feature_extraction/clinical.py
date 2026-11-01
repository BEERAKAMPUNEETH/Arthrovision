import numpy as np
import pandas as pd

class ClinicalFeaturePreprocessor:
    def __init__(self):
        self.means = {}
        self.mins = {}
        self.maxs = {}
        self.is_fit = False
        
    def fit(self, df):
        """Fits scale parameters on train clinical data to prevent data leakage."""
        for col in ["age", "bmi", "pain_score"]:
            # Handle missing value imputation parameters
            self.means[col] = df[col].mean()
            self.mins[col] = df[col].min()
            self.maxs[col] = df[col].max()
        self.is_fit = True
        return self
        
    def transform(self, df):
        """Transforms clinical features into a standardized clinical feature vector."""
        if not self.is_fit:
            raise ValueError("ClinicalFeaturePreprocessor must be fitted first.")
            
        processed_records = []
        for idx, row in df.iterrows():
            # Standard missing value handling
            age = row["age"] if not pd.isna(row["age"]) else self.means["age"]
            bmi = row["bmi"] if not pd.isna(row["bmi"]) else self.means["bmi"]
            pain = row["pain_score"] if not pd.isna(row["pain_score"]) else self.means["pain_score"]
            gender = row["gender"] if not pd.isna(row["gender"]) else 0 # 0: default Female
            
            # Normalization (Min-Max mapping to [0, 1])
            age_norm = (age - self.mins["age"]) / (self.maxs["age"] - self.mins["age"] + 1e-9)
            bmi_norm = (bmi - self.mins["bmi"]) / (self.maxs["bmi"] - self.mins["bmi"] + 1e-9)
            pain_norm = (pain - self.mins["pain_score"]) / (self.maxs["pain_score"] - self.mins["pain_score"] + 1e-9)
            
            # Vector representation: [age, bmi, pain_score, gender]
            feat_vec = np.array([age_norm, bmi_norm, pain_norm, float(gender)], dtype=np.float32)
            processed_records.append(feat_vec)
            
        return np.vstack(processed_records)
        
    def fit_transform(self, df):
        return self.fit(df).transform(df)
