import numpy as np

class XrayTopologicalScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """Fits mean and std deviation on training TDA features only."""
        X_arr = np.array(X)
        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        # Avoid division by zero
        self.std[self.std == 0.0] = 1.0
        
    def transform(self, X):
        """Transforms TDA features using fitted mean and std."""
        X_arr = np.array(X)
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transforming features.")
        return (X_arr - self.mean) / self.std
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def save(self, filepath):
        """Saves scaler parameters to a file."""
        np.savez(filepath, mean=self.mean, std=self.std)
        
    def load(self, filepath):
        """Loads scaler parameters from a file."""
        data = np.load(filepath)
        self.mean = data["mean"]
        self.std = data["std"]
