import numpy as np

class TopologicalFeatureExtractor:
    def __init__(self, thresholds_h0=[2.0, 10.0], thresholds_h1=[5.0, 20.0]):
        self.thresholds_h0 = thresholds_h0
        self.thresholds_h1 = thresholds_h1
        
    def compute_persistence_entropy(self, intervals):
        """Computes the persistence entropy of a set of intervals."""
        if len(intervals) == 0:
            return 0.0
            
        lifetimes = intervals[:, 1] - intervals[:, 0]
        # Keep only positive lifetimes
        lifetimes = lifetimes[lifetimes > 0]
        if len(lifetimes) == 0:
            return 0.0
            
        sum_lifetimes = lifetimes.sum()
        probs = lifetimes / sum_lifetimes
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        return float(entropy)
        
    def extract_features(self, simplex_tree):
        """
        Extracts a fixed-size 16-dimensional Topological Feature Vector from GUDHI SimplexTree.
        Dimensions:
          0-5: H0 features (mean life, max life, count, entropy, Betti0_th1, Betti0_th2)
          6-11: H1 features (mean life, max life, count, entropy, Betti1_th1, Betti1_th2)
          12-15: H2 features (mean life, max life, count, entropy)
        """
        features = []
        
        # Helper to compute stats for a given dimension
        def get_dim_features(dim, thresh_list=None):
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            # Filter out infinite values for calculation
            cleaned = []
            for b, d in intervals:
                if np.isinf(d):
                    cleaned.append([b, 100.0]) # Replace inf with a maximum filtration value
                else:
                    cleaned.append([b, d])
            intervals = np.array(cleaned)
            
            if len(intervals) == 0:
                dim_stats = [0.0, 0.0, 0.0, 0.0]
                if thresh_list:
                    dim_stats.extend([0.0] * len(thresh_list))
                return dim_stats
                
            lifetimes = intervals[:, 1] - intervals[:, 0]
            mean_life = float(np.mean(lifetimes))
            max_life = float(np.max(lifetimes))
            count = float(len(intervals))
            entropy = self.compute_persistence_entropy(intervals)
            
            dim_stats = [mean_life, max_life, count, entropy]
            
            if thresh_list:
                for th in thresh_list:
                    # Betti number at threshold th: count how many intervals cross th
                    betti_th = np.sum((intervals[:, 0] <= th) & (intervals[:, 1] > th))
                    dim_stats.append(float(betti_th))
                    
            return dim_stats
            
        # H0
        features.extend(get_dim_features(0, self.thresholds_h0))
        # H1
        features.extend(get_dim_features(1, self.thresholds_h1))
        # H2
        features.extend(get_dim_features(2))
        
        # Ensure we have exactly 16 features
        features = np.array(features, dtype=np.float32)
        if len(features) < 16:
            pad = 16 - len(features)
            features = np.concatenate([features, np.zeros(pad, dtype=np.float32)])
        elif len(features) > 16:
            features = features[:16]
            
        return features

class TopologicalFeatureScaler:
    def __init__(self):
        self.means = None
        self.stds = None
        self.is_fit = False
        
    def fit(self, features):
        """Fits scale parameters on train topological data to prevent data leakage."""
        features = np.array(features)
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
        # Avoid division by zero
        self.stds[self.stds == 0] = 1.0
        self.is_fit = True
        return self
        
    def transform(self, features):
        """Standardizes topological feature vectors using training statistics."""
        if not self.is_fit:
            raise ValueError("TopologicalFeatureScaler must be fitted first.")
        features = np.array(features)
        return (features - self.means) / (self.stds + 1e-9)
        
    def fit_transform(self, features):
        return self.fit(features).transform(features)
