import numpy as np
import matplotlib.pyplot as plt
import gudhi
import cv2
import io

class XrayTopologicalAnalyzer:
    def __init__(self, max_points=400, downsample_size=(64, 64)):
        self.max_points = max_points
        self.downsample_size = downsample_size
        
    def extract_point_cloud_from_edges(self, img_gray):
        """
        Extracts 2D coordinates from Canny edges.
        Keeps a fixed maximum number of points to stabilize topological complexity.
        """
        # Apply bilateral filter to smooth noise while preserving bone edges
        smoothed = cv2.bilateralFilter(img_gray, 9, 75, 75)
        # Canny edge detection
        edges = cv2.Canny(smoothed, 30, 100)
        
        # Get coordinates of edge pixels
        y_indices, x_indices = np.where(edges > 0)
        points = np.stack([x_indices, y_indices], axis=1).astype(np.float32)
        
        if len(points) == 0:
            # Fallback to a grid of points if no edges are detected
            return np.random.rand(10, 2) * img_gray.shape[0]
            
        if len(points) > self.max_points:
            # Uniformly subsample points
            indices = np.linspace(0, len(points) - 1, self.max_points, dtype=int)
            points = points[indices]
            
        return points
        
    def compute_persistence_alpha(self, points):
        """
        Computes persistent homology (H0, H1) using AlphaComplex on 2D coordinates.
        """
        alpha_complex = gudhi.AlphaComplex(points=points)
        simplex_tree = alpha_complex.create_simplex_tree()
        persistence = simplex_tree.persistence()
        return simplex_tree, persistence
        
    def compute_persistence_cubical(self, img_gray):
        """
        Computes persistent homology using CubicalComplex on intensity grid.
        Downsamples image to avoid excessive computation times.
        """
        resized = cv2.resize(img_gray, self.downsample_size, interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        grid = resized.astype(np.float32) / 255.0
        # Invert so that dark regions (like joint spaces/gaps) correspond to higher filtration values
        grid = 1.0 - grid
        
        cubical_complex = gudhi.CubicalComplex(dimensions=self.downsample_size, top_dimensional_cells=grid.flatten())
        persistence = cubical_complex.persistence()
        return cubical_complex, persistence
        
    def get_intervals(self, complex_obj, dimension, is_cubical=False):
        """Extracts birth-death intervals for a homology dimension."""
        intervals = complex_obj.persistence_intervals_in_dimension(dimension)
        # Clean infinite intervals
        cleaned = []
        max_val = 1.0 if is_cubical else 10000.0
        for birth, death in intervals:
            if np.isinf(death):
                cleaned.append([birth, max_val])
            else:
                cleaned.append([birth, death])
        return np.array(cleaned) if len(cleaned) > 0 else np.empty((0, 2))
        
    def extract_topological_features(self, complex_obj, is_cubical=False, mode="standard"):
        """
        Extracts stable fixed-length numerical features from persistence intervals.
        Returns a vector of 16 features (mode='standard') or 32 features (mode='expanded').
        """
        features = []
        feature_len = 8 if mode == "standard" else 16
        
        for dim in [0, 1]:
            intervals = self.get_intervals(complex_obj, dim, is_cubical)
            if len(intervals) == 0:
                features.extend([0.0] * feature_len)
                continue
                
            births = intervals[:, 0]
            deaths = intervals[:, 1]
            lifetimes = deaths - births
            
            # Feature extraction
            num_intervals = len(intervals)
            max_lifetime = np.max(lifetimes)
            mean_lifetime = np.mean(lifetimes)
            std_lifetime = np.std(lifetimes)
            sum_lifetime = np.sum(lifetimes)
            mean_birth = np.mean(births)
            mean_death = np.mean(deaths)
            
            # Persistence Entropy
            p_i = lifetimes / (np.sum(lifetimes) + 1e-8)
            entropy = -np.sum(p_i * np.log(p_i + 1e-8))
            
            features.extend([
                float(num_intervals),
                float(max_lifetime),
                float(mean_lifetime),
                float(std_lifetime),
                float(sum_lifetime),
                float(mean_birth),
                float(mean_death),
                float(entropy)
            ])
            
            if mode == "expanded":
                var_lifetime = np.var(lifetimes)
                median_lifetime = np.median(lifetimes)
                q25_lifetime = np.percentile(lifetimes, 25)
                q75_lifetime = np.percentile(lifetimes, 75)
                max_birth = np.max(births)
                max_death = np.max(deaths)
                std_birth = np.std(births)
                std_death = np.std(deaths)
                
                features.extend([
                    float(var_lifetime),
                    float(median_lifetime),
                    float(q25_lifetime),
                    float(q75_lifetime),
                    float(max_birth),
                    float(max_death),
                    float(std_birth),
                    float(std_death)
                ])
                
        return np.array(features, dtype=np.float32)

    def plot_persistence_diagram(self, complex_obj, save_path, is_cubical=False):
        """Plots and saves the 2D persistence diagram."""
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        max_val = 1.0 if is_cubical else 200.0
        
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        colors = {0: 'r', 1: 'g'}
        markers = {0: 'o', 1: 's'}
        
        for dim in [0, 1]:
            intervals = self.get_intervals(complex_obj, dim, is_cubical)
            if len(intervals) > 0:
                # Limit values for clean display
                b_plot = np.clip(intervals[:, 0], 0, max_val)
                d_plot = np.clip(intervals[:, 1], 0, max_val)
                ax.scatter(
                    b_plot, 
                    d_plot, 
                    c=colors[dim], 
                    marker=markers[dim], 
                    label=f"H{dim}", 
                    alpha=0.7, 
                    edgecolors='none'
                )
                
        ax.set_xlim(-0.05 * max_val, max_val)
        ax.set_ylim(-0.05 * max_val, max_val)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("2D Persistence Diagram")
        ax.legend()
        plt.tight_layout()
        if isinstance(save_path, str):
            plt.savefig(save_path, dpi=150)
        else:
            plt.savefig(save_path, format='png', dpi=150)
        plt.close()

    def plot_persistence_barcode(self, complex_obj, save_path, is_cubical=False):
        """Plots and saves the 2D persistence barcode."""
        fig, ax = plt.subplots(figsize=(6, 3))
        y_offset = 0
        colors = {0: 'r', 1: 'g'}
        max_val = 1.0 if is_cubical else 200.0
        
        for dim in [0, 1]:
            intervals = self.get_intervals(complex_obj, dim, is_cubical)
            if len(intervals) > 0:
                lifetimes = intervals[:, 1] - intervals[:, 0]
                sort_idx = np.argsort(lifetimes)
                intervals = intervals[sort_idx]
                
                # Limit barcode plots to top 50 longest-lived components to keep UI clean
                for birth, death in intervals[-50:]:
                    birth_plot = np.clip(birth, 0, max_val)
                    death_plot = np.clip(death, 0, max_val)
                    ax.plot([birth_plot, death_plot], [y_offset, y_offset], c=colors[dim], alpha=0.7, linewidth=1.5)
                    y_offset += 1
                    
        ax.set_xlabel("Filtration Value")
        ax.set_ylabel("Interval Index")
        ax.set_title("2D Persistence Barcode")
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='r', label='H0 (Connected Components)'),
            Patch(facecolor='g', label='H1 (Loops)')
        ]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        if isinstance(save_path, str):
            plt.savefig(save_path, dpi=150)
        else:
            plt.savefig(save_path, format='png', dpi=150)
        plt.close()
