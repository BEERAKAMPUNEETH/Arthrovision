import numpy as np
import matplotlib.pyplot as plt
import gudhi

class TopologicalAnalyzer:
    def __init__(self, max_alpha_square=100.0):
        self.max_alpha_square = max_alpha_square
        
    def compute_persistence(self, points):
        """
        Computes persistent homology on the point cloud using GUDHI AlphaComplex.
        Returns:
            simplex_tree: GUDHI SimplexTree containing the filtration.
            persistence: list of (dimension, (birth, death)) tuples.
        """
        # Construct Alpha Complex
        alpha_complex = gudhi.AlphaComplex(points=points)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=self.max_alpha_square)
        
        # Compute persistent homology
        persistence = simplex_tree.persistence()
        return simplex_tree, persistence
        
    def get_intervals(self, simplex_tree, dimension):
        """Returns the birth-death intervals for a specific homology dimension."""
        intervals = simplex_tree.persistence_intervals_in_dimension(dimension)
        # Filter out infinite intervals (death == inf) or replace inf with a large number
        cleaned = []
        for birth, death in intervals:
            if np.isinf(death):
                cleaned.append([birth, self.max_alpha_square])
            else:
                cleaned.append([birth, death])
        return np.array(cleaned)

    def plot_persistence_diagram(self, simplex_tree, save_path):
        """Plots and saves the persistence diagram."""
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        
        # Draw diagonal
        ax.plot([0, self.max_alpha_square], [0, self.max_alpha_square], 'k--', alpha=0.5)
        
        colors = {0: 'r', 1: 'g', 2: 'b'}
        markers = {0: 'o', 1: 's', 2: '^'}
        
        for dim in [0, 1, 2]:
            intervals = self.get_intervals(simplex_tree, dim)
            if len(intervals) > 0:
                ax.scatter(
                    intervals[:, 0], 
                    intervals[:, 1], 
                    c=colors[dim], 
                    marker=markers[dim], 
                    label=f"H{dim}", 
                    alpha=0.7, 
                    edgecolors='none'
                )
                
        ax.set_xlim(-0.5, self.max_alpha_square * 0.5)
        ax.set_ylim(-0.5, self.max_alpha_square * 0.5)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("Persistence Diagram")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_persistence_barcode(self, simplex_tree, save_path):
        """Plots and saves the persistence barcode."""
        fig, ax = plt.subplots(figsize=(6, 3))
        
        y_offset = 0
        colors = {0: 'r', 1: 'g', 2: 'b'}
        
        for dim in [0, 1, 2]:
            intervals = self.get_intervals(simplex_tree, dim)
            if len(intervals) > 0:
                # Sort by lifetime
                lifetimes = intervals[:, 1] - intervals[:, 0]
                sort_idx = np.argsort(lifetimes)
                intervals = intervals[sort_idx]
                
                for birth, death in intervals:
                    ax.plot([birth, death], [y_offset, y_offset], c=colors[dim], alpha=0.7, linewidth=1.5)
                    y_offset += 1
                    
        ax.set_xlabel("Filtration Value (Alpha^2)")
        ax.set_ylabel("Interval Index")
        ax.set_title("Persistence Barcode")
        
        # Create custom legend handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='r', label='H0'),
            Patch(facecolor='g', label='H1'),
            Patch(facecolor='b', label='H2')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
