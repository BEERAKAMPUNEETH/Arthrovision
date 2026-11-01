import os
import shutil

def setup_directories():
    base_dir = "results"
    subdirs = [
        "development_subset",
        "full_baseline",
        "full_tda",
        "full_taff",
        "full_ordinal",
        "ablation",
        "cross_validation",
        "dataset_statistics"
    ]
    
    # Create directories
    for s in subdirs:
        path = os.path.join(base_dir, s)
        os.makedirs(path, exist_ok=True)
        print(f"Verified directory: {path}")
        
    # Copy existing results to development_subset if it exists in root
    src_file = os.path.join(base_dir, "experimental_results_xray.json")
    dst_file = os.path.join(base_dir, "development_subset", "experimental_results_xray.json")
    
    if os.path.exists(src_file) and not os.path.exists(dst_file):
        shutil.copy2(src_file, dst_file)
        print(f"Copied existing development results to {dst_file}")
        
if __name__ == "__main__":
    setup_directories()
