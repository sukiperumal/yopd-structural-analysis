"""
Configuration file for YOPD Structural Analysis Pipeline
========================================================

This file contains all configuration parameters for the structural MRI analysis
pipeline following VBM and surface-based analysis guidelines.
"""

import os
from pathlib import Path

# ===== PATHS =====
# Data paths - Windows compatible
import platform
if platform.system() == "Windows":
    # Try to use D:\ drive, fallback to local directory if not accessible
    try:
        test_path = "D:\\data_NIMHANS"
        os.makedirs(test_path, exist_ok=True)
        DATA_ROOT = test_path
        OUTPUT_ROOT = "D:\\data_NIMHANS\\outputs"
    except (OSError, PermissionError):
        # D:\ drive not accessible, use local directory
        DATA_ROOT = os.path.join(os.getcwd(), "data_NIMHANS")
        OUTPUT_ROOT = os.path.join(os.getcwd(), "outputs")
else:
    DATA_ROOT = "/mnt/d/data_NIMHANS"  # WSL path
    OUTPUT_ROOT = "/mnt/d/data_NIMHANS/outputs"  # WSL path
    
BIDS_ROOT = DATA_ROOT

# Subject groups
GROUPS = {
    'HC': 'HC',
    'PIGD': 'PIGD', 
    'TDPD': 'TDPD'
}

# Demographics file
DEMOGRAPHICS_FILE = os.path.join(DATA_ROOT, "age_gender.xlsx")

# ===== ANALYSIS PARAMETERS =====

# VBM Parameters
VBM_CONFIG = {
    'smoothing_fwhm': 8,  # mm, Gaussian kernel for smoothing
    'normalization_template': 'MNI152NLin2009cAsym',
    'voxel_size': (2, 2, 2),  # mm, target voxel size after normalization
    'bias_field_correction': True,
    'skull_strip': True
}

# Surface Analysis Parameters  
SURFACE_CONFIG = {
    'smoothing_fwhm': 15,  # mm, surface smoothing
    'parcellation': 'aparc',  # Desikan-Killiany atlas
    'measure': 'thickness'
}

# Statistical Parameters
STATS_CONFIG = {
    'alpha_level': 0.05,
    'multiple_comparison_method': 'fdr_bh',  # False Discovery Rate
    'normality_test': 'shapiro',
    'permutation_n': 5000,
    'cluster_threshold': 0.001,  # p-value for cluster forming
    'cluster_correction': 'fwe'  # Family-wise error correction
}

# Quality Control Parameters
QC_CONFIG = {
    'outlier_threshold': 3,  # Standard deviations for outlier detection
    'min_image_quality': 0.7,  # Minimum acceptable image quality score
    'visual_inspection': True
}

# Covariates for statistical models
COVARIATES = ['age', 'sex', 'TIV']  # Total Intracranial Volume

# Network Analysis Parameters
NETWORK_CONFIG = {
    'parcellation_atlas': 'schaefer',
    'n_parcels': 200,
    'correlation_threshold': 0.1,
    'density_threshold': 0.1,  # Keep top 10% of connections
    'graph_metrics': ['efficiency', 'clustering', 'path_length', 'degree', 'betweenness']
}

# ===== OUTPUT DIRECTORIES =====
OUTPUT_DIRS = {
    'preprocessed': os.path.join(OUTPUT_ROOT, "01_preprocessed"),
    'qc': os.path.join(OUTPUT_ROOT, "02_quality_control"),
    'vbm': os.path.join(OUTPUT_ROOT, "03_vbm_analysis"),
    'surface': os.path.join(OUTPUT_ROOT, "04_surface_analysis"),
    'roi': os.path.join(OUTPUT_ROOT, "05_roi_analysis"), 
    'network': os.path.join(OUTPUT_ROOT, "06_network_analysis"),
    'stats': os.path.join(OUTPUT_ROOT, "07_statistics"),
    'figures': os.path.join(OUTPUT_ROOT, "08_figures"),
    'logs': os.path.join(OUTPUT_ROOT, "logs")
}

# Create output directories
for dir_path in OUTPUT_DIRS.values():
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create {dir_path}, will use local fallback")
        # Continue silently, we'll handle fallbacks in individual scripts

# ===== LOGGING CONFIGURATION =====
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(OUTPUT_DIRS['logs'], 'structural_analysis.log')
}

print("Configuration loaded successfully!")
print(f"Data root: {DATA_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Groups: {list(GROUPS.keys())}")
