"""
Utility functions for YOPD Structural Analysis Pipeline
======================================================

This module contains helper functions for logging, data loading, and common operations
used throughout the structural analysis pipeline.
"""

import os
import logging
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import LOGGING_CONFIG, OUTPUT_DIRS, GROUPS, DATA_ROOT


def setup_logging(module_name: str) -> logging.Logger:
    """
    Set up logging for a specific module.
    
    Parameters:
    -----------
    module_name : str
        Name of the module for the logger
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    try:
        Path(OUTPUT_DIRS['logs']).mkdir(parents=True, exist_ok=True)
        log_file = LOGGING_CONFIG['file']
    except Exception:
        # Fallback to local directory if D: drive has issues
        log_dir = "./outputs/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "structural_analysis.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(module_name)
    return logger


def get_subject_list() -> Dict[str, List[str]]:
    """
    Get list of subjects organized by group.
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with group names as keys and subject lists as values
    """
    logger = setup_logging(__name__)
    subjects_by_group = {}
    
    for group_name, group_dir in GROUPS.items():
        group_path = os.path.join(DATA_ROOT, group_dir)
        if os.path.exists(group_path):
            subjects = [d for d in os.listdir(group_path) if d.startswith('sub-')]
            subjects.sort()
            subjects_by_group[group_name] = subjects
            logger.info(f"Found {len(subjects)} subjects in {group_name} group")
        else:
            logger.warning(f"Group directory not found: {group_path}")
            subjects_by_group[group_name] = []
    
    total_subjects = sum(len(subjs) for subjs in subjects_by_group.values())
    logger.info(f"Total subjects found: {total_subjects}")
    
    return subjects_by_group


def find_t1_files() -> Dict[str, Dict[str, str]]:
    """
    Find T1w anatomical files for all subjects.
    
    Returns:
    --------
    Dict[str, Dict[str, str]]
        Nested dictionary: {group: {subject: t1_file_path}}
    """
    logger = setup_logging(__name__)
    t1_files = {}
    
    subjects_by_group = get_subject_list()
    
    for group_name, subjects in subjects_by_group.items():
        t1_files[group_name] = {}
        
        for subject in subjects:
            # Look for T1w file in BIDS structure
            subject_path = os.path.join(DATA_ROOT, group_name, subject)
            
            # Search for T1w files
            t1_pattern = f"{subject}*T1w.nii.gz"
            t1_candidates = []
            
            for root, dirs, files in os.walk(subject_path):
                for file in files:
                    if file.endswith('T1w.nii.gz') and subject in file:
                        t1_candidates.append(os.path.join(root, file))
            
            if t1_candidates:
                # Use the first T1w file found (should only be one per subject)
                t1_files[group_name][subject] = t1_candidates[0]
                logger.debug(f"Found T1w for {subject}: {os.path.basename(t1_candidates[0])}")
            else:
                logger.warning(f"No T1w file found for {subject} in {group_name}")
    
    # Count total files found
    total_files = sum(len(files) for files in t1_files.values())
    logger.info(f"Found T1w files for {total_files} subjects")
    
    return t1_files


def load_demographics() -> pd.DataFrame:
    """
    Load demographic and clinical data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing demographic information
    """
    logger = setup_logging(__name__)
    
    try:
        demo_file = os.path.join(DATA_ROOT, "age_gender.xlsx")
        if os.path.exists(demo_file):
            df = pd.read_excel(demo_file)
            logger.info(f"Loaded demographics for {len(df)} subjects")
            logger.info(f"Demographic columns: {list(df.columns)}")
            return df
        else:
            logger.warning(f"Demographics file not found: {demo_file}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading demographics: {str(e)}")
        return pd.DataFrame()


def check_image_exists(file_path: str) -> bool:
    """
    Check if an image file exists and is readable.
    
    Parameters:
    -----------
    file_path : str
        Path to the image file
        
    Returns:
    --------
    bool
        True if file exists and is readable
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        img = nib.load(file_path)
        return True
    except Exception:
        return False


def basic_image_info(file_path: str) -> Dict:
    """
    Get basic information about a NIfTI image.
    
    Parameters:
    -----------
    file_path : str
        Path to the NIfTI file
        
    Returns:
    --------
    Dict
        Dictionary containing image information
    """
    try:
        img = nib.load(file_path)
        header = img.header
        
        info = {
            'file_path': file_path,
            'shape': img.shape,
            'voxel_size': header.get_zooms()[:3],
            'data_type': header.get_data_dtype(),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
        return info
    except Exception as e:
        return {'error': str(e)}


def test_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Test normality of data using Shapiro-Wilk test.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to test
    alpha : float
        Significance level
        
    Returns:
    --------
    Tuple[bool, float]
        (is_normal, p_value)
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 3:
        return False, np.nan
    
    try:
        statistic, p_value = stats.shapiro(clean_data)
        is_normal = p_value > alpha
        return is_normal, p_value
    except Exception:
        return False, np.nan


def create_summary_plot(data_dict: Dict[str, np.ndarray], 
                       title: str, 
                       ylabel: str,
                       save_path: str) -> None:
    """
    Create boxplot comparing groups.
    
    Parameters:
    -----------
    data_dict : Dict[str, np.ndarray]
        Dictionary with group names as keys and data arrays as values
    title : str
        Plot title
    ylabel : str
        Y-axis label
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    plot_data = []
    group_labels = []
    
    for group, data in data_dict.items():
        clean_data = data[~np.isnan(data)]
        plot_data.extend(clean_data)
        group_labels.extend([group] * len(clean_data))
    
    # Create DataFrame for seaborn
    df_plot = pd.DataFrame({
        'Group': group_labels,
        'Value': plot_data
    })
    
    # Create boxplot
    sns.boxplot(data=df_plot, x='Group', y='Value')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_analysis_summary(analysis_name: str, 
                          subjects_analyzed: int,
                          subjects_excluded: int = 0,
                          notes: str = "") -> None:
    """
    Print a formatted summary of an analysis step.
    
    Parameters:
    -----------
    analysis_name : str
        Name of the analysis step
    subjects_analyzed : int
        Number of subjects successfully analyzed
    subjects_excluded : int
        Number of subjects excluded
    notes : str
        Additional notes
    """
    logger = setup_logging(__name__)
    
    print("\n" + "="*60)
    print(f"ANALYSIS SUMMARY: {analysis_name}")
    print("="*60)
    print(f"Subjects analyzed: {subjects_analyzed}")
    if subjects_excluded > 0:
        print(f"Subjects excluded: {subjects_excluded}")
    print(f"Success rate: {subjects_analyzed/(subjects_analyzed + subjects_excluded)*100:.1f}%")
    if notes:
        print(f"Notes: {notes}")
    print("="*60)
    
    # Log the summary
    logger.info(f"{analysis_name} completed: {subjects_analyzed} analyzed, {subjects_excluded} excluded")


def log_analysis_summary(analysis_name: str, 
                        subjects_analyzed: int,
                        subjects_excluded: int = 0,
                        notes: str = "") -> None:
    """
    Log a formatted summary of an analysis step.
    
    Parameters:
    -----------
    analysis_name : str
        Name of the analysis step
    subjects_analyzed : int
        Number of subjects successfully analyzed
    subjects_excluded : int
        Number of subjects excluded
    notes : str
        Additional notes
    """
    logger = setup_logging(__name__)
    
    # Log the summary
    logger.info(f"{analysis_name} completed: {subjects_analyzed} analyzed, {subjects_excluded} excluded")
    if notes:
        logger.info(f"Notes: {notes}")


if __name__ == "__main__":
    # Test utility functions
    logger = setup_logging(__name__)
    logger.info("Testing utility functions...")
    
    # Test subject discovery
    subjects = get_subject_list()
    
    # Test T1 file discovery
    t1_files = find_t1_files()
    
    # Test demographics loading
    demographics = load_demographics()
    
    print("\nUtility functions test completed!")
