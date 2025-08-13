#!/usr/bin/env python3
"""
Create missing preprocessing results CSV file
"""

import os
import pandas as pd
from pathlib import Path

def create_preprocessing_results_csv():
    """Create the missing preprocessing_results.csv file"""
    
    # Path to preprocessed data
    preproc_dir = "D:\\data_NIMHANS\\outputs\\01_preprocessed"
    output_file = "D:\\data_NIMHANS\\outputs\\02_quality_control\\preprocessing_results.csv"
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(preproc_dir) 
                   if os.path.isdir(os.path.join(preproc_dir, d)) and d.startswith('sub-')]
    
    print(f"Found {len(subject_dirs)} preprocessed subjects")
    
    # Create results dataframe
    results = []
    for subject_id in sorted(subject_dirs):
        subject_path = os.path.join(preproc_dir, subject_id)
        
        # Check for required files
        gm_file = os.path.join(subject_path, f"{subject_id}_GM_mask.nii.gz")
        wm_file = os.path.join(subject_path, f"{subject_id}_WM_mask.nii.gz")
        csf_file = os.path.join(subject_path, f"{subject_id}_CSF_mask.nii.gz")
        brain_file = os.path.join(subject_path, f"{subject_id}_T1w_brain.nii.gz")
        
        success = all(os.path.exists(f) for f in [gm_file, wm_file, csf_file, brain_file])
        
        results.append({
            'subject_id': subject_id,
            'success': success,
            'gm_path': gm_file if success else '',
            'wm_path': wm_file if success else '',
            'csf_path': csf_file if success else '',
            'brain_path': brain_file if success else '',
            'group': 'HC' if 'HC' in subject_id else ('PIGD' if 'PIGD' in subject_id else 'TDPD')
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"Created preprocessing results CSV: {output_file}")
    print(f"Successful subjects: {df['success'].sum()}/{len(df)}")
    print(f"Groups: {df['group'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    create_preprocessing_results_csv()
