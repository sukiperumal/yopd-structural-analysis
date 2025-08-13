#!/usr/bin/env python3
"""
YOPD Structural Analysis - Step 02: Basic Preprocessing Pipeline
================================================================

This script implements a basic preprocessing pipeline for structural MRI analysis:
1. Load T1 images and check quality
2. Calculate basic tissue volumes
3. Generate preprocessing report

Simplified version for initial testing.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration and utilities
import config
from utils import setup_logging, log_analysis_summary

def main():
    """Main preprocessing pipeline"""
    logger = setup_logging("step02_preprocessing_basic")
    
    print("Starting basic preprocessing pipeline...")
    logger.info("=" * 60)
    logger.info("STARTING STEP 02: BASIC PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Create output directory
        preproc_output_dir = config.OUTPUT_DIRS['preprocessed']
        print(f"Creating output directory: {preproc_output_dir}")
        Path(preproc_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data inventory
        inventory_path = os.path.join(config.OUTPUT_DIRS['qc'], 'data_inventory.csv')
        print(f"Loading data inventory from: {inventory_path}")
        
        if not os.path.exists(inventory_path):
            logger.error(f"Data inventory not found: {inventory_path}")
            logger.error("Please run step01_data_inventory.py first")
            return
        
        logger.info(f"Loading data inventory from: {inventory_path}")
        inventory_df = pd.read_csv(inventory_path)
        print(f"Loaded inventory for {len(inventory_df)} subjects")
        
        # Filter for subjects with readable T1 images
        valid_subjects = inventory_df[inventory_df['t1_file_readable'] == True].copy()
        logger.info(f"Found {len(valid_subjects)} subjects with valid T1 images")
        print(f"Processing {len(valid_subjects)} valid subjects...")
        
        # Process each subject (basic version)
        results = []
        for idx, row in valid_subjects.iterrows():
            subject_id = row['subject_id']
            t1_path = row['t1_file_path']
            group = row['group']
            
            print(f"Processing {subject_id} ({idx+1}/{len(valid_subjects)})...")
            logger.info(f"Processing subject: {subject_id}")
            
            try:
                # Load T1 image
                t1_nii = nib.load(t1_path)
                t1_data = t1_nii.get_fdata()
                
                # Calculate basic metrics
                voxel_size = t1_nii.header.get_zooms()[:3]
                voxel_volume = np.prod(voxel_size)  # mm³
                
                # Simple brain mask (intensity thresholding)
                threshold = 0.1 * np.max(t1_data)
                brain_mask = t1_data > threshold
                brain_volume = np.sum(brain_mask) * voxel_volume / 1000  # ml
                
                # Basic intensity statistics
                brain_intensities = t1_data[brain_mask]
                mean_intensity = np.mean(brain_intensities)
                std_intensity = np.std(brain_intensities)
                
                result = {
                    'subject_id': subject_id,
                    'group': group,
                    'success': True,
                    'error': None,
                    'voxel_size_x': voxel_size[0],
                    'voxel_size_y': voxel_size[1], 
                    'voxel_size_z': voxel_size[2],
                    'voxel_volume_mm3': voxel_volume,
                    'brain_volume_ml': brain_volume,
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'image_shape_x': t1_data.shape[0],
                    'image_shape_y': t1_data.shape[1],
                    'image_shape_z': t1_data.shape[2]
                }
                
                logger.info(f"Subject {subject_id} processed successfully - Brain volume: {brain_volume:.1f} ml")
                
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {e}")
                result = {
                    'subject_id': subject_id,
                    'group': group,
                    'success': False,
                    'error': str(e),
                    'voxel_size_x': None,
                    'voxel_size_y': None,
                    'voxel_size_z': None,
                    'voxel_volume_mm3': None,
                    'brain_volume_ml': None,
                    'mean_intensity': None,
                    'std_intensity': None,
                    'image_shape_x': None,
                    'image_shape_y': None,
                    'image_shape_z': None
                }
            
            results.append(result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(preproc_output_dir, 'basic_preprocessing_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Basic preprocessing results saved: {results_path}")
        print(f"Results saved to: {results_path}")
        
        # Create summary
        successful = results_df['success'].sum()
        total = len(results_df)
        success_rate = 100 * successful / total
        
        # Create summary plot
        if successful > 0:
            successful_data = results_df[results_df['success']].copy()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Basic Preprocessing Summary', fontsize=16)
            
            # Brain volume by group
            sns.boxplot(data=successful_data, x='group', y='brain_volume_ml', ax=axes[0, 0])
            axes[0, 0].set_title('Brain Volume by Group')
            axes[0, 0].set_ylabel('Brain Volume (ml)')
            
            # Mean intensity by group
            sns.boxplot(data=successful_data, x='group', y='mean_intensity', ax=axes[0, 1])
            axes[0, 1].set_title('Mean Intensity by Group')
            axes[0, 1].set_ylabel('Mean Intensity')
            
            # Voxel volume distribution
            axes[1, 0].hist(successful_data['voxel_volume_mm3'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Voxel Volume Distribution')
            axes[1, 0].set_xlabel('Voxel Volume (mm³)')
            axes[1, 0].set_ylabel('Count')
            
            # Success rate by group
            success_by_group = results_df.groupby('group')['success'].agg(['sum', 'count'])
            success_by_group['rate'] = 100 * success_by_group['sum'] / success_by_group['count']
            
            axes[1, 1].bar(success_by_group.index, success_by_group['rate'])
            axes[1, 1].set_title('Success Rate by Group')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].set_ylim(0, 100)
            
            plt.tight_layout()
            summary_plot_path = os.path.join(preproc_output_dir, 'basic_preprocessing_summary.png')
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Summary plot saved: {summary_plot_path}")
            print(f"Summary plot saved to: {summary_plot_path}")
        
        # Generate text report
        report_path = os.path.join(preproc_output_dir, 'basic_preprocessing_report.txt')
        with open(report_path, 'w') as f:
            f.write("YOPD Structural Analysis - Basic Preprocessing Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"total_subjects: {total}\n")
            f.write(f"successful_preprocessing: {successful}\n")
            f.write(f"success_rate: {success_rate:.1f}%\n\n")
            
            if successful > 0:
                successful_data = results_df[results_df['success']].copy()
                
                f.write("GROUP-WISE STATISTICS\n")
                f.write("-" * 30 + "\n")
                for group in successful_data['group'].unique():
                    group_data = successful_data[successful_data['group'] == group]
                    f.write(f"{group}: {len(group_data)} subjects\n")
                    f.write(f"  Mean brain volume: {group_data['brain_volume_ml'].mean():.1f} ± {group_data['brain_volume_ml'].std():.1f} ml\n")
                    f.write(f"  Mean intensity: {group_data['mean_intensity'].mean():.1f} ± {group_data['mean_intensity'].std():.1f}\n")
                
                f.write(f"\nVOXEL SIZE STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"X: {successful_data['voxel_size_x'].mean():.3f} ± {successful_data['voxel_size_x'].std():.3f} mm\n")
                f.write(f"Y: {successful_data['voxel_size_y'].mean():.3f} ± {successful_data['voxel_size_y'].std():.3f} mm\n")
                f.write(f"Z: {successful_data['voxel_size_z'].mean():.3f} ± {successful_data['voxel_size_z'].std():.3f} mm\n")
                f.write(f"Volume: {successful_data['voxel_volume_mm3'].mean():.3f} ± {successful_data['voxel_volume_mm3'].std():.3f} mm³\n")
        
        logger.info(f"Basic preprocessing report saved: {report_path}")
        print(f"Report saved to: {report_path}")
        
        # Log summary
        logger.info("=" * 60)
        logger.info("BASIC PREPROCESSING PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Subjects processed: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Log analysis summary
        summary_notes = f"Success rate: {success_rate:.1f}%"
        if successful < total:
            failed_subjects = results_df[~results_df['success']]['subject_id'].tolist()
            summary_notes += f", Failed: {failed_subjects}"
        
        log_analysis_summary("Basic Preprocessing Pipeline", total, total - successful, summary_notes)
        
        # Print key findings
        print("\n" + "="*60)
        print("BASIC PREPROCESSING COMPLETED")
        print("="*60)
        print(f"• Processing completed for {successful}/{total} subjects")
        if successful > 0:
            successful_data = results_df[results_df['success']]
            mean_brain_vol = successful_data['brain_volume_ml'].mean()
            std_brain_vol = successful_data['brain_volume_ml'].std()
            print(f"• Mean brain volume: {mean_brain_vol:.1f} ± {std_brain_vol:.1f} ml")
            
            by_group = successful_data.groupby('group')['brain_volume_ml'].agg(['count', 'mean', 'std'])
            for group in by_group.index:
                print(f"• {group}: {by_group.loc[group, 'count']} subjects, "
                      f"Brain volume = {by_group.loc[group, 'mean']:.1f} ± {by_group.loc[group, 'std']:.1f} ml")
        print("="*60)
        
        logger.info("Step 02 basic preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Basic preprocessing pipeline failed: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
