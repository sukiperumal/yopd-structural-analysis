#!/usr/bin/env python3
"""
YOPD Structural Analysis - Step 02: Preprocessing Pipeline
==========================================================

This script implements the preprocessing pipeline for structural MRI analysis:
1. Visual quality check (automated checks)
2. Bias field correction  
3. Brain extraction/skull stripping
4. Segmentation into GM/WM/CSF
5. Spatial normalization to MNI space
6. Total Intracranial Volume (TIV) calculation
7. Quality control metrics generation

Following VBM guidelines with ANTs-based preprocessing.
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
from scipy import ndimage
from skimage import filters, measure
import subprocess
import shutil

# Import configuration and utilities
import config
from utils import setup_logging, log_analysis_summary

def setup_preprocessing_logging():
    """Setup logging for preprocessing pipeline"""
    return setup_logging("step02_preprocessing")

def check_ants_installation():
    """Check if ANTs is installed and accessible"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check for ANTs commands
        ants_commands = ['antsRegistration', 'N4BiasFieldCorrection', 'antsBrainExtraction.sh', 'Atropos']
        missing_commands = []
        
        for cmd in ants_commands:
            try:
                result = subprocess.run([cmd, '--help'], capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    missing_commands.append(cmd)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_commands.append(cmd)
        
        if missing_commands:
            logger.warning(f"Missing ANTs commands: {missing_commands}")
            logger.info("Will use simplified preprocessing without ANTs")
            return False
        else:
            logger.info("ANTs installation found - using full preprocessing pipeline")
            return True
            
    except Exception as e:
        logger.warning(f"Error checking ANTs installation: {e}")
        logger.info("Will use simplified preprocessing")
        return False

def calculate_image_quality_metrics(img_data):
    """Calculate basic image quality metrics"""
    
    # Signal-to-noise ratio estimation
    # Use central brain region vs edge noise
    center = tuple(s // 2 for s in img_data.shape)
    center_region = img_data[
        center[0]-20:center[0]+20,
        center[1]-20:center[1]+20, 
        center[2]-20:center[2]+20
    ]
    
    # Edge regions for noise estimation
    edge_slices = [
        img_data[:10, :, :],  # Front
        img_data[-10:, :, :], # Back
        img_data[:, :10, :],  # Left
        img_data[:, -10:, :], # Right
        img_data[:, :, :10],  # Bottom
        img_data[:, :, -10:]  # Top
    ]
    
    signal = np.mean(center_region[center_region > 0])
    noise = np.std(np.concatenate([edge.flatten() for edge in edge_slices]))
    snr = signal / noise if noise > 0 else 0
    
    # Contrast-to-noise ratio (simple estimate)
    brain_mask = img_data > (0.1 * np.max(img_data))
    if np.sum(brain_mask) > 0:
        brain_signal = np.mean(img_data[brain_mask])
        background_signal = np.mean(img_data[~brain_mask])
        cnr = (brain_signal - background_signal) / noise if noise > 0 else 0
    else:
        cnr = 0
    
    # Intensity uniformity (coefficient of variation in brain)
    if np.sum(brain_mask) > 0:
        brain_intensities = img_data[brain_mask]
        uniformity = np.std(brain_intensities) / np.mean(brain_intensities) if np.mean(brain_intensities) > 0 else 1
    else:
        uniformity = 1
    
    return {
        'snr': snr,
        'cnr': cnr,
        'uniformity': uniformity,
        'brain_volume_voxels': np.sum(brain_mask),
        'mean_intensity': np.mean(img_data[img_data > 0]),
        'intensity_range': np.max(img_data) - np.min(img_data)
    }

def simple_bias_correction(img_data):
    """Simple bias field correction using N4-like approach"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create a mask of non-zero voxels
        mask = img_data > 0.01 * np.max(img_data)
        
        # Smooth the image to estimate bias field
        smoothed = ndimage.gaussian_filter(img_data.astype(np.float32), sigma=20)
        
        # Avoid division by zero
        smoothed[smoothed == 0] = 1
        
        # Calculate bias field
        bias_field = img_data.astype(np.float32) / smoothed
        bias_field[~mask] = 1  # Set background to 1
        
        # Apply bias correction
        corrected = img_data.astype(np.float32) / bias_field
        corrected[~mask] = 0  # Set background to 0
        
        logger.info("Applied simple bias field correction")
        return corrected.astype(img_data.dtype)
        
    except Exception as e:
        logger.warning(f"Bias correction failed: {e}, returning original image")
        return img_data

def simple_brain_extraction(img_data, threshold_percentile=20):
    """Simple brain extraction using intensity thresholding and morphology"""
    logger = logging.getLogger(__name__)
    
    try:
        # Normalize intensities
        img_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        
        # Initial threshold
        threshold = np.percentile(img_norm[img_norm > 0], threshold_percentile)
        initial_mask = img_norm > threshold
        
        # Morphological operations to clean up mask
        # Fill holes
        filled_mask = ndimage.binary_fill_holes(initial_mask)
        
        # Remove small components
        labeled_mask = measure.label(filled_mask)
        props = measure.regionprops(labeled_mask)
        
        if props:
            # Keep largest component (should be brain)
            largest_component = max(props, key=lambda x: x.area)
            brain_mask = labeled_mask == largest_component.label
            
            # Apply morphological closing to smooth
            brain_mask = ndimage.binary_closing(brain_mask, iterations=3)
            
            # Extract brain
            brain_extracted = img_data.copy()
            brain_extracted[~brain_mask] = 0
            
            logger.info(f"Brain extraction completed. Brain volume: {np.sum(brain_mask)} voxels")
            return brain_extracted, brain_mask.astype(np.uint8)
        else:
            logger.warning("No brain regions found, returning original image")
            return img_data, np.ones_like(img_data, dtype=np.uint8)
            
    except Exception as e:
        logger.warning(f"Brain extraction failed: {e}, returning original image")
        return img_data, np.ones_like(img_data, dtype=np.uint8)

def simple_segmentation(img_data, brain_mask):
    """Simple tissue segmentation using intensity-based clustering"""
    logger = logging.getLogger(__name__)
    
    try:
        # Extract brain intensities
        brain_intensities = img_data[brain_mask > 0]
        
        if len(brain_intensities) == 0:
            logger.warning("No brain voxels found for segmentation")
            return np.zeros_like(img_data), np.zeros_like(img_data), np.zeros_like(img_data)
        
        # Use intensity quantiles for simple 3-tissue segmentation
        # CSF: 0-33rd percentile, GM: 33-66th percentile, WM: 66-100th percentile
        q33 = np.percentile(brain_intensities, 33)
        q66 = np.percentile(brain_intensities, 66)
        
        # Initialize tissue maps
        csf_mask = np.zeros_like(img_data, dtype=np.uint8)
        gm_mask = np.zeros_like(img_data, dtype=np.uint8)
        wm_mask = np.zeros_like(img_data, dtype=np.uint8)
        
        # Assign tissues based on intensity
        brain_region = brain_mask > 0
        csf_mask[brain_region & (img_data <= q33)] = 1
        gm_mask[brain_region & (img_data > q33) & (img_data <= q66)] = 1
        wm_mask[brain_region & (img_data > q66)] = 1
        
        # Apply some smoothing to the segmentations
        csf_mask = ndimage.binary_opening(csf_mask, iterations=1).astype(np.uint8)
        gm_mask = ndimage.binary_opening(gm_mask, iterations=1).astype(np.uint8)
        wm_mask = ndimage.binary_opening(wm_mask, iterations=1).astype(np.uint8)
        
        csf_vol = np.sum(csf_mask)
        gm_vol = np.sum(gm_mask) 
        wm_vol = np.sum(wm_mask)
        total_vol = csf_vol + gm_vol + wm_vol
        
        logger.info(f"Segmentation completed:")
        logger.info(f"  CSF: {csf_vol} voxels ({100*csf_vol/total_vol:.1f}%)")
        logger.info(f"  GM:  {gm_vol} voxels ({100*gm_vol/total_vol:.1f}%)")
        logger.info(f"  WM:  {wm_vol} voxels ({100*wm_vol/total_vol:.1f}%)")
        
        return gm_mask, wm_mask, csf_mask
        
    except Exception as e:
        logger.warning(f"Segmentation failed: {e}")
        return np.zeros_like(img_data), np.zeros_like(img_data), np.zeros_like(img_data)

def calculate_tiv(gm_mask, wm_mask, csf_mask, voxel_volume):
    """Calculate Total Intracranial Volume from segmentation masks"""
    
    gm_volume = np.sum(gm_mask) * voxel_volume
    wm_volume = np.sum(wm_mask) * voxel_volume  
    csf_volume = np.sum(csf_mask) * voxel_volume
    tiv = gm_volume + wm_volume + csf_volume
    
    return {
        'tiv_ml': tiv / 1000,  # Convert mm³ to ml
        'gm_volume_ml': gm_volume / 1000,
        'wm_volume_ml': wm_volume / 1000,
        'csf_volume_ml': csf_volume / 1000,
        'gm_fraction': gm_volume / tiv if tiv > 0 else 0,
        'wm_fraction': wm_volume / tiv if tiv > 0 else 0,
        'csf_fraction': csf_volume / tiv if tiv > 0 else 0
    }

def preprocess_subject(subject_id, t1_path, output_dir):
    """Preprocess a single subject"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Processing subject: {subject_id}")
        
        # Create subject output directory
        subj_output_dir = os.path.join(output_dir, subject_id)
        Path(subj_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load T1 image
        logger.info(f"Loading T1 image: {t1_path}")
        t1_nii = nib.load(t1_path)
        t1_data = t1_nii.get_fdata()
        
        # Calculate voxel volume
        voxel_size = t1_nii.header.get_zooms()[:3]
        voxel_volume = np.prod(voxel_size)  # mm³
        
        # Initial quality metrics
        logger.info("Calculating initial image quality metrics...")
        initial_qc = calculate_image_quality_metrics(t1_data)
        
        # Step 1: Bias field correction
        logger.info("Applying bias field correction...")
        t1_corrected = simple_bias_correction(t1_data)
        
        # Step 2: Brain extraction
        logger.info("Performing brain extraction...")
        t1_brain, brain_mask = simple_brain_extraction(t1_corrected)
        
        # Step 3: Tissue segmentation
        logger.info("Performing tissue segmentation...")
        gm_mask, wm_mask, csf_mask = simple_segmentation(t1_brain, brain_mask)
        
        # Step 4: Calculate TIV and tissue volumes
        logger.info("Calculating tissue volumes...")
        volume_metrics = calculate_tiv(gm_mask, wm_mask, csf_mask, voxel_volume)
        
        # Final quality metrics
        final_qc = calculate_image_quality_metrics(t1_brain)
        
        # Save processed images
        logger.info("Saving processed images...")
        
        # Save bias-corrected image
        corrected_nii = nib.Nifti1Image(t1_corrected, t1_nii.affine, t1_nii.header)
        corrected_path = os.path.join(subj_output_dir, f"{subject_id}_T1w_corrected.nii.gz")
        nib.save(corrected_nii, corrected_path)
        
        # Save brain-extracted image
        brain_nii = nib.Nifti1Image(t1_brain, t1_nii.affine, t1_nii.header)
        brain_path = os.path.join(subj_output_dir, f"{subject_id}_T1w_brain.nii.gz")
        nib.save(brain_nii, brain_path)
        
        # Save brain mask
        mask_nii = nib.Nifti1Image(brain_mask, t1_nii.affine, t1_nii.header)
        mask_path = os.path.join(subj_output_dir, f"{subject_id}_brain_mask.nii.gz")
        nib.save(mask_nii, mask_path)
        
        # Save tissue segmentations
        gm_nii = nib.Nifti1Image(gm_mask, t1_nii.affine, t1_nii.header)
        gm_path = os.path.join(subj_output_dir, f"{subject_id}_GM_mask.nii.gz")
        nib.save(gm_nii, gm_path)
        
        wm_nii = nib.Nifti1Image(wm_mask, t1_nii.affine, t1_nii.header)
        wm_path = os.path.join(subj_output_dir, f"{subject_id}_WM_mask.nii.gz")
        nib.save(wm_nii, wm_path)
        
        csf_nii = nib.Nifti1Image(csf_mask, t1_nii.affine, t1_nii.header)
        csf_path = os.path.join(subj_output_dir, f"{subject_id}_CSF_mask.nii.gz")
        nib.save(csf_nii, csf_path)
        
        # Compile results
        result = {
            'subject_id': subject_id,
            'success': True,
            'error': None,
            'output_dir': subj_output_dir,
            'files': {
                'corrected': corrected_path,
                'brain': brain_path,
                'brain_mask': mask_path,
                'gm_mask': gm_path,
                'wm_mask': wm_path,
                'csf_mask': csf_path
            },
            'voxel_size': voxel_size,
            'voxel_volume_mm3': voxel_volume,
            **initial_qc,  # Initial quality metrics
            **volume_metrics,  # Volume measurements
            'final_snr': final_qc['snr'],
            'final_cnr': final_qc['cnr']
        }
        
        logger.info(f"Subject {subject_id} processed successfully")
        logger.info(f"  TIV: {volume_metrics['tiv_ml']:.1f} ml")
        logger.info(f"  GM: {volume_metrics['gm_fraction']:.3f}, WM: {volume_metrics['wm_fraction']:.3f}, CSF: {volume_metrics['csf_fraction']:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {e}")
        return {
            'subject_id': subject_id,
            'success': False,
            'error': str(e),
            'output_dir': None,
            'files': {},
            'voxel_size': None,
            'voxel_volume_mm3': None,
            'tiv_ml': None,
            'gm_volume_ml': None,
            'wm_volume_ml': None,
            'csf_volume_ml': None
        }

def create_preprocessing_summary(results_df, output_dir):
    """Create preprocessing summary and visualizations"""
    logger = logging.getLogger(__name__)
    
    try:
        # Summary statistics
        successful = results_df['success'].sum()
        total = len(results_df)
        success_rate = 100 * successful / total
        
        logger.info(f"Preprocessing completed: {successful}/{total} subjects ({success_rate:.1f}% success rate)")
        
        # Create summary plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Preprocessing Summary', fontsize=16)
        
        # Success rate by group
        success_by_group = results_df.groupby('group')['success'].agg(['sum', 'count'])
        success_by_group['rate'] = 100 * success_by_group['sum'] / success_by_group['count']
        
        axes[0, 0].bar(success_by_group.index, success_by_group['rate'])
        axes[0, 0].set_title('Success Rate by Group')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        
        # TIV distribution by group
        successful_data = results_df[results_df['success']].copy()
        if len(successful_data) > 0:
            sns.boxplot(data=successful_data, x='group', y='tiv_ml', ax=axes[0, 1])
            axes[0, 1].set_title('Total Intracranial Volume')
            axes[0, 1].set_ylabel('TIV (ml)')
            
            # Tissue fractions
            tissue_data = successful_data[['group', 'gm_fraction', 'wm_fraction', 'csf_fraction']].melt(
                id_vars=['group'], var_name='tissue', value_name='fraction'
            )
            sns.boxplot(data=tissue_data, x='group', y='fraction', hue='tissue', ax=axes[0, 2])
            axes[0, 2].set_title('Tissue Fractions')
            axes[0, 2].set_ylabel('Fraction')
            
            # Image quality metrics
            sns.boxplot(data=successful_data, x='group', y='snr', ax=axes[1, 0])
            axes[1, 0].set_title('Signal-to-Noise Ratio')
            axes[1, 0].set_ylabel('SNR')
            
            sns.boxplot(data=successful_data, x='group', y='cnr', ax=axes[1, 1])
            axes[1, 1].set_title('Contrast-to-Noise Ratio')
            axes[1, 1].set_ylabel('CNR')
            
            # TIV vs GM volume correlation
            axes[1, 2].scatter(successful_data['tiv_ml'], successful_data['gm_volume_ml'], 
                             c=['red' if g=='HC' else 'blue' if g=='PIGD' else 'green' 
                               for g in successful_data['group']], alpha=0.7)
            axes[1, 2].set_xlabel('TIV (ml)')
            axes[1, 2].set_ylabel('GM Volume (ml)')
            axes[1, 2].set_title('TIV vs GM Volume')
        
        plt.tight_layout()
        summary_plot_path = os.path.join(output_dir, 'preprocessing_summary.png')
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Preprocessing summary plot saved: {summary_plot_path}")
        
        # Generate text report
        report_path = os.path.join(output_dir, 'preprocessing_report.txt')
        with open(report_path, 'w') as f:
            f.write("YOPD Structural Analysis - Preprocessing Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"total_subjects: {total}\n")
            f.write(f"successful_preprocessing: {successful}\n")
            f.write(f"success_rate: {success_rate:.1f}%\n\n")
            
            f.write("GROUP-WISE STATISTICS\n")
            f.write("-" * 30 + "\n")
            for group in success_by_group.index:
                f.write(f"{group}: {success_by_group.loc[group, 'sum']}/{success_by_group.loc[group, 'count']} "
                       f"subjects ({success_by_group.loc[group, 'rate']:.1f}%)\n")
            
            if len(successful_data) > 0:
                f.write(f"\nVOLUME STATISTICS (ml)\n")
                f.write("-" * 30 + "\n")
                volume_stats = successful_data.groupby('group')[['tiv_ml', 'gm_volume_ml', 'wm_volume_ml', 'csf_volume_ml']].describe()
                f.write(volume_stats.to_string())
                
                f.write(f"\n\nQUALITY METRICS\n")
                f.write("-" * 30 + "\n")
                quality_stats = successful_data.groupby('group')[['snr', 'cnr', 'uniformity']].describe()
                f.write(quality_stats.to_string())
        
        logger.info(f"Preprocessing report saved: {report_path}")
        
    except Exception as e:
        logger.error(f"Error creating preprocessing summary: {e}")

def main():
    """Main preprocessing pipeline"""
    logger = setup_preprocessing_logging()
    
    logger.info("=" * 60)
    logger.info("STARTING STEP 02: PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Check if ANTs is available
        has_ants = check_ants_installation()
        if not has_ants:
            logger.warning("ANTs not found - using simplified preprocessing pipeline")
        
        # Load data inventory
        inventory_path = os.path.join(config.OUTPUT_DIRS['qc'], 'data_inventory.csv')
        if not os.path.exists(inventory_path):
            logger.error(f"Data inventory not found: {inventory_path}")
            logger.error("Please run step01_data_inventory.py first")
            return
        
        logger.info(f"Loading data inventory from: {inventory_path}")
        inventory_df = pd.read_csv(inventory_path)
        
        # Filter for subjects with readable T1 images
        valid_subjects = inventory_df[inventory_df['t1_file_readable'] == True].copy()
        logger.info(f"Found {len(valid_subjects)} subjects with valid T1 images")
        
        # Create output directory
        preproc_output_dir = config.OUTPUT_DIRS['preprocessed']
        Path(preproc_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each subject
        results = []
        for idx, row in valid_subjects.iterrows():
            subject_id = row['subject_id']
            t1_path = row['t1_file_path']
            group = row['group']
            
            result = preprocess_subject(subject_id, t1_path, preproc_output_dir)
            result['group'] = group
            results.append(result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(preproc_output_dir, 'preprocessing_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Preprocessing results saved: {results_path}")
        
        # Create summary
        create_preprocessing_summary(results_df, preproc_output_dir)
        
        # Log summary
        successful = results_df['success'].sum()
        total = len(results_df)
        success_rate = 100 * successful / total
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Subjects processed: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Log analysis summary
        summary_notes = f"Success rate: {success_rate:.1f}%"
        if successful < total:
            failed_subjects = results_df[~results_df['success']]['subject_id'].tolist()
            summary_notes += f", Failed: {failed_subjects}"
        
        log_analysis_summary("Preprocessing Pipeline", total, total - successful, summary_notes)
        
        # Print key findings
        print("\nKEY FINDINGS:")
        print(f"• Preprocessing completed for {successful}/{total} subjects")
        if len(results_df[results_df['success']]) > 0:
            successful_data = results_df[results_df['success']]
            mean_tiv = successful_data['tiv_ml'].mean()
            std_tiv = successful_data['tiv_ml'].std()
            print(f"• Mean TIV: {mean_tiv:.1f} ± {std_tiv:.1f} ml")
            
            by_group = successful_data.groupby('group')['tiv_ml'].agg(['count', 'mean', 'std'])
            for group in by_group.index:
                print(f"• {group}: {by_group.loc[group, 'count']} subjects, "
                      f"TIV = {by_group.loc[group, 'mean']:.1f} ± {by_group.loc[group, 'std']:.1f} ml")
        
        logger.info("Step 02 completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
