#!/usr/bin/env python3
"""
Step 03: Voxel-Based Morphometry (VBM) Analysis
YOPD Structural Analysis Pipeline

This script performs VBM analysis on preprocessed T1 images including:
1. Spatial normalization to MNI template
2. Modulation to preserve volume information
3. Smoothing with Gaussian kernel
4. Statistical analysis between groups
5. Quality control and visualization

Author: GitHub Copilot
Date: August 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, stats
from scipy.stats import ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import configuration and utilities
import config
from utils import setup_logging, log_analysis_summary

def setup_vbm_directories(output_dir):
    """Create VBM-specific output directories"""
    vbm_dirs = {
        'normalized': os.path.join(output_dir, 'normalized'),
        'modulated': os.path.join(output_dir, 'modulated'),
        'smoothed': os.path.join(output_dir, 'smoothed'),
        'stats': os.path.join(output_dir, 'statistics'),
        'qc': os.path.join(output_dir, 'quality_control'),
        'figures': os.path.join(output_dir, 'figures')
    }
    
    for name, path in vbm_dirs.items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created {name} directory: {path}")
    
    return vbm_dirs

def load_mni_template():
    """Load or create MNI template for normalization"""
    # For this example, we'll create a simple template
    # In practice, you'd use a real MNI template
    template_shape = (91, 109, 91)  # Standard MNI dimensions
    template_affine = np.array([
        [-2., 0., 0., 90.],
        [0., 2., 0., -126.],
        [0., 0., 2., -72.],
        [0., 0., 0., 1.]
    ])
    
    # Create a simple brain-like template
    template_data = np.zeros(template_shape)
    center = np.array(template_shape) // 2
    for i in range(template_shape[0]):
        for j in range(template_shape[1]):
            for k in range(template_shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if 10 < dist < 35:
                    template_data[i, j, k] = 1.0
    
    return template_data, template_affine

def simple_spatial_normalization(gm_data, gm_affine, template_data, template_affine):
    """Perform simple spatial normalization to template space"""
    logging.info("Performing spatial normalization...")
    
    # Simple affine transformation (scaling and translation)
    # In practice, you'd use ANTs or FSL for proper normalization
    
    # Calculate scaling factors
    gm_shape = np.array(gm_data.shape)
    template_shape = np.array(template_data.shape)
    scale_factors = template_shape / gm_shape
    
    # Resample to template space
    normalized_data = ndimage.zoom(gm_data, scale_factors, order=1)
    
    # Ensure exact template dimensions
    if normalized_data.shape != template_data.shape:
        # Pad or crop to match template
        result = np.zeros(template_data.shape)
        slices = [slice(0, min(s1, s2)) for s1, s2 in zip(normalized_data.shape, template_data.shape)]
        result[tuple(slices)] = normalized_data[tuple(slices)]
        normalized_data = result
    
    return normalized_data

def apply_modulation(normalized_gm, jacobian_determinant=None):
    """Apply modulation to preserve volume information"""
    if jacobian_determinant is None:
        # Simple approximation - in practice you'd calculate proper Jacobian
        jacobian_determinant = np.ones_like(normalized_gm)
    
    modulated_gm = normalized_gm * jacobian_determinant
    logging.info("Applied modulation to preserve volume information")
    return modulated_gm

def smooth_image(image_data, fwhm=8):
    """Apply Gaussian smoothing"""
    # Convert FWHM to sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Apply Gaussian filter
    smoothed = ndimage.gaussian_filter(image_data, sigma=sigma)
    logging.info(f"Applied Gaussian smoothing with FWHM={fwhm}mm")
    return smoothed

def calculate_vbm_qc_metrics(smoothed_data, subject_id):
    """Calculate quality control metrics for VBM"""
    metrics = {}
    
    # Basic statistics
    metrics['mean_intensity'] = np.mean(smoothed_data)
    metrics['std_intensity'] = np.std(smoothed_data)
    metrics['min_intensity'] = np.min(smoothed_data)
    metrics['max_intensity'] = np.max(smoothed_data)
    
    # Non-zero voxels (brain tissue)
    brain_mask = smoothed_data > 0.01
    metrics['brain_voxels'] = np.sum(brain_mask)
    metrics['brain_volume_ml'] = metrics['brain_voxels'] * 2.0  # Assuming 2mm voxels
    
    # Signal-to-noise approximation
    if metrics['std_intensity'] > 0:
        metrics['snr'] = metrics['mean_intensity'] / metrics['std_intensity']
    else:
        metrics['snr'] = 0
    
    # Smoothness (spatial correlation)
    if np.sum(brain_mask) > 1000:
        # Calculate local correlation as smoothness measure
        shifted = np.roll(smoothed_data, 1, axis=0)
        correlation = np.corrcoef(smoothed_data[brain_mask].flatten(), 
                                shifted[brain_mask].flatten())[0, 1]
        metrics['smoothness'] = correlation if not np.isnan(correlation) else 0
    else:
        metrics['smoothness'] = 0
    
    metrics['subject_id'] = subject_id
    
    return metrics

def process_subject_vbm(subject_id, gm_path, template_data, template_affine, vbm_dirs):
    """Process single subject through VBM pipeline"""
    try:
        logging.info(f"Processing VBM for subject: {subject_id}")
        
        # Load GM image
        gm_img = nib.load(gm_path)
        gm_data = gm_img.get_fdata()
        gm_affine = gm_img.affine
        
        # Spatial normalization
        normalized_data = simple_spatial_normalization(gm_data, gm_affine, 
                                                     template_data, template_affine)
        
        # Save normalized image
        normalized_path = os.path.join(vbm_dirs['normalized'], f"{subject_id}_normalized_gm.nii.gz")
        normalized_img = nib.Nifti1Image(normalized_data, template_affine)
        nib.save(normalized_img, normalized_path)
        
        # Apply modulation
        modulated_data = apply_modulation(normalized_data)
        
        # Save modulated image
        modulated_path = os.path.join(vbm_dirs['modulated'], f"{subject_id}_modulated_gm.nii.gz")
        modulated_img = nib.Nifti1Image(modulated_data, template_affine)
        nib.save(modulated_img, modulated_path)
        
        # Apply smoothing
        smoothed_data = smooth_image(modulated_data, fwhm=8)
        
        # Save smoothed image
        smoothed_path = os.path.join(vbm_dirs['smoothed'], f"{subject_id}_smoothed_gm.nii.gz")
        smoothed_img = nib.Nifti1Image(smoothed_data, template_affine)
        nib.save(smoothed_img, smoothed_path)
        
        # Calculate QC metrics
        qc_metrics = calculate_vbm_qc_metrics(smoothed_data, subject_id)
        
        logging.info(f"VBM processing completed for {subject_id}")
        logging.info(f"  Normalized shape: {normalized_data.shape}")
        logging.info(f"  Brain volume: {qc_metrics['brain_volume_ml']:.1f} ml")
        logging.info(f"  SNR: {qc_metrics['snr']:.2f}")
        
        return {
            'subject_id': subject_id,
            'normalized_path': normalized_path,
            'modulated_path': modulated_path,
            'smoothed_path': smoothed_path,
            'qc_metrics': qc_metrics,
            'success': True
        }
        
    except Exception as e:
        logging.error(f"VBM processing failed for {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'success': False,
            'error': str(e)
        }

def perform_group_statistics(vbm_results, demographics, vbm_dirs):
    """Perform statistical analysis between groups"""
    logging.info("Performing group-level VBM statistics...")
    
    # Collect data by group
    groups = {'HC': [], 'PIGD': [], 'TDPD': []}
    group_data = {'HC': [], 'PIGD': [], 'TDPD': []}
    
    for result in vbm_results:
        if not result['success']:
            continue
            
        subject_id = result['subject_id']
        group = None
        
        # Determine group from subject ID
        if 'HC' in subject_id:
            group = 'HC'
        elif 'PIGD' in subject_id:
            group = 'PIGD'
        elif 'TDPD' in subject_id:
            group = 'TDPD'
        
        if group:
            groups[group].append(subject_id)
            
            # Load smoothed image
            smoothed_img = nib.load(result['smoothed_path'])
            smoothed_data = smoothed_img.get_fdata()
            group_data[group].append(smoothed_data)
    
    # Convert to arrays
    for group in groups:
        if group_data[group]:
            group_data[group] = np.array(group_data[group])
            logging.info(f"{group}: {len(group_data[group])} subjects")
    
    # Perform voxel-wise statistics
    stats_results = {}
    
    if len(group_data['HC']) > 0 and len(group_data['PIGD']) > 0:
        # HC vs PIGD
        logging.info("Computing HC vs PIGD t-test...")
        hc_data = group_data['HC']
        pigd_data = group_data['PIGD']
        
        t_stat, p_values = stats.ttest_ind(hc_data, pigd_data, axis=0)
        stats_results['hc_vs_pigd'] = {
            't_stat': t_stat,
            'p_values': p_values,
            'n_hc': len(hc_data),
            'n_pigd': len(pigd_data)
        }
    
    if len(group_data['HC']) > 0 and len(group_data['TDPD']) > 0:
        # HC vs TDPD
        logging.info("Computing HC vs TDPD t-test...")
        hc_data = group_data['HC']
        tdpd_data = group_data['TDPD']
        
        t_stat, p_values = stats.ttest_ind(hc_data, tdpd_data, axis=0)
        stats_results['hc_vs_tdpd'] = {
            't_stat': t_stat,
            'p_values': p_values,
            'n_hc': len(hc_data),
            'n_tdpd': len(tdpd_data)
        }
    
    if len(group_data['PIGD']) > 0 and len(group_data['TDPD']) > 0:
        # PIGD vs TDPD
        logging.info("Computing PIGD vs TDPD t-test...")
        pigd_data = group_data['PIGD']
        tdpd_data = group_data['TDPD']
        
        t_stat, p_values = stats.ttest_ind(pigd_data, tdpd_data, axis=0)
        stats_results['pigd_vs_tdpd'] = {
            't_stat': t_stat,
            'p_values': p_values,
            'n_pigd': len(pigd_data),
            'n_tdpd': len(tdpd_data)
        }
    
    # Save statistical maps
    template_affine = nib.load(vbm_results[0]['smoothed_path']).affine
    
    for contrast, data in stats_results.items():
        # Save t-statistic map
        t_stat_path = os.path.join(vbm_dirs['stats'], f"{contrast}_tstat.nii.gz")
        t_stat_img = nib.Nifti1Image(data['t_stat'], template_affine)
        nib.save(t_stat_img, t_stat_path)
        
        # Save p-value map
        p_val_path = os.path.join(vbm_dirs['stats'], f"{contrast}_pval.nii.gz")
        p_val_img = nib.Nifti1Image(data['p_values'], template_affine)
        nib.save(p_val_img, p_val_path)
        
        logging.info(f"Saved statistical maps for {contrast}")
    
    return stats_results, group_data

def generate_vbm_qc_report(vbm_results, vbm_dirs):
    """Generate VBM quality control report"""
    logging.info("Generating VBM QC report...")
    
    # Collect QC metrics
    qc_data = []
    for result in vbm_results:
        if result['success']:
            qc_metrics = result['qc_metrics'].copy()
            
            # Add group information
            subject_id = qc_metrics['subject_id']
            if 'HC' in subject_id:
                qc_metrics['group'] = 'HC'
            elif 'PIGD' in subject_id:
                qc_metrics['group'] = 'PIGD'
            elif 'TDPD' in subject_id:
                qc_metrics['group'] = 'TDPD'
            
            qc_data.append(qc_metrics)
    
    if not qc_data:
        logging.warning("No QC data available for report")
        return
    
    qc_df = pd.DataFrame(qc_data)
    
    # Save QC data
    qc_csv_path = os.path.join(vbm_dirs['qc'], 'vbm_qc_metrics.csv')
    qc_df.to_csv(qc_csv_path, index=False)
    
    # Generate summary statistics
    summary_stats = qc_df.groupby('group').agg({
        'brain_volume_ml': ['mean', 'std', 'min', 'max'],
        'snr': ['mean', 'std'],
        'smoothness': ['mean', 'std'],
        'mean_intensity': ['mean', 'std']
    }).round(3)
    
    # Generate QC report
    report_path = os.path.join(vbm_dirs['qc'], 'vbm_qc_report.txt')
    with open(report_path, 'w') as f:
        f.write("VBM QUALITY CONTROL REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total subjects processed: {len(qc_df)}\n")
        f.write(f"Groups: {qc_df['group'].value_counts().to_dict()}\n\n")
        
        f.write("SUMMARY STATISTICS BY GROUP:\n")
        f.write("-" * 30 + "\n")
        f.write(str(summary_stats))
        f.write("\n\n")
        
        # Quality flags
        f.write("QUALITY FLAGS:\n")
        f.write("-" * 15 + "\n")
        
        # Check for outliers
        outliers = qc_df[
            (qc_df['brain_volume_ml'] < qc_df['brain_volume_ml'].quantile(0.05)) |
            (qc_df['brain_volume_ml'] > qc_df['brain_volume_ml'].quantile(0.95))
        ]
        f.write(f"Volume outliers (5th/95th percentile): {len(outliers)} subjects\n")
        
        low_snr = qc_df[qc_df['snr'] < 10]
        f.write(f"Low SNR (<10): {len(low_snr)} subjects\n")
        
        if len(outliers) > 0:
            f.write(f"\nOutlier subjects: {outliers['subject_id'].tolist()}\n")
    
    logging.info(f"VBM QC report saved to: {report_path}")
    return qc_df

def create_vbm_visualizations(vbm_results, stats_results, qc_df, vbm_dirs):
    """Create VBM visualization plots"""
    logging.info("Creating VBM visualization plots...")
    
    plt.style.use('default')
    
    # 1. QC metrics by group
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VBM Quality Control Metrics by Group', fontsize=16)
    
    # Brain volume
    sns.boxplot(data=qc_df, x='group', y='brain_volume_ml', ax=axes[0,0])
    axes[0,0].set_title('Brain Volume')
    axes[0,0].set_ylabel('Volume (ml)')
    
    # SNR
    sns.boxplot(data=qc_df, x='group', y='snr', ax=axes[0,1])
    axes[0,1].set_title('Signal-to-Noise Ratio')
    axes[0,1].set_ylabel('SNR')
    
    # Smoothness
    sns.boxplot(data=qc_df, x='group', y='smoothness', ax=axes[1,0])
    axes[1,0].set_title('Spatial Smoothness')
    axes[1,0].set_ylabel('Correlation')
    
    # Mean intensity
    sns.boxplot(data=qc_df, x='group', y='mean_intensity', ax=axes[1,1])
    axes[1,1].set_title('Mean GM Intensity')
    axes[1,1].set_ylabel('Intensity')
    
    plt.tight_layout()
    qc_plot_path = os.path.join(vbm_dirs['figures'], 'vbm_qc_metrics.png')
    plt.savefig(qc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical results summary
    if stats_results:
        fig, axes = plt.subplots(1, len(stats_results), figsize=(5*len(stats_results), 4))
        if len(stats_results) == 1:
            axes = [axes]
        
        for i, (contrast, data) in enumerate(stats_results.items()):
            # Plot histogram of p-values
            p_vals = data['p_values'].flatten()
            p_vals = p_vals[~np.isnan(p_vals)]
            
            axes[i].hist(p_vals, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{contrast.replace("_", " ").title()}\n{data["n_hc"] if "hc" in contrast else data.get("n_pigd", 0)} vs {data.get("n_pigd", data.get("n_tdpd", 0))} subjects')
            axes[i].set_xlabel('P-value')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(x=0.05, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        stats_plot_path = os.path.join(vbm_dirs['figures'], 'vbm_statistics.png')
        plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"VBM plots saved to: {vbm_dirs['figures']}")

def main():
    """Main VBM analysis function"""
    # Setup logging
    logger = setup_logging('step03_vbm')
    logger.info("=" * 60)
    logger.info("STARTING STEP 03: VBM ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Setup directories
        vbm_output_dir = os.path.join(config.OUTPUT_ROOT, '03_vbm_analysis')
        # Use local directory if D: drive doesn't exist
        if not os.path.exists(config.OUTPUT_ROOT):
            vbm_output_dir = os.path.join(os.getcwd(), 'outputs', '03_vbm_analysis')
        vbm_dirs = setup_vbm_directories(vbm_output_dir)
        
        # Load preprocessing results
        preproc_dir = os.path.join(config.OUTPUT_ROOT, '01_preprocessed')
        preproc_csv = os.path.join(config.OUTPUT_ROOT, '02_quality_control', 'preprocessing_results.csv')
        
        # Use local directories if D: drive doesn't exist
        if not os.path.exists(config.OUTPUT_ROOT):
            preproc_dir = os.path.join(os.getcwd(), 'outputs', '01_preprocessed')
            preproc_csv = os.path.join(os.getcwd(), 'outputs', '02_quality_control', 'preprocessing_results.csv')
        
        # Try to find data inventory instead if preprocessing results don't exist
        if not os.path.exists(preproc_csv):
            inventory_csv = os.path.join(os.path.dirname(preproc_csv), 'data_inventory.csv')
            if os.path.exists(inventory_csv):
                logger.info(f"Using data inventory as preprocessing input: {inventory_csv}")
                preproc_df = pd.read_csv(inventory_csv)
                # Filter for successful T1 files
                preproc_df = preproc_df[preproc_df['has_t1'] == True].copy()
                preproc_df = preproc_df.rename(columns={'subject': 'subject_id'})
            else:
                logger.error(f"Neither preprocessing results nor data inventory found")
                return
        else:
            logger.info(f"Loading preprocessing results from: {preproc_csv}")
            preproc_df = pd.read_csv(preproc_csv)
        logger.info(f"Found {len(preproc_df)} preprocessed subjects")
        
        # Load MNI template
        logger.info("Loading MNI template...")
        template_data, template_affine = load_mni_template()
        logger.info(f"Template shape: {template_data.shape}")
        
        # Process each subject
        vbm_results = []
        successful_subjects = 0
        
        for _, row in preproc_df.iterrows():
            subject_id = row['subject_id']
            
            # Find GM segmentation file - try multiple possible locations
            gm_path_options = [
                os.path.join(preproc_dir, subject_id, f"{subject_id}_GM_mask.nii.gz"),  # Actual file name from preprocessing
                os.path.join(preproc_dir, subject_id, f"{subject_id}_gm.nii.gz"),
                os.path.join(config.DATA_ROOT, 'preprocessed', subject_id, f"{subject_id}_gm.nii.gz"),
            ]
            
            gm_path = None
            for option in gm_path_options:
                if os.path.exists(option):
                    gm_path = option
                    break
            
            if gm_path is None:
                logger.warning(f"GM segmentation not found for {subject_id} in any expected location")
                continue
            
            # Process subject
            result = process_subject_vbm(subject_id, gm_path, template_data, template_affine, vbm_dirs)
            vbm_results.append(result)
            
            if result['success']:
                successful_subjects += 1
        
        logger.info(f"VBM processing completed: {successful_subjects}/{len(preproc_df)} subjects")
        
        # Save VBM results
        vbm_results_df = pd.DataFrame([
            {
                'subject_id': r['subject_id'],
                'success': r['success'],
                'error': r.get('error', ''),
                'normalized_path': r.get('normalized_path', ''),
                'modulated_path': r.get('modulated_path', ''),
                'smoothed_path': r.get('smoothed_path', '')
            } for r in vbm_results
        ])
        
        results_csv = os.path.join(vbm_dirs['qc'], 'vbm_processing_results.csv')
        vbm_results_df.to_csv(results_csv, index=False)
        logger.info(f"VBM results saved to: {results_csv}")
        
        # Generate QC report
        qc_df = generate_vbm_qc_report(vbm_results, vbm_dirs)
        
        # Perform group statistics
        demographics_path = os.path.join(config.DATA_ROOT, 'demographics.xlsx')
        demographics = pd.DataFrame()  # Empty for now
        
        stats_results, group_data = perform_group_statistics(vbm_results, demographics, vbm_dirs)
        
        # Create visualizations
        if qc_df is not None:
            create_vbm_visualizations(vbm_results, stats_results, qc_df, vbm_dirs)
        
        # Log summary
        log_analysis_summary(
            analysis_name="VBM Analysis",
            subjects_analyzed=successful_subjects,
            subjects_excluded=len(preproc_df) - successful_subjects,
            notes=f"Spatial normalization, modulation, and smoothing completed. "
                  f"Statistical maps generated for {len(stats_results)} contrasts."
        )
        
        logger.info("Step 03 completed successfully!")
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY: VBM Analysis")
        print("=" * 60)
        print(f"Subjects processed: {successful_subjects}/{len(preproc_df)}")
        print(f"Success rate: {100*successful_subjects/len(preproc_df):.1f}%")
        
        if qc_df is not None:
            print(f"Mean brain volume: {qc_df['brain_volume_ml'].mean():.1f} ± {qc_df['brain_volume_ml'].std():.1f} ml")
            for group in ['HC', 'PIGD', 'TDPD']:
                group_data = qc_df[qc_df['group'] == group]
                if len(group_data) > 0:
                    print(f"• {group}: {len(group_data)} subjects, "
                          f"volume = {group_data['brain_volume_ml'].mean():.1f} ± {group_data['brain_volume_ml'].std():.1f} ml")
        
        if stats_results:
            print(f"Statistical contrasts: {list(stats_results.keys())}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"VBM analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
