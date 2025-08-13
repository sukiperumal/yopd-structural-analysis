#!/usr/bin/env python3
"""
Step 04: Surface-based Cortical Thickness Analysis
YOPD Structural Analysis Pipeline

This script performs surface-based cortical thickness analysis including:
1. Simulated FreeSurfer recon-all pipeline
2. Cortical thickness extraction
3. Surface-based smoothing
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
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import configuration and utilities
import config
from utils import setup_logging, log_analysis_summary

def setup_surface_directories(output_dir):
    """Create surface analysis-specific output directories"""
    surface_dirs = {
        'recon': os.path.join(output_dir, 'recon_all'),
        'thickness': os.path.join(output_dir, 'cortical_thickness'),
        'parcellation': os.path.join(output_dir, 'parcellation'),
        'smoothed': os.path.join(output_dir, 'smoothed_surfaces'),
        'stats': os.path.join(output_dir, 'statistics'),
        'qc': os.path.join(output_dir, 'quality_control'),
        'figures': os.path.join(output_dir, 'figures')
    }
    
    for name, path in surface_dirs.items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created {name} directory: {path}")
    
    return surface_dirs

def simulate_freesurfer_recon(subject_id, t1_path, surface_dirs):
    """
    Simulate FreeSurfer recon-all pipeline
    In practice, this would run: recon-all -i subject_T1.nii -s <subjID> -all
    """
    logging.info(f"Simulating FreeSurfer recon-all for {subject_id}")
    
    try:
        # Load T1 image
        t1_img = nib.load(t1_path)
        t1_data = t1_img.get_fdata()
        
        # Simulate cortical surface extraction and thickness calculation
        # This is a simplified simulation of what FreeSurfer does
        
        # Create brain mask (simple thresholding)
        brain_mask = t1_data > np.percentile(t1_data[t1_data > 0], 20)
        
        # Simulate cortical surface by finding tissue boundaries
        # Apply morphological operations to get cortical ribbon
        eroded = ndimage.binary_erosion(brain_mask, iterations=3)
        cortical_mask = brain_mask & ~eroded
        
        # Simulate thickness calculation using distance transforms
        # This is a very simplified version of cortical thickness estimation
        inner_surface = ndimage.binary_erosion(cortical_mask, iterations=1)
        outer_surface = ndimage.binary_dilation(cortical_mask, iterations=1)
        
        # Calculate distance-based thickness
        inner_dist = ndimage.distance_transform_edt(~inner_surface)
        outer_dist = ndimage.distance_transform_edt(~outer_surface)
        thickness_map = (inner_dist + outer_dist) * cortical_mask
        
        # Apply realistic thickness constraints (1-5mm)
        thickness_map = np.clip(thickness_map, 0, 5)
        
        # Create parcellation using simplified anatomical regions
        parcellation_map = create_simplified_parcellation(brain_mask, t1_data.shape)
        
        # Calculate Euler number (topology quality metric)
        euler_number = calculate_euler_number(cortical_mask)
        
        # Save outputs
        subject_recon_dir = os.path.join(surface_dirs['recon'], subject_id)
        os.makedirs(subject_recon_dir, exist_ok=True)
        
        # Save thickness map
        thickness_path = os.path.join(subject_recon_dir, f"{subject_id}_thickness.nii.gz")
        thickness_img = nib.Nifti1Image(thickness_map, t1_img.affine)
        nib.save(thickness_img, thickness_path)
        
        # Save parcellation
        parcellation_path = os.path.join(subject_recon_dir, f"{subject_id}_parcellation.nii.gz")
        parcellation_img = nib.Nifti1Image(parcellation_map.astype(np.int16), t1_img.affine)
        nib.save(parcellation_img, parcellation_path)
        
        # Save cortical mask
        cortical_path = os.path.join(subject_recon_dir, f"{subject_id}_cortical_mask.nii.gz")
        cortical_img = nib.Nifti1Image(cortical_mask.astype(np.uint8), t1_img.affine)
        nib.save(cortical_img, cortical_path)
        
        # Calculate summary statistics
        mean_thickness = np.mean(thickness_map[cortical_mask])
        total_cortical_volume = np.sum(cortical_mask) * np.prod(t1_img.header.get_zooms()[:3])
        
        logging.info(f"FreeSurfer recon completed for {subject_id}")
        logging.info(f"  Mean cortical thickness: {mean_thickness:.2f} mm")
        logging.info(f"  Cortical volume: {total_cortical_volume:.1f} mm³")
        logging.info(f"  Euler number: {euler_number}")
        
        return {
            'subject_id': subject_id,
            'thickness_path': thickness_path,
            'parcellation_path': parcellation_path,
            'cortical_path': cortical_path,
            'mean_thickness': mean_thickness,
            'cortical_volume': total_cortical_volume,
            'euler_number': euler_number,
            'success': True
        }
        
    except Exception as e:
        logging.error(f"FreeSurfer recon failed for {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'success': False,
            'error': str(e)
        }

def create_simplified_parcellation(brain_mask, shape):
    """Create a simplified anatomical parcellation"""
    parcellation = np.zeros(shape, dtype=np.int16)
    
    # Define anatomical regions based on position
    center_x, center_y, center_z = np.array(shape) // 2
    
    # Create basic anatomical regions
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if brain_mask[i, j, k]:
                    # Simplified anatomical assignment based on coordinates
                    if i < center_x * 0.7:  # Frontal
                        if j < center_y:  # Left hemisphere
                            parcellation[i, j, k] = 1  # Left frontal
                        else:
                            parcellation[i, j, k] = 2  # Right frontal
                    elif i < center_x * 1.3:  # Parietal
                        if j < center_y:
                            parcellation[i, j, k] = 3  # Left parietal
                        else:
                            parcellation[i, j, k] = 4  # Right parietal
                    else:  # Occipital
                        if j < center_y:
                            parcellation[i, j, k] = 5  # Left occipital
                        else:
                            parcellation[i, j, k] = 6  # Right occipital
                    
                    # Add temporal regions
                    if k < center_z * 0.8:
                        if j < center_y:
                            parcellation[i, j, k] = 7  # Left temporal
                        else:
                            parcellation[i, j, k] = 8  # Right temporal
    
    return parcellation

def calculate_euler_number(binary_mask):
    """Calculate Euler number for topology assessment"""
    # Simplified Euler number calculation
    # In practice, FreeSurfer uses more sophisticated methods
    labeled, num_components = ndimage.label(binary_mask)
    return num_components - 1  # Simplified approximation

def apply_surface_smoothing(thickness_map, cortical_mask, fwhm=15):
    """Apply surface-based smoothing to thickness maps"""
    # Convert FWHM to sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Apply Gaussian smoothing within cortical mask
    smoothed = ndimage.gaussian_filter(thickness_map, sigma=sigma)
    smoothed = smoothed * cortical_mask  # Preserve cortical boundaries
    
    logging.info(f"Applied surface smoothing with FWHM={fwhm}mm")
    return smoothed

def extract_roi_thickness(thickness_map, parcellation_map, roi_labels):
    """Extract mean thickness for each ROI"""
    roi_thickness = {}
    
    for roi_id, roi_name in roi_labels.items():
        roi_mask = parcellation_map == roi_id
        if np.sum(roi_mask) > 0:
            roi_thickness[roi_name] = np.mean(thickness_map[roi_mask])
        else:
            roi_thickness[roi_name] = np.nan
    
    return roi_thickness

def process_subject_surface(subject_id, surface_dirs, roi_labels):
    """Process single subject through surface analysis pipeline"""
    try:
        logging.info(f"Processing surface analysis for subject: {subject_id}")
        
        # Load thickness and parcellation maps
        subject_recon_dir = os.path.join(surface_dirs['recon'], subject_id)
        thickness_path = os.path.join(subject_recon_dir, f"{subject_id}_thickness.nii.gz")
        parcellation_path = os.path.join(subject_recon_dir, f"{subject_id}_parcellation.nii.gz")
        cortical_path = os.path.join(subject_recon_dir, f"{subject_id}_cortical_mask.nii.gz")
        
        if not all(os.path.exists(p) for p in [thickness_path, parcellation_path, cortical_path]):
            raise FileNotFoundError("Required surface files not found")
        
        # Load data
        thickness_img = nib.load(thickness_path)
        thickness_data = thickness_img.get_fdata()
        
        parcellation_img = nib.load(parcellation_path)
        parcellation_data = parcellation_img.get_fdata()
        
        cortical_img = nib.load(cortical_path)
        cortical_mask = cortical_img.get_fdata().astype(bool)
        
        # Apply surface smoothing
        smoothed_thickness = apply_surface_smoothing(thickness_data, cortical_mask, fwhm=15)
        
        # Save smoothed thickness
        smoothed_path = os.path.join(surface_dirs['smoothed'], f"{subject_id}_smoothed_thickness.nii.gz")
        smoothed_img = nib.Nifti1Image(smoothed_thickness, thickness_img.affine)
        nib.save(smoothed_img, smoothed_path)
        
        # Extract ROI thickness values
        roi_thickness = extract_roi_thickness(smoothed_thickness, parcellation_data, roi_labels)
        
        # Calculate quality metrics
        mean_thickness = np.mean(smoothed_thickness[cortical_mask])
        std_thickness = np.std(smoothed_thickness[cortical_mask])
        
        logging.info(f"Surface processing completed for {subject_id}")
        logging.info(f"  Mean thickness: {mean_thickness:.2f} ± {std_thickness:.2f} mm")
        
        return {
            'subject_id': subject_id,
            'smoothed_path': smoothed_path,
            'mean_thickness': mean_thickness,
            'std_thickness': std_thickness,
            'roi_thickness': roi_thickness,
            'success': True
        }
        
    except Exception as e:
        logging.error(f"Surface processing failed for {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'success': False,
            'error': str(e)
        }

def perform_surface_statistics(surface_results, roi_labels, surface_dirs):
    """Perform statistical analysis on surface data"""
    logging.info("Performing surface-based statistical analysis...")
    
    # Collect data by group
    groups = {'HC': [], 'PIGD': [], 'TDPD': []}
    roi_data = {roi_name: {'HC': [], 'PIGD': [], 'TDPD': []} for roi_name in roi_labels.values()}
    
    for result in surface_results:
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
            groups[group].append(result['mean_thickness'])
            
            # Collect ROI data
            for roi_name, thickness_value in result['roi_thickness'].items():
                if not np.isnan(thickness_value):
                    roi_data[roi_name][group].append(thickness_value)
    
    # Perform statistical tests
    stats_results = {}
    
    # Global thickness comparisons
    if len(groups['HC']) > 0 and len(groups['PIGD']) > 0:
        t_stat, p_val = stats.ttest_ind(groups['HC'], groups['PIGD'])
        stats_results['global_hc_vs_pigd'] = {
            't_stat': t_stat,
            'p_value': p_val,
            'mean_diff': np.mean(groups['HC']) - np.mean(groups['PIGD']),
            'effect_size': t_stat / np.sqrt(len(groups['HC']) + len(groups['PIGD']))
        }
    
    if len(groups['HC']) > 0 and len(groups['TDPD']) > 0:
        t_stat, p_val = stats.ttest_ind(groups['HC'], groups['TDPD'])
        stats_results['global_hc_vs_tdpd'] = {
            't_stat': t_stat,
            'p_value': p_val,
            'mean_diff': np.mean(groups['HC']) - np.mean(groups['TDPD']),
            'effect_size': t_stat / np.sqrt(len(groups['HC']) + len(groups['TDPD']))
        }
    
    # ROI-wise comparisons
    roi_stats = {}
    for roi_name in roi_labels.values():
        roi_stats[roi_name] = {}
        
        if len(roi_data[roi_name]['HC']) > 0 and len(roi_data[roi_name]['PIGD']) > 0:
            t_stat, p_val = stats.ttest_ind(roi_data[roi_name]['HC'], roi_data[roi_name]['PIGD'])
            roi_stats[roi_name]['hc_vs_pigd'] = {
                't_stat': t_stat,
                'p_value': p_val,
                'mean_diff': np.mean(roi_data[roi_name]['HC']) - np.mean(roi_data[roi_name]['PIGD'])
            }
    
    # Apply FDR correction for ROI tests
    roi_p_values = []
    roi_names = []
    for roi_name, tests in roi_stats.items():
        for test_name, test_result in tests.items():
            roi_p_values.append(test_result['p_value'])
            roi_names.append(f"{roi_name}_{test_name}")
    
    if roi_p_values:
        from statsmodels.stats.multitest import fdrcorrection
        reject, pvals_corrected = fdrcorrection(roi_p_values, alpha=0.05)
        
        # Update results with corrected p-values
        for i, (roi_name, corrected_p) in enumerate(zip(roi_names, pvals_corrected)):
            parts = roi_name.split('_')
            roi = '_'.join(parts[:-2])
            test = '_'.join(parts[-2:])
            if roi in roi_stats and test in roi_stats[roi]:
                roi_stats[roi][test]['p_corrected'] = corrected_p
                roi_stats[roi][test]['significant'] = reject[i]
    
    # Save results
    stats_summary = {
        'global_tests': stats_results,
        'roi_tests': roi_stats,
        'group_means': {group: np.mean(values) for group, values in groups.items() if values},
        'group_stds': {group: np.std(values) for group, values in groups.items() if values}
    }
    
    # Save detailed results
    stats_file = os.path.join(surface_dirs['stats'], 'surface_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("SURFACE-BASED CORTICAL THICKNESS STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("GLOBAL THICKNESS COMPARISONS:\n")
        f.write("-" * 30 + "\n")
        for test_name, result in stats_results.items():
            f.write(f"{test_name}:\n")
            f.write(f"  t-statistic: {result['t_stat']:.3f}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Mean difference: {result['mean_diff']:.3f} mm\n")
            f.write(f"  Effect size: {result['effect_size']:.3f}\n\n")
        
        f.write("ROI-WISE COMPARISONS (FDR corrected):\n")
        f.write("-" * 40 + "\n")
        for roi_name, tests in roi_stats.items():
            f.write(f"{roi_name}:\n")
            for test_name, result in tests.items():
                f.write(f"  {test_name}:\n")
                f.write(f"    t-stat: {result['t_stat']:.3f}, ")
                f.write(f"p-uncorr: {result['p_value']:.6f}, ")
                f.write(f"p-FDR: {result.get('p_corrected', 'N/A'):.6f}\n")
            f.write("\n")
    
    logging.info(f"Surface statistics saved to: {stats_file}")
    return stats_summary

def generate_surface_qc_report(surface_results, surface_dirs):
    """Generate surface analysis quality control report"""
    logging.info("Generating surface QC report...")
    
    # Collect QC data
    qc_data = []
    for result in surface_results:
        if result['success']:
            subject_id = result['subject_id']
            group = 'HC' if 'HC' in subject_id else ('PIGD' if 'PIGD' in subject_id else 'TDPD')
            
            qc_data.append({
                'subject_id': subject_id,
                'group': group,
                'mean_thickness': result['mean_thickness'],
                'std_thickness': result['std_thickness']
            })
    
    if not qc_data:
        logging.warning("No QC data available for surface report")
        return None
    
    qc_df = pd.DataFrame(qc_data)
    
    # Save QC data
    qc_csv_path = os.path.join(surface_dirs['qc'], 'surface_qc_metrics.csv')
    qc_df.to_csv(qc_csv_path, index=False)
    
    # Generate summary statistics
    summary_stats = qc_df.groupby('group').agg({
        'mean_thickness': ['mean', 'std', 'min', 'max'],
        'std_thickness': ['mean', 'std']
    }).round(3)
    
    # Generate QC report
    report_path = os.path.join(surface_dirs['qc'], 'surface_qc_report.txt')
    with open(report_path, 'w') as f:
        f.write("SURFACE ANALYSIS QUALITY CONTROL REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total subjects processed: {len(qc_df)}\n")
        f.write(f"Groups: {qc_df['group'].value_counts().to_dict()}\n\n")
        
        f.write("CORTICAL THICKNESS SUMMARY BY GROUP:\n")
        f.write("-" * 40 + "\n")
        f.write(str(summary_stats))
        f.write("\n\n")
        
        # Quality flags
        f.write("QUALITY FLAGS:\n")
        f.write("-" * 15 + "\n")
        
        # Check for outliers
        thickness_outliers = qc_df[
            (qc_df['mean_thickness'] < qc_df['mean_thickness'].quantile(0.05)) |
            (qc_df['mean_thickness'] > qc_df['mean_thickness'].quantile(0.95))
        ]
        f.write(f"Thickness outliers (5th/95th percentile): {len(thickness_outliers)} subjects\n")
        
        if len(thickness_outliers) > 0:
            f.write(f"Outlier subjects: {thickness_outliers['subject_id'].tolist()}\n")
    
    logging.info(f"Surface QC report saved to: {report_path}")
    return qc_df

def create_surface_visualizations(surface_results, stats_summary, qc_df, surface_dirs):
    """Create surface analysis visualization plots"""
    logging.info("Creating surface visualization plots...")
    
    plt.style.use('default')
    
    # 1. Cortical thickness by group
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Cortical Thickness Analysis', fontsize=16)
    
    # Mean thickness boxplot
    sns.boxplot(data=qc_df, x='group', y='mean_thickness', ax=axes[0])
    axes[0].set_title('Mean Cortical Thickness by Group')
    axes[0].set_ylabel('Thickness (mm)')
    
    # Thickness variability
    sns.boxplot(data=qc_df, x='group', y='std_thickness', ax=axes[1])
    axes[1].set_title('Cortical Thickness Variability')
    axes[1].set_ylabel('Standard Deviation (mm)')
    
    plt.tight_layout()
    thickness_plot_path = os.path.join(surface_dirs['figures'], 'cortical_thickness_analysis.png')
    plt.savefig(thickness_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical results summary
    if 'global_tests' in stats_summary and stats_summary['global_tests']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        tests = []
        p_values = []
        effect_sizes = []
        
        for test_name, result in stats_summary['global_tests'].items():
            tests.append(test_name.replace('global_', '').replace('_', ' ').title())
            p_values.append(result['p_value'])
            effect_sizes.append(abs(result['effect_size']))
        
        # Create scatter plot of effect size vs p-value
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        ax.scatter(effect_sizes, [-np.log10(p) for p in p_values], c=colors, s=100)
        
        for i, test in enumerate(tests):
            ax.annotate(test, (effect_sizes[i], -np.log10(p_values[i])), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Effect Size')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('Surface-based Statistical Results')
        
        stats_plot_path = os.path.join(surface_dirs['figures'], 'surface_statistics.png')
        plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Surface plots saved to: {surface_dirs['figures']}")

def main():
    """Main surface analysis function"""
    # Setup logging
    logger = setup_logging('step04_surface')
    logger.info("=" * 60)
    logger.info("STARTING STEP 04: SURFACE-BASED CORTICAL THICKNESS ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Setup directories
        surface_output_dir = os.path.join(config.OUTPUT_ROOT, '04_surface_analysis')
        surface_dirs = setup_surface_directories(surface_output_dir)
        
        # Define ROI labels (simplified)
        roi_labels = {
            1: 'left_frontal',
            2: 'right_frontal',
            3: 'left_parietal',
            4: 'right_parietal',
            5: 'left_occipital',
            6: 'right_occipital',
            7: 'left_temporal',
            8: 'right_temporal'
        }
        
        # Load preprocessing results to get brain-extracted T1 images
        preproc_dir = os.path.join(config.OUTPUT_ROOT, '01_preprocessed')
        
        # Get all preprocessed subjects
        subject_dirs = [d for d in os.listdir(preproc_dir) 
                       if os.path.isdir(os.path.join(preproc_dir, d)) and d.startswith('sub-')]
        
        logger.info(f"Found {len(subject_dirs)} preprocessed subjects")
        
        # Phase 1: FreeSurfer recon-all simulation
        logger.info("Phase 1: Running FreeSurfer recon-all simulation...")
        recon_results = []
        
        for subject_id in sorted(subject_dirs):
            # Find brain-extracted T1 file
            t1_brain_path = os.path.join(preproc_dir, subject_id, f"{subject_id}_T1w_brain.nii.gz")
            
            if not os.path.exists(t1_brain_path):
                logger.warning(f"Brain-extracted T1 not found for {subject_id}: {t1_brain_path}")
                continue
            
            # Run FreeSurfer simulation
            result = simulate_freesurfer_recon(subject_id, t1_brain_path, surface_dirs)
            recon_results.append(result)
        
        successful_recon = sum(1 for r in recon_results if r['success'])
        logger.info(f"FreeSurfer recon completed: {successful_recon}/{len(subject_dirs)} subjects")
        
        # Phase 2: Surface processing and smoothing
        logger.info("Phase 2: Surface processing and smoothing...")
        surface_results = []
        
        for result in recon_results:
            if not result['success']:
                continue
            
            surface_result = process_subject_surface(result['subject_id'], surface_dirs, roi_labels)
            surface_results.append(surface_result)
        
        successful_surface = sum(1 for r in surface_results if r['success'])
        logger.info(f"Surface processing completed: {successful_surface}/{successful_recon} subjects")
        
        # Save surface processing results
        surface_results_df = pd.DataFrame([
            {
                'subject_id': r['subject_id'],
                'success': r['success'],
                'mean_thickness': r.get('mean_thickness', np.nan),
                'std_thickness': r.get('std_thickness', np.nan),
                'error': r.get('error', '')
            } for r in surface_results
        ])
        
        results_csv = os.path.join(surface_dirs['qc'], 'surface_processing_results.csv')
        surface_results_df.to_csv(results_csv, index=False)
        logger.info(f"Surface results saved to: {results_csv}")
        
        # Phase 3: Statistical analysis
        logger.info("Phase 3: Statistical analysis...")
        stats_summary = perform_surface_statistics(surface_results, roi_labels, surface_dirs)
        
        # Phase 4: Quality control and visualization
        logger.info("Phase 4: Quality control and visualization...")
        qc_df = generate_surface_qc_report(surface_results, surface_dirs)
        
        if qc_df is not None:
            create_surface_visualizations(surface_results, stats_summary, qc_df, surface_dirs)
        
        # Log summary
        log_analysis_summary(
            analysis_name="Surface-based Cortical Thickness Analysis",
            subjects_analyzed=successful_surface,
            subjects_excluded=len(subject_dirs) - successful_surface,
            notes=f"FreeSurfer recon simulation, cortical thickness extraction, and statistical analysis completed. "
                  f"ROI-wise analysis performed for {len(roi_labels)} regions."
        )
        
        logger.info("Step 04 completed successfully!")
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY: Surface-based Cortical Thickness Analysis")
        print("=" * 60)
        print(f"Subjects processed: {successful_surface}/{len(subject_dirs)}")
        print(f"Success rate: {100*successful_surface/len(subject_dirs):.1f}%")
        
        if qc_df is not None:
            print(f"Mean cortical thickness: {qc_df['mean_thickness'].mean():.2f} ± {qc_df['mean_thickness'].std():.2f} mm")
            for group in ['HC', 'PIGD', 'TDPD']:
                group_data = qc_df[qc_df['group'] == group]
                if len(group_data) > 0:
                    print(f"• {group}: {len(group_data)} subjects, "
                          f"thickness = {group_data['mean_thickness'].mean():.2f} ± {group_data['mean_thickness'].std():.2f} mm")
        
        if 'global_tests' in stats_summary:
            print(f"Statistical tests: {len(stats_summary['global_tests'])} global comparisons")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Surface analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
