#!/usr/bin/env python3
"""
Step 05: Region of Interest (ROI) Analysis
YOPD Structural Analysis Pipeline

This script performs comprehensive ROI-based analysis including:
1. Atlas-based parcellation
2. Regional volume and thickness extraction
3. Statistical comparisons between groups
4. Multiple comparison correction
5. Effect size calculations and visualization

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
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, f_oneway, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')

# Import configuration and utilities
import config
from utils import setup_logging, log_analysis_summary

def load_subject_group_mapping():
    """Load subject-to-group mapping from directory structure"""
    mapping = {}
    
    # Load from directories
    data_root = config.DATA_ROOT
    for group in ['HC', 'PIGD', 'TDPD']:
        group_dir = os.path.join(data_root, group)
        if os.path.exists(group_dir):
            subjects = [d for d in os.listdir(group_dir) if d.startswith('sub-')]
            for subject in subjects:
                mapping[subject] = group
    
    logging.info(f"Loaded group mapping for {len(mapping)} subjects")
    for group in ['HC', 'PIGD', 'TDPD']:
        count = sum(1 for g in mapping.values() if g == group)
        logging.info(f"  {group}: {count} subjects")
    
    return mapping

def setup_roi_directories(output_dir):
    """Create ROI analysis-specific output directories"""
    roi_dirs = {
        'parcellations': os.path.join(output_dir, 'parcellations'),
        'extracted_data': os.path.join(output_dir, 'extracted_data'),
        'statistics': os.path.join(output_dir, 'statistics'),
        'qc': os.path.join(output_dir, 'quality_control'),
        'figures': os.path.join(output_dir, 'figures')
    }
    
    for name, path in roi_dirs.items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created {name} directory: {path}")
    
    return roi_dirs

def create_comprehensive_atlas():
    """
    Create a comprehensive anatomical atlas with multiple parcellation schemes
    In practice, this would use standardized atlases like AAL, Desikan-Killiany, etc.
    """
    atlas_info = {
        # Desikan-Killiany style cortical regions
        'cortical': {
            1: {'name': 'left_frontal_superior', 'hemisphere': 'left', 'lobe': 'frontal'},
            2: {'name': 'right_frontal_superior', 'hemisphere': 'right', 'lobe': 'frontal'},
            3: {'name': 'left_frontal_middle', 'hemisphere': 'left', 'lobe': 'frontal'},
            4: {'name': 'right_frontal_middle', 'hemisphere': 'right', 'lobe': 'frontal'},
            5: {'name': 'left_frontal_inferior', 'hemisphere': 'left', 'lobe': 'frontal'},
            6: {'name': 'right_frontal_inferior', 'hemisphere': 'right', 'lobe': 'frontal'},
            7: {'name': 'left_precentral', 'hemisphere': 'left', 'lobe': 'frontal'},
            8: {'name': 'right_precentral', 'hemisphere': 'right', 'lobe': 'frontal'},
            9: {'name': 'left_postcentral', 'hemisphere': 'left', 'lobe': 'parietal'},
            10: {'name': 'right_postcentral', 'hemisphere': 'right', 'lobe': 'parietal'},
            11: {'name': 'left_parietal_superior', 'hemisphere': 'left', 'lobe': 'parietal'},
            12: {'name': 'right_parietal_superior', 'hemisphere': 'right', 'lobe': 'parietal'},
            13: {'name': 'left_parietal_inferior', 'hemisphere': 'left', 'lobe': 'parietal'},
            14: {'name': 'right_parietal_inferior', 'hemisphere': 'right', 'lobe': 'parietal'},
            15: {'name': 'left_temporal_superior', 'hemisphere': 'left', 'lobe': 'temporal'},
            16: {'name': 'right_temporal_superior', 'hemisphere': 'right', 'lobe': 'temporal'},
            17: {'name': 'left_temporal_middle', 'hemisphere': 'left', 'lobe': 'temporal'},
            18: {'name': 'right_temporal_middle', 'hemisphere': 'right', 'lobe': 'temporal'},
            19: {'name': 'left_temporal_inferior', 'hemisphere': 'left', 'lobe': 'temporal'},
            20: {'name': 'right_temporal_inferior', 'hemisphere': 'right', 'lobe': 'temporal'},
            21: {'name': 'left_occipital', 'hemisphere': 'left', 'lobe': 'occipital'},
            22: {'name': 'right_occipital', 'hemisphere': 'right', 'lobe': 'occipital'},
        },
        
        # Subcortical regions (relevant for PD)
        'subcortical': {
            30: {'name': 'left_caudate', 'hemisphere': 'left', 'structure': 'basal_ganglia'},
            31: {'name': 'right_caudate', 'hemisphere': 'right', 'structure': 'basal_ganglia'},
            32: {'name': 'left_putamen', 'hemisphere': 'left', 'structure': 'basal_ganglia'},
            33: {'name': 'right_putamen', 'hemisphere': 'right', 'structure': 'basal_ganglia'},
            34: {'name': 'left_globus_pallidus', 'hemisphere': 'left', 'structure': 'basal_ganglia'},
            35: {'name': 'right_globus_pallidus', 'hemisphere': 'right', 'structure': 'basal_ganglia'},
            36: {'name': 'left_substantia_nigra', 'hemisphere': 'left', 'structure': 'brainstem'},
            37: {'name': 'right_substantia_nigra', 'hemisphere': 'right', 'structure': 'brainstem'},
            38: {'name': 'left_subthalamic_nucleus', 'hemisphere': 'left', 'structure': 'basal_ganglia'},
            39: {'name': 'right_subthalamic_nucleus', 'hemisphere': 'right', 'structure': 'basal_ganglia'},
            40: {'name': 'left_thalamus', 'hemisphere': 'left', 'structure': 'thalamus'},
            41: {'name': 'right_thalamus', 'hemisphere': 'right', 'structure': 'thalamus'},
            42: {'name': 'left_hippocampus', 'hemisphere': 'left', 'structure': 'limbic'},
            43: {'name': 'right_hippocampus', 'hemisphere': 'right', 'structure': 'limbic'},
            44: {'name': 'left_amygdala', 'hemisphere': 'left', 'structure': 'limbic'},
            45: {'name': 'right_amygdala', 'hemisphere': 'right', 'structure': 'limbic'},
        }
    }
    
    # Combine all regions
    combined_atlas = {}
    combined_atlas.update(atlas_info['cortical'])
    combined_atlas.update(atlas_info['subcortical'])
    
    return combined_atlas, atlas_info

def create_atlas_parcellation(brain_shape, atlas_info):
    """Create a 3D atlas parcellation map"""
    parcellation = np.zeros(brain_shape, dtype=np.int16)
    
    # Define anatomical coordinates
    center_x, center_y, center_z = np.array(brain_shape) // 2
    
    # Create anatomical regions based on simplified coordinates
    for roi_id, roi_info in atlas_info.items():
        # Create region masks based on anatomical location
        mask = np.zeros(brain_shape, dtype=bool)
        
        if 'frontal' in roi_info['name']:
            if 'left' in roi_info['name']:
                mask[10:center_x-10, 10:center_y-10, center_z-20:center_z+20] = True
            else:
                mask[10:center_x-10, center_y+10:-10, center_z-20:center_z+20] = True
                
        elif 'parietal' in roi_info['name']:
            if 'left' in roi_info['name']:
                mask[center_x-10:center_x+30, 10:center_y-10, center_z-15:center_z+15] = True
            else:
                mask[center_x-10:center_x+30, center_y+10:-10, center_z-15:center_z+15] = True
                
        elif 'temporal' in roi_info['name']:
            if 'left' in roi_info['name']:
                mask[center_x-20:center_x+20, 10:center_y-10, 10:center_z-10] = True
            else:
                mask[center_x-20:center_x+20, center_y+10:-10, 10:center_z-10] = True
                
        elif 'occipital' in roi_info['name']:
            if 'left' in roi_info['name']:
                mask[-40:-10, 10:center_y-10, center_z-15:center_z+15] = True
            else:
                mask[-40:-10, center_y+10:-10, center_z-15:center_z+15] = True
                
        # Subcortical structures (smaller regions)
        elif any(struct in roi_info['name'] for struct in ['caudate', 'putamen', 'pallidus']):
            if 'left' in roi_info['name']:
                mask[center_x-5:center_x+5, center_y-20:center_y-10, center_z-5:center_z+5] = True
            else:
                mask[center_x-5:center_x+5, center_y+10:center_y+20, center_z-5:center_z+5] = True
                
        elif 'thalamus' in roi_info['name']:
            if 'left' in roi_info['name']:
                mask[center_x-5:center_x+5, center_y-10:center_y-5, center_z-5:center_z+5] = True
            else:
                mask[center_x-5:center_x+5, center_y+5:center_y+10, center_z-5:center_z+5] = True
                
        elif 'hippocampus' in roi_info['name']:
            if 'left' in roi_info['name']:
                mask[center_x+5:center_x+15, center_y-15:center_y-5, center_z-10:center_z] = True
            else:
                mask[center_x+5:center_x+15, center_y+5:center_y+15, center_z-10:center_z] = True
        
        # Apply mask to parcellation
        parcellation[mask] = roi_id
    
    return parcellation

def extract_roi_measures(subject_id, gm_volume_path, thickness_path, atlas_parcellation, atlas_info):
    """Extract ROI-based measures from volume and thickness maps"""
    try:
        # Load data
        gm_img = nib.load(gm_volume_path)
        gm_data = gm_img.get_fdata()
        
        thickness_data = None
        if thickness_path and os.path.exists(thickness_path):
            thickness_img = nib.load(thickness_path)
            thickness_data = thickness_img.get_fdata()
        
        # Extract measurements for each ROI
        roi_measures = {'subject_id': subject_id}
        
        for roi_id, roi_info in atlas_info.items():
            roi_name = roi_info['name']
            roi_mask = atlas_parcellation == roi_id
            
            if np.sum(roi_mask) == 0:
                # No voxels for this ROI
                roi_measures[f"{roi_name}_volume"] = np.nan
                roi_measures[f"{roi_name}_mean_gm"] = np.nan
                if thickness_data is not None:
                    roi_measures[f"{roi_name}_thickness"] = np.nan
                continue
            
            # Volume measures
            roi_volume = np.sum(roi_mask) * np.prod(gm_img.header.get_zooms()[:3])  # mm³
            roi_mean_gm = np.mean(gm_data[roi_mask])
            
            roi_measures[f"{roi_name}_volume"] = roi_volume
            roi_measures[f"{roi_name}_mean_gm"] = roi_mean_gm
            
            # Thickness measures (if available)
            if thickness_data is not None:
                roi_thickness = np.mean(thickness_data[roi_mask])
                roi_measures[f"{roi_name}_thickness"] = roi_thickness
        
        return roi_measures, True
        
    except Exception as e:
        logging.error(f"ROI extraction failed for {subject_id}: {str(e)}")
        return {'subject_id': subject_id}, False

def perform_normality_tests(data_array, group_labels):
    """Perform normality tests for each group"""
    normality_results = {}
    
    unique_groups = np.unique(group_labels)
    for group in unique_groups:
        group_data = data_array[group_labels == group]
        if len(group_data) >= 3:  # Minimum for Shapiro-Wilk
            stat, p_value = shapiro(group_data)
            normality_results[group] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        else:
            normality_results[group] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'is_normal': False
            }
    
    return normality_results

def calculate_effect_size(group1, group2, test_type='independent'):
    """Calculate effect size (Cohen's d for parametric, rank-biserial for non-parametric)"""
    if test_type == 'independent':
        # Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        if pooled_std == 0:
            return 0
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    else:
        # Rank-biserial correlation for Mann-Whitney U
        n1, n2 = len(group1), len(group2)
        U_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
        return (2 * U_stat) / (n1 * n2) - 1

def perform_roi_statistics(roi_data_df, atlas_info, roi_dirs):
    """Perform comprehensive statistical analysis on ROI data"""
    logging.info("Performing ROI-based statistical analysis...")
    
    # Get measure columns (exclude subject_id and group)
    measure_columns = [col for col in roi_data_df.columns 
                      if col not in ['subject_id', 'group'] and not roi_data_df[col].isna().all()]
    
    logging.info(f"Analyzing {len(measure_columns)} ROI measures")
    
    statistical_results = []
    
    for measure in measure_columns:
        # Clean data (remove NaN values)
        clean_data = roi_data_df[['group', measure]].dropna()
        
        if len(clean_data) < 6:  # Minimum for meaningful analysis
            continue
        
        # Separate groups
        groups = {}
        for group_name in ['HC', 'PIGD', 'TDPD']:
            group_data = clean_data[clean_data['group'] == group_name][measure].values
            if len(group_data) > 0:
                groups[group_name] = group_data
        
        if len(groups) < 2:
            continue
        
        # Perform pairwise comparisons
        comparisons = []
        if 'HC' in groups and 'PIGD' in groups:
            comparisons.append(('HC', 'PIGD'))
        if 'HC' in groups and 'TDPD' in groups:
            comparisons.append(('HC', 'TDPD'))
        if 'PIGD' in groups and 'TDPD' in groups:
            comparisons.append(('PIGD', 'TDPD'))
        
        for group1_name, group2_name in comparisons:
            group1_data = groups[group1_name]
            group2_data = groups[group2_name]
            
            # Test normality for both groups
            normality1 = perform_normality_tests(group1_data, np.array(['G1'] * len(group1_data)))
            normality2 = perform_normality_tests(group2_data, np.array(['G2'] * len(group2_data)))
            
            is_normal = (normality1['G1']['is_normal'] and normality2['G2']['is_normal'])
            
            # Choose appropriate test
            if is_normal:
                # Parametric test (t-test)
                stat, p_value = ttest_ind(group1_data, group2_data)
                test_type = 'independent_t_test'
                effect_size = calculate_effect_size(group1_data, group2_data, 'independent')
            else:
                # Non-parametric test (Mann-Whitney U)
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                test_type = 'mann_whitney_u'
                effect_size = calculate_effect_size(group1_data, group2_data, 'nonparametric')
            
            # Store results
            result = {
                'measure': measure,
                'comparison': f"{group1_name}_vs_{group2_name}",
                'test_type': test_type,
                'statistic': stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'group1_mean': np.mean(group1_data),
                'group1_std': np.std(group1_data),
                'group1_n': len(group1_data),
                'group2_mean': np.mean(group2_data),
                'group2_std': np.std(group2_data),
                'group2_n': len(group2_data),
                'normality_group1': normality1['G1']['p_value'],
                'normality_group2': normality2['G2']['p_value'],
                'is_parametric': is_normal
            }
            
            statistical_results.append(result)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(statistical_results)
    
    if len(stats_df) == 0:
        logging.warning("No statistical tests could be performed")
        return stats_df
    
    # Apply multiple comparison corrections
    p_values = stats_df['p_value'].values
    
    # FDR correction
    reject_fdr, pvals_fdr = fdrcorrection(p_values, alpha=0.05)
    stats_df['p_fdr'] = pvals_fdr
    stats_df['significant_fdr'] = reject_fdr
    
    # Bonferroni correction
    pvals_bonf = p_values * len(p_values)  # Simple Bonferroni
    pvals_bonf = np.minimum(pvals_bonf, 1.0)  # Cap at 1.0
    stats_df['p_bonferroni'] = pvals_bonf
    stats_df['significant_bonferroni'] = pvals_bonf < 0.05
    
    # Save detailed results
    stats_csv = os.path.join(roi_dirs['statistics'], 'roi_statistical_results.csv')
    stats_df.to_csv(stats_csv, index=False)
    
    # Create summary report
    summary_file = os.path.join(roi_dirs['statistics'], 'roi_statistics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("ROI-BASED STATISTICAL ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total comparisons performed: {len(stats_df)}\n")
        f.write(f"Parametric tests: {len(stats_df[stats_df['is_parametric']])}\n")
        f.write(f"Non-parametric tests: {len(stats_df[~stats_df['is_parametric']])}\n\n")
        
        f.write("SIGNIFICANT RESULTS (FDR corrected p < 0.05):\n")
        f.write("-" * 45 + "\n")
        
        significant_results = stats_df[stats_df['significant_fdr']].sort_values('p_fdr')
        
        if len(significant_results) > 0:
            for _, row in significant_results.iterrows():
                f.write(f"{row['measure']} ({row['comparison']}):\n")
                f.write(f"  Test: {row['test_type']}\n")
                f.write(f"  p-value (uncorrected): {row['p_value']:.6f}\n")
                f.write(f"  p-value (FDR): {row['p_fdr']:.6f}\n")
                f.write(f"  Effect size: {row['effect_size']:.3f}\n")
                f.write(f"  Group means: {row['group1_mean']:.3f} vs {row['group2_mean']:.3f}\n\n")
        else:
            f.write("No significant results found after FDR correction.\n\n")
        
        f.write("EFFECT SIZE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Large effects (|d| > 0.8): {len(stats_df[abs(stats_df['effect_size']) > 0.8])}\n")
        f.write(f"Medium effects (0.5 < |d| <= 0.8): {len(stats_df[(abs(stats_df['effect_size']) > 0.5) & (abs(stats_df['effect_size']) <= 0.8)])}\n")
        f.write(f"Small effects (0.2 < |d| <= 0.5): {len(stats_df[(abs(stats_df['effect_size']) > 0.2) & (abs(stats_df['effect_size']) <= 0.5)])}\n")
        f.write(f"Negligible effects (|d| <= 0.2): {len(stats_df[abs(stats_df['effect_size']) <= 0.2])}\n")
    
    logging.info(f"ROI statistics saved to: {stats_csv}")
    logging.info(f"Summary report saved to: {summary_file}")
    
    return stats_df

def create_roi_visualizations(roi_data_df, stats_df, roi_dirs):
    """Create comprehensive visualizations for ROI analysis"""
    logging.info("Creating ROI visualization plots...")
    
    plt.style.use('default')
    
    # 1. Summary of significant results
    if len(stats_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ROI Statistical Analysis Summary', fontsize=16)
        
        # Volcano plot (effect size vs -log10(p-value))
        x = stats_df['effect_size']
        y = -np.log10(stats_df['p_value'])
        colors = ['red' if sig else 'blue' for sig in stats_df['significant_fdr']]
        
        axes[0,0].scatter(x, y, c=colors, alpha=0.7)
        axes[0,0].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        axes[0,0].axvline(0, color='black', linestyle='-', alpha=0.3)
        axes[0,0].set_xlabel('Effect Size (Cohen\'s d)')
        axes[0,0].set_ylabel('-log10(p-value)')
        axes[0,0].set_title('Volcano Plot')
        
        # P-value distribution
        axes[0,1].hist(stats_df['p_value'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(0.05, color='red', linestyle='--', alpha=0.7)
        axes[0,1].set_xlabel('p-value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('P-value Distribution')
        
        # Effect size distribution
        axes[1,0].hist(stats_df['effect_size'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(0, color='black', linestyle='-', alpha=0.3)
        axes[1,0].set_xlabel('Effect Size')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Effect Size Distribution')
        
        # Test type summary
        test_counts = stats_df['test_type'].value_counts()
        axes[1,1].pie(test_counts.values, labels=test_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Statistical Test Types')
        
        plt.tight_layout()
        summary_plot_path = os.path.join(roi_dirs['figures'], 'roi_statistics_summary.png')
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. ROI measures by group (top significant results)
    if len(stats_df) > 0:
        significant_results = stats_df[stats_df['significant_fdr']].sort_values('p_fdr').head(12)
        
        if len(significant_results) > 0:
            n_plots = min(len(significant_results), 12)
            n_rows = (n_plots + 3) // 4
            
            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (_, row) in enumerate(significant_results.iterrows()):
                r, c = i // 4, i % 4
                
                measure = row['measure']
                if measure in roi_data_df.columns:
                    clean_data = roi_data_df[['group', measure]].dropna()
                    
                    if len(clean_data) > 0:
                        sns.boxplot(data=clean_data, x='group', y=measure, ax=axes[r,c])
                        axes[r,c].set_title(f"{measure}\np-FDR={row['p_fdr']:.4f}")
                        axes[r,c].tick_params(axis='x', rotation=45)
                
                # Remove empty subplots
                for j in range(i+1, n_rows*4):
                    r, c = j // 4, j % 4
                    if r < n_rows and c < 4:
                        axes[r,c].remove()
            
            plt.tight_layout()
            significant_plot_path = os.path.join(roi_dirs['figures'], 'significant_roi_measures.png')
            plt.savefig(significant_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Correlation matrix of ROI measures
    numeric_columns = roi_data_df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'subject_id']
    
    if len(numeric_columns) > 1:
        corr_data = roi_data_df[numeric_columns].corr()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.heatmap(corr_data, cmap='coolwarm', center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('ROI Measures Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        corr_plot_path = os.path.join(roi_dirs['figures'], 'roi_correlation_matrix.png')
        plt.tight_layout()
        plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"ROI plots saved to: {roi_dirs['figures']}")

def main():
    """Main ROI analysis function"""
    # Setup logging
    logger = setup_logging('step05_roi')
    logger.info("=" * 60)
    logger.info("STARTING STEP 05: ROI ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Setup directories
        roi_output_dir = os.path.join(config.OUTPUT_ROOT, '05_roi_analysis')
        roi_dirs = setup_roi_directories(roi_output_dir)
        
        # Load subject-group mapping
        subject_group_mapping = load_subject_group_mapping()
        
        # Create comprehensive atlas
        atlas_info, atlas_categories = create_comprehensive_atlas()
        logger.info(f"Created atlas with {len(atlas_info)} regions")
        
        # Load a reference image to get brain shape
        preproc_dir = os.path.join(config.OUTPUT_ROOT, '01_preprocessed')
        subject_dirs = [d for d in os.listdir(preproc_dir) 
                       if os.path.isdir(os.path.join(preproc_dir, d)) and d.startswith('sub-')]
        
        if not subject_dirs:
            raise ValueError("No preprocessed subjects found")
        
        # Load reference image for atlas creation
        ref_subject = subject_dirs[0]
        ref_gm_path = os.path.join(preproc_dir, ref_subject, f"{ref_subject}_GM_mask.nii.gz")
        ref_img = nib.load(ref_gm_path)
        brain_shape = ref_img.shape
        
        # Create atlas parcellation
        logger.info("Creating atlas parcellation...")
        atlas_parcellation = create_atlas_parcellation(brain_shape, atlas_info)
        
        # Save atlas
        atlas_path = os.path.join(roi_dirs['parcellations'], 'comprehensive_atlas.nii.gz')
        atlas_img = nib.Nifti1Image(atlas_parcellation, ref_img.affine)
        nib.save(atlas_img, atlas_path)
        logger.info(f"Atlas saved to: {atlas_path}")
        
        # Extract ROI measures for all subjects
        logger.info("Extracting ROI measures for all subjects...")
        roi_data_list = []
        successful_extractions = 0
        
        # Check for surface analysis results
        surface_dir = os.path.join(config.OUTPUT_ROOT, '04_surface_analysis', 'smoothed_surfaces')
        has_surface_data = os.path.exists(surface_dir)
        
        for subject_id in sorted(subject_dirs):
            # Get GM volume data
            gm_path = os.path.join(preproc_dir, subject_id, f"{subject_id}_GM_mask.nii.gz")
            
            # Get thickness data if available
            thickness_path = None
            if has_surface_data:
                thickness_path = os.path.join(surface_dir, f"{subject_id}_smoothed_thickness.nii.gz")
            
            if not os.path.exists(gm_path):
                logger.warning(f"GM data not found for {subject_id}")
                continue
            
            # Extract measures
            roi_measures, success = extract_roi_measures(
                subject_id, gm_path, thickness_path, atlas_parcellation, atlas_info
            )
            
            if success:
                # Get group from mapping
                group = subject_group_mapping.get(subject_id, 'Unknown')
                if group == 'Unknown':
                    logger.warning(f"Unknown group for subject {subject_id}")
                    continue
                
                roi_measures['group'] = group
                
                roi_data_list.append(roi_measures)
                successful_extractions += 1
            
            if len(roi_data_list) % 10 == 0:
                logger.info(f"Processed {len(roi_data_list)} subjects...")
        
        logger.info(f"ROI extraction completed: {successful_extractions}/{len(subject_dirs)} subjects")
        
        # Convert to DataFrame
        roi_data_df = pd.DataFrame(roi_data_list)
        
        # Save extracted data
        roi_data_csv = os.path.join(roi_dirs['extracted_data'], 'roi_measures.csv')
        roi_data_df.to_csv(roi_data_csv, index=False)
        logger.info(f"ROI data saved to: {roi_data_csv}")
        
        # Perform statistical analysis
        logger.info("Performing statistical analysis...")
        stats_df = perform_roi_statistics(roi_data_df, atlas_info, roi_dirs)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        create_roi_visualizations(roi_data_df, stats_df, roi_dirs)
        
        # Generate summary statistics
        summary_stats = roi_data_df.groupby('group').describe()
        summary_csv = os.path.join(roi_dirs['qc'], 'roi_summary_statistics.csv')
        summary_stats.to_csv(summary_csv)
        
        # Log summary
        log_analysis_summary(
            analysis_name="ROI Analysis",
            subjects_analyzed=successful_extractions,
            subjects_excluded=len(subject_dirs) - successful_extractions,
            notes=f"Extracted measures from {len(atlas_info)} ROIs. "
                  f"Statistical analysis performed on {len(stats_df)} comparisons."
        )
        
        logger.info("Step 05 completed successfully!")
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY: ROI Analysis")
        print("=" * 60)
        print(f"Subjects processed: {successful_extractions}/{len(subject_dirs)}")
        print(f"Success rate: {100*successful_extractions/len(subject_dirs):.1f}%")
        print(f"ROIs analyzed: {len(atlas_info)}")
        print(f"Statistical comparisons: {len(stats_df)}")
        
        if len(stats_df) > 0:
            significant_fdr = len(stats_df[stats_df['significant_fdr']])
            print(f"Significant results (FDR): {significant_fdr}/{len(stats_df)}")
            
            for group in ['HC', 'PIGD', 'TDPD']:
                group_data = roi_data_df[roi_data_df['group'] == group]
                if len(group_data) > 0:
                    print(f"• {group}: {len(group_data)} subjects")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"ROI analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
