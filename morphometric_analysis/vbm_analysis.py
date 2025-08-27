#!/usr/bin/env python3
"""
Voxel-Based Morphometry (VBM) Analysis Pipeline
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from nilearn import plotting, image, masking
from scipy import stats, ndimage
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class EfficientVBMAnalyzer:
    """Efficient VBM analysis with optimized memory usage and parallel processing."""
    
    def __init__(self, gm_images_dir, subjects_file, output_dir, n_jobs=-1):
        """Initialize the efficient VBM analyzer."""
        self.gm_images_dir = Path(gm_images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load subjects info
        self.subjects_df = self._load_subjects_file(subjects_file)
        
        # Set up parallel processing
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        print(f"Using {self.n_jobs} CPU cores for parallel processing")
        
        # VBM parameters
        self.smoothing_fwhm = 8  # Standard VBM smoothing
        self.cluster_threshold = 50  # Minimum cluster size in voxels
        
        # ROI definitions for targeted analysis (MNI coordinates)
        self.roi_atlas = {
            'frontal_cortex': {'center': (0, 40, 20), 'radius': 20},
            'motor_cortex': {'center': (0, -25, 60), 'radius': 15},
            'parietal_cortex': {'center': (0, -60, 50), 'radius': 15},
            'temporal_cortex': {'center': (50, -20, -10), 'radius': 20},
            'occipital_cortex': {'center': (0, -90, 0), 'radius': 15},
            'basal_ganglia': {'center': (0, 0, 0), 'radius': 15},
            'putamen_l': {'center': (-24, 0, 0), 'radius': 8},
            'putamen_r': {'center': (24, 0, 0), 'radius': 8},
            'caudate_l': {'center': (-14, 10, 10), 'radius': 6},
            'caudate_r': {'center': (14, 10, 10), 'radius': 6},
            'thalamus_l': {'center': (-12, -18, 8), 'radius': 6},
            'thalamus_r': {'center': (12, -18, 8), 'radius': 6},
            'hippocampus_l': {'center': (-28, -20, -15), 'radius': 8},
            'hippocampus_r': {'center': (28, -20, -15), 'radius': 8},
            'amygdala_l': {'center': (-22, -5, -15), 'radius': 5},
            'amygdala_r': {'center': (22, -5, -15), 'radius': 5},
            'insula_l': {'center': (-35, 0, 5), 'radius': 10},
            'insula_r': {'center': (35, 0, 5), 'radius': 10},
            'anterior_cingulate': {'center': (0, 32, 20), 'radius': 10},
            'posterior_cingulate': {'center': (0, -55, 25), 'radius': 10}
        }
        
    def _load_subjects_file(self, subjects_file):
        """Load and validate subjects file."""
        try:
            # Check if the file has headers
            with open(subjects_file, 'r') as f:
                first_line = f.readline().strip()
            
            if "subject_id" in first_line.lower() or "group" in first_line.lower():
                df = pd.read_csv(subjects_file)
            else:
                df = pd.read_csv(subjects_file, header=None, names=['subject_id', 'group'])
            
            print(f"Loaded {len(df)} subjects from {subjects_file}")
            print(f"Groups: {df['group'].value_counts().to_dict()}")
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading subjects file: {e}")
    
    def _find_gm_images(self):
        """Find and validate gray matter images."""
        gm_files = []
        groups = []
        missing_subjects = []
        
        for _, row in self.subjects_df.iterrows():
            subject_id = str(row['subject_id'])
            group = row['group']
            
            # Look for GM image with various naming patterns
            potential_files = [
                self.gm_images_dir / f"{subject_id}_gm.nii.gz",
                self.gm_images_dir / f"{subject_id}_gm.nii",
                self.gm_images_dir / f"sub-{subject_id}_gm.nii.gz",
                self.gm_images_dir / f"sub-{subject_id}_gm.nii"
            ]
            
            found_file = None
            for potential_file in potential_files:
                if potential_file.exists():
                    found_file = potential_file
                    break
            
            if found_file:
                gm_files.append(str(found_file))
                groups.append(group)
            else:
                missing_subjects.append(subject_id)
        
        if missing_subjects:
            print(f"Warning: Could not find GM images for {len(missing_subjects)} subjects:")
            print(missing_subjects[:10])  # Show first 10
        
        print(f"Found {len(gm_files)} valid GM images")
        return gm_files, groups
    
    def _load_and_validate_image(self, img_path):
        """Load and validate a single image."""
        try:
            img = nib.load(img_path)
            data = img.get_fdata()
            
            # Check for valid 3D image
            if len(data.shape) != 3:
                return None, f"Not 3D: {data.shape}"
            
            # Check for reasonable data range
            if np.all(data == 0) or np.isnan(data).all():
                return None, "Empty or all NaN data"
            
            return img, "valid"
            
        except Exception as e:
            return None, str(e)
    
    def _create_efficient_mask(self, gm_files, max_images=5):
        """Create an efficient analysis mask using subset of images."""
        print("Creating analysis mask...")
        
        # Use subset for mask creation to save time
        subset_files = gm_files[:min(max_images, len(gm_files))]
        valid_images = []
        
        for file_path in subset_files:
            img, status = self._load_and_validate_image(file_path)
            if img is not None:
                valid_images.append(img)
        
        if not valid_images:
            raise ValueError("No valid images found for mask creation")
        
        # Create mask using nilearn's efficient method
        mask = masking.compute_multi_brain_mask(valid_images, threshold=0.2)
        
        # Optimize mask size for memory efficiency
        mask_data = mask.get_fdata().astype(bool)
        n_voxels = np.sum(mask_data)
        
        print(f"Initial mask: {n_voxels} voxels")
        
        # If mask is too large, apply mild erosion
        if n_voxels > 400000:
            mask_data = ndimage.binary_erosion(mask_data, iterations=1)
            n_voxels = np.sum(mask_data)
            print(f"Optimized mask: {n_voxels} voxels")
        
        # Create final mask image
        final_mask = nib.Nifti1Image(mask_data.astype(np.uint8), mask.affine, mask.header)
        
        # Save mask
        mask_file = self.output_dir / 'analysis_mask.nii.gz'
        nib.save(final_mask, mask_file)
        print(f"Saved analysis mask: {mask_file}")
        
        return final_mask, n_voxels
    
    def _extract_data_batch(self, file_batch, mask_data):
        """Extract data from a batch of files."""
        batch_data = []
        batch_info = []
        
        for file_path, group in file_batch:
            img, status = self._load_and_validate_image(file_path)
            
            if img is not None:
                try:
                    # Resample to mask if needed
                    if img.shape != mask_data.shape:
                        img_resampled = image.resample_to_img(img, 
                                                            nib.Nifti1Image(mask_data.astype(np.uint8), img.affine),
                                                            interpolation='linear')
                        data = img_resampled.get_fdata()
                    else:
                        data = img.get_fdata()
                    
                    # Extract masked data
                    masked_values = data[mask_data].astype(np.float32)
                    batch_data.append(masked_values)
                    batch_info.append({'file': file_path, 'group': group, 'status': 'success'})
                    
                except Exception as e:
                    batch_info.append({'file': file_path, 'group': group, 'status': f'error: {e}'})
            else:
                batch_info.append({'file': file_path, 'group': group, 'status': f'invalid: {status}'})
        
        return batch_data, batch_info
    
    def _extract_all_data(self, gm_files, groups, mask, batch_size=10):
        """Extract data from all images using parallel processing."""
        print("Extracting voxel data...")
        
        mask_data = mask.get_fdata().astype(bool)
        n_voxels = np.sum(mask_data)
        n_subjects = len(gm_files)
        
        # Prepare batches
        file_group_pairs = list(zip(gm_files, groups))
        batches = [file_group_pairs[i:i+batch_size] for i in range(0, len(file_group_pairs), batch_size)]
        
        # Process batches in parallel
        all_data = []
        all_groups = []
        processing_info = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._extract_data_batch, batch, mask_data): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                batch_idx = future_to_batch[future]
                try:
                    batch_data, batch_info = future.result()
                    all_data.extend(batch_data)
                    processing_info.extend(batch_info)
                    
                    # Extract groups for successful extractions
                    for info in batch_info:
                        if info['status'] == 'success':
                            # Find corresponding group
                            file_path = info['file']
                            for orig_file, orig_group in file_group_pairs:
                                if orig_file == file_path:
                                    all_groups.append(orig_group)
                                    break
                                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
        
        # Convert to array
        if all_data:
            data_array = np.vstack(all_data)
            print(f"Extracted data shape: {data_array.shape}")
            print(f"Successfully processed: {len(all_data)}/{n_subjects} subjects")
        else:
            raise ValueError("No data could be extracted from any images")
        
        # Save processing info
        info_df = pd.DataFrame(processing_info)
        info_df.to_csv(self.output_dir / 'processing_info.csv', index=False)
        
        return data_array, all_groups, info_df
    
    def _run_statistical_tests(self, data_array, groups):
        """Run efficient statistical tests."""
        print("Running statistical analysis...")
        
        unique_groups = list(set(groups))
        n_voxels = data_array.shape[1]
        
        results = {
            'n_subjects': len(groups),
            'n_voxels': n_voxels,
            'groups': unique_groups,
            'group_counts': {g: groups.count(g) for g in unique_groups}
        }
        
        if len(unique_groups) >= 2:
            # Two-sample t-tests
            group1, group2 = unique_groups[0], unique_groups[1]
            idx1 = [i for i, g in enumerate(groups) if g == group1]
            idx2 = [i for i, g in enumerate(groups) if g == group2]
            
            data1 = data_array[idx1]
            data2 = data_array[idx2]
            
            # Vectorized t-test
            t_stats, p_values = stats.ttest_ind(data1, data2, axis=0, equal_var=False)
            
            # Handle NaN values
            nan_mask = np.isnan(t_stats) | np.isnan(p_values)
            t_stats[nan_mask] = 0
            p_values[nan_mask] = 1
            
            # FDR correction
            _, p_fdr = fdrcorrection(p_values)
            
            # Effect sizes (Cohen's d)
            pooled_std = np.sqrt(((len(idx1) - 1) * np.var(data1, axis=0, ddof=1) + 
                                 (len(idx2) - 1) * np.var(data2, axis=0, ddof=1)) / 
                                (len(idx1) + len(idx2) - 2))
            
            # Avoid division by zero
            pooled_std[pooled_std == 0] = 1e-10
            cohens_d = (np.mean(data1, axis=0) - np.mean(data2, axis=0)) / pooled_std
            
            results.update({
                'comparison': f'{group1} vs {group2}',
                't_stats': t_stats,
                'p_values': p_values,
                'p_fdr': p_fdr,
                'cohens_d': cohens_d,
                'n_significant_uncorrected': np.sum(p_values < 0.05),
                'n_significant_fdr': np.sum(p_fdr < 0.05),
                'max_t': np.max(np.abs(t_stats)),
                'mean_effect_size': np.mean(np.abs(cohens_d))
            })
        
        # Multi-group analysis if more than 2 groups
        if len(unique_groups) > 2:
            print("Running ANOVA for multiple groups...")
            f_stats = np.zeros(n_voxels)
            p_anova = np.ones(n_voxels)
            
            for v in range(n_voxels):
                try:
                    # Prepare data for ANOVA
                    group_data = [data_array[np.array(groups) == g, v] for g in unique_groups]
                    f_stat, p_val = stats.f_oneway(*group_data)
                    f_stats[v] = f_stat
                    p_anova[v] = p_val
                except:
                    pass
            
            # FDR correction for ANOVA
            _, p_anova_fdr = fdrcorrection(p_anova)
            
            results.update({
                'f_stats': f_stats,
                'p_anova': p_anova,
                'p_anova_fdr': p_anova_fdr,
                'n_significant_anova': np.sum(p_anova < 0.05),
                'n_significant_anova_fdr': np.sum(p_anova_fdr < 0.05)
            })
        
        return results
    
    def _extract_roi_metrics(self, data_array, groups, mask):
        """Extract region of interest metrics."""
        print("Extracting ROI metrics...")
        
        roi_results = []
        mask_data = mask.get_fdata().astype(bool)
        mask_affine = mask.affine
        
        # Get voxel coordinates
        voxel_coords = np.where(mask_data)
        
        # Convert voxel coordinates to MNI coordinates
        mni_coords = nib.affines.apply_affine(mask_affine, 
                                            np.column_stack(voxel_coords))
        
        for roi_name, roi_info in self.roi_atlas.items():
            center = np.array(roi_info['center'])
            radius = roi_info['radius']
            
            # Find voxels within ROI
            distances = np.linalg.norm(mni_coords - center, axis=1)
            roi_mask = distances <= radius
            
            if np.sum(roi_mask) > 0:
                # Extract ROI data
                roi_data = data_array[:, roi_mask]
                roi_mean = np.mean(roi_data, axis=1)  # Mean across voxels for each subject
                
                # Group statistics
                unique_groups = list(set(groups))
                group_means = {}
                group_stds = {}
                
                for group in unique_groups:
                    group_indices = [i for i, g in enumerate(groups) if g == group]
                    group_values = roi_mean[group_indices]
                    group_means[group] = np.mean(group_values)
                    group_stds[group] = np.std(group_values)
                
                # Statistical test if two groups
                p_value = 1.0
                t_stat = 0.0
                cohens_d = 0.0
                
                if len(unique_groups) >= 2:
                    group1, group2 = unique_groups[0], unique_groups[1]
                    idx1 = [i for i, g in enumerate(groups) if g == group1]
                    idx2 = [i for i, g in enumerate(groups) if g == group2]
                    
                    if len(idx1) > 1 and len(idx2) > 1:
                        t_stat, p_value = stats.ttest_ind(roi_mean[idx1], roi_mean[idx2])
                        
                        # Cohen's d
                        pooled_std = np.sqrt(((len(idx1) - 1) * np.var(roi_mean[idx1], ddof=1) + 
                                            (len(idx2) - 1) * np.var(roi_mean[idx2], ddof=1)) / 
                                           (len(idx1) + len(idx2) - 2))
                        if pooled_std > 0:
                            cohens_d = (np.mean(roi_mean[idx1]) - np.mean(roi_mean[idx2])) / pooled_std
                
                roi_result = {
                    'roi_name': roi_name,
                    'center_x': center[0],
                    'center_y': center[1], 
                    'center_z': center[2],
                    'radius': radius,
                    'n_voxels': np.sum(roi_mask),
                    't_stat': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d
                }
                
                # Add group-specific metrics
                for group in unique_groups:
                    roi_result[f'{group}_mean'] = group_means[group]
                    roi_result[f'{group}_std'] = group_stds[group]
                
                roi_results.append(roi_result)
        
        roi_df = pd.DataFrame(roi_results)
        
        # FDR correction for ROI p-values
        if len(roi_df) > 0 and 'p_value' in roi_df.columns:
            _, roi_df['p_fdr'] = fdrcorrection(roi_df['p_value'])
        
        return roi_df
    
    def _create_statistical_maps(self, results, mask):
        """Create and save statistical maps."""
        print("Creating statistical maps...")
        
        maps_dir = self.output_dir / 'statistical_maps'
        maps_dir.mkdir(exist_ok=True, parents=True)
        
        if 't_stats' in results:
            # T-statistic map
            t_map = masking.unmask(results['t_stats'], mask)
            nib.save(t_map, maps_dir / 't_statistic.nii.gz')
            
            # P-value map (negative log)
            p_map = masking.unmask(-np.log10(np.maximum(results['p_values'], 1e-10)), mask)
            nib.save(p_map, maps_dir / 'p_values_log.nii.gz')
            
            # FDR-corrected P-value map
            p_fdr_map = masking.unmask(-np.log10(np.maximum(results['p_fdr'], 1e-10)), mask)
            nib.save(p_fdr_map, maps_dir / 'p_fdr_log.nii.gz')
            
            # Effect size map (Cohen's d)
            d_map = masking.unmask(results['cohens_d'], mask)
            nib.save(d_map, maps_dir / 'cohens_d.nii.gz')
            
            # Thresholded maps
            # FDR p < 0.05
            thresh_fdr_05 = masking.unmask(results['t_stats'] * (results['p_fdr'] < 0.05), mask)
            nib.save(thresh_fdr_05, maps_dir / 'significant_fdr_05.nii.gz')
            
            # FDR p < 0.1
            thresh_fdr_01 = masking.unmask(results['t_stats'] * (results['p_fdr'] < 0.1), mask)
            nib.save(thresh_fdr_01, maps_dir / 'significant_fdr_01.nii.gz')
            
            # Uncorrected p < 0.05
            thresh_uncorr = masking.unmask(results['t_stats'] * (results['p_values'] < 0.05), mask)
            nib.save(thresh_uncorr, maps_dir / 'significant_uncorrected.nii.gz')
        
        if 'f_stats' in results:
            # F-statistic map for ANOVA
            f_map = masking.unmask(results['f_stats'], mask)
            nib.save(f_map, maps_dir / 'f_statistic.nii.gz')
        
        return maps_dir
    
    def _find_clusters(self, results, mask):
        """Find and characterize significant clusters."""
        print("Finding significant clusters...")
        
        clusters_info = []
        
        if 't_stats' not in results:
            return pd.DataFrame(clusters_info)
        
        # Try different thresholds
        thresholds = [
            ('fdr_05', results['p_fdr'] < 0.05),
            ('fdr_01', results['p_fdr'] < 0.1),
            ('uncorrected', results['p_values'] < 0.05)
        ]
        
        mask_data = mask.get_fdata().astype(bool)
        
        for thresh_name, thresh_mask in thresholds:
            if np.sum(thresh_mask) > 0:
                # Create 3D threshold map
                thresh_3d = np.zeros(mask_data.shape)
                thresh_3d[mask_data] = thresh_mask
                
                # Find connected components
                labeled_array, n_clusters = ndimage.label(thresh_3d)
                
                for cluster_id in range(1, n_clusters + 1):
                    cluster_mask_3d = labeled_array == cluster_id
                    cluster_size = np.sum(cluster_mask_3d)
                    
                    if cluster_size >= self.cluster_threshold:
                        # Get cluster statistics
                        cluster_mask_1d = cluster_mask_3d[mask_data]
                        cluster_t_values = results['t_stats'][cluster_mask_1d]
                        
                        # Cluster center of mass
                        coords = np.where(cluster_mask_3d)
                        center_x = int(np.mean(coords[0]))
                        center_y = int(np.mean(coords[1]))
                        center_z = int(np.mean(coords[2]))
                        
                        # Convert to MNI coordinates
                        mni_coords = nib.affines.apply_affine(mask.affine, [[center_x, center_y, center_z]])[0]
                        
                        cluster_info = {
                            'threshold': thresh_name,
                            'cluster_id': cluster_id,
                            'size_voxels': cluster_size,
                            'max_t': np.max(np.abs(cluster_t_values)),
                            'mean_t': np.mean(cluster_t_values),
                            'center_voxel_x': center_x,
                            'center_voxel_y': center_y,
                            'center_voxel_z': center_z,
                            'center_mni_x': mni_coords[0],
                            'center_mni_y': mni_coords[1],
                            'center_mni_z': mni_coords[2]
                        }
                        
                        clusters_info.append(cluster_info)
        
        return pd.DataFrame(clusters_info)
    
    def _create_visualizations(self, results, mask):
        """Create visualization plots."""
        print("Creating visualizations...")
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        maps_dir = self.output_dir / 'statistical_maps'
        
        try:
            # Significant results plot
            sig_map_file = maps_dir / 'significant_fdr_01.nii.gz'
            if sig_map_file.exists():
                plotting.plot_stat_map(
                    sig_map_file,
                    threshold=0.1,
                    colorbar=True,
                    cut_coords=(0, 0, 0),
                    title=f"VBM Results: {results.get('comparison', 'Group Differences')}",
                    output_file=viz_dir / 'statistical_map.png'
                )
                plt.close()
                
                # Glass brain view
                plotting.plot_glass_brain(
                    sig_map_file,
                    colorbar=True,
                    threshold=0.1,
                    plot_abs=False,
                    title=f"VBM Results: {results.get('comparison', 'Group Differences')}",
                    output_file=viz_dir / 'glass_brain.png'
                )
                plt.close()
        except Exception as e:
            print(f"Error creating statistical visualizations: {e}")
        
        # Summary statistics plot
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # P-value histogram
            if 'p_values' in results:
                axes[0, 0].hist(results['p_values'], bins=50, alpha=0.7, edgecolor='black')
                axes[0, 0].set_xlabel('P-value')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('P-value Distribution')
                axes[0, 0].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
                axes[0, 0].legend()
            
            # T-statistic histogram
            if 't_stats' in results:
                axes[0, 1].hist(results['t_stats'], bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('T-statistic')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('T-statistic Distribution')
                axes[0, 1].axvline(0, color='red', linestyle='--')
            
            # Effect size histogram
            if 'cohens_d' in results:
                axes[1, 0].hist(results['cohens_d'], bins=50, alpha=0.7, edgecolor='black')
                axes[1, 0].set_xlabel("Cohen's d")
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Effect Size Distribution')
                axes[1, 0].axvline(0, color='red', linestyle='--')
            
            # Summary statistics
            summary_text = f"""
            Analysis Summary:
            • Subjects: {results.get('n_subjects', 'N/A')}
            • Voxels: {results.get('n_voxels', 'N/A')}
            • Groups: {', '.join(results.get('groups', []))}
            • Significant (p<0.05): {results.get('n_significant_uncorrected', 0)}
            • Significant (FDR): {results.get('n_significant_fdr', 0)}
            • Max |T|: {results.get('max_t', 0):.3f}
            • Mean |Effect Size|: {results.get('mean_effect_size', 0):.3f}
            """
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating summary plots: {e}")
    
    def _save_comprehensive_metrics(self, results, roi_results, clusters_df, processing_info):
        """Save comprehensive metrics to CSV files."""
        print("Saving comprehensive metrics...")
        
        # Main results summary
        summary_metrics = {
            'analysis_timestamp': pd.Timestamp.now(),
            'n_subjects_total': len(self.subjects_df),
            'n_subjects_analyzed': results.get('n_subjects', 0),
            'n_voxels_analyzed': results.get('n_voxels', 0),
            'groups_analyzed': ', '.join(results.get('groups', [])),
            'comparison': results.get('comparison', 'N/A'),
            'n_significant_uncorrected_p05': results.get('n_significant_uncorrected', 0),
            'n_significant_fdr_p05': results.get('n_significant_fdr', 0),
            'max_t_statistic': results.get('max_t', 0),
            'mean_effect_size': results.get('mean_effect_size', 0),
            'n_clusters_fdr05': len(clusters_df[clusters_df['threshold'] == 'fdr_05']) if len(clusters_df) > 0 else 0,
            'n_clusters_fdr01': len(clusters_df[clusters_df['threshold'] == 'fdr_01']) if len(clusters_df) > 0 else 0,
            'n_clusters_uncorrected': len(clusters_df[clusters_df['threshold'] == 'uncorrected']) if len(clusters_df) > 0 else 0,
        }
        
        # Add group-specific counts
        group_counts = results.get('group_counts', {})
        for group, count in group_counts.items():
            summary_metrics[f'n_subjects_{group}'] = count
        
        # Add ANOVA results if available
        if 'n_significant_anova' in results:
            summary_metrics.update({
                'n_significant_anova_p05': results['n_significant_anova'],
                'n_significant_anova_fdr_p05': results['n_significant_anova_fdr'],
            })
        
        # Add ROI summary statistics
        if len(roi_results) > 0:
            summary_metrics.update({
                'n_rois_analyzed': len(roi_results),
                'n_rois_significant_p05': np.sum(roi_results['p_value'] < 0.05),
                'n_rois_significant_fdr_p05': np.sum(roi_results.get('p_fdr', [1]) < 0.05),
                'max_roi_effect_size': np.max(np.abs(roi_results['cohens_d'])) if 'cohens_d' in roi_results.columns else 0,
                'mean_roi_effect_size': np.mean(np.abs(roi_results['cohens_d'])) if 'cohens_d' in roi_results.columns else 0,
            })
        
        # Save summary
        summary_df = pd.DataFrame([summary_metrics])
        summary_df.to_csv(self.output_dir / 'vbm_analysis_summary.csv', index=False)
        
        # Save detailed ROI results
        if len(roi_results) > 0:
            roi_results.to_csv(self.output_dir / 'roi_analysis_results.csv', index=False)
        
        # Save cluster results
        if len(clusters_df) > 0:
            clusters_df.to_csv(self.output_dir / 'cluster_analysis_results.csv', index=False)
        
        # Save processing information
        processing_info.to_csv(self.output_dir / 'processing_log.csv', index=False)
        
        # Create a detailed voxel-wise results file (top results only to save space)
        if 't_stats' in results:
            n_top_voxels = min(10000, len(results['t_stats']))  # Top 10k voxels max
            
            # Sort by absolute t-statistic
            abs_t_stats = np.abs(results['t_stats'])
            top_indices = np.argsort(abs_t_stats)[-n_top_voxels:]
            
            voxel_results = pd.DataFrame({
                'voxel_index': top_indices,
                't_statistic': results['t_stats'][top_indices],
                'p_value': results['p_values'][top_indices],
                'p_fdr': results['p_fdr'][top_indices],
                'cohens_d': results['cohens_d'][top_indices],
                'abs_t_stat': abs_t_stats[top_indices]
            })
            
            # Sort by significance (p_fdr, then p_value, then abs_t_stat)
            voxel_results = voxel_results.sort_values(['p_fdr', 'p_value', 'abs_t_stat'], 
                                                    ascending=[True, True, False])
            
            voxel_results.to_csv(self.output_dir / 'top_voxel_results.csv', index=False)
        
        print(f"Saved comprehensive metrics to {self.output_dir}")
        
        return summary_df
    
    def run_complete_analysis(self):
        """Run the complete efficient VBM analysis pipeline."""
        start_time = time.time()
        print("="*60)
        print("EFFICIENT VBM ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Find and validate images
        print("\n1. Finding and validating gray matter images...")
        gm_files, groups = self._find_gm_images()
        
        if len(gm_files) == 0:
            raise ValueError("No valid gray matter images found!")
        
        # Step 2: Create analysis mask
        print("\n2. Creating analysis mask...")
        mask, n_voxels = self._create_efficient_mask(gm_files)
        
        # Step 3: Extract all data
        print("\n3. Extracting voxel data from all subjects...")
        data_array, final_groups, processing_info = self._extract_all_data(gm_files, groups, mask)
        
        # Step 4: Run statistical tests
        print("\n4. Running statistical analysis...")
        results = self._run_statistical_tests(data_array, final_groups)
        
        # Step 5: Extract ROI metrics
        print("\n5. Extracting ROI-based metrics...")
        roi_results = self._extract_roi_metrics(data_array, final_groups, mask)
        
        # Step 6: Find significant clusters
        print("\n6. Finding significant clusters...")
        clusters_df = self._find_clusters(results, mask)
        
        # Step 7: Create statistical maps
        print("\n7. Creating statistical maps...")
        maps_dir = self._create_statistical_maps(results, mask)
        
        # Step 8: Create visualizations
        print("\n8. Creating visualizations...")
        self._create_visualizations(results, mask)
        
        # Step 9: Save comprehensive metrics
        print("\n9. Saving comprehensive metrics...")
        summary_df = self._save_comprehensive_metrics(results, roi_results, clusters_df, processing_info)
        
        # Analysis complete
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Subjects analyzed: {results.get('n_subjects', 0)}")
        print(f"Voxels analyzed: {results.get('n_voxels', 0):,}")
        print(f"Results saved to: {self.output_dir}")
        
        if 'comparison' in results:
            print(f"\nKey Results:")
            print(f"  • Comparison: {results['comparison']}")
            print(f"  • Significant voxels (uncorrected p<0.05): {results.get('n_significant_uncorrected', 0):,}")
            print(f"  • Significant voxels (FDR p<0.05): {results.get('n_significant_fdr', 0):,}")
            print(f"  • Max |T-statistic|: {results.get('max_t', 0):.3f}")
            print(f"  • Mean |Effect Size|: {results.get('mean_effect_size', 0):.3f}")
        
        if len(clusters_df) > 0:
            print(f"  • Significant clusters found: {len(clusters_df)}")
        
        if len(roi_results) > 0:
            n_sig_rois = np.sum(roi_results.get('p_fdr', [1]) < 0.05)
            print(f"  • ROIs analyzed: {len(roi_results)}")
            print(f"  • Significant ROIs (FDR p<0.05): {n_sig_rois}")
        
        print(f"\nOutput files:")
        print(f"  • Summary: vbm_analysis_summary.csv")
        print(f"  • ROI results: roi_analysis_results.csv")
        print(f"  • Cluster results: cluster_analysis_results.csv")
        print(f"  • Statistical maps: statistical_maps/")
        print(f"  • Visualizations: visualizations/")
        
        return {
            'summary': summary_df,
            'roi_results': roi_results,
            'clusters': clusters_df,
            'processing_info': processing_info,
            'results': results,
            'output_dir': self.output_dir,
            'processing_time': total_time
        }


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Efficient VBM Analysis Pipeline")
    parser.add_argument('--gm_dir', default='D:/data_NIMHANS/t1w_vbm_atlas/gm_images',
                       help='Directory containing gray matter images')
    parser.add_argument('--subjects_file', required=True, 
                       help='CSV file with subject IDs and groups')
    parser.add_argument('--output', default='./efficient_vbm_results', 
                       help='Output directory for results')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
    args = parser.parse_args()
    
    # Validate inputs
    gm_dir = Path(args.gm_dir)
    if not gm_dir.exists():
        print(f"Error: GM images directory not found: {gm_dir}")
        return 1
    
    subjects_file = Path(args.subjects_file)
    if not subjects_file.exists():
        print(f"Error: Subjects file not found: {subjects_file}")
        return 1
    
    try:
        # Run analysis
        analyzer = EfficientVBMAnalyzer(
            gm_images_dir=gm_dir,
            subjects_file=subjects_file,
            output_dir=args.output,
            n_jobs=args.n_jobs
        )
        
        results = analyzer.run_complete_analysis()
        
        print("\nAnalysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())