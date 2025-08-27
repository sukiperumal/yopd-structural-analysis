#!/usr/bin/env python3
"""
Subcortical Shape Analysis
==========================

This script performs subcortical shape analysis using real brain atlases 
(Harvard-Oxford subcortical atlas) on registered MRI data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats, ndimage
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nilearn import datasets, image, masking, plotting
import nibabel as nib
from datetime import datetime
from tqdm import tqdm

class SubcorticalMorphometry:
    def __init__(self, atlas: str = "harvard_oxford", atlas_resample: bool = True):
        """
        Initialize with atlas name.
        Options: "harvard_oxford" for subcortical regions.
        """
        self.atlas = atlas
        self.atlas_resample = atlas_resample
        self.atlas_img, self.atlas_labels = self._load_atlas()

    def _load_atlas(self):
        """Fetch subcortical atlas and return (atlas_img, labels)."""
        if self.atlas == "harvard_oxford":
            # Use Harvard-Oxford subcortical atlas
            atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
            return nib.load(atlas.filename), atlas.labels
        else:
            raise ValueError(f"Unsupported atlas: {self.atlas}")

    def extract_metrics(self, subject_img_path: str, brain_mask_path: str = None, output_csv: str = None) -> pd.DataFrame:
        """
        Extract subcortical morphometry metrics for each atlas region.
        subject_img_path : str -> registered T1 in MNI space
        brain_mask_path  : str -> HD-BET brain mask (same space)
        """
        # Load subject image
        subj_img = nib.load(subject_img_path)
        subj_data = subj_img.get_fdata()
        
        # Load or create brain mask
        if brain_mask_path and os.path.exists(brain_mask_path):
            # If it's a brain-extracted image (not a mask), create mask from it
            try:
                brain_img = nib.load(brain_mask_path)
                brain_data = brain_img.get_fdata()
                # Resample to subject space
                brain_resamp = image.resample_to_img(brain_img, subj_img, 
                                                   interpolation="nearest", copy_header=True)
                brain_resamp_data = brain_resamp.get_fdata()
                # Create mask (threshold > 0 for brain-extracted images)
                mask_data = brain_resamp_data > 0
                print(f"Using brain mask derived from {os.path.basename(brain_mask_path)}")
            except Exception as e:
                print(f"Error processing brain mask: {e}, using threshold mask")
                mask_data = subj_data > np.percentile(subj_data[subj_data > 0], 25)
        else:
            # Create simple mask if none provided (threshold-based)
            print("No brain mask provided, creating threshold mask from registered image")
            # Use a more conservative threshold for registered images
            mask_data = subj_data > np.percentile(subj_data[subj_data > 0], 10)

        # Resample atlas to subject space if needed
        atlas_img = self.atlas_img
        if self.atlas_resample:
            atlas_img = image.resample_to_img(self.atlas_img, subj_img, interpolation="nearest", 
                                             force_resample=True, copy_header=True)

        atlas_data = atlas_img.get_fdata().astype(int)

        # Extract metrics for each region
        results = []
        for idx, label in enumerate(self.atlas_labels):
            if idx == 0 or not label:  # Skip background and empty labels
                continue

            # Create region mask (intersection with brain mask)
            region_mask = (atlas_data == idx) & mask_data
            if not np.any(region_mask):
                continue

            region_vals = subj_data[region_mask]
            
            # Calculate voxel dimensions for volume calculations
            voxel_volume = np.prod(subj_img.header.get_zooms()[:3])

            # --- Morphometric Metrics ---
            # Volume (number of voxels * voxel volume)
            volume = np.sum(region_mask) * voxel_volume
            
            # Mean intensity (tissue density proxy)
            mean_intensity = float(np.mean(region_vals))
            
            # Intensity standard deviation (tissue heterogeneity)
            std_intensity = float(np.std(region_vals))
            
            # Surface area estimation (boundary voxels)
            eroded = ndimage.binary_erosion(region_mask)
            boundary = region_mask & ~eroded
            surface_voxels = np.sum(boundary)
            # Approximate surface area (assuming cubic voxels)
            voxel_face_area = (subj_img.header.get_zooms()[0] * subj_img.header.get_zooms()[1])
            surface_area = surface_voxels * voxel_face_area
            
            # Shape index (surface area to volume ratio)
            if volume > 0:
                shape_index = surface_area / (volume**(2/3))
            else:
                shape_index = 0.0
            
            # Mean displacement (using intensity gradient as proxy for shape irregularity)
            if len(region_vals) > 10:  # Need sufficient voxels for gradient calculation
                # Calculate gradient within the region
                region_coords = np.where(region_mask)
                if len(region_coords[0]) > 10:
                    # Sample a subset for efficiency
                    sample_size = min(1000, len(region_coords[0]))
                    sample_indices = np.random.choice(len(region_coords[0]), sample_size, replace=False)
                    
                    # Calculate local gradients
                    gradients = []
                    for i in sample_indices:
                        x, y, z = region_coords[0][i], region_coords[1][i], region_coords[2][i]
                        # Calculate gradient in 3x3x3 neighborhood
                        if (x > 0 and x < subj_data.shape[0]-1 and 
                            y > 0 and y < subj_data.shape[1]-1 and 
                            z > 0 and z < subj_data.shape[2]-1):
                            
                            grad_x = subj_data[x+1,y,z] - subj_data[x-1,y,z]
                            grad_y = subj_data[x,y+1,z] - subj_data[x,y-1,z]
                            grad_z = subj_data[x,y,z+1] - subj_data[x,y,z-1]
                            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                            gradients.append(grad_mag)
                    
                    mean_displacement = float(np.mean(gradients)) if gradients else 0.0
                else:
                    mean_displacement = 0.0
            else:
                mean_displacement = 0.0

            # Add metrics for this region
            results.append({
                "region_id": idx,
                "region_label": label,
                "volume": volume,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "surface_area": surface_area,
                "shape_index": shape_index,
                "mean_displacement": mean_displacement
            })

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Add ICV (intracranial volume)
        icv = np.sum(mask_data) * np.prod(subj_img.header.get_zooms()[:3])
        df["icv"] = icv

        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)

        return df
    
def normalize_by_icv(df, metrics_to_normalize=None):
    """Normalize regional volumes by intracranial volume."""
    if metrics_to_normalize is None:
        # Default: normalize volume measurements
        metrics_to_normalize = [col for col in df.columns if 'volume' in col]
    
    if 'icv' not in df.columns:
        print("Warning: ICV not found in dataframe. Normalization skipped.")
        return df
    
    df_norm = df.copy()
    
    for metric in metrics_to_normalize:
        if metric in df.columns:
            # Calculate normalized value (value/icv * mean_icv)
            mean_icv = df['icv'].mean()
            df_norm[f"{metric}_norm"] = df[metric] / df['icv'] * mean_icv
    
    return df_norm

def calculate_laterality_indices(df):
    """Calculate laterality indices for paired subcortical regions."""
    # Identify paired regions (left/right)
    left_regions = [col for col in df.columns if 'Left' in col and any(term in col for term in ['volume', 'mean_intensity', 'shape_index'])]
    
    for left_region in left_regions:
        # Construct matching right hemisphere region name
        right_region = left_region.replace('Left', 'Right')
        
        if right_region in df.columns:
            # Calculate laterality index: (L-R)/(L+R) * 100
            lat_idx_name = f"lat_idx_{left_region.replace('Left_', '')}"
            denominator = df[left_region] + df[right_region]
            # Avoid division by zero
            denominator = denominator.replace(0, np.nan)
            df[lat_idx_name] = ((df[left_region] - df[right_region]) / denominator) * 100
    
    return df
    
    def create_structure_masks(self, shape):
        """Create masks for subcortical structures based on approximate MNI coordinates."""
        masks = {}
        
        # Check if the shape is 2D or 3D
        is_3d = len(shape) == 3
        
        if not is_3d:
            # For 2D images, create simplified structure masks
            print(f"Creating simplified 2D structure masks for shape {shape}")
            
            # Define structure locations in 2D space
            # These are very approximate positions for illustration
            structure_coords_2d = {
                'L_Thal': {'center': (shape[0]//3, shape[1]//2), 'radius': shape[0]//10},
                'R_Thal': {'center': (2*shape[0]//3, shape[1]//2), 'radius': shape[0]//10},
                'L_Caud': {'center': (shape[0]//3, shape[1]//3), 'radius': shape[0]//12},
                'R_Caud': {'center': (2*shape[0]//3, shape[1]//3), 'radius': shape[0]//12},
                'L_Puta': {'center': (shape[0]//4, shape[1]//2), 'radius': shape[0]//12},
                'R_Puta': {'center': (3*shape[0]//4, shape[1]//2), 'radius': shape[0]//12},
                'L_Pall': {'center': (shape[0]//4, 2*shape[1]//3), 'radius': shape[0]//16},
                'R_Pall': {'center': (3*shape[0]//4, 2*shape[1]//3), 'radius': shape[0]//16},
                'L_Hipp': {'center': (shape[0]//3, 3*shape[1]//4), 'radius': shape[0]//12},
                'R_Hipp': {'center': (2*shape[0]//3, 3*shape[1]//4), 'radius': shape[0]//12},
                'L_Amyg': {'center': (shape[0]//4, 4*shape[1]//5), 'radius': shape[0]//16},
                'R_Amyg': {'center': (3*shape[0]//4, 4*shape[1]//5), 'radius': shape[0]//16},
                'L_Accu': {'center': (shape[0]//3, shape[1]//4), 'radius': shape[0]//16},
                'R_Accu': {'center': (2*shape[0]//3, shape[1]//4), 'radius': shape[0]//16}
            }
            
            # Create masks for 2D images
            for structure, coords in structure_coords_2d.items():
                mask = np.zeros(shape, dtype=bool)
                
                # Create coordinate grid
                xx, yy = np.mgrid[:shape[0], :shape[1]]
                
                # Create circle
                circle = (xx - coords['center'][0])**2 + (yy - coords['center'][1])**2 <= coords['radius']**2
                
                # Assign to mask
                mask[circle] = True
                
                masks[structure] = mask
            
            return masks
        
        # For 3D images, continue with normal processing
        # Define structure locations in normalized space
        # These are approximate locations in a standard brain
        structure_coords = {
            'L_Thal': {'center': (-10, -20, 8), 'radius': 10},
            'R_Thal': {'center': (10, -20, 8), 'radius': 10},
            'L_Caud': {'center': (-15, 8, 10), 'radius': 8},
            'R_Caud': {'center': (15, 8, 10), 'radius': 8},
            'L_Puta': {'center': (-25, 2, 0), 'radius': 8},
            'R_Puta': {'center': (25, 2, 0), 'radius': 8},
            'L_Pall': {'center': (-20, 0, 0), 'radius': 5},
            'R_Pall': {'center': (20, 0, 0), 'radius': 5},
            'L_Hipp': {'center': (-30, -20, -10), 'radius': 8},
            'R_Hipp': {'center': (30, -20, -10), 'radius': 8},
            'L_Amyg': {'center': (-25, -10, -15), 'radius': 5},
            'R_Amyg': {'center': (25, -10, -15), 'radius': 5},
            'L_Accu': {'center': (-10, 8, -10), 'radius': 4},
            'R_Accu': {'center': (10, 8, -10), 'radius': 4}
        }
        
        # Convert to voxel coordinates
        center_x, center_y, center_z = shape[0]//2, shape[1]//2, shape[2]//2
        
        # Create masks
        for structure, coords in structure_coords.items():
            mask = np.zeros(shape, dtype=bool)
            
            # Convert MNI coordinates to voxel coordinates
            # This is a simplified transformation assuming the image is roughly aligned to MNI space
            voxel_x = center_x + coords['center'][0] // 2  # Assuming 2mm voxels
            voxel_y = center_y + coords['center'][1] // 2
            voxel_z = center_z + coords['center'][2] // 2
            
            # Create a spherical mask
            radius_voxels = coords['radius'] // 2  # Convert mm to voxels (assuming 2mm)
            
            # Create coordinate grid
            xx, yy, zz = np.mgrid[:shape[0], :shape[1], :shape[2]]
            
            # Create sphere
            sphere = (xx - voxel_x)**2 + (yy - voxel_y)**2 + (zz - voxel_z)**2 <= radius_voxels**2
            
            # Assign to mask
            mask[sphere] = True
            
            masks[structure] = mask
        
        return masks
    
    def extract_shape_metrics(self, subject_id, group):
        """Extract shape metrics from a real brain image."""
        # Find the subject's brain image
        brain_files = list(self.data_dir.glob(f"{subject_id}*brain*.nii*")) + \
                     list(self.data_dir.glob(f"sub-{subject_id}*brain*.nii*"))
        
        if not brain_files:
            print(f"No brain image found for subject {subject_id}")
            return None
        
        # Use the first found image
        brain_file = brain_files[0]
        print(f"Processing {brain_file}")
        
        # Extract metrics from the image
        metrics = self.analyze_brain_image(brain_file, subject_id, group)
        
        return metrics
    
    def process_all_subjects(self):
        """Process all subjects in the dataset."""
        all_subject_data = []
        
        for _, row in tqdm(self.subjects_df.iterrows(), total=len(self.subjects_df)):
            subject_id, group = row['subject_id'], row['group']
            subject_data = self.extract_shape_metrics(subject_id, group)
            if subject_data:
                all_subject_data.append(subject_data)
        
        return pd.DataFrame(all_subject_data)

def run_group_comparisons(df, group_col='group', covariates=None):
    """Run statistical comparisons between groups with covariates."""
    if covariates is None:
        covariates = ['age', 'sex', 'icv']
    
    # Filter covariates that actually exist in the dataframe
    covariates = [cov for cov in covariates if cov in df.columns]
    
    # Select metric columns (excluding covariates, subject_id, etc.)
    metric_cols = [col for col in df.columns if any(
        term in col for term in ['volume', 'mean_intensity', 'std_intensity', 'surface_area', 'shape_index', 'mean_displacement', 'lat_idx'])]
    
    results = []
    
    for metric in metric_cols:
        # Skip if metric has too many NaN values
        if df[metric].isna().sum() > len(df) * 0.5:
            continue
            
        # Build formula for linear model
        formula = f"{metric} ~ {group_col}"
        if covariates:
            formula += " + " + " + ".join(covariates)
        
        try:
            # Remove rows with NaN values for this specific analysis
            analysis_df = df[[metric, group_col] + covariates].dropna()
            
            if len(analysis_df) < 10:  # Need minimum samples
                continue
                
            # Fit the model
            model = smf.ols(formula=formula, data=analysis_df).fit()
            
            # Extract results for group comparisons
            for group_val in df[group_col].unique():
                if group_val != 'HC':  # Assuming HC is reference
                    group_term = f"{group_col}[T.{group_val}]"
                    if group_term in model.params:
                        result = {
                            'metric': metric,
                            'group': group_val,
                            'coefficient': model.params[group_term],
                            'std_error': model.bse[group_term],
                            't_value': model.tvalues[group_term],
                            'p_value': model.pvalues[group_term],
                            'adjusted_r2': model.rsquared_adj,
                            'formula': formula,
                            'n_subjects': len(analysis_df)
                        }
                        results.append(result)
        except Exception as e:
            print(f"Error in model for {metric}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if len(results_df) > 0:
        _, results_df['fdr_p_value'] = fdrcorrection(results_df['p_value'].values)
    
    return results_df

def create_visualizations(df, results_df, output_dir):
    """Create visualizations of subcortical shape analysis results."""
    try:
        # Create figures directory
        figures_dir = Path(output_dir) / 'figures'
        figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Box plots of significant metrics by group
        if len(results_df) > 0:
            sig_metrics = results_df[results_df['fdr_p_value'] < 0.05]['metric'].unique()
            for i, metric in enumerate(sig_metrics[:10]):  # Limit to 10 plots
                if metric in df.columns:
                    try:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x='group', y=metric, data=df)
                        sns.stripplot(x='group', y=metric, data=df, size=4, color='.3', alpha=0.6)
                        plt.title(f'Distribution of {metric} by group')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(figures_dir / f'boxplot_{metric.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"Error creating boxplot for {metric}: {e}")
                        plt.close()
        
        # 2. Volume comparison for all regions
        volume_cols = [col for col in df.columns if 'volume' in col and 'norm' not in col]
        if volume_cols:
            try:
                # Create a melted dataframe for easier plotting
                volume_df = df[['subject_id', 'group'] + volume_cols].melt(
                    id_vars=['subject_id', 'group'],
                    value_vars=volume_cols,
                    var_name='region', value_name='volume'
                )
                
                plt.figure(figsize=(15, 8))
                sns.boxplot(x='region', y='volume', hue='group', data=volume_df)
                plt.xticks(rotation=90)
                plt.title('Subcortical Volumes by Region and Group')
                plt.tight_layout()
                plt.savefig(figures_dir / 'subcortical_volumes.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error creating volume plot: {e}")
                plt.close()
        
        # 3. Shape index comparison
        shape_cols = [col for col in df.columns if 'shape_index' in col]
        if shape_cols:
            try:
                shape_df = df[['subject_id', 'group'] + shape_cols].melt(
                    id_vars=['subject_id', 'group'],
                    value_vars=shape_cols,
                    var_name='region', value_name='shape_index'
                )
                
                plt.figure(figsize=(15, 8))
                sns.boxplot(x='region', y='shape_index', hue='group', data=shape_df)
                plt.xticks(rotation=90)
                plt.title('Shape Index by Region and Group')
                plt.tight_layout()
                plt.savefig(figures_dir / 'shape_indices.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error creating shape index plot: {e}")
                plt.close()
        
        # 4. Laterality indices if available
        lat_idx_cols = [col for col in df.columns if col.startswith('lat_idx_')]
        if lat_idx_cols:
            try:
                lat_df = df[['subject_id', 'group'] + lat_idx_cols].melt(
                    id_vars=['subject_id', 'group'],
                    value_vars=lat_idx_cols,
                    var_name='region', value_name='laterality_index'
                )
                
                # Remove NaN values
                lat_df = lat_df.dropna(subset=['laterality_index'])
                
                if len(lat_df) > 0:
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x='region', y='laterality_index', hue='group', data=lat_df)
                    plt.xticks(rotation=90)
                    plt.title('Laterality Indices by Region and Group')
                    plt.tight_layout()
                    plt.savefig(figures_dir / 'laterality_indices.png', dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"Error creating laterality plot: {e}")
                plt.close()
        
        # 5. Summary plot of effect sizes
        if len(results_df) > 0:
            try:
                # Filter to significant results and sort by effect size
                sig_results = results_df[results_df['fdr_p_value'] < 0.05].copy()
                if len(sig_results) > 0:
                    # Calculate effect size
                    sig_results['effect_size'] = sig_results['coefficient'] / sig_results['std_error']
                    sig_results = sig_results.sort_values('effect_size')
                    
                    # Plot top 20 effect sizes
                    top_results = sig_results.tail(min(20, len(sig_results)))
                    
                    if len(top_results) > 0:
                        plt.figure(figsize=(12, 10))
                        sns.barplot(x='effect_size', y='metric', hue='group', data=top_results)
                        plt.title('Top Effect Sizes for Significant Results')
                        plt.tight_layout()
                        plt.savefig(figures_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
                        plt.close()
            except Exception as e:
                print(f"Error creating effect size plot: {e}")
                plt.close()
        
        # 6. Summary statistics table
        try:
            summary_stats = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['subject_id']:
                    stats_dict = {
                        'metric': col,
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'n_valid': df[col].count()
                    }
                    summary_stats.append(stats_dict)
            
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv(figures_dir / 'summary_statistics.csv', index=False)
            print(f"Saved summary statistics to {figures_dir}/summary_statistics.csv")
        except Exception as e:
            print(f"Error creating summary statistics: {e}")
        
        print(f"Visualizations saved to {figures_dir}")
        
    except Exception as e:
        print(f"Error in visualization creation: {e}")

def process_cohort(data_dir, demographics_file, output_dir, atlas="harvard_oxford"):
    """Process a cohort of subjects and extract subcortical metrics using atlas ROIs."""
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Load demographics file
    try:
        # Check if the file has headers by reading the first line
        with open(demographics_file, 'r') as f:
            first_line = f.readline().strip()
        
        # If first line contains "subject_id" or "group", assume it has headers
        if "subject_id" in first_line or "group" in first_line:
            demo_df = pd.read_csv(demographics_file)
        else:
            # Otherwise assume it's a simple two-column file without headers
            demo_df = pd.read_csv(demographics_file, header=None, names=['subject_id', 'group'])
        
        print(f"Loaded demographics for {len(demo_df)} subjects")
    except Exception as e:
        print(f"Error loading demographics file: {e}")
        return None, None
    
    # Initialize subcortical morphometry analyzer
    analyzer = SubcorticalMorphometry(atlas=atlas)
    
    # Process each subject
    all_metrics = []
    for _, row in tqdm(demo_df.iterrows(), total=len(demo_df)):
        subject_id = row['subject_id']
        group = row['group'] if 'group' in row else 'HC'  # Default to HC if no group specified
        
        # Find the subject's registered brain image
        possible_patterns = [
            f"{subject_id}*registered*.nii*",
            f"sub-{subject_id}*registered*.nii*",
            f"{subject_id}*.nii*",
            f"sub-{subject_id}*.nii*"
        ]
        
        # Find brain image and mask
        brain_files = []
        for pattern in possible_patterns:
            brain_files.extend(list(Path(data_dir).glob(pattern)))
            if brain_files:  # Stop at first successful pattern
                break
        
        if not brain_files:
            print(f"No brain image found for subject {subject_id}")
            continue
        
        # Use the first found image
        brain_file = brain_files[0]
        print(f"Processing {brain_file}")
        
        # Find corresponding brain mask - since we're using registered images,
        # we need to look for the corresponding brain-extracted image to create a mask
        mask_file = None
        brain_extracted_patterns = [
            f"{subject_id}*brain*.nii*",
            f"sub-{subject_id}*brain*.nii*",
        ]
        
        # Look for brain-extracted images in the HD-BET directory
        brain_extracted_dir = Path("D:/data_NIMHANS/t1w_brain_extracted_hdbet")
        if brain_extracted_dir.exists():
            for pattern in brain_extracted_patterns:
                brain_matches = list(brain_extracted_dir.glob(pattern))
                if brain_matches:
                    # We'll use this brain image to create a mask by registering it to MNI space
                    mask_file = brain_matches[0]
                    break
        
        try:
            # Extract metrics from the actual image
            subject_metrics = analyzer.extract_metrics(
                str(brain_file), 
                str(mask_file) if mask_file else None,
                str(output_dir_path / f"{subject_id}_subcortical_metrics.csv")
            )
            
            # Add subject ID and group info
            subject_metrics['subject_id'] = subject_id
            subject_metrics['group'] = group
            
            # Add demographic info
            for col in demo_df.columns:
                if col not in ['subject_id', 'group'] and col not in subject_metrics:
                    subject_metrics[col] = row[col]
            
            # Append to all metrics
            all_metrics.append(subject_metrics)
            
        except Exception as e:
            print(f"Error processing {brain_file}: {e}")
    
    if not all_metrics:
        print("No subjects were successfully processed")
        return None, None
    
    # Combine all subject metrics
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    print(f"Generated metrics for {len(metrics_df['subject_id'].unique())} subjects")
    
    # Normalize volume measurements by ICV
    metrics_df = normalize_by_icv(metrics_df)
    
    # Calculate laterality indices
    metrics_df = calculate_laterality_indices(metrics_df)
    
    # Save raw metrics
    metrics_df.to_csv(output_dir_path / 'subcortical_metrics.csv', index=False)
    print(f"Saved raw metrics to {output_dir}/subcortical_metrics.csv")
    
    # Run group comparisons if group column exists
    if 'group' in metrics_df.columns and len(metrics_df['group'].unique()) > 1:
        # Standard analysis with all subjects
        results_df = run_group_comparisons(metrics_df)
        results_df.to_csv(output_dir_path / 'group_comparisons.csv', index=False)
        print(f"Saved group comparison results to {output_dir}/group_comparisons.csv")
        
        # Create summary tables by metric type
        for metric_type in ['volume_norm', 'mean_intensity', 'shape_index', 'mean_displacement']:
            type_results = results_df[results_df['metric'].str.contains(metric_type)]
            if len(type_results) > 0:
                # Filter for significant results
                sig_results = type_results[type_results['fdr_p_value'] < 0.05]
                if len(sig_results) > 0:
                    sig_results.to_csv(output_dir_path / f'significant_{metric_type}_results.csv', index=False)
                    print(f"Saved significant {metric_type} results to {output_dir}/significant_{metric_type}_results.csv")
        
        # Create visualizations
        create_visualizations(metrics_df, results_df, output_dir)
    
    return metrics_df, results_df if 'group' in metrics_df.columns and len(metrics_df['group'].unique()) > 1 else None

def main():
    parser = argparse.ArgumentParser(description="Subcortical morphometry analysis using atlas ROIs")
    parser.add_argument('--data_dir', required=True, help='Directory containing registered T1 image data')
    parser.add_argument('--subjects_file', required=True, help='CSV file with subject IDs and groups')
    parser.add_argument('--output', default='./subcortical_morphometry_results', 
                        help='Output directory for results')
    parser.add_argument('--atlas', default='harvard_oxford', 
                        choices=['harvard_oxford'],
                        help='Brain atlas to use for subcortical region definitions')
    
    args = parser.parse_args()
    
    print("Subcortical Morphometry Analysis with Atlas ROIs")
    print("===============================================")
    print(f"Data Directory: {args.data_dir}")
    print(f"Subjects File: {args.subjects_file}")
    print(f"Output Directory: {args.output}")
    print(f"Atlas: {args.atlas}")
    print()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory {args.data_dir} does not exist!")
        return 1
    
    # Check if subjects file exists
    if not Path(args.subjects_file).exists():
        print(f"Error: Subjects file {args.subjects_file} does not exist!")
        return 1
    
    try:
        # Process all subjects
        metrics_df, results_df = process_cohort(
            args.data_dir, args.subjects_file, args.output, args.atlas)
        
        if metrics_df is not None:
            print(f"\nAnalysis complete. Processed {len(metrics_df['subject_id'].unique())} subjects.")
            if results_df is not None and len(results_df) > 0:
                sig_results = results_df[results_df['fdr_p_value'] < 0.05]
                print(f"Found {len(sig_results)} significant results after FDR correction.")
                
                # Print some key findings
                if len(sig_results) > 0:
                    print("\nTop significant findings:")
                    top_findings = sig_results.nsmallest(5, 'fdr_p_value')[['metric', 'group', 'coefficient', 'fdr_p_value']]
                    print(top_findings.to_string(index=False))
            else:
                print("No group comparisons performed (single group or insufficient data).")
        else:
            print("Analysis failed - no metrics generated.")
            return 1
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())