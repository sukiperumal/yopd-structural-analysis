#!/usr/bin/env python3
"""
Surface-based Morphometry Module
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

class SurfaceMorphometry:
    def __init__(self, atlas: str = "harvard_oxford", atlas_resample: bool = True):
        """
        Initialize with atlas name.
        Options: "harvard_oxford", "aal", "schaefer".
        """
        self.atlas = atlas
        self.atlas_resample = atlas_resample
        self.atlas_img, self.atlas_labels = self._load_atlas()

    def _load_atlas(self):
        """Fetch atlas and return (atlas_img, labels)."""
        if self.atlas == "harvard_oxford":
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            return nib.load(atlas.filename), atlas.labels
        elif self.atlas == "aal":
            atlas = datasets.fetch_atlas_aal()
            return nib.load(atlas.maps), atlas.labels
        elif self.atlas == "schaefer":
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
            return nib.load(atlas.maps), atlas.labels
        else:
            raise ValueError(f"Unsupported atlas: {self.atlas}")

    def extract_metrics(self, subject_img_path: str, brain_mask_path: str = None, output_csv: str = None) -> pd.DataFrame:
        """
        Extract surface morphometry metrics for each atlas region.
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
                
                # Resample the brain image to match the registered image space
                brain_img_registered = image.resample_to_img(brain_img, subj_img, interpolation="linear",
                                                           force_resample=True, copy_header=True)
                brain_data_registered = brain_img_registered.get_fdata()
                # Create binary mask from registered brain image
                mask_data = brain_data_registered > (np.max(brain_data_registered) * 0.1)
                print(f"Using brain mask derived from {os.path.basename(brain_mask_path)}")
            except Exception as e:
                print(f"Error processing brain mask: {e}, using threshold mask")
                # Fallback to threshold mask
                mask_data = subj_data > np.percentile(subj_data[subj_data > 0], 10)
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

            # --- Metrics ---
            thickness = float(np.mean(region_vals))          # proxy for thickness
            area = int(np.sum(region_mask))                  # voxel count as area proxy
            curvature = float(np.std(region_vals))           # variability as curvature proxy

            # Add metrics for this region
            results.append({
                "region_id": idx,
                "region_label": label,
                "thickness": thickness,
                "area": area,
                "curvature": curvature
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
        
    def preprocess_image_data(self, data):
        """Preprocess image data to handle different dimensions"""
        # Handle 4D images by taking the first volume
        if len(data.shape) == 4:
            print(f"4D image detected (shape: {data.shape}), taking first volume")
            data = data[:, :, :, 0]
        elif len(data.shape) == 2:
            print(f"2D image detected (shape: {data.shape})")
        elif len(data.shape) != 3:
            print(f"Unexpected image dimensions: {data.shape}")
            return None
            
        return data
        
    def extract_metrics_from_image(self, image_path, subject_id, group):
        """Extract surface-based metrics from a real brain image"""
        try:
            # Load the brain image
            img = nib.load(str(image_path))
            original_data = img.get_fdata()
            
            # Preprocess to handle different dimensions
            data = self.preprocess_image_data(original_data)
            if data is None:
                return self.get_default_metrics(subject_id, group)
            
            # Extract basic properties of the image
            shape = data.shape
            voxel_dims = img.header.get_zooms()[:len(shape)]  # Only use available dimensions
            
            # Check if the image is 2D or 3D
            is_3d = len(shape) == 3
            
            # Calculate total brain volume (proxy for ICV)
            brain_mask = data > np.percentile(data[data > 0], 25)  # Only consider non-zero voxels
            brain_voxels = np.sum(brain_mask)
            voxel_volume = np.prod(voxel_dims)  # Use only available dimensions
            total_volume = brain_voxels * voxel_volume
            
            # Create metrics dictionary
            metrics = {
                'subject_id': subject_id,
                'group': group,
                'icv': total_volume
            }
            
            # If the image is 2D, use a simplified approach
            if not is_3d:
                print(f"Image {image_path} is 2D (shape: {shape}). Using simplified 2D analysis.")
                # Add values based on actual 2D analysis
                for region in self.regions:
                    region_mask = self.create_region_mask_2d(shape, region)
                    
                    # Calculate metrics for left and right sides
                    lh_mask = region_mask.copy()
                    lh_mask[:shape[0]//2, :] = 0  # Keep only right half (left hemisphere)
                    rh_mask = region_mask.copy()
                    rh_mask[shape[0]//2:, :] = 0  # Keep only left half (right hemisphere)
                    
                    lh_data = data * lh_mask
                    rh_data = data * rh_mask
                    
                    metrics[f"lh_{region}_thickness"] = self.estimate_thickness_2d(lh_data, lh_mask)
                    metrics[f"lh_{region}_area"] = self.estimate_surface_area_2d(lh_mask, voxel_dims)
                    metrics[f"lh_{region}_curvature"] = self.estimate_curvature_2d(lh_data, lh_mask)
                    metrics[f"rh_{region}_thickness"] = self.estimate_thickness_2d(rh_data, rh_mask)
                    metrics[f"rh_{region}_area"] = self.estimate_surface_area_2d(rh_mask, voxel_dims)
                    metrics[f"rh_{region}_curvature"] = self.estimate_curvature_2d(rh_data, rh_mask)
                
                # Add demographic information
                metrics['age'] = 50  # Default value
                metrics['sex'] = 0   # Default value
                
                return metrics
            
            # For 3D images, continue with region-based analysis
            for region in self.regions:
                try:
                    region_mask = self.create_region_mask_3d(shape, region)
                    
                    # For left hemisphere
                    lh_mask = region_mask.copy()
                    lh_mask[:shape[0]//2, :, :] = 0  # Keep only right half of image (left hemisphere)
                    lh_data = data * lh_mask
                    
                    # For right hemisphere
                    rh_mask = region_mask.copy()
                    rh_mask[shape[0]//2:, :, :] = 0  # Keep only left half of image (right hemisphere)
                    rh_data = data * rh_mask
                    
                    # Calculate metrics for left hemisphere
                    lh_thickness = self.estimate_thickness_3d(lh_data, lh_mask)
                    lh_area = self.estimate_surface_area_3d(lh_mask, voxel_dims)
                    lh_curvature = self.estimate_curvature_3d(lh_data, lh_mask)
                    
                    # Calculate metrics for right hemisphere
                    rh_thickness = self.estimate_thickness_3d(rh_data, rh_mask)
                    rh_area = self.estimate_surface_area_3d(rh_mask, voxel_dims)
                    rh_curvature = self.estimate_curvature_3d(rh_data, rh_mask)
                    
                    # Store metrics
                    metrics[f"lh_{region}_thickness"] = lh_thickness
                    metrics[f"lh_{region}_area"] = lh_area
                    metrics[f"lh_{region}_curvature"] = lh_curvature
                    metrics[f"rh_{region}_thickness"] = rh_thickness
                    metrics[f"rh_{region}_area"] = rh_area
                    metrics[f"rh_{region}_curvature"] = rh_curvature
                except Exception as e:
                    print(f"Error processing region {region}: {e}")
                    # Use reference values if processing fails
                    metrics[f"lh_{region}_thickness"] = self.reference_values['thickness']['mean']
                    metrics[f"lh_{region}_area"] = self.reference_values['area']['mean']
                    metrics[f"lh_{region}_curvature"] = self.reference_values['curvature']['mean']
                    metrics[f"rh_{region}_thickness"] = self.reference_values['thickness']['mean']
                    metrics[f"rh_{region}_area"] = self.reference_values['area']['mean']
                    metrics[f"rh_{region}_curvature"] = self.reference_values['curvature']['mean']
            
            # Add demographic information
            metrics['age'] = 50  # Default value
            metrics['sex'] = 0   # Default value
            
            return metrics
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return self.get_default_metrics(subject_id, group)
    
    def get_default_metrics(self, subject_id, group):
        """Return default metrics dictionary"""
        metrics = {
            'subject_id': subject_id,
            'group': group,
            'icv': 1000000  # Default ICV value
        }
        
        # Add dummy values for regions to allow pipeline to continue
        for region in self.regions:
            metrics[f"lh_{region}_thickness"] = self.reference_values['thickness']['mean']
            metrics[f"lh_{region}_area"] = self.reference_values['area']['mean']
            metrics[f"lh_{region}_curvature"] = self.reference_values['curvature']['mean']
            metrics[f"rh_{region}_thickness"] = self.reference_values['thickness']['mean']
            metrics[f"rh_{region}_area"] = self.reference_values['area']['mean']
            metrics[f"rh_{region}_curvature"] = self.reference_values['curvature']['mean']
        
        # Add demographic information
        metrics['age'] = 50
        metrics['sex'] = 0
        
        return metrics
    
    def create_region_mask_2d(self, shape, region):
        """Create a mask for a region in 2D"""
        mask = np.zeros(shape)
        if region == 'frontal':
            mask[:shape[0]//2, :shape[1]//3] = 1
        elif region == 'parietal':
            mask[:shape[0]//2, shape[1]//3:2*shape[1]//3] = 1
        elif region == 'temporal':
            mask[shape[0]//2:, :shape[1]//3] = 1
        elif region == 'occipital':
            mask[shape[0]//2:, 2*shape[1]//3:] = 1
        elif region == 'cingulate':
            midline = shape[0]//2
            mask[midline-5:midline+5, shape[1]//4:3*shape[1]//4] = 1
        elif region == 'insula':
            mask[shape[0]//3:2*shape[0]//3, shape[1]//3:2*shape[1]//3] = 1
        elif region == 'cerebellum':
            mask[3*shape[0]//4:, 2*shape[1]//3:] = 1
        return mask
    
    def create_region_mask_3d(self, shape, region):
        """Create a mask for a region in 3D"""
        mask = np.zeros(shape)
        if region == 'frontal':
            mask[:, :shape[1]//3, shape[2]//3:2*shape[2]//3] = 1
        elif region == 'parietal':
            mask[:, shape[1]//3:2*shape[1]//3, 2*shape[2]//3:] = 1
        elif region == 'temporal':
            mask[:, shape[1]//3:2*shape[1]//3, :shape[2]//3] = 1
        elif region == 'occipital':
            mask[:, 2*shape[1]//3:, shape[2]//3:2*shape[2]//3] = 1
        elif region == 'cingulate':
            midline = shape[0]//2
            mask[midline-5:midline+5, shape[1]//4:3*shape[1]//4, 2*shape[2]//3:] = 1
        elif region == 'insula':
            mask[shape[0]//3:2*shape[0]//3, shape[1]//3:2*shape[1]//3, shape[2]//3:2*shape[2]//3] = 1
        elif region == 'cerebellum':
            mask[:, 2*shape[1]//3:, :shape[2]//3] = 1
        return mask
    
    def estimate_thickness_2d(self, region_data, region_mask):
        """Estimate cortical thickness for a region in 2D"""
        if np.sum(region_mask) == 0:
            return self.reference_values['thickness']['mean']
        
        try:
            # Calculate gradient for 2D
            grad_y, grad_x = np.gradient(region_data)
            gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            
            # Extract values where mask is positive
            mask_indices = np.where(region_mask > 0)
            if len(mask_indices[0]) > 0:
                valid_gradients = gradient_magnitude[mask_indices[0], mask_indices[1]]
                if len(valid_gradients) > 0:
                    thickness = np.mean(valid_gradients) * 0.1  # Scale factor
                else:
                    thickness = self.reference_values['thickness']['mean']
            else:
                thickness = self.reference_values['thickness']['mean']
            
            # Apply sanity check
            if not np.isfinite(thickness) or thickness < 0.5 or thickness > 5.0:
                thickness = self.reference_values['thickness']['mean']
            
            return thickness
        except Exception as e:
            print(f"Error in 2D thickness estimation: {e}")
            return self.reference_values['thickness']['mean']
    
    def estimate_thickness_3d(self, region_data, region_mask):
        """Estimate cortical thickness for a region in 3D"""
        if np.sum(region_mask) == 0:
            return self.reference_values['thickness']['mean']
        
        try:
            # Calculate 3D gradient
            grad_z, grad_y, grad_x = np.gradient(region_data)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            
            # Extract values where mask is positive
            mask_indices = np.where(region_mask > 0)
            if len(mask_indices[0]) > 0:
                valid_gradients = gradient_magnitude[mask_indices[0], mask_indices[1], mask_indices[2]]
                if len(valid_gradients) > 0:
                    thickness = np.mean(valid_gradients) * 0.1  # Scale factor
                else:
                    thickness = self.reference_values['thickness']['mean']
            else:
                thickness = self.reference_values['thickness']['mean']
            
            # Apply sanity check
            if not np.isfinite(thickness) or thickness < 0.5 or thickness > 5.0:
                thickness = self.reference_values['thickness']['mean']
            
            return thickness
        except Exception as e:
            print(f"Error in 3D thickness estimation: {e}")
            return self.reference_values['thickness']['mean']
    
    def estimate_surface_area_2d(self, region_mask, voxel_dims):
        """Estimate surface area for a region in 2D"""
        if np.sum(region_mask) == 0:
            return self.reference_values['area']['mean']
        
        try:
            # Find boundary voxels
            eroded = ndimage.binary_erosion(region_mask)
            # Convert to boolean arrays to avoid type mismatch in bitwise operation
            region_mask_bool = region_mask.astype(bool)
            eroded_bool = eroded.astype(bool)
            boundary = np.logical_and(region_mask_bool, ~eroded_bool)
            boundary_voxels = np.sum(boundary)
            
            # Calculate area
            if len(voxel_dims) >= 2:
                voxel_area = voxel_dims[0] * voxel_dims[1]
            else:
                voxel_area = 1.0
            
            area = boundary_voxels * voxel_area
            
            # Apply sanity check
            if not np.isfinite(area) or area < 100 or area > 5000:
                area = self.reference_values['area']['mean']
            
            return area
        except Exception as e:
            print(f"Error in 2D surface area estimation: {e}")
            return self.reference_values['area']['mean']
    
    def estimate_surface_area_3d(self, region_mask, voxel_dims):
        """Estimate surface area for a region in 3D"""
        if np.sum(region_mask) == 0:
            return self.reference_values['area']['mean']
        
        try:
            # Find boundary voxels
            eroded = ndimage.binary_erosion(region_mask)
            # Convert to boolean arrays to avoid type mismatch in bitwise operation
            region_mask_bool = region_mask.astype(bool)
            eroded_bool = eroded.astype(bool)
            boundary = np.logical_and(region_mask_bool, ~eroded_bool)
            boundary_voxels = np.sum(boundary)
            
            # Calculate area
            if len(voxel_dims) >= 3:
                voxel_area = voxel_dims[0] * voxel_dims[1]  # Simplified for 3D
            else:
                voxel_area = 1.0
            
            area = boundary_voxels * voxel_area
            
            # Apply sanity check
            if not np.isfinite(area) or area < 100 or area > 5000:
                area = self.reference_values['area']['mean']
            
            return area
        except Exception as e:
            print(f"Error in 3D surface area estimation: {e}")
            return self.reference_values['area']['mean']
    
    def estimate_curvature_2d(self, region_data, region_mask):
        """Estimate curvature for a region in 2D"""
        if np.sum(region_mask) == 0:
            return self.reference_values['curvature']['mean']
        
        try:
            # Calculate 2D Laplacian
            laplacian = ndimage.laplace(region_data)
            
            # Extract values where mask is positive
            mask_indices = np.where(region_mask > 0)
            if len(mask_indices[0]) > 0:
                valid_laplacians = laplacian[mask_indices[0], mask_indices[1]]
                if len(valid_laplacians) > 0:
                    curvature = np.mean(np.abs(valid_laplacians)) * 0.01  # Scale factor
                else:
                    curvature = self.reference_values['curvature']['mean']
            else:
                curvature = self.reference_values['curvature']['mean']
            
            # Apply sanity check
            if not np.isfinite(curvature) or curvature < 0.01 or curvature > 0.5:
                curvature = self.reference_values['curvature']['mean']
            
            return curvature
        except Exception as e:
            print(f"Error in 2D curvature estimation: {e}")
            return self.reference_values['curvature']['mean']
    
    def estimate_curvature_3d(self, region_data, region_mask):
        """Estimate curvature for a region in 3D"""
        if np.sum(region_mask) == 0:
            return self.reference_values['curvature']['mean']
        
        try:
            # Calculate 3D Laplacian
            laplacian = ndimage.laplace(region_data)
            
            # Extract values where mask is positive
            mask_indices = np.where(region_mask > 0)
            if len(mask_indices[0]) > 0:
                valid_laplacians = laplacian[mask_indices[0], mask_indices[1], mask_indices[2]]
                if len(valid_laplacians) > 0:
                    curvature = np.mean(np.abs(valid_laplacians)) * 0.01  # Scale factor
                else:
                    curvature = self.reference_values['curvature']['mean']
            else:
                curvature = self.reference_values['curvature']['mean']
            
            # Apply sanity check
            if not np.isfinite(curvature) or curvature < 0.01 or curvature > 0.5:
                curvature = self.reference_values['curvature']['mean']
            
            return curvature
        except Exception as e:
            print(f"Error in 3D curvature estimation: {e}")
            return self.reference_values['curvature']['mean']
    
    def extract_regional_metrics(self, subject_id, group):
        """Extract regional metrics from a real brain image"""
        # Find the subject's brain image - more flexible pattern matching
        possible_patterns = [
            f"{subject_id}*brain*.nii*",
            f"sub-{subject_id}*brain*.nii*",
            f"{subject_id}*.nii*",
            f"sub-{subject_id}*.nii*"
        ]
        
        brain_files = []
        for pattern in possible_patterns:
            brain_files.extend(list(self.data_dir.glob(pattern)))
            if brain_files:  # Stop at first successful pattern
                break
        
        if not brain_files:
            print(f"No brain image found for subject {subject_id}")
            return self.get_default_metrics(subject_id, group)
        
        # Use the first found image
        brain_file = brain_files[0]
        print(f"Processing {brain_file}")
        
        try:
            # Extract metrics from the actual image
            metrics = self.extract_metrics_from_image(brain_file, subject_id, group)
            return metrics
            
        except Exception as e:
            print(f"Error processing {brain_file}: {e}")
            return self.get_default_metrics(subject_id, group)

def normalize_by_icv(df, metrics_to_normalize=None):
    """Normalize regional volumes by intracranial volume."""
    if metrics_to_normalize is None:
        # Default: normalize area measurements
        metrics_to_normalize = [col for col in df.columns if 'area' in col]
    
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
    """Calculate laterality indices for paired regions."""
    # Identify paired regions (left/right)
    lh_regions = [col for col in df.columns if 'Left' in col]
    
    for lh_region in lh_regions:
        # Construct matching right hemisphere region name
        rh_region = lh_region.replace('Left', 'Right')
        
        if rh_region in df.columns:
            # Calculate laterality index: (L-R)/(L+R) * 100
            lat_idx_name = f"lat_idx_{lh_region.replace('Left_', '')}"
            denominator = df[lh_region] + df[rh_region]
            # Avoid division by zero
            denominator = denominator.replace(0, np.nan)
            df[lat_idx_name] = ((df[lh_region] - df[rh_region]) / denominator) * 100
    
    return df

def run_group_comparisons(df, group_col='group', covariates=None):
    """Run statistical comparisons between groups with covariates."""
    if covariates is None:
        covariates = ['age', 'sex', 'icv']
    
    # Filter covariates that actually exist in the dataframe
    covariates = [cov for cov in covariates if cov in df.columns]
    
    # Select metric columns (excluding covariates, subject_id, etc.)
    metric_cols = [col for col in df.columns if any(
        term in col for term in ['area', 'thickness', 'curvature', 'lat_idx'])]
    
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

def process_cohort(data_dir, demographics_file, output_dir, atlas="harvard_oxford"):
    """Process a cohort of subjects and extract surface-based metrics using atlas ROIs."""
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
    
    # Initialize surface morphometry analyzer
    analyzer = SurfaceMorphometry(atlas=atlas)
    
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
                str(output_dir_path / f"{subject_id}_metrics.csv")
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
    
    # Normalize area measurements by ICV
    metrics_df = normalize_by_icv(metrics_df)
    
    # Calculate laterality indices
    metrics_df = calculate_laterality_indices(metrics_df)
    
    # Save raw metrics
    metrics_df.to_csv(output_dir_path / 'surface_metrics.csv', index=False)
    print(f"Saved raw metrics to {output_dir}/surface_metrics.csv")
    
    # Run group comparisons if group column exists
    if 'group' in metrics_df.columns and len(metrics_df['group'].unique()) > 1:
        # Standard analysis with all subjects
        results_df = run_group_comparisons(metrics_df)
        results_df.to_csv(output_dir_path / 'group_comparisons.csv', index=False)
        print(f"Saved group comparison results to {output_dir}/group_comparisons.csv")
        
        # Create summary tables by region type
        for metric_type in ['thickness', 'area_norm', 'curvature']:
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

def create_visualizations(metrics_df, results_df, output_dir):
    """Create visualizations of the results."""
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
                if metric in metrics_df.columns:
                    try:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x='group', y=metric, data=metrics_df)
                        sns.stripplot(x='group', y=metric, data=metrics_df, size=4, color='.3', alpha=0.6)
                        plt.title(f'Distribution of {metric} by group')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(figures_dir / f'boxplot_{metric.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"Error creating boxplot for {metric}: {e}")
                        plt.close()
        
        # 2. Brain map of significant regions (if nilearn is available)
        try:
            if len(results_df) > 0:
                sig_results = results_df[results_df['fdr_p_value'] < 0.05]
                if len(sig_results) > 0:
                    # Get atlas
                    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
                    atlas_img = nib.load(atlas.filename)
                    
                    # Create a map of significant p-values
                    p_values = np.ones(len(atlas.labels))
                    
                    for _, row in sig_results.iterrows():
                        metric = row['metric']
                        # Extract region ID from metric name
                        if 'region_id' in metric:
                            region_id = int(metric.split('_')[-1])
                            p_values[region_id] = row['fdr_p_value']
                    
                    # Plot brain map
                    plt.figure(figsize=(15, 10))
                    plotting.plot_roi(atlas_img, title='Significant brain regions')
                    plt.savefig(figures_dir / 'significant_regions.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
        except Exception as e:
            print(f"Error creating brain map: {e}")
            plt.close()
        
        # 3. Correlation matrix of significant metrics
        if len(results_df) > 0:
            sig_metrics = results_df[results_df['fdr_p_value'] < 0.05]['metric'].unique()
            available_metrics = [m for m in sig_metrics if m in metrics_df.columns]
            
            if len(available_metrics) > 1:
                try:
                    corr_data = metrics_df[available_metrics].select_dtypes(include=[np.number])
                    if not corr_data.empty:
                        corr_matrix = corr_data.corr()
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                        plt.title('Correlation Matrix of Significant Metrics')
                        plt.tight_layout()
                        plt.savefig(figures_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
                        plt.close()
                except Exception as e:
                    print(f"Error creating correlation matrix: {e}")
                    plt.close()
        
        # 4. Summary plot of effect sizes
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
        
        # 5. Summary statistics table
        try:
            summary_stats = []
            numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['subject_id']:
                    stats_dict = {
                        'metric': col,
                        'mean': metrics_df[col].mean(),
                        'std': metrics_df[col].std(),
                        'min': metrics_df[col].min(),
                        'max': metrics_df[col].max(),
                        'n_valid': metrics_df[col].count()
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

def main():
    parser = argparse.ArgumentParser(description="Surface-based morphometry analysis using atlas ROIs")
    parser.add_argument('--data_dir', required=True, help='Directory containing registered T1 image data')
    parser.add_argument('--subjects_file', required=True, help='CSV file with subject IDs and groups')
    parser.add_argument('--output', default='./surface_morphometry_results', 
                        help='Output directory for results')
    parser.add_argument('--atlas', default='harvard_oxford', 
                        choices=['harvard_oxford', 'aal', 'schaefer'],
                        help='Brain atlas to use for region definitions')
    
    args = parser.parse_args()
    
    print("Surface-Based Morphometry Analysis with Atlas ROIs")
    print("================================================")
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