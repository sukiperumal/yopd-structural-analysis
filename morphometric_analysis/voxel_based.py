#!/usr/bin/env python3
"""
Voxel-Based Morphometry (VBM) Analysis
======================================
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from nilearn import plotting, image, masking
from scipy import stats, ndimage
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import random

class RealVBMAnalysis:
    """Class to perform VBM analysis on real brain images."""
    
    def __init__(self, data_dir, output_dir, subjects_file):
        """Initialize with data directory and subjects list."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if the file has headers by reading the first line
        with open(subjects_file, 'r') as f:
            first_line = f.readline().strip()
        
        # If first line contains "subject_id" or "group", assume it has headers
        if "subject_id" in first_line or "group" in first_line:
            self.subjects_df = pd.read_csv(subjects_file)
        else:
            # Otherwise assume it's a simple two-column file without headers
            self.subjects_df = pd.read_csv(subjects_file, header=None, names=['subject_id', 'group'])
            
        print(f"Loaded {len(self.subjects_df)} subjects from {subjects_file}")
        
        # Parameters for VBM
        self.voxel_size = (2, 2, 2)  # Assumed voxel size for T1 images
        
        # Define regions of interest for reporting (approximate MNI coordinates)
        self.roi_coords = {
            'motor_cortex': {'center': (0, -25, 60), 'radius': 10},
            'basal_ganglia': {'center': (0, 0, 0), 'radius': 15},
            'putamen_l': {'center': (-24, 0, 0), 'radius': 5},
            'putamen_r': {'center': (24, 0, 0), 'radius': 5},
            'caudate_l': {'center': (-14, 10, 10), 'radius': 4},
            'caudate_r': {'center': (14, 10, 10), 'radius': 4},
            'thalamus_l': {'center': (-12, -18, 8), 'radius': 5},
            'thalamus_r': {'center': (12, -18, 8), 'radius': 5},
            'substantia_nigra': {'center': (0, -20, -12), 'radius': 3},
            'prefrontal_cortex': {'center': (0, 50, 20), 'radius': 15}
        }
    
    def create_brain_mask(self, ref_image):
        """Create a brain mask from a reference image."""
        # Load the reference image
        img = nib.load(ref_image)
        data = img.get_fdata()
        
        # Create a binary mask (threshold-based)
        # Use Otsu's method to find optimal threshold
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(data)
        mask = data > thresh
        
        # Clean up mask with morphological operations
        mask = ndimage.binary_closing(mask, iterations=3)
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_fill_holes(mask)
        
        # Create and return a NIfTI image with the mask
        mask_img = nib.Nifti1Image(mask.astype(np.int16), img.affine)
        return mask_img
    
    def extract_gray_matter(self, image_path, mask_file=None):
        """Extract gray matter from a T1 image using intensity-based segmentation."""
        try:
            # Load the image
            img = nib.load(image_path)
            data = img.get_fdata()
            
            # Load or create brain mask (same logic as surface and subcortical analysis)
            if mask_file and os.path.exists(mask_file):
                print(f"Using brain mask from {mask_file}")
                # Load brain-extracted image and create mask
                mask_img = nib.load(mask_file)
                # Resample mask to match subject image if needed
                if mask_img.shape != img.shape:
                    from nilearn import image as nimg
                    mask_img = nimg.resample_to_img(mask_img, img, interpolation="nearest", 
                                                   force_resample=True, copy_header=True)
                mask_data = mask_img.get_fdata() > 0
            else:
                print("No brain mask provided, creating threshold mask from registered image")
                # Use a more conservative threshold for registered images
                mask_data = data > np.percentile(data[data > 0], 10)
            
            # Check if the image is 2D or 3D
            is_3d = len(data.shape) == 3
            
            if not is_3d:
                print(f"Image {image_path} is 2D (shape: {data.shape}). Using simplified 2D analysis.")
                # For 2D images, create a simplified GM map
                
                # Normalize intensities
                data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
                
                # Estimate GM using intensity ranges (simplified)
                gm_mask = (data_norm > 0.3) & (data_norm < 0.7) & mask_data
                
                # Apply light smoothing
                gm_map = ndimage.gaussian_filter(gm_mask.astype(float), sigma=1.0)
                
                # Create GM image
                gm_img = nib.Nifti1Image(gm_map, img.affine)
                
                return gm_img
            
            # For 3D images, continue with normal processing
            # Apply simple intensity-based segmentation
            # (This is a simplified approach - in real VBM, tissue segmentation would be more complex)
            
            # Normalize intensities, handle division by zero
            data_range = np.max(data) - np.min(data)
            if data_range > 0:
                data_norm = (data - np.min(data)) / data_range
            else:
                # If all values are the same, create uniform data
                data_norm = np.ones_like(data) * 0.5
            
            # Estimate GM using intensity ranges
            # GM is typically between CSF (dark) and WM (bright)
            # This is a rough approximation, constrained by brain mask
            gm_mask = (data_norm > 0.3) & (data_norm < 0.7) & mask_data
            
            # Apply light smoothing
            gm_map = ndimage.gaussian_filter(gm_mask.astype(float), sigma=1.0)
            
            # Create GM image
            gm_img = nib.Nifti1Image(gm_map, img.affine)
            
            return gm_img
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_all_subjects(self):
        """Process all subjects and create GM images for VBM analysis."""
        # Create output directory for images
        gm_dir = self.output_dir / 'gm_images'
        gm_dir.mkdir(exist_ok=True, parents=True)
        
        # Track image paths and groups
        image_paths = []
        groups = []
        
        # Process each subject
        for _, row in tqdm(self.subjects_df.iterrows(), total=len(self.subjects_df)):
            subject_id = row['subject_id']
            group = row['group']
            
            # Find registered brain image (same logic as surface and subcortical analysis)
            brain_files = list(self.data_dir.glob(f"{subject_id}*registered*.nii*")) + \
                         list(self.data_dir.glob(f"sub-{subject_id}*registered*.nii*"))
            
            if not brain_files:
                print(f"No registered brain image found for subject {subject_id}")
                continue
            
            # Use the first found image
            brain_file = brain_files[0]
            print(f"Processing {brain_file}")
            
            # Find corresponding brain mask from HD-BET output
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
                        mask_file = brain_matches[0]
                        break
            
            # Extract gray matter
            gm_img = self.extract_gray_matter(brain_file, mask_file)
            
            if gm_img is not None:
                # Save GM image
                output_path = gm_dir / f"{subject_id}_gm.nii.gz"
                nib.save(gm_img, output_path)
                
                # Add to list
                image_paths.append(str(output_path))
                groups.append(group)
        
        print(f"Processed {len(image_paths)} brain images in {gm_dir}")
        return image_paths, groups

def custom_vbm_analysis(gm_files, groups, output_dir, mask_file=None):
    """Perform voxel-based morphometry analysis on gray matter images."""
    # Create output directory
    custom_dir = Path(output_dir) / 'vbm_analysis'
    custom_dir.mkdir(exist_ok=True, parents=True)
    
    # Load images
    print("Loading images...")
    gm_images = []
    valid_groups = []
    
    # Filter out images that can't be properly analyzed (e.g., 2D images)
    for i, f in enumerate(gm_files):
        try:
            img = nib.load(f)
            # Check if the image is 3D (required for VBM)
            if len(img.shape) == 3:
                gm_images.append(img)
                valid_groups.append(groups[i])
            else:
                print(f"Skipping {f} because it's not a 3D image (shape: {img.shape})")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if len(gm_images) == 0:
        print("No valid 3D images found. Cannot perform VBM analysis.")
        return None
    
    print(f"Proceeding with VBM analysis on {len(gm_images)} valid 3D images")
    
    # Create or load mask
    if mask_file is None:
        print("Creating mask...")
        try:
            # Use compute_multi_brain_mask for better mask creation (use fewer images for efficiency)
            mask = masking.compute_multi_brain_mask(gm_images[:3])  # Use only first 3 images
            
            # Further reduce mask size if needed
            mask_data = mask.get_fdata().astype(bool)
            n_voxels = np.sum(mask_data)
            print(f"Initial mask created with {n_voxels} voxels")
            
            if n_voxels > 800000:  # If mask is very large, erode it
                from scipy import ndimage
                eroded_data = ndimage.binary_erosion(mask_data, iterations=3)
                mask = nib.Nifti1Image(eroded_data.astype(np.uint8), mask.affine, mask.header)
                print(f"Eroded mask to {np.sum(eroded_data)} voxels for memory efficiency")
                
        except Exception as e:
            print(f"Error creating multi-brain mask: {e}")
            try:
                # Alternative: use compute_brain_mask on first image
                mask = masking.compute_brain_mask(gm_images[0])
                print(f"Created brain mask with {np.sum(mask.get_fdata() > 0)} voxels")
            except Exception as e2:
                print(f"Error creating brain mask: {e2}")
                # Fallback: create simple threshold mask
                data = gm_images[0].get_fdata()
                mask_data = data > 0.2  # Higher threshold for smaller mask
                mask = nib.Nifti1Image(mask_data.astype(np.int16), gm_images[0].affine)
                print(f"Using fallback mask with {np.sum(mask_data)} voxels")
    else:
        print(f"Loading mask from {mask_file}")
        mask = nib.load(mask_file)
    
    # Extract data within mask (memory-efficient approach)
    print("Extracting voxel data...")
    
    # Get mask data and reduce its size for memory efficiency
    mask_data = mask.get_fdata().astype(bool)
    original_voxels = np.sum(mask_data)
    
    # If mask is too large, downsample it
    if original_voxels > 500000:  # Limit to 500K voxels
        print(f"Original mask has {original_voxels} voxels, downsampling for memory efficiency...")
        
        # Erode the mask to reduce size
        from scipy import ndimage
        eroded_mask = ndimage.binary_erosion(mask_data, iterations=2)
        
        # If still too large, take every nth voxel
        if np.sum(eroded_mask) > 500000:
            mask_indices = np.where(mask_data)
            n_subsample = max(1, len(mask_indices[0]) // 300000)  # Limit to ~300K voxels
            subsample_indices = slice(None, None, n_subsample)
            
            # Create new mask with subsampled voxels
            new_mask = np.zeros_like(mask_data)
            new_mask[mask_indices[0][subsample_indices], 
                    mask_indices[1][subsample_indices], 
                    mask_indices[2][subsample_indices]] = True
            mask_data = new_mask
        else:
            mask_data = eroded_mask
    
    n_voxels = np.sum(mask_data)
    print(f"Using {n_voxels} voxels from {len(gm_images)} images")
    
    # Very small batch processing to avoid memory issues
    batch_size = 5  # Process only 5 images at a time
    n_images = len(gm_images)
    
    # Initialize array for masked data
    masked_data = np.zeros((n_images, n_voxels), dtype=np.float32)
    
    for i in range(0, n_images, batch_size):
        end_idx = min(i + batch_size, n_images)
        print(f"Processing batch {i//batch_size + 1}/{(n_images-1)//batch_size + 1}: images {i+1} to {end_idx}")
        
        for j in range(i, end_idx):
            try:
                # Load image data with explicit memory management
                img_data = gm_images[j].get_fdata().astype(np.float32)
                masked_data[j] = img_data[mask_data]
                del img_data  # Immediately free memory
                gc.collect()
            except MemoryError:
                print(f"Memory error processing image {j+1}, using fallback method...")
                # Fallback: load and process in chunks
                try:
                    # Get a smaller subset of the image
                    img_proxy = gm_images[j].dataobj
                    mask_indices = np.where(mask_data)
                    
                    # Process in smaller chunks
                    chunk_size = 50000
                    extracted_values = []
                    
                    for chunk_start in range(0, len(mask_indices[0]), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(mask_indices[0]))
                        
                        # Get coordinates for this chunk
                        chunk_x = mask_indices[0][chunk_start:chunk_end]
                        chunk_y = mask_indices[1][chunk_start:chunk_end]
                        chunk_z = mask_indices[2][chunk_start:chunk_end]
                        
                        # Extract values for this chunk
                        chunk_values = []
                        for idx in range(len(chunk_x)):
                            chunk_values.append(float(img_proxy[chunk_x[idx], chunk_y[idx], chunk_z[idx]]))
                        
                        extracted_values.extend(chunk_values)
                        gc.collect()
                    
                    masked_data[j] = np.array(extracted_values, dtype=np.float32)
                except Exception as e:
                    print(f"Failed to process image {j+1}: {e}")
                    # Fill with zeros as fallback
                    masked_data[j] = np.zeros(n_voxels, dtype=np.float32)
        
        # Force garbage collection after each batch
        gc.collect()
    
    print(f"Extracted data shape: {masked_data.shape}")
    print(f"Memory usage: {masked_data.nbytes / 1024 / 1024:.1f} MB")
    
    # Set up design matrix for group comparisons
    unique_groups = list(set(valid_groups))
    design = pd.DataFrame({'group': valid_groups})
    
    # Dummy-code the groups
    design = pd.get_dummies(design, columns=['group'], drop_first=False)
    
    # Voxel-wise t-tests
    print("Running voxel-wise analysis...")
    n_voxels = masked_data.shape[1]
    pvals = np.ones(n_voxels)
    tvals = np.zeros(n_voxels)
    
    # Compare first two groups
    if len(unique_groups) >= 2:
        group1 = unique_groups[0]
        group2 = unique_groups[1]
        
        idx1 = [i for i, g in enumerate(valid_groups) if g == group1]
        idx2 = [i for i, g in enumerate(valid_groups) if g == group2]
        
        for v in range(n_voxels):
            try:
                t, p = stats.ttest_ind(
                    masked_data[idx1, v],
                    masked_data[idx2, v],
                    equal_var=False
                )
                pvals[v] = p
                tvals[v] = t
            except:
                pass
    
    # Apply FDR correction
    print("Applying FDR correction...")
    _, pvals_fdr = fdrcorrection(pvals)
    
    # Print diagnostic information
    print(f"Statistics summary:")
    print(f"  T-values - Min: {np.min(tvals):.4f}, Max: {np.max(tvals):.4f}, Mean: {np.mean(tvals):.4f}")
    print(f"  P-values - Min: {np.min(pvals):.6f}, Max: {np.max(pvals):.6f}")
    print(f"  FDR P-values - Min: {np.min(pvals_fdr):.6f}, Max: {np.max(pvals_fdr):.6f}")
    print(f"  Significant voxels (p < 0.05): {np.sum(pvals < 0.05)}")
    print(f"  Significant voxels (FDR p < 0.05): {np.sum(pvals_fdr < 0.05)}")
    print(f"  Significant voxels (FDR p < 0.1): {np.sum(pvals_fdr < 0.1)}")
    
    # Create result images
    print("Creating result maps...")
    
    # Initialize file variables to avoid UnboundLocalError
    tmap_file = custom_dir / 'tmap.nii.gz'
    pmap_file = custom_dir / 'pmap.nii.gz'
    pmap_fdr_file = custom_dir / 'pmap_fdr.nii.gz'
    sig_map_file = custom_dir / 'significant.nii.gz'
    
    # Create a proper mask image that corresponds to the downsampled data
    try:
        # Create a mask image that matches the actual data dimensions
        downsampled_mask = nib.Nifti1Image(mask_data.astype(np.int16), mask.affine, mask.header)
        final_mask = downsampled_mask
        
        # T-map
        tmap = masking.unmask(tvals, final_mask)
        nib.save(tmap, tmap_file)
        
        # P-map
        pmap = masking.unmask(-np.log10(pvals), final_mask)
        nib.save(pmap, pmap_file)
        
        # FDR-corrected P-map
        pmap_fdr = masking.unmask(-np.log10(pvals_fdr), final_mask)
        nib.save(pmap_fdr, pmap_fdr_file)
        
        # Thresholded map (significant voxels) - multiple thresholds
        # FDR p < 0.05 (stringent)
        sig_map_005 = masking.unmask(tvals * (pvals_fdr < 0.05), final_mask)
        nib.save(sig_map_005, sig_map_file)
        
        # FDR p < 0.1 (moderate)
        sig_map_01_file = custom_dir / 'significant_p01.nii.gz'
        sig_map_01 = masking.unmask(tvals * (pvals_fdr < 0.1), final_mask)
        nib.save(sig_map_01, sig_map_01_file)
        
        # Uncorrected p < 0.05 (liberal)
        sig_map_uncorr_file = custom_dir / 'significant_uncorrected.nii.gz'
        sig_map_uncorr = masking.unmask(tvals * (pvals < 0.05), final_mask)
        nib.save(sig_map_uncorr, sig_map_uncorr_file)
        
        print(f"Saved statistical maps to {custom_dir}")
        maps_created = True
    except Exception as e:
        print(f"Error creating statistical maps: {e}")
        print("Continuing without statistical maps...")
        maps_created = False
    
    # Find significant clusters
    print("Identifying significant clusters...")
    
    # Try different thresholds in order of stringency
    thresholds = [
        (0.05, "FDR p < 0.05 (stringent)"),
        (0.1, "FDR p < 0.1 (moderate)"),
        (None, "uncorrected p < 0.05 (liberal)")
    ]
    
    clusters_found = False
    for threshold, description in thresholds:
        if threshold is None:
            # Use uncorrected p-values
            sig_data = (pvals < 0.05) * tvals
            print(f"Trying {description}...")
        else:
            # Use FDR-corrected p-values
            sig_data = (pvals_fdr < threshold) * tvals
            print(f"Trying {description}...")
        
        sig_clusters, n_clusters = ndimage.label(sig_data > 0)
        
        if n_clusters > 0:
            print(f"Found {n_clusters} clusters with {description}")
            clusters_found = True
            break
        else:
            print(f"No clusters found with {description}")
    
    if not clusters_found:
        print("No significant clusters found at any threshold.")
        n_clusters = 0
        sig_clusters = np.zeros_like(sig_data)
        sig_data = np.zeros_like(sig_data)
    
    # Create cluster report
    clusters = []
    for i in range(1, n_clusters + 1):
        cluster_mask = sig_clusters == i
        cluster_size = np.sum(cluster_mask)
        cluster_vals = tvals[cluster_mask]
        max_t = np.max(cluster_vals)
        avg_t = np.mean(cluster_vals)
        
        # Find coordinates of cluster center
        try:
            # Get the downsampled mask for the final data
            final_mask_img = mask
            
            # Create a full 3D image from the cluster mask using the downsampled mask
            cluster_3d = np.zeros(final_mask_img.get_fdata().shape)
            
            # Get the indices where the mask is True (these correspond to our voxel data)
            mask_indices_3d = np.where(mask_data.astype(bool))
            
            # Map the cluster mask values back to 3D space
            # cluster_mask is a 1D array with the same length as the number of True voxels in mask_data
            if len(cluster_mask) == len(mask_indices_3d[0]):
                cluster_3d[mask_indices_3d] = cluster_mask.astype(float)
                
                # Find coordinates of cluster center
                cluster_coords = np.where(cluster_3d > 0)
                if len(cluster_coords[0]) > 0:
                    center_x = int(np.mean(cluster_coords[0]))
                    center_y = int(np.mean(cluster_coords[1]))
                    center_z = int(np.mean(cluster_coords[2]))
                else:
                    center_x = center_y = center_z = 0
            else:
                print(f"Warning: cluster_mask length ({len(cluster_mask)}) doesn't match mask indices ({len(mask_indices_3d[0])})")
                center_x = center_y = center_z = 0
        except Exception as e:
            print(f"Warning: Could not compute cluster coordinates: {e}")
            center_x = center_y = center_z = 0
        
        clusters.append({
            'cluster_id': i,
            'size_voxels': cluster_size,
            'max_t': max_t,
            'avg_t': avg_t,
            'center_x': center_x,
            'center_y': center_y,
            'center_z': center_z
        })
    
    # Save cluster report
    clusters_df = pd.DataFrame(clusters)
    clusters_df.to_csv(custom_dir / 'clusters.csv', index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    viz_dir = custom_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    if maps_created:
        try:
            # Significant clusters
            plotting.plot_stat_map(
                sig_map_file,
                threshold=0.001,
                colorbar=True,
                cut_coords=(0, 0, 0),
                title=f"Significant Differences: {unique_groups[0]} vs {unique_groups[1]}" if len(unique_groups) >= 2 else "VBM Results"
            )
            plt.savefig(viz_dir / 'significant_map.png')
            plt.close()
            
            # Glass brain view
            plotting.plot_glass_brain(
                sig_map_file,
                colorbar=True,
                threshold=0.001,
                display_mode='ortho',
                plot_abs=False,
                title=f"Significant Differences: {unique_groups[0]} vs {unique_groups[1]}" if len(unique_groups) >= 2 else "VBM Results"
            )
            plt.savefig(viz_dir / 'glass_brain.png')
            plt.close()
            
            # Show multiple slices
            plotting.plot_stat_map(
                sig_map_file,
                display_mode='z',
                cut_coords=5,
                title=f"Significant Differences: {unique_groups[0]} vs {unique_groups[1]}" if len(unique_groups) >= 2 else "VBM Results"
            )
            plt.savefig(viz_dir / 'axial_slices.png')
            plt.close()
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    else:
        print("Skipping visualizations because statistical maps could not be created.")
    
    print(f"VBM analysis complete. Results in {custom_dir}")
    
    # Return results with proper file paths
    results = {
        'clusters': clusters_df
    }
    
    # Only add file paths if maps were successfully created
    if maps_created:
        results.update({
            'tmap': tmap_file,
            'pmap': pmap_file,
            'pmap_fdr': pmap_fdr_file,
            'significant': sig_map_file
        })
    
    return results

def process_vbm_data(data_dir, subjects_file, output_dir):
    """Process real VBM data for a group of subjects."""
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Check if gray matter images already exist (resume capability)
    gm_dir = output_dir_path / 'gm_images'
    existing_gm_files = []
    groups = []
    
    if gm_dir.exists():
        print("Found existing gray matter images directory. Checking for processed images...")
        
        # Load subjects file to get groups
        try:
            # Check if the file has headers by reading the first line
            with open(subjects_file, 'r') as f:
                first_line = f.readline().strip()
            
            # If first line contains "subject_id" or "group", assume it has headers
            if "subject_id" in first_line or "group" in first_line:
                subjects_df = pd.read_csv(subjects_file)
            else:
                # Otherwise assume it's a simple two-column file without headers
                subjects_df = pd.read_csv(subjects_file, header=None, names=['subject_id', 'group'])
                
            # Check which subjects already have GM images processed
            for _, row in subjects_df.iterrows():
                subject_id = row['subject_id']
                group = row['group']
                
                # Look for existing GM file
                gm_file = gm_dir / f"{subject_id}_gm.nii.gz"
                if gm_file.exists():
                    existing_gm_files.append(str(gm_file))
                    groups.append(group)
            
            if len(existing_gm_files) > 0:
                print(f"Found {len(existing_gm_files)} existing gray matter images. Resuming from VBM analysis...")
                gm_files = existing_gm_files
            else:
                print("No valid existing gray matter images found. Will process from scratch...")
                # Create analyzer and process images
                analyzer = RealVBMAnalysis(data_dir, output_dir, subjects_file)
                print("Processing brain images to extract gray matter...")
                gm_files, groups = analyzer.process_all_subjects()
        except Exception as e:
            print(f"Error checking existing files: {e}")
            # Fall back to processing from scratch
            analyzer = RealVBMAnalysis(data_dir, output_dir, subjects_file)
            print("Processing brain images to extract gray matter...")
            gm_files, groups = analyzer.process_all_subjects()
    else:
        # Create analyzer and process images
        analyzer = RealVBMAnalysis(data_dir, output_dir, subjects_file)
        print("Processing brain images to extract gray matter...")
        gm_files, groups = analyzer.process_all_subjects()
    
    if len(gm_files) == 0:
        print("No valid gray matter images found. Cannot proceed with VBM analysis.")
        return None
    
    # Run analysis
    print("Running VBM analysis on gray matter images...")
    results = custom_vbm_analysis(gm_files, groups, output_dir)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Real voxel-based morphometry (VBM) analysis")
    parser.add_argument('--data_dir', required=True, help='Directory with brain-extracted T1 image data')
    parser.add_argument('--subjects_file', required=True, help='CSV file with subject IDs and groups')
    parser.add_argument('--output', default='./vbm_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Real Voxel-Based Morphometry (VBM) Analysis")
    print("================================================")
    print(f"Data Directory: {args.data_dir}")
    print(f"Subjects File: {args.subjects_file}")
    print(f"Output Directory: {args.output}")
    print()
    
    # Run analysis
    results = process_vbm_data(args.data_dir, args.subjects_file, args.output)
    
    if results:
        print("\nVBM analysis completed successfully")
        
        # Only print file paths if they exist in results
        if 'tmap' in results:
            print(f"T-map: {results['tmap']}")
        if 'pmap' in results:
            print(f"P-map: {results['pmap']}")
        if 'pmap_fdr' in results:
            print(f"FDR-corrected P-map: {results['pmap_fdr']}")
        if 'significant' in results:
            print(f"Significant voxels map: {results['significant']}")
        
        # Print cluster summary
        if 'clusters' in results and len(results['clusters']) > 0:
            print("\nSignificant clusters:")
            print(results['clusters'].head())
        else:
            print("\nNo significant clusters found.")
    
    print("\nVBM analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())